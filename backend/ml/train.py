"""
train_v2.py — Training GAT v2 dengan type-aware sampling
==========================================================
Perbaikan utama terhadap `train.py` (v1):

1. **Type-aware balanced sampling.** v1 memperlakukan seluruh 69.036 positif
   secara seragam, sehingga 433 contoh laundering SAML-D asli hanya menempati
   0,63% slot positif dan praktis tidak mempengaruhi gradien. v2 menjatah
   `RARE_POSITIVE_BATCH_RATIO` (35%) slot positif khusus untuk tipologi langka.

2. **Bobot loss per sampel.** Positif langka mendapat pengali
   `RARE_POSITIVE_LOSS_WEIGHT` di atas cost-sensitive focal loss.

3. **Seleksi model berbasis anggaran FPR.** Alih-alih skor constraint v1,
   model dipilih berdasarkan "berapa banyak laundering yang tertangkap bila
   kita hanya boleh salah menandai 0,1% transaksi normal" — tujuan operasional
   yang sebenarnya. Recall tipologi langka dilaporkan terpisah tiap epoch.

Catatan: `Laundering_type` dipakai SEMATA untuk bobot sampling/loss saat
training. Ia tidak pernah masuk sebagai fitur input model.

Usage:
    python -m backend.ml.train
"""

import gc
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ml.model import EdgeGATModelV2, WeightedFocalLoss, count_parameters
from backend.ml.utils import format_time, get_device, sanitize_for_json, set_seed
from backend.config import (
    CURRENCY_EMB_DIM,
    DATA_V2_PATH,
    DATA_V3_PATH,
    DATA_V4_PATH,
    DROPOUT_V2,
    EDGE_NUMERIC_DIM_V2,
    ENCODERS_V2_PATH,
    ENCODERS_V3_PATH,
    ENCODERS_V4_PATH,
    EPOCHS_V2,
    FN_COST_V2,
    FOCAL_ALPHA_V2,
    FOCAL_GAMMA_V2,
    FP_COST_V2,
    HARD_MINING_RATIO_V2,
    HARD_MINING_START_EPOCH_V2,
    HARD_NEG_TOPK_V2,
    HARD_POS_TOPK_V2,
    HIDDEN_DIM_V2,
    LABEL_SMOOTHING_V2,
    LOCATION_EMB_DIM,
    LR_V2,
    MARGIN_LOSS_WEIGHT_V2,
    MARGIN_VALUE_V2,
    MAX_FNR,
    MAX_FPR,
    METRICS_V2_PATH,
    METRICS_V3_PATH,
    METRICS_V4_PATH,
    MODEL_V2_PATH,
    MODEL_V3_PATH,
    MODEL_V4_PATH,
    NODE_FEAT_DIM_V2,
    NUM_HEADS_V2,
    NUM_LAYERS_V2,
    OVERSAMPLE_RATIO_V2,
    PATIENCE_V2,
    PAYMENT_TYPE_EMB_DIM,
    POS_SLOTS_PER_STEP_V2,
    RARE_POSITIVE_BATCH_RATIO,
    RARE_POSITIVE_LOSS_WEIGHT,
    SEED,
    DATA_V5_PATH,
    ENCODERS_V5_PATH,
    MODEL_V5_PATH,
    METRICS_V5_PATH,
    EDGE_NUMERIC_DIM_V5,
    CAT_EMBED_DROPOUT_V5,
    STEPS_PER_EPOCH_V2,
    MULTITASK_WEIGHT,
    TYPE_BALANCED_SAMPLING,
    USE_EDGE_ATTENTION,
    USE_MOTIF_FEATURES,
    USE_MULTITASK,
    USE_LAYER_NORM_V2,
    USE_RESIDUAL_V2,
    WEIGHT_DECAY_V2,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training_v2.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

torch.set_num_threads(4)

# Dataset aktif dipilih lewat --version. v3 memakai dataset yang seluruh
# tipologinya sudah terwakili (lihat backend/scripts/generate_missing_patterns.py).
PATHS = {
    "v2": {"graph": DATA_V2_PATH, "artifacts": ENCODERS_V2_PATH,
           "model": MODEL_V2_PATH, "metrics": METRICS_V2_PATH, "probs": "test_probs_v2.npy"},
    # Graf v3 berisi 1,67 juta edge (37% lebih besar dari v2). Attention layer
    # pertama menyimpan buffer sebesar edge x head x hidden_dim, sehingga jumlah
    # head diturunkan menjadi 2 agar tetap muat di VRAM 8 GB.
    "v3": {"graph": DATA_V3_PATH, "artifacts": ENCODERS_V3_PATH,
           "model": MODEL_V3_PATH, "metrics": METRICS_V3_PATH, "probs": "test_probs_v3.npy",
           "num_heads": 2},
    # v4: lima pola skripsi dengan atribut realistis, ditambah tiga teknik baru
    # (fitur motif, atensi sadar-edge, multi-task). Head tetap 2 karena atensi
    # sadar-edge menambah buffer sebesar edge x head x hidden_dim.
    "v4": {"graph": DATA_V4_PATH, "artifacts": ENCODERS_V4_PATH,
           "model": MODEL_V4_PATH, "metrics": METRICS_V4_PATH, "probs": "test_probs_v4.npy",
           "num_heads": 2, "edge_attention": True, "multitask": True},
    # v5: data sama dengan v4, tetapi ditambah fitur ketersediaan konteks dan
    # dropout embedding kategorikal -- keduanya menyasar kegagalan pada akun
    # berderajat rendah. Dilatih beberapa kali dengan seed berbeda lalu
    # probabilitasnya dirata-ratakan.
    "v5": {"graph": DATA_V5_PATH, "artifacts": ENCODERS_V5_PATH,
           "model": MODEL_V5_PATH, "metrics": METRICS_V5_PATH, "probs": "test_probs_v5.npy",
           "num_heads": 2, "edge_attention": True, "multitask": True,
           "edge_numeric_dim": EDGE_NUMERIC_DIM_V5,
           "cat_embed_dropout": CAT_EMBED_DROPOUT_V5},
}


# ─── Sampling ─────────────────────────────────────────────────────────────────
def build_type_aware_batch(
    pools: Dict[str, torch.Tensor],
    n_pos_slots: int,
    oversample_ratio: float = 4.0,
    rare_ratio: float = 0.35,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Susun indeks batch dengan jatah khusus untuk positif langka.

    Args:
        pools: Dict berisi indeks 'rare_pos', 'common_pos', dan 'neg' hasil
            `build_index_pools` (dihitung sekali agar tidak diulang tiap step).
        n_pos_slots: Jumlah slot positif dalam batch.
        oversample_ratio: Jumlah negatif per satu slot positif.
        rare_ratio: Porsi slot positif yang dijatahkan untuk positif langka.

    Returns:
        Tensor indeks edge yang sudah diacak.
    """
    rare_pos = pools["rare_pos"]
    common_pos = pools["common_pos"]
    neg_all = pools["neg"]

    if len(rare_pos) == 0 and len(common_pos) == 0:
        return neg_all[torch.randperm(len(neg_all), generator=generator)[:10_000]]

    # Bila kolam per tipologi tersedia, bagi slot positif sama rata antar tipologi
    by_type = pools.get("_by_type")
    if by_type:
        per_type = max(1, n_pos_slots // len(by_type))
        parts = []
        for members in by_type:
            pick = torch.randint(len(members), (per_type,), generator=generator)
            parts.append(members[pick])
        n_neg = min(int(n_pos_slots * oversample_ratio), len(neg_all))
        parts.append(neg_all[torch.randperm(len(neg_all), generator=generator)[:n_neg]])
        batch = torch.cat(parts)
        return batch[torch.randperm(len(batch), generator=generator)]

    n_rare_slots = int(n_pos_slots * rare_ratio) if len(rare_pos) > 0 else 0
    n_common_slots = n_pos_slots - n_rare_slots

    parts = []
    if n_rare_slots > 0:
        # Sampling dengan pengembalian: 433 contoh mengisi ~24K slot
        pick = torch.randint(len(rare_pos), (n_rare_slots,), generator=generator)
        parts.append(rare_pos[pick])
    if n_common_slots > 0 and len(common_pos) > 0:
        pick = torch.randint(len(common_pos), (n_common_slots,), generator=generator)
        parts.append(common_pos[pick])

    n_neg = min(int(n_pos_slots * oversample_ratio), len(neg_all))
    parts.append(neg_all[torch.randperm(len(neg_all), generator=generator)[:n_neg]])

    batch = torch.cat(parts)
    return batch[torch.randperm(len(batch), generator=generator)]


def build_index_pools(
    y: torch.Tensor, mask: torch.Tensor, is_rare: torch.Tensor,
    laundering_type: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pisahkan indeks train menjadi kolam negatif dan kolam positif per tipologi.

    Menyeimbangkan per tipologi, bukan sekadar langka/umum: pada dataset v3,
    tipologi lemah seperti Cycle hanya menyumbang 4.000 dari 94.000 positif,
    sehingga dengan sampling proporsional ia nyaris tidak pernah masuk batch.
    Dengan porsi yang sama rata, tiap tipologi memperoleh perhatian setara.
    """
    idx = mask.nonzero(as_tuple=True)[0]
    labels = y[idx]
    rare_sel = is_rare[idx]
    pools: Dict[str, torch.Tensor] = {
        "rare_pos": idx[(labels == 1) & rare_sel],
        "common_pos": idx[(labels == 1) & ~rare_sel],
        "neg": idx[labels == 0],
    }

    if laundering_type is not None:
        pos_idx = idx[labels == 1]
        types = laundering_type[pos_idx]
        by_type = []
        for t in torch.unique(types):
            members = pos_idx[types == t]
            if len(members) > 0:
                by_type.append(members)
        pools["_by_type"] = by_type  # type: ignore[assignment]
    return pools


def select_hard_examples(
    y: torch.Tensor,
    mask: torch.Tensor,
    probs: torch.Tensor,
    k_neg: int = 20_000,
    k_pos: int = 20_000,
) -> torch.Tensor:
    """
    Ambil contoh sulit sebagai **top-K**, bukan proporsi.

    Versi sebelumnya mengambil 35% teratas (≈150.000 negatif), terlalu menyebar:
    yang benar-benar menentukan letak threshold hanyalah beberapa ratus negatif
    berskor tertinggi. Fokus top-K membuat gradien mengarah tepat ke sana.
    """
    idx = mask.nonzero(as_tuple=True)[0]
    labels = y[idx].cpu().numpy()
    p = probs[idx].cpu().numpy()

    parts = []
    for is_pos, k in [(True, k_pos), (False, k_neg)]:
        sel = (labels == 1) if is_pos else (labels == 0)
        cand = idx[sel]
        scores = p[sel]
        if len(scores) == 0:
            continue
        k = min(max(1, k), len(scores))
        # positif tersulit = skor terendah; negatif tersulit = skor tertinggi
        order = np.argpartition(scores if is_pos else -scores, k - 1)[:k]
        parts.append(cand[torch.from_numpy(order)])

    if not parts:
        return torch.tensor([], dtype=torch.long)
    hard = torch.cat(parts)
    return hard[torch.randperm(len(hard))]


def budget_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    max_fpr: float,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Loss yang langsung mengoptimalkan "recall pada anggaran FPR tertentu".

    Focal loss menilai tiap sampel sendiri-sendiri, sehingga positif yang sudah
    berada di p≈0,93 hampir tidak menghasilkan gradien ((1-p)^gamma ≈ 0,005) —
    padahal justru merekalah yang tertinggal tepat di bawah threshold. Loss ini
    bersifat *relatif*: ia mengambil skor negatif pada batas anggaran FPR sebagai
    acuan, lalu menghukum setiap positif yang belum berada `margin` di atasnya.
    Gradiennya mendorong positif naik sekaligus menekan negatif batas itu turun.
    """
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return logits.sum() * 0.0

    neg_logits = logits[neg_mask]
    k = max(1, int(neg_logits.numel() * max_fpr))
    reference = torch.topk(neg_logits, k).values.mean()

    pos_logits = logits[pos_mask]
    pos_weights = weights[pos_mask]
    violation = torch.relu(margin - (pos_logits - reference))
    return (violation * pos_weights).sum() / pos_weights.sum().clamp(min=1.0)


def mix_hard(base: torch.Tensor, hard: torch.Tensor, ratio: float) -> torch.Tensor:
    """Ganti sebagian indeks batch dengan contoh sulit."""
    if len(hard) == 0 or ratio <= 0:
        return base
    n_hard = min(int(len(base) * ratio), len(hard))
    keep = base[torch.randperm(len(base))[: len(base) - n_hard]]
    chosen = hard[torch.randperm(len(hard))[:n_hard]]
    mixed = torch.cat([keep, chosen])
    return mixed[torch.randperm(len(mixed))]


# ─── Evaluasi ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_all_edges(
    model: EdgeGATModelV2,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """Inferensi seluruh edge (chunked, tanpa gradien). Dipakai validasi + hard mining."""
    model.eval()
    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        h = model.encode_nodes(
            tensors["x"], tensors["edge_index"], tensors["edge_attr"], tensors["edge_cat"]
        )
        logits = model.score_edges(
            h, tensors["edge_index"], tensors["edge_attr"], tensors["edge_cat"], chunked=True
        )
    return torch.sigmoid(logits.float()).cpu()


def slice_by_mask(
    probs: torch.Tensor, data, mask: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ambil (probs, labels, is_rare) untuk satu split."""
    m = mask.cpu().numpy()
    return probs.numpy()[m], data.y.numpy()[m], data.is_rare_positive.numpy()[m]


def threshold_at_fpr_budget(
    probs: np.ndarray, labels: np.ndarray, max_fpr: float = 0.001
) -> Tuple[float, Dict]:
    """
    Cari threshold tertinggi yang masih memenuhi FPR <= max_fpr, lalu laporkan FNR di titik itu.

    Ini menjawab pertanyaan operasional yang sebenarnya: "bila kita hanya boleh
    salah menandai 0,1% transaksi normal, berapa banyak laundering yang lolos?"
    """
    neg = np.sort(probs[labels == 0])[::-1]
    n_neg = len(neg)
    budget = int(n_neg * max_fpr)

    if n_neg == 0:
        thr = 0.5
    elif budget <= 0:
        thr = float(neg[0]) + 1e-6
    else:
        thr = float(neg[min(budget, n_neg - 1)])

    thr = min(max(thr, 1e-6), 1.0 - 1e-6)
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    return thr, {
        "threshold": thr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / max(tn + fp, 1),
        "fnr": fn / max(tp + fn, 1),
    }


def metrics_at(probs: np.ndarray, labels: np.ndarray, is_rare: np.ndarray, thr: float) -> Dict:
    """Confusion matrix + rincian recall untuk kelompok sintetis vs langka."""
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    rare_pos = (labels == 1) & is_rare
    synth_pos = (labels == 1) & ~is_rare
    rare_hit = int((pred[rare_pos] == 1).sum())
    synth_hit = int((pred[synth_pos] == 1).sum())

    return {
        "threshold": float(thr),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fp / max(tn + fp, 1),
        "fnr": fn / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "rare_total": int(rare_pos.sum()),
        "rare_caught": rare_hit,
        "rare_recall": rare_hit / max(int(rare_pos.sum()), 1),
        "synthetic_total": int(synth_pos.sum()),
        "synthetic_caught": synth_hit,
        "synthetic_recall": synth_hit / max(int(synth_pos.sum()), 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(version: str = "v2", seed: int = None, override: Dict = None) -> Dict:
    """
    `override` menimpa hyperparameter untuk satu kali jalan.

    Dipakai membangun ensemble beragam: anggota dengan konfigurasi berbeda
    membuat kesalahan yang berbeda pula, sehingga rata-ratanya lebih kuat
    daripada beberapa seed dengan konfigurasi identik. Kunci yang dikenali:
    fn_cost, dropout, hidden_dim, label_smoothing, focal_alpha.
    """
    paths = PATHS[version]
    seed = SEED if seed is None else seed
    ov = override or {}
    hp_fn_cost = ov.get("fn_cost", FN_COST_V2)
    hp_dropout = ov.get("dropout", DROPOUT_V2)
    hp_hidden = ov.get("hidden_dim", HIDDEN_DIM_V2)
    hp_smooth = ov.get("label_smoothing", LABEL_SMOOTHING_V2)
    hp_alpha = ov.get("focal_alpha", FOCAL_ALPHA_V2)
    hp_tag = ov.get("tag", "")
    logger.info("=" * 70)
    logger.info(f"AML Detection — GAT v2 (type-aware sampling, dataset {version})")
    logger.info("=" * 70)

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    if not os.path.exists(paths["graph"]):
        logger.error(f"  {paths['graph']} tidak ditemukan. Jalankan data_pipeline_v2 lebih dulu.")
        return {}

    # Tiap seed menulis ke berkasnya sendiri agar bisa dirata-ratakan nanti
    sufiks = hp_tag or (f"seed{seed}" if seed != SEED else "")
    if sufiks:
        paths = dict(paths)
        for k, ext in (("model", ".pt"), ("metrics", ".json")):
            paths[k] = paths[k].replace(ext, f"_{sufiks}{ext}")
        paths["probs"] = paths["probs"].replace(".npy", f"_{sufiks}.npy")

    data = torch.load(paths["graph"], weights_only=False)
    logger.info(f"  Node: {data.num_nodes:,} | Edge: {data.edge_index.shape[1]:,}")
    n_rare_train = int((data.is_rare_positive & data.train_mask).sum())
    n_pos_train = int(data.y[data.train_mask].sum())
    logger.info(f"  Positif train: {n_pos_train:,} ({n_rare_train} di antaranya tipologi langka)")

    device = get_device()
    use_amp = device.type == "cuda"

    import joblib
    artifacts = joblib.load(paths["artifacts"])
    cards = artifacts["cat_cardinalities"]

    # Multi-task: jumlah kelas jenis pola (termasuk kelas Normal)
    use_edge_attn = paths.get("edge_attention", False) and USE_EDGE_ATTENTION
    use_mt = paths.get("multitask", False) and USE_MULTITASK and hasattr(data, "laundering_type")
    n_typologies = int(data.laundering_type.max()) + 1 if use_mt else 0
    emb_dims = (CURRENCY_EMB_DIM, CURRENCY_EMB_DIM, LOCATION_EMB_DIM, LOCATION_EMB_DIM, PAYMENT_TYPE_EMB_DIM)

    model = EdgeGATModelV2(
        node_feat_dim=NODE_FEAT_DIM_V2,
        edge_numeric_dim=paths.get("edge_numeric_dim", EDGE_NUMERIC_DIM_V2),
        cat_cardinalities=cards,
        cat_emb_dims=emb_dims,
        hidden_dim=hp_hidden,
        num_heads=paths.get("num_heads", NUM_HEADS_V2),
        num_layers=NUM_LAYERS_V2,
        dropout=hp_dropout,
        use_layer_norm=USE_LAYER_NORM_V2,
        use_residual=USE_RESIDUAL_V2,
        use_edge_attention=paths.get("edge_attention", False) and USE_EDGE_ATTENTION,
        num_typologies=n_typologies,
        cat_embed_dropout=paths.get("cat_embed_dropout", 0.0),
    ).to(device)
    logger.info(f"  Parameter: {count_parameters(model):,}")
    logger.info(
        f"  Teknik aktif: motif={'ya' if USE_MOTIF_FEATURES else 'tidak'} | "
        f"atensi sadar-edge={'ya' if use_edge_attn else 'tidak'} | "
        f"multi-task={'ya (' + str(n_typologies) + ' kelas)' if n_typologies else 'tidak'}"
    )
    logger.info(f"  Kardinalitas kategorikal: {cards}")

    criterion = WeightedFocalLoss(
        alpha=hp_alpha,
        gamma=FOCAL_GAMMA_V2,
        fn_cost=hp_fn_cost,
        fp_cost=FP_COST_V2,
        label_smoothing=hp_smooth,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR_V2, weight_decay=WEIGHT_DECAY_V2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Bobot loss per edge: positif langka diberi pengali tambahan
    weights_all = torch.ones(data.y.shape[0], dtype=torch.float32)
    weights_all[data.is_rare_positive] = RARE_POSITIVE_LOSS_WEIGHT
    weights_all = weights_all.to(device)

    logger.info(
        f"\n[Training] epochs={EPOCHS_V2} patience={PATIENCE_V2} "
        f"rare_slot={RARE_POSITIVE_BATCH_RATIO:.0%} rare_w={RARE_POSITIVE_LOSS_WEIGHT} "
        f"FN_cost={hp_fn_cost} AMP={'ON' if use_amp else 'OFF'}"
    )
    logger.info(
        f"  {'Ep':>4} | {'Loss':>8} | {'thr':>6} | {'FP':>5} | {'FN':>5} | "
        f"{'FPR%':>7} | {'FNR%':>7} | {'Langka':>12} | {'Waktu':>6}"
    )
    logger.info("  " + "-" * 92)

    best_score = -1.0
    best_epoch = 0
    patience = 0
    history: List[Dict] = []
    hard_idx: Optional[torch.Tensor] = None

    ckpt_dir = os.path.join(os.path.dirname(paths["model"]), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    x_dev = data.x.to(device)
    ei_dev = data.edge_index.to(device)
    ea_dev = data.edge_attr.to(device)
    ec_dev = data.edge_cat.to(device)
    y_dev = data.y.to(device)
    tensors = {"x": x_dev, "edge_index": ei_dev, "edge_attr": ea_dev, "edge_cat": ec_dev}
    type_dev = data.laundering_type.to(device) if use_mt else None

    pools = build_index_pools(
        data.y, data.train_mask, data.is_rare_positive,
        laundering_type=getattr(data, "laundering_type", None) if TYPE_BALANCED_SAMPLING else None,
    )
    n_types = len(pools.get("_by_type") or [])
    logger.info(
        f"  Kolam sampling: langka={len(pools['rare_pos']):,} "
        f"umum={len(pools['common_pos']):,} negatif={len(pools['neg']):,}"
        + (f" | seimbang per tipologi ({n_types} tipologi)" if n_types else "")
    )
    logger.info(
        f"  Batch per step: {POS_SLOTS_PER_STEP_V2:,} positif "
        f"({int(POS_SLOTS_PER_STEP_V2 * RARE_POSITIVE_BATCH_RATIO):,} slot langka) "
        f"+ {int(POS_SLOTS_PER_STEP_V2 * OVERSAMPLE_RATIO_V2):,} negatif "
        f"x {STEPS_PER_EPOCH_V2} step/epoch"
    )

    for epoch in range(1, EPOCHS_V2 + 1):
        t0 = time.time()
        model.train()
        step_losses = []

        for _ in range(STEPS_PER_EPOCH_V2):
            optimizer.zero_grad(set_to_none=True)

            batch_idx = build_type_aware_batch(
                pools,
                n_pos_slots=POS_SLOTS_PER_STEP_V2,
                oversample_ratio=OVERSAMPLE_RATIO_V2,
                rare_ratio=RARE_POSITIVE_BATCH_RATIO,
                generator=gen,
            )
            if hard_idx is not None and epoch >= HARD_MINING_START_EPOCH_V2:
                batch_idx = mix_hard(batch_idx, hard_idx, HARD_MINING_RATIO_V2)
            bi = batch_idx.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Encoding node atas graf penuh, tetapi gradien edge hanya
                # untuk batch -- kunci agar model v2 muat di VRAM 8 GB.
                h = model.encode_nodes(x_dev, ei_dev, ea_dev, ec_dev)
                scored = model.score_edges(h, ei_dev, ea_dev, ec_dev, subset=bi,
                                           chunked=False, return_typology=use_mt)
                if use_mt:
                    logits, type_logits = scored
                else:
                    logits, type_logits = scored, None

                focal = criterion(logits, y_dev[bi], sample_weights=weights_all[bi])
                margin = budget_margin_loss(
                    logits.float(), y_dev[bi], weights_all[bi],
                    max_fpr=MAX_FPR, margin=MARGIN_VALUE_V2,
                )
                loss = focal + MARGIN_LOSS_WEIGHT_V2 * margin

                # Tugas tambahan: mengenali JENIS pola, bukan sekadar
                # laundering/bukan. Memaksa representasi khas tiap pola.
                if type_logits is not None:
                    mt = torch.nn.functional.cross_entropy(
                        type_logits.float(), type_dev[bi]
                    )
                    loss = loss + MULTITASK_WEIGHT * mt

            if use_amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step_losses.append(float(loss.item()))
            del h, logits, loss, focal, margin, scored

        scheduler.step()
        loss_value = float(np.mean(step_losses))

        # Satu inferensi penuh melayani validasi sekaligus hard mining
        all_probs = infer_all_edges(model, tensors, device, use_amp)
        if epoch >= HARD_MINING_START_EPOCH_V2 - 1:
            hard_idx = select_hard_examples(
                data.y, data.train_mask, all_probs,
                k_neg=HARD_NEG_TOPK_V2, k_pos=HARD_POS_TOPK_V2,
            )

        # ── Validasi ──
        vp, vy, vr = slice_by_mask(all_probs, data, data.val_mask)
        thr, budget_metrics = threshold_at_fpr_budget(vp, vy, max_fpr=MAX_FPR)
        vm = metrics_at(vp, vy, vr, thr)
        score = 1.0 - vm["fnr"]

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": loss_value,
            "val_threshold": thr,
            "val_fp": vm["confusion_matrix"]["fp"],
            "val_fn": vm["confusion_matrix"]["fn"],
            "val_fpr": vm["fpr"],
            "val_fnr": vm["fnr"],
            "val_rare_recall": vm["rare_recall"],
            "val_synthetic_recall": vm["synthetic_recall"],
            "val_precision": vm["precision"],
            "val_recall": vm["recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        })

        if epoch <= 5 or epoch % 5 == 0 or score > best_score:
            logger.info(
                f"  {epoch:4d} | {loss_value:8.5f} | {thr:6.3f} | "
                f"{vm['confusion_matrix']['fp']:5d} | {vm['confusion_matrix']['fn']:5d} | "
                f"{vm['fpr']*100:7.4f} | {vm['fnr']*100:7.4f} | "
                f"{vm['rare_caught']:4d}/{vm['rare_total']:<4d} ({vm['rare_recall']*100:3.0f}%) | "
                f"{format_time(elapsed)}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), paths["model"])
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "score": score},
                os.path.join(ckpt_dir, f"best_checkpoint_{version}.pt"),
            )
        else:
            patience += 1

        if patience >= PATIENCE_V2:
            logger.info(f"\n  Early stopping di epoch {epoch} (terbaik: {best_epoch})")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"\n  Skor validasi terbaik: {best_score:.6f} (epoch {best_epoch})")

    # ── Evaluasi test ──
    logger.info("\n[Evaluasi test] ...")
    model.load_state_dict(torch.load(paths["model"], map_location=device, weights_only=True))

    final_probs = infer_all_edges(model, tensors, device, use_amp)
    vp, vy, vr = slice_by_mask(final_probs, data, data.val_mask)
    thr, _ = threshold_at_fpr_budget(vp, vy, max_fpr=MAX_FPR)
    val_metrics = metrics_at(vp, vy, vr, thr)
    logger.info(f"  Threshold dari validasi: {thr:.4f}")

    tp_, ty, tr_ = slice_by_mask(final_probs, data, data.test_mask)
    test_metrics = metrics_at(tp_, ty, tr_, thr)
    cm = test_metrics["confusion_matrix"]

    logger.info("\n  == Confusion Matrix (test) ==")
    logger.info(f"    TN {cm['tn']:>8,}  |  FP {cm['fp']:>6,}")
    logger.info(f"    FN {cm['fn']:>8,}  |  TP {cm['tp']:>6,}")
    logger.info(f"  FPR {test_metrics['fpr']*100:.4f}%  FNR {test_metrics['fnr']*100:.4f}%  "
                f"Akurasi {test_metrics['accuracy']*100:.4f}%")
    logger.info(f"  Precision {test_metrics['precision']:.4f}  Recall {test_metrics['recall']:.4f}")
    logger.info(f"  Recall tipologi langka : {test_metrics['rare_caught']}/{test_metrics['rare_total']} "
                f"({test_metrics['rare_recall']*100:.1f}%)")
    logger.info(f"  Recall pola sintetis   : {test_metrics['synthetic_caught']:,}/{test_metrics['synthetic_total']:,} "
                f"({test_metrics['synthetic_recall']*100:.2f}%)")

    target_ok = test_metrics["fpr"] < MAX_FPR and test_metrics["fnr"] < MAX_FNR
    logger.info(f"\n  Target FPR<{MAX_FPR:.1%} & FNR<{MAX_FNR:.1%}: "
                f"{'TERCAPAI' if target_ok else 'BELUM tercapai'}")

    pred_bin = (tp_ >= thr).astype(int)
    output = {
        "version": version,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "optimal_threshold": float(thr),
        "threshold_selection": "fpr_budget",
        "test_metrics": {
            "precision": float(precision_score(ty, pred_bin, zero_division=0)),
            "recall": float(recall_score(ty, pred_bin, zero_division=0)),
            "f1": float(f1_score(ty, pred_bin, zero_division=0)),
        },
        "confusion_matrix": cm,
        "error_rates": {
            "fpr": test_metrics["fpr"],
            "fnr": test_metrics["fnr"],
            "tpr": test_metrics["recall"],
            "tnr": 1.0 - test_metrics["fpr"],
            "accuracy": test_metrics["accuracy"],
        },
        "breakdown": {
            "rare_total": test_metrics["rare_total"],
            "rare_caught": test_metrics["rare_caught"],
            "rare_recall": test_metrics["rare_recall"],
            "synthetic_total": test_metrics["synthetic_total"],
            "synthetic_caught": test_metrics["synthetic_caught"],
            "synthetic_recall": test_metrics["synthetic_recall"],
        },
        "val_metrics": val_metrics,
        "target_achieved": bool(target_ok),
        "hyperparameters": {
            "epochs_trained": len(history),
            "lr": LR_V2,
            "weight_decay": WEIGHT_DECAY_V2,
            "focal_alpha": hp_alpha,
            "focal_gamma": FOCAL_GAMMA_V2,
            "fn_cost": hp_fn_cost,
            "fp_cost": FP_COST_V2,
            "hidden_dim": hp_hidden,
            "num_heads": paths.get("num_heads", NUM_HEADS_V2),
            "num_layers": NUM_LAYERS_V2,
            "dropout": hp_dropout,
            "oversample_ratio": OVERSAMPLE_RATIO_V2,
            "rare_positive_batch_ratio": RARE_POSITIVE_BATCH_RATIO,
            "rare_positive_loss_weight": RARE_POSITIVE_LOSS_WEIGHT,
            "hard_neg_topk": HARD_NEG_TOPK_V2,
            "hard_pos_topk": HARD_POS_TOPK_V2,
            "margin_loss_weight": MARGIN_LOSS_WEIGHT_V2,
            "margin_value": MARGIN_VALUE_V2,
            "seed": seed,
            "use_motif_features": USE_MOTIF_FEATURES,
            "use_edge_attention": use_edge_attn,
            "use_multitask": bool(n_typologies),
            "multitask_weight": MULTITASK_WEIGHT if n_typologies else 0.0,
        },
        "training_history": sanitize_for_json(history),
    }

    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    np.save(os.path.join(os.path.dirname(paths["metrics"]), paths["probs"]), tp_)
    logger.info(f"\n  Model  : {paths['model']}")
    logger.info(f"  Metrik : {paths['metrics']}")
    logger.info("\n[OK] Training v2 selesai.")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training GAT v2.")
    parser.add_argument("--version", choices=["v2", "v3", "v4", "v5"], default="v2",
                        help="Dataset: v2, v3, v4, atau v5 (fitur konteks + ensemble).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed acak. Seed selain bawaan menulis ke berkas terpisah "
                             "sehingga beberapa run dapat dirata-ratakan menjadi ensemble.")
    args = parser.parse_args()
    main(args.version, args.seed)
