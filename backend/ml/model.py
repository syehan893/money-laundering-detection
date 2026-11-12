"""
model_v2.py — EdgeGATModelV2 (embedding kategorikal + GAT dalam)
==================================================================
Perbaikan terhadap `model.py` (v1):

1. **Embedding untuk kategorikal.** v1 memberi makan indeks LabelEncoder
   yang sudah di-StandardScaler ke Linear layer, sehingga "Euro" dan "Yen"
   hanya berbeda pada besaran angka. v2 memakai `nn.Embedding` terpisah
   untuk mata uang, lokasi bank, dan tipe pembayaran — pasangan seperti
   "Mexican Peso -> UK pounds" bisa dipelajari sebagai kombinasi, bukan
   sebagai selisih angka.

2. **GAT 3 layer + LayerNorm + residual.** Jangkauan tetangga 3 hop
   diperlukan untuk pola berlapis (layering) yang melewati akun perantara.

3. **Interaksi eksplisit src-dst.** Selain konkatenasi, representasi edge
   menyertakan hasil perkalian elemen (src * dst) dan selisih |src - dst|,
   sehingga classifier dapat menilai *kecocokan* dua akun, bukan sekadar
   profil masing-masing.

4. **Loss sadar-subkelompok.** `WeightedFocalLoss` menerima bobot per
   sampel, dipakai untuk memberi penalti ekstra pada tipologi laundering
   langka (lihat RARE_POSITIVE_LOSS_WEIGHT).
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv


class WeightedFocalLoss(nn.Module):
    """
    Focal loss cost-sensitive dengan dukungan bobot per sampel.

    FL(p_t) = -alpha_t * cost_t * w_i * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Bobot kelas positif.
        gamma: Parameter fokus.
        fn_cost: Penalti untuk false negative (dikenakan pada sampel positif).
        fp_cost: Penalti untuk false positive (dikenakan pada sampel negatif).
        label_smoothing: Pelunakan target keras.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 2.0,
        fn_cost: float = 1.0,
        fp_cost: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets.clamp(0.0, 1.0)
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        cost_t = targets * self.fn_cost + (1 - targets) * self.fp_cost

        loss = alpha_t * cost_t * (1 - p_t) ** self.gamma * bce

        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EdgeGATModelV2(nn.Module):
    """
    GAT untuk klasifikasi level-edge dengan embedding kategorikal.

    Args:
        node_feat_dim: Jumlah fitur node numerik.
        edge_numeric_dim: Jumlah fitur edge numerik.
        cat_cardinalities: Jumlah kategori per kolom kategorikal
            (urutan: payment_currency, received_currency,
            sender_location, receiver_location, payment_type).
        cat_emb_dims: Dimensi embedding untuk tiap kolom kategorikal.
        hidden_dim: Dimensi tersembunyi per head GAT.
        num_heads: Jumlah attention head pada layer pertama.
        num_layers: Kedalaman encoder GAT (>= 2).
        dropout: Tingkat dropout.
        use_layer_norm: Pakai LayerNorm (True) atau BatchNorm (False).
        use_residual: Aktifkan residual connection antar layer GAT.
    """

    def __init__(
        self,
        node_feat_dim: int = 20,
        edge_numeric_dim: int = 16,
        cat_cardinalities: Sequence[int] = (19, 19, 24, 24, 10),
        cat_emb_dims: Sequence[int] = (12, 12, 12, 12, 8),
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.25,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_edge_attention: bool = False,
        edge_attn_dim: int = 16,
        num_typologies: int = 0,
        cat_embed_dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers harus >= 2")
        if len(cat_cardinalities) != len(cat_emb_dims):
            raise ValueError("cat_cardinalities dan cat_emb_dims harus sama panjang")

        self.dropout = dropout
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_edge_attention = use_edge_attention
        self.num_typologies = num_typologies
        self.cat_embed_dropout = cat_embed_dropout

        # ── Embedding kategorikal ──
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, cat_emb_dims)
        ])
        cat_total_dim = int(sum(cat_emb_dims))

        # ── Encoder GAT ──
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # Atensi sadar-edge: fitur transaksi ikut menentukan bobot atensi antar
        # akun. Tanpa ini, `edge_attr` sama sekali tidak menyentuh perhitungan
        # atensi -- bobot murni berasal dari embedding node, sehingga nominal dan
        # waktu transaksi tidak berpengaruh pada seberapa besar satu akun
        # memperhatikan akun lain.
        edge_dim = edge_attn_dim if use_edge_attention else None
        if use_edge_attention:
            self.edge_attn_proj = nn.Sequential(
                nn.Linear(edge_numeric_dim + cat_total_dim, edge_attn_dim),
                nn.LayerNorm(edge_attn_dim),
                nn.ELU(),
            )

        # Layer pertama multi-head (concat), layer berikutnya single-head dengan
        # dimensi tetap hidden_dim -- menjaga jejak memori pada graf 321K node
        # sekaligus membuat residual connection berlaku mulai layer ketiga.
        self.gat_layers.append(
            GATConv(node_feat_dim, hidden_dim, heads=num_heads, dropout=dropout,
                    concat=True, edge_dim=edge_dim)
        )
        for i in range(1, num_layers):
            in_dim = hidden_dim * num_heads if i == 1 else hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=1, dropout=dropout,
                        concat=False, edge_dim=edge_dim)
            )

        for i in range(num_layers):
            out_dim = hidden_dim * num_heads if i == 0 else hidden_dim
            self.gat_norms.append(
                nn.LayerNorm(out_dim) if use_layer_norm else nn.BatchNorm1d(out_dim)
            )

        # ── Transformasi fitur edge ──
        edge_in_dim = edge_numeric_dim + cat_total_dim
        edge_proj_dim = 64
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_in_dim, 96),
            nn.LayerNorm(96),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(96, edge_proj_dim),
            nn.ELU(),
        )

        # ── Classifier ──
        # [src, dst, src*dst, |src-dst|, edge] -> 4 * hidden_dim + edge_proj_dim
        classifier_in = hidden_dim * 4 + edge_proj_dim
        # Trunk dipisah dari kepala agar representasi yang sama dipakai bersama
        # oleh tugas biner dan tugas pengenalan jenis pola (multi-task).
        self.trunk = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(64, 1)
        # Kepala tambahan: memprediksi jenis pola, bukan sekadar laundering/bukan.
        # Memaksa model membentuk representasi khas tiap pola alih-alih satu
        # batas keputusan umum.
        self.typology_head = nn.Linear(64, num_typologies) if num_typologies > 0 else None

        self.edge_chunk_size = 150_000

        # Gradient checkpointing: aktivasi tiap layer GAT dihitung ulang saat
        # backward alih-alih disimpan. Pada graf 1,22 juta edge, menyimpan
        # attention message layer pertama saja butuh >1,4 GB -- melebihi sisa
        # VRAM 8 GB. Biayanya ~30% waktu training.
        self.use_grad_checkpoint = True

    def _gat_layer(
        self, gat: nn.Module, norm: nn.Module, h: torch.Tensor,
        edge_index: torch.Tensor, edge_attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Satu blok GAT: konvolusi -> normalisasi -> ELU -> dropout."""
        out = gat(h, edge_index, edge_attr=edge_attn) if edge_attn is not None \
            else gat(h, edge_index)
        out = norm(out)
        out = F.elu(out)
        return F.dropout(out, p=self.dropout, training=self.training)

    def _gat_forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Jalankan encoder GAT dengan residual bila dimensi cocok."""
        h = x
        use_ckpt = self.use_grad_checkpoint and self.training and torch.is_grad_enabled()

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if use_ckpt:
                h_out = checkpoint(
                    self._gat_layer, gat, norm, h, edge_index, edge_attn, use_reentrant=False
                )
            else:
                h_out = self._gat_layer(gat, norm, h, edge_index, edge_attn)

            if self.use_residual and i > 0 and h.shape == h_out.shape:
                h_out = h_out + h
            h = h_out
        return h

    def _cat_embeddings(self, edge_cat: torch.Tensor) -> List[torch.Tensor]:
        """
        Embedding kategorikal, dengan dropout tingkat-kolom saat training.

        Seluruh vektor embedding sebuah kolom dinolkan untuk sebagian edge yang
        dipilih acak. Berbeda dari dropout elemen biasa, ini menghapus kolom itu
        secara utuh sehingga model tidak bisa menyusun ulang isinya dari dimensi
        yang tersisa, dan terpaksa mengandalkan struktur graf serta nominal.

        Alasannya terukur: pada v4, Payment_type "Cash Deposit" menyumbang 323
        dari 480 false positive (FPR 3,24%, 28x rata-rata) meski laundering
        rate-nya normal (14,03%). Model menghafal prior kategorikal dari data
        SAML-D asli, tempat Cash Deposit memang 10x lebih sering dipakai
        pencucian uang, lalu menerapkannya pada akun yang grafnya kosong.
        """
        embs = [emb(edge_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        if self.training and self.cat_embed_dropout > 0.0:
            kept = []
            for e in embs:
                mask = torch.rand(e.shape[0], 1, device=e.device) >= self.cat_embed_dropout
                # Diskalakan seperti dropout biasa agar besaran rata-rata tetap
                kept.append(e * mask / (1.0 - self.cat_embed_dropout))
            embs = kept
        return embs

    def _edge_repr(self, edge_numeric: torch.Tensor, edge_cat: torch.Tensor) -> torch.Tensor:
        """Gabungkan fitur numerik dengan embedding kategorikal."""
        return self.edge_transform(torch.cat([edge_numeric] + self._cat_embeddings(edge_cat), dim=1))

    def encode_nodes(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hasilkan embedding node lewat encoder GAT (message passing graf penuh).

        Bila atensi sadar-edge aktif, `edge_attr` dan `edge_cat` wajib diisi
        supaya fitur transaksi ikut membentuk bobot atensi.
        """
        edge_attn = None
        if self.use_edge_attention and edge_attr is not None and edge_cat is not None:
            edge_attn = self.edge_attn_proj(
                torch.cat([edge_attr] + self._cat_embeddings(edge_cat), dim=1))
        return self._gat_forward(x, edge_index, edge_attn=edge_attn)

    def score_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_cat: torch.Tensor,
        subset: Optional[torch.Tensor] = None,
        chunked: bool = False,
        return_typology: bool = False,
    ) -> torch.Tensor:
        """
        Skor edge dari embedding node yang sudah dihitung.

        Memisahkan tahap ini dari encoding memungkinkan training hanya
        menyimpan graf autograd untuk edge yang benar-benar masuk batch —
        tanpa ini, 1,22 juta edge sekaligus melebihi VRAM 8 GB.

        Args:
            h: Embedding node dari `encode_nodes`.
            subset: Indeks edge yang ingin diskor (None = seluruh edge).
            chunked: Proses per potongan (dipakai saat inferensi no_grad).
        """
        if subset is not None:
            src = edge_index[0][subset]
            dst = edge_index[1][subset]
            attr = edge_attr[subset]
            cat = edge_cat[subset]
        else:
            src, dst, attr, cat = edge_index[0], edge_index[1], edge_attr, edge_cat

        def _score(a, b, ea, ec):
            s, d = h[a], h[b]
            e = self._edge_repr(ea, ec)
            feat = self.trunk(torch.cat([s, d, s * d, (s - d).abs(), e], dim=1))
            logit = self.classifier_head(feat).squeeze(-1)
            if return_typology and self.typology_head is not None:
                return logit, self.typology_head(feat)
            return logit

        if not chunked:
            return _score(src, dst, attr, cat)

        outs: List[torch.Tensor] = []
        type_outs: List[torch.Tensor] = []
        for i in range(0, src.shape[0], self.edge_chunk_size):
            j = min(i + self.edge_chunk_size, src.shape[0])
            result = _score(src[i:j], dst[i:j], attr[i:j], cat[i:j])
            if isinstance(result, tuple):
                outs.append(result[0])
                type_outs.append(result[1])
            else:
                outs.append(result)
        if type_outs:
            return torch.cat(outs), torch.cat(type_outs)
        return torch.cat(outs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_cat: torch.Tensor,
        subset: Optional[torch.Tensor] = None,
        chunked: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Fitur node (num_nodes, node_feat_dim).
            edge_index: Indeks edge (2, num_edges).
            edge_attr: Fitur edge numerik (num_edges, edge_numeric_dim).
            edge_cat: Indeks kategorikal edge (num_edges, num_cat_cols).
            subset: Bila diisi, hanya edge tersebut yang diskor.

        Returns:
            Logit mentah per edge.
        """
        h = self.encode_nodes(x, edge_index, edge_attr, edge_cat)
        return self.score_edges(h, edge_index, edge_attr, edge_cat, subset=subset, chunked=chunked)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Mode inferensi — mengembalikan probabilitas 0–1."""
        return torch.sigmoid(self.forward(x, edge_index, edge_attr, edge_cat))


def count_parameters(model: nn.Module) -> int:
    """Hitung parameter yang dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("EdgeGATModelV2 — ringkasan arsitektur")
    print("=" * 52)
    model = EdgeGATModelV2()
    print(f"Parameter terlatih: {count_parameters(model):,}")

    x = torch.randn(200, 20)
    edge_index = torch.randint(0, 200, (2, 1000))
    edge_attr = torch.randn(1000, 16)
    edge_cat = torch.stack([
        torch.randint(0, c, (1000,)) for c in (19, 19, 24, 24, 10)
    ], dim=1)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, edge_cat)
        probs = model.predict(x, edge_index, edge_attr, edge_cat)
    print(f"Logits: {tuple(logits.shape)} rentang [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Probs : {tuple(probs.shape)} rentang [{probs.min():.3f}, {probs.max():.3f}]")
    print("\n[OK] Sanity check lolos.")
