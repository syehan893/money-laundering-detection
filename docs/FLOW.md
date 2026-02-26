# 📊 AML Guard — System Flow Diagrams

Dokumen ini menjelaskan alur kerja sistem AML Guard secara visual menggunakan diagram.

---

## 1. 🔄 Keseluruhan Alur Sistem

Dari data mentah → model ML → API → dashboard:

```mermaid
flowchart LR
    A[/"📄 SAML-D.csv\n152K transaksi"/] --> B["🔧 augment_data.py\nInjeksi pola AML sintetis"]
    B --> C[/"📄 Augmented CSV"/]
    C --> D["⚙️ data_pipeline.py\nPreprocessing + Graph"]
    D --> E[/"📦 processed_data.pt"/]
    D --> F[/"📦 encoders.pkl"/]
    E --> G["🧠 train.py\nTraining GAT Model"]
    G --> H[/"🏆 best_model.pt"/]
    G --> I[/"📊 training_metrics.json"/]
    G -->|"Populate"| DB[("🍃 MongoDB\ntransactions\naccounts\nmetrics")]
    H --> J["🚀 main.py\nFastAPI Server"]
    F --> J
    E --> J
    DB --> J
    J -->|"REST API :8000"| K["🖥️ React Dashboard\n:5173"]
```

### Penjelasan Alur:

| Langkah | File | Input | Output | Deskripsi |
|---------|------|-------|--------|-----------|
| 1 | `augment_data.py` | SAML-D.csv | Augmented CSV | Menambahkan pola laundering sintetis |
| 2 | `data_pipeline.py` | Augmented CSV | processed_data.pt, encoders.pkl | Membersihkan data, encode fitur, bangun graph |
| 3 | `train.py` | processed_data.pt | best_model.pt + **MongoDB** | Melatih model GAT, populate MongoDB |
| 4 | `main.py` | Model + MongoDB | REST API | Query MongoDB, serve API |
| 5 | `frontend/` | REST API | Dashboard | Menampilkan data secara visual |

---

## 2. 🧠 Alur Model (EdgeGATModel)

Bagaimana model GAT memproses data graph untuk mendeteksi transaksi mencurigakan:

```mermaid
flowchart TB
    subgraph INPUT["① Input"]
        direction LR
        NF["🔵 Node Features\n7 dimensi per akun"]
        EF["🟡 Edge Features\n8 dimensi per transaksi"]
    end

    subgraph ENCODE["② GAT Encoder — Belajar representasi node"]
        NF --> G1["GAT Layer 1\n4 attention heads × 64 dims\nTotal output: 256 dims"]
        G1 --> N1["BatchNorm → ELU → Dropout"]
        N1 --> G2["GAT Layer 2\n1 head × 64 dims\nOutput: 64 dims"]
        G2 --> N2["BatchNorm → ELU → Dropout"]
        N2 --> EMB["Node Embeddings\n64 dims per node"]
    end

    subgraph EDGE["③ Edge Processing — Gabungkan info untuk tiap transaksi"]
        EMB --> SRC["Embedding Pengirim\n64 dims"]
        EMB --> DST["Embedding Penerima\n64 dims"]
        EF --> ET["Edge MLP\n8 → 32 dims"]
        SRC --> CONCAT["Concatenate\n64 + 64 + 32 = 160 dims"]
        DST --> CONCAT
        ET --> CONCAT
    end

    subgraph CLASSIFY["④ Klasifikasi — Prediksi per transaksi"]
        CONCAT --> L1["Dense Layer\n160 → 64"]
        L1 --> L2["Dense Layer\n64 → 32"]
        L2 --> L3["Dense Layer\n32 → 1"]
        L3 --> SIG["Sigmoid\n0.0 sampai 1.0"]
    end

    subgraph RESULT["⑤ Hasil"]
        SIG --> R1["🟢 Low\nP < 0.30"]
        SIG --> R2["🟡 Moderate\n0.30 ≤ P < 0.70"]
        SIG --> R3["🔴 High\n0.70 ≤ P < 0.90"]
        SIG --> R4["⛔ Critical\nP ≥ 0.90"]
    end
```

### Node Features (7 dimensi per akun):

```
┌─────────────────────────────────────────────────────────────┐
│  1. total_sent           → Total uang dikirim               │
│  2. total_received       → Total uang diterima              │
│  3. tx_count_sent        → Jumlah transaksi keluar          │
│  4. tx_count_received    → Jumlah transaksi masuk           │
│  5. unique_partners      → Jumlah mitra unik                │
│  6. foreign_currency_ratio → Rasio transaksi valas          │
│  7. cross_border_ratio   → Rasio transaksi lintas negara    │
└─────────────────────────────────────────────────────────────┘
```

### Edge Features (8 dimensi per transaksi):

```
┌─────────────────────────────────────────────────────────────┐
│  1. Payment_currency     → Mata uang pembayaran (encoded)   │
│  2. Received_currency    → Mata uang diterima (encoded)     │
│  3. Sender_bank_location → Lokasi bank pengirim (encoded)   │
│  4. Receiver_bank_location → Lokasi bank penerima (encoded) │
│  5. Payment_type         → Jenis pembayaran (encoded)       │
│  6. Amount               → Jumlah transaksi                 │
│  7. Temporal_weight      → Amount / (Δt + 1)               │
│  8. Is_laundering        → Label ground truth               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 🚀 Alur Backend (FastAPI)

Bagaimana server API memproses request:

```mermaid
flowchart TB
    subgraph BOOT["① Startup — Connect MongoDB + Load Model"]
        S1["Muat best_model.pt\nModel GAT terlatih"] --> S2["Muat encoders.pkl\nLabel encoders"]
        S2 --> S3["Muat processed_data.pt\nGraph data (untuk /graph-stats)"]
        S3 --> S4["Connect MongoDB\nmotor async client"]
    end

    subgraph ENDPOINTS["② API Endpoints — Query MongoDB"]
        S4 --> E1
        S4 --> E2
        S4 --> E3
        S4 --> E4
        S4 --> E5
        S4 --> E6

        E1["GET /api/health\n→ Status sistem + DB"]
        E2["GET /api/summary\n→ Overview + KPI\n(aggregate query)"]
        E3["GET /api/accounts\n→ Daftar akun + filter\n(find + sort + skip)"]
        E4["GET /api/accounts/:id\n→ Detail akun + graph\n(find + lookup)"]
        E5["GET /api/metrics\n→ Performa model\n(find_one)"]
        E6["GET /api/graph-stats\n→ Statistik graph\n(dari processed_data.pt)"]
    end

    subgraph PREDICT["③ Prediksi Real-time + Simpan ke MongoDB"]
        P0["POST /api/predict\n→ Input transaksi baru"]
        P0 --> P1["Encode kategorikal\ndengan encoders.pkl"]
        P1 --> P2["Forward pass\nmelalui GAT model"]
        P2 --> P3["Hitung probabilitas\nSigmoid output"]
        P3 --> P4["Klasifikasi risiko\nLow/Moderate/High/Critical"]
        P4 --> P5["Simpan ke MongoDB\ncollection: predictions"]
        P5 --> P6["Return JSON\npredictions + summary"]
    end
```

### Contoh Alur Request:

```
Browser → GET /api/accounts/8724731955

  1. Server menerima request
  2. Query MongoDB: db.accounts.find_one({account_id: "8724731955"})
  3. Query MongoDB: db.transactions.find({sender/receiver: "8724731955"})
  4. Bangun graph data (nodes + edges) dari transaksi
  5. Return JSON: { account, transaction_summary, transactions, graph }

Browser ← JSON Response (< 100ms)
```

---

## 4. 🖥️ Alur Frontend (React Dashboard)

Bagaimana halaman frontend berinteraksi dengan API:

```mermaid
flowchart TB
    subgraph APP["① App Entry"]
        M["main.jsx"] --> R["App.jsx\nSidebar + Router"]
    end

    subgraph NAV["② Navigasi — 5 Halaman"]
        R --> P1["📊 Dashboard\nRute: /"]
        R --> P2["👥 Accounts\nRute: /accounts"]
        R --> P3["🔍 Account Detail\nRute: /accounts/:id"]
        R --> P4["📈 Model Performance\nRute: /model"]
        R --> P5["⚡ Predict\nRute: /predict"]
    end

    subgraph DASH["③ Dashboard — Halaman Utama"]
        P1 --> D1["KPI Cards\nTotal akun, transaksi,\nflagged, F1 score"]
        P1 --> D2["Risk Donut Chart\nDistribusi Low/Mod/High/Critical"]
        P1 --> D3["Currency Bar Chart\nTop mata uang"]
        P1 --> D4["Flagged Table\nAkun berisiko tinggi"]
    end

    subgraph DETAIL["④ Account Detail — Investigasi"]
        P3 --> A1["Profile Card\nRisk gauge + statistik"]
        P3 --> A2["Transaction Table\nRiwayat 20 transaksi terakhir"]
        P3 --> A3["Network Graph\nCanvas force-directed\nPanah = arah aliran uang\nWarna = tingkat risiko"]
    end

    subgraph MODEL["⑤ Model Performance"]
        P4 --> M1["Metric Cards\nPrecision, Recall, F1, Accuracy"]
        P4 --> M2["Training History\nLine chart: Loss vs F1"]
        P4 --> M3["Confusion Matrix\nTP, FP, TN, FN"]
        P4 --> M4["Hyperparameters\nTabel konfigurasi model"]
    end

    subgraph API["⑥ API Layer"]
        API_JS["api.js"]
        API_JS -->|"GET /api/summary\nGET /api/graph-stats"| P1
        API_JS -->|"GET /api/accounts"| P2
        API_JS -->|"GET /api/accounts/:id"| P3
        API_JS -->|"GET /api/metrics"| P4
        API_JS -->|"POST /api/predict"| P5
    end
```

### Komponen Utama per Halaman:

```
Dashboard (/)
├── 4x KPI Cards          → Total akun, transaksi, flagged, F1 score
├── Donut Chart           → Distribusi risiko (Recharts PieChart)
├── Bar Chart             → Statistik mata uang (Recharts BarChart)
└── Flagged Table         → 10 akun paling berisiko

Accounts (/accounts)
├── Search Bar            → Cari berdasarkan Account ID
├── Category Filter       → Tab: All / Low / Moderate / High / Critical
├── Accounts Table        → Sortable, risk badges
└── Pagination            → 20 akun per halaman

Account Detail (/accounts/:id)
├── Profile Card          → Risk gauge (conic-gradient), sent/received stats
├── Transaction History   → Tabel dengan direction badges
└── Network Graph         → Canvas force-directed graph
    ├── Directed arrows   → Panah menunjukkan arah uang
    ├── Risk colors       → Merah/kuning/hijau berdasarkan risiko
    ├── Edge thickness    → Ketebalan ∝ jumlah transaksi
    └── Hover tooltips    → Detail saat mouse hover

Model Performance (/model)
├── 4x Metric Cards       → Precision, Recall, F1 Score, Accuracy
├── Training History       → Line chart (Loss + Val F1 per epoch)
├── Confusion Matrix       → 2×2 grid (TN, FP, FN, TP)
└── Hyperparameters Table  → lr, hidden_dim, heads, focal_alpha, dll.

Predict (/predict)
├── Transaction Form       → Input 10 field transaksi
├── Submit Button          → POST ke /api/predict
└── Result Card            → Probabilitas + risk category + faktor
```

---

## 5. 🔄 Alur Data End-to-End (Ringkasan)

```
    CSV File                  Graph Neural Network            MongoDB               REST API              Web Browser
    ────────                  ────────────────────            ───────               ────────              ───────────
    
    SAML-D.csv ──────────┐
    (152K transaksi)     │
                         ▼
                   ┌───────────────┐
                   │ augment_data  │ ← Injeksi 4 pola AML sintetis
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │ data_pipeline │ ← Bersihkan, encode, bangun graph
                   └───────┬───────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             processed_data   encoders
                .pt             .pkl
                    │             │
                    ▼             │
              ┌──────────┐       │
              │  train   │       │
              │  (GAT)   │       │
              └────┬─────┘       │
                   │             │
             ┌─────┼─────┐      │
             ▼     │     ▼      │
        best_model │  metrics   │
          .pt      │   .json    │
             │     │     │      │
             │     ▼     │      │
             │  ┌──────────────────┐
             │  │    MongoDB       │ ← Persistent data store
             │  │  ├─ transactions │   (152K+ transaksi)
             │  │  ├─ accounts     │   (52K+ akun)
             │  │  ├─ metrics      │   (training results)
             │  │  └─ predictions  │   (real-time predictions)
             │  └────────┬─────────┘
             │           │         │
             └─────┬─────┘         │
                   ▼               ▼
            ┌─────────────────────┐
            │   FastAPI Server    │ ← Query MongoDB for GET endpoints
            │   (port 8000)       │   Model inference for POST /predict
            └─────────┬───────────┘
                      │ JSON
                      ▼
            ┌─────────────────────┐
            │   React Dashboard   │ ← 5 halaman interaktif
            │   (port 5173)       │
            └─────────────────────┘
```

---

<div align="center">
  <sub>AML Guard — Anti-Money Laundering Detection System</sub><br>
  <sub>Muhammad Syehan</sub>
</div>
