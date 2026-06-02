# 📊 AML Guard — Diagram untuk Skripsi

Dokumen ini berisi diagram-diagram yang disederhanakan untuk keperluan penulisan skripsi.

---

## 1. Arsitektur Sistem Keseluruhan

Gambaran umum arsitektur sistem AML Guard dari data hingga dashboard.

```mermaid
flowchart LR
    subgraph A["Data Layer"]
        A1["Dataset SAML-D\n(152K transaksi)"]
        A2["Augmentasi Data\n(4 pola AML sintetis)"]
    end

    subgraph B["ML Pipeline"]
        B1["Preprocessing &\nKonstruksi Graph"]
        B2["Training Model\nGAT"]
    end

    subgraph C["Application Layer"]
        C1["REST API\n(FastAPI)"]
        C2["Database\n(MongoDB)"]
        C3["Dashboard\n(React)"]
    end

    A1 --> A2 --> B1 --> B2
    B2 --> C2
    C2 --> C1
    B2 --> C1
    C1 -->|"JSON"| C3
```

---

## 2. Alur Data Pipeline

Tahapan pemrosesan data dari CSV mentah hingga model terlatih.

```mermaid
flowchart TD
    A["CSV SAML-D\n152K baris"] --> B["Augmentasi Data"]
    B --> C["Pembersihan &\nParsing Tanggal"]
    C --> D["Encoding Fitur\nKategorikal"]
    D --> E["Pembangunan\nNode Features\n(7 dimensi)"]
    D --> F["Pembangunan\nEdge Features\n(8 dimensi)"]
    E --> G["Konstruksi\nGraph (PyG)"]
    F --> G
    G --> H["Split Data\nTrain 70% / Val 15% / Test 15%"]
    H --> I["Training\nGAT Model"]
    I --> J["Model Terlatih\n(best_model.pt)"]
    I --> K["Metrik Evaluasi\n(training_metrics.json)"]
```

---

## 3. Arsitektur Model GAT (EdgeGATModel)

Alur forward pass model Graph Attention Network untuk klasifikasi edge.

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        NF["Node Features\n(7 dimensi per akun)"]
        EF["Edge Features\n(8 dimensi per transaksi)"]
    end

    subgraph ENCODER["GAT Encoder"]
        G1["GAT Layer 1\n4 heads × 64 dims"]
        G2["GAT Layer 2\n1 head × 64 dims"]
        NE["Node Embeddings\n(64 dims)"]
    end

    subgraph EDGE_PROC["Pemrosesan Edge"]
        SRC["Embedding\nPengirim (64)"]
        DST["Embedding\nPenerima (64)"]
        ET["Edge MLP\n8 → 32 dims"]
        CAT["Concatenate\n(160 dims)"]
    end

    subgraph KLASIFIKASI["Klasifikasi"]
        MLP["Classifier MLP\n160 → 64 → 32 → 1"]
        SIG["Sigmoid"]
        OUT["P(laundering)\n0.0 — 1.0"]
    end

    NF --> G1 --> G2 --> NE
    NE --> SRC --> CAT
    NE --> DST --> CAT
    EF --> ET --> CAT
    CAT --> MLP --> SIG --> OUT
```

---

## 4. Pola Augmentasi Data (Money Laundering Patterns)

Empat pola pencucian uang sintetis yang diinjeksikan ke dataset.

```mermaid
flowchart LR
    subgraph SM["Smurfing"]
        SM1["Satu Pengirim"] --> SM2["Banyak Transaksi\nKecil & Cepat"]
        SM2 --> SM3["Banyak\nPenerima"]
    end

    subgraph RT["Round-Tripping"]
        RT1["A"] --> RT2["B"]
        RT2 --> RT3["C"]
        RT3 --> RT4["D"]
        RT4 --> RT1
    end

    subgraph CB["Cross-Border\nLayering"]
        CB1["Negara A"] --> CB2["Negara B"]
        CB2 --> CB3["Negara C"]
        CB3 --> CB4["Negara D"]
    end

    subgraph FO["Fan-Out / Fan-In"]
        FO1["Sumber"] --> FO2["Perantara 1"]
        FO1 --> FO3["Perantara 2"]
        FO1 --> FO4["Perantara 3"]
        FO2 --> FO5["Kolektor"]
        FO3 --> FO5
        FO4 --> FO5
    end
```

---

## 5. Alur Prediksi Real-Time

Bagaimana sistem memproses transaksi baru untuk prediksi risiko.

```mermaid
flowchart LR
    A["Input\nTransaksi Baru"] --> B["Encoding\nFitur"]
    B --> C["Forward Pass\nModel GAT"]
    C --> D["Probabilitas\n(Sigmoid)"]
    D --> E{"Klasifikasi\nRisiko"}
    E -->|"P < 0.30"| F["🟢 Low"]
    E -->|"0.30 ≤ P < 0.70"| G["🟡 Moderate"]
    E -->|"0.70 ≤ P < 0.90"| H["🔴 High"]
    E -->|"P ≥ 0.90"| I["⛔ Critical"]
```

---

## 6. Arsitektur Backend API

Struktur endpoint FastAPI dan alur pemrosesan request.

```mermaid
flowchart TD
    subgraph STARTUP["Startup Server"]
        S1["Load Model & Encoder"]
        S2["Koneksi MongoDB"]
    end

    subgraph EP["API Endpoints"]
        E1["GET /api/summary\nRingkasan & KPI"]
        E2["GET /api/accounts\nDaftar Akun"]
        E3["GET /api/accounts/:id\nDetail Akun"]
        E4["GET /api/metrics\nPerforma Model"]
        E5["POST /api/predict\nPrediksi Real-time"]
    end

    S1 --> S2
    S2 --> E1
    S2 --> E2
    S2 --> E3
    S2 --> E4
    S2 --> E5

    E5 --> P1["Encode → Inference → Klasifikasi"]
    P1 --> P2["JSON Response"]
```

---

## 7. Arsitektur Frontend Dashboard

Struktur halaman dashboard React dan komponen utamanya.

```mermaid
flowchart TD
    APP["App.jsx\n(Router + Sidebar)"]

    APP --> D["📊 Dashboard"]
    APP --> AC["👥 Daftar Akun"]
    APP --> AD["🔍 Detail Akun"]
    APP --> MP["📈 Performa Model"]
    APP --> PR["⚡ Prediksi"]

    D --> D1["KPI Cards"]
    D --> D2["Risk Distribution Chart"]
    D --> D3["Flagged Accounts Table"]

    AD --> AD1["Profil & Risk Gauge"]
    AD --> AD2["Riwayat Transaksi"]
    AD --> AD3["Network Graph"]

    MP --> MP1["Metric Cards"]
    MP --> MP2["Training History Chart"]
    MP --> MP3["Confusion Matrix"]
```

---

## 8. Diagram Training Pipeline

Detail proses training model GAT dengan teknik optimasi.

```mermaid
flowchart TD
    A["Graph Data\n(processed_data.pt)"] --> B["Balanced Sampling\n(Oversample positif 3:1)"]
    B --> C["Forward Pass\n(Mixed Precision / AMP)"]
    C --> D["Focal Loss\n(α=0.9, γ=2.0)"]
    D --> E["Backpropagation"]
    E --> F["Cosine Annealing LR\n(Warm Restarts)"]
    F --> G{"Validasi F1\nMembaik?"}
    G -->|"Ya"| H["Simpan Best Model"]
    G -->|"Tidak (15 epoch)"| I["Early Stopping"]
    H --> J{"Epoch\nSelesai?"}
    J -->|"Belum"| B
    J -->|"Selesai"| K["Threshold Tuning\n(Sweep 0.1 - 0.95)"]
    K --> L["Model Final\n+ Metrik Evaluasi"]
    L --> M["Populate MongoDB\n(Transaksi + Akun)"]
```

---

## 9. Representasi Graph Transaksi

Bagaimana transaksi dimodelkan sebagai graph.

```mermaid
graph LR
    subgraph GRAPH["Transaction Graph"]
        N1((Akun A))
        N2((Akun B))
        N3((Akun C))
        N4((Akun D))

        N1 -->|"Rp 5.000\nWire Transfer"| N2
        N1 -->|"Rp 3.000\nCash"| N3
        N2 -->|"Rp 8.000\nWire Transfer"| N4
        N3 -->|"Rp 2.500\nCash"| N4
        N4 -->|"Rp 7.000\nWire Transfer"| N1
    end
```

**Keterangan:**
- **Node** = Akun (dengan 7 fitur: total_sent, total_received, dll.)
- **Edge** = Transaksi (dengan 8 fitur: amount, currency, dll.)
- Setiap edge diklasifikasikan secara independen oleh model GAT

---

## 10. Alur End-to-End (Ringkasan)

```mermaid
flowchart LR
    A["📄 Dataset\nCSV"] ==> B["🔧 Preprocessing\n& Augmentasi"]
    B ==> C["🧠 Training\nGAT Model"]
    C ==> D["🍃 MongoDB\nData Store"]
    D ==> E["🚀 FastAPI\nREST API"]
    E ==> F["🖥️ React\nDashboard"]
```

---

<div align="center">
  <sub>AML Guard — Anti-Money Laundering Detection System</sub><br>
  <sub>Dokumen diagram untuk keperluan skripsi</sub>
</div>
