import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Create presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle):
    slide_layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(slide_layout)
    
    # Background shape
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)  # dark navy
    bg.line.fill.background()
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(1))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(24)
    p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p.alignment = PP_ALIGN.CENTER
    
    # Accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.5), Inches(5.3), Inches(2.333), Inches(0.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    bar.line.fill.background()
    
    return slide

def add_content_slide(prs, title, bullets, note=None, accent_color=RGBColor(0x3B, 0x82, 0xF6)):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # White background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bg.line.fill.background()
    
    # Accent header bar
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2))
    header.fill.solid()
    header.fill.fore_color.rgb = accent_color
    header.line.fill.background()
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    # Content
    content_box = slide.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(12), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(22)
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        p.space_after = Pt(18)
        p.level = 0
    
    if note:
        note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.5), Inches(12), Inches(0.8))
        tf = note_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"💡 {note}"
        p.font.size = Pt(16)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    
    return slide

def add_diagram_slide(prs, title, steps, note=None):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    colors = [
        RGBColor(0x3B, 0x82, 0xF6),  # blue
        RGBColor(0x10, 0xB9, 0x81),  # green
        RGBColor(0xF5, 0x9E, 0x0B),  # yellow
        RGBColor(0xEF, 0x44, 0x44),  # red
        RGBColor(0x8B, 0x5C, 0xF6),  # purple
        RGBColor(0x06, 0xB6, 0xD4),  # cyan
    ]
    
    n = len(steps)
    box_width = Inches(11.5 / n)
    start_x = Inches(0.9)
    
    for i, (step_title, step_desc) in enumerate(steps):
        x = start_x + i * (box_width + Inches(0.15))
        
        # Box
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.5), box_width, Inches(4.5))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        box.line.color.rgb = colors[i % len(colors)]
        box.line.width = Pt(3)
        
        # Number circle
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + box_width/2 - Inches(0.3), Inches(1.2), Inches(0.6), Inches(0.6))
        circle.fill.solid()
        circle.fill.fore_color.rgb = colors[i % len(colors)]
        circle.line.fill.background()
        
        num_box = slide.shapes.add_textbox(x + box_width/2 - Inches(0.3), Inches(1.25), Inches(0.6), Inches(0.5))
        tf = num_box.text_frame
        p = tf.paragraphs[0]
        p.text = str(i + 1)
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = PP_ALIGN.CENTER
        
        # Step title
        st_box = slide.shapes.add_textbox(x + Inches(0.1), Inches(2.0), box_width - Inches(0.2), Inches(0.6))
        tf = st_box.text_frame
        p = tf.paragraphs[0]
        p.text = step_title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = colors[i % len(colors)]
        p.alignment = PP_ALIGN.CENTER
        
        # Step description
        sd_box = slide.shapes.add_textbox(x + Inches(0.15), Inches(2.6), box_width - Inches(0.3), Inches(3.0))
        tf = sd_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = step_desc
        p.font.size = Pt(13)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.alignment = PP_ALIGN.CENTER
        
        # Arrow between boxes (except last)
        if i < n - 1:
            arrow_x = x + box_width + Inches(0.02)
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arrow_x, Inches(3.2), Inches(0.12), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
            arrow.line.fill.background()
    
    if note:
        note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.3), Inches(12), Inches(0.8))
        tf = note_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"💡 {note}"
        p.font.size = Pt(15)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    return slide

# ==========================================
# SLIDE 1: Title
# ==========================================
add_title_slide(prs, 
    "AML Guard", 
    "Deteksi Pencucian Uang dengan AI & Graph Neural Network\nMuhammad Syehan")

# ==========================================
# SLIDE 2: Masalah
# ==========================================
add_content_slide(prs,
    "Apa Masalah yang Diselesaikan?",
    [
        "Pencucian uang (Money Laundering) menyamar dalam transaksi normal",
        "Tradisional: Aturan statis (rule-based) → mudah diakali kriminal",
        "Volume transaksi sangat besar — jutaan per hari",
        "Pola AML kompleks: Fan-Out, Fan-In, Mixer, Round-Tripping",
        "Butuh sistem yang bisa belajar pola dari data historis"
    ],
    "Analogi: Seperti mencari jarum di tumpukan jerami — tapi jarumnya berubah bentuk setiap hari",
    RGBColor(0xEF, 0x44, 0x44))

# ==========================================
# SLIDE 3: Data Transaksi
# ==========================================
add_content_slide(prs,
    "Data Apa yang Digunakan?",
    [
        "Dataset: Transaksi bank internasional (SAML-D)",
        "1 juta+ baris transaksi dari berbagai bank dunia",
        "Setiap baris berisi: Pengirim, Penerima, Nominal, Mata Uang, Lokasi, Jenis Pembayaran, Waktu",
        "Sebagian besar transaksi NORMAL, sebagian kecil mencurigakan (label 0 atau 1)",
        "Rasio imbalance: ~99 normal : 1 mencurigakan — seperti 99 transaksi biasa dan 1 transaksi jahat"
    ],
    "Bayangkan: 1 juta struk transfer bank yang harus disortir satu per satu oleh detektif",
    RGBColor(0x3B, 0x82, 0xF6))

# ==========================================
# SLIDE 4: Big Picture Flow Diagram
# ==========================================
add_diagram_slide(prs,
    "Alur Kerja Sistem — Dari Data ke Klasifikasi",
    [
        ("Data Mentah", "1 juta+ transaksi CSV\nPengirim, Penerima, Nominal, Mata Uang, Waktu"),
        ("Persiapan", "Bersihkan data\nUbah teks jadi angka\nHitung profil tiap akun"),
        ("Jaringan", "Node = Akun Bank\nEdge = Transfer Uang\nBangun Graph Network"),
        ("AI Belajar", "GAT Model\n2 Lapisan Attention\nBelajar pola dari historis"),
        ("Klasifikasi", "Setiap transaksi\ndapat skor risiko\n0% → 100%"),
        ("Dashboard", "Tampilkan hasil\nFilter risiko\nInvestigasi akun")
    ],
    "Sama seperti: Detektif belajar dari kasus lama → menilai kasus baru → laporkan ke meja kerja")

# ==========================================
# SLIDE 5: Step 1 - Data Preparation
# ==========================================
add_content_slide(prs,
    "Tahap 1: Persiapan Data (Preprocessing)",
    [
        "Hapus data rusak / kosong — seperti membuang struk yang robek",
        "Urutkan berdasarkan waktu — dari yang terlama ke terbaru",
        "Ubah teks jadi angka: US Dollar=0, Euro=1, Wire Transfer=1, Cash Deposit=4",
        "Hitung 'Temporal Weight': Transaksi besar dalam waktu singkat = lebih mencurigakan",
        "Hitung profil setiap akun: total kirim/terima, jumlah transaksi, rasio valas, rasio lintas negara"
    ],
    "Seperti membuat 'kartu identitas' untuk setiap pemilik rekening berdasarkan riwayat transaksi",
    RGBColor(0x10, 0xB9, 0x81))

# ==========================================
# SLIDE 6: Step 2 - Graph Construction
# ==========================================
add_content_slide(prs,
    "Tahap 2: Membangun Jaringan Transaksi (Graph)",
    [
        "Node (Titik): Setiap akun bank unik → 52.000+ node",
        "Edge (Garis): Setiap transfer uang antar akun → 196.000+ edge",
        "Setiap node punya 7 angka profil (fitur akun)",
        "Setiap edge punya 8 angka detail (fitur transaksi)",
        "Data dipisah: 70% untuk latih, 15% validasi, 15% uji — agar tidak mencontek saat ujian!"
    ],
    "Analogi: Seperti peta jaringan narkoba — titik = orang, garis = transaksi antar orang",
    RGBColor(0x8B, 0x5C, 0xF6))

# ==========================================
# SLIDE 7: Step 3 - GAT Model
# ==========================================
add_content_slide(prs,
    "Tahap 3: AI Belajar dengan Graph Attention Network (GAT)",
    [
        "Lapisan 1: Setiap akun 'bertanya' ke tetangganya — 'Seberapa penting kamu bagiku?'",
        "Lapisan 2: Perkuat pemahaman — gabungkan info dari tetangga-tetangganya",
        "Focal Loss: Beri perhatian ekstra pada kasus langka (fraud) — jangan biarkan tenggelam",
        "Early Stopping: Berhenti latih jika sudah cukup pandai — hindari hafalan",
        "Proses: 150 epoch maksimal, dicek tiap epoch — simpan model terbaik"
    ],
    "Seperti detektif yang wawancara semua tetangga: 'Apa yang kamu tahu tentang orang ini?' lalu merangkum",
    RGBColor(0x06, 0xB6, 0xD4))

# ==========================================
# SLIDE 8: Step 4 - Classification
# ==========================================
add_content_slide(prs,
    "Tahap 4: Klasifikasi Risiko — Dari Angka ke Kategori",
    [
        "Setiap transaksi diproses melalui Model GAT → output angka logit",
        "Fungsi Sigmoid mengubah logit menjadi probabilitas 0% sampai 100%",
        "🟢 LOW (< 30%) — Transaksi wajar, tidak perlu tindakan",
        "🟡 MODERATE (30–70%) — Pantau, mungkin mencurigakan",
        "🔴 HIGH (70–90%) — Berisiko tinggi, perlu investigasi",
        "⚫ CRITICAL (> 90%) — Hampir pasti pencucian uang!"
    ],
    "Seperti termometer: 0% = suhu normal, 100% = panas membara → segera tangani!",
    RGBColor(0xF5, 0x9E, 0x0B))

# ==========================================
# SLIDE 9: Results
# ==========================================
add_content_slide(prs,
    "Hasil & Performa Model",
    [
        "Data Uji (Test Set): 500.000+ transaksi yang BELUM PERNAH dilihat model",
        "Precision: 99.73% — Dari 100 yang di-flag, 99 benar-benar fraud",
        "Recall: 99.58% — Dari semua fraud yang ada, 99.6% berhasil tertangkap",
        "F1 Score: 99.66% — Keseimbangan sempurna antara Precision dan Recall",
        "Confusion Matrix: 429.447 TN | 187 FP | 296 FN | 70.070 TP",
        "Kesimpulan: Model sangat akurat dalam membedakan normal vs pencucian uang"
    ],
    "Dari 500.000 transaksi, cuma 187 yang salah dianggap jahat (false alarm) dan 296 yang lolos",
    RGBColor(0x10, 0xB9, 0x81))

# ==========================================
# SLIDE 10: Dashboard
# ==========================================
add_content_slide(prs,
    "Dashboard AML Guard — Apa yang Dilihat Analis?",
    [
        "🏠 Dashboard: Ringkasan total akun, transaksi, yang di-flag, skor F1 model",
        "📋 Daftar Akun: Filter risiko (Low/Moderate/High/Critical), cari ID akun, sortir",
        "🔍 Detail Akun: Riwayat transaksi + visualisasi jaringan koneksi (graph interaktif)",
        "🔮 Prediksi Baru: Masukkan data transaksi baru → langsung dapat skor risiko",
        "📊 Performa Model: Precision, Recall, F1, Confusion Matrix, riwayat training"
    ],
    "Meja kerja analis bank — semua informasi dalam satu layar, tanpa hitung manual",
    RGBColor(0x3B, 0x82, 0xF6))

# ==========================================
# SLIDE 11: Summary
# ==========================================
add_content_slide(prs,
    "Kesimpulan: Dari Data ke Deteksi",
    [
        "1. Data mentah → dibersihkan dan diubah menjadi angka yang komputer mengerti",
        "2. Setiap akun bank mendapat 'profil risiko' berdasarkan riwayat transaksi",
        "3. Transaksi diubah menjadi jaringan (graph) — akun sebagai titik, transfer sebagai garis",
        "4. Model GAT 'wawancara' tetangga-tetangga untuk memahami pola jaringan",
        "5. Output: Setiap transaksi dapat skor 0–100% → dikelompokkan Low/Moderate/High/Critical",
        "6. Analis bank melihat hasil di dashboard — fokus investigasi pada yang Critical & High"
    ],
    "AI tidak menggantikan manusia — AI membantu manusia fokus pada transaksi yang benar-benar mencurigakan",
    RGBColor(0x1E, 0x29, 0x3B))

# ==========================================
# SLIDE 12: Patterns
# ==========================================
add_content_slide(prs,
    "5 Pola Pencucian Uang yang Terdeteksi",
    [
        "🔀 Fan-Out: Satu akun → banyak tujuan (pecah dana ke banyak rekening)",
        "🔁 Fan-In: Banyak akun → satu tujuan (kumpulkan dana dari banyak sumber)",
        "💨 Scatter-Gather (Smurfing): Banyak sumber → perantara → banyak tujuan",
        "🔄 Gather-Scatter (Mixer): Kumpulkan → satukan → pecahkan lagi (Akun 601)",
        "⭕ Simple Cycle: Uang berputar dan kembali ke pengirim (buat jejak palsu)"
    ],
    "Semua pola ini terdeteksi karena model melihat jaringan keseluruhan, bukan transaksi individual",
    RGBColor(0xEF, 0x44, 0x44))

def add_flowchart_slide_simple(prs, title, steps, colors_list, note=None):
    """Simple flowchart with big boxes and clear arrows - for normies"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    n = len(steps)
    if n <= 5:
        box_width = Inches(2.2)
        box_height = Inches(1.8)
        gap = Inches(0.25)
        start_x = (prs.slide_width - n * box_width - (n - 1) * gap) / 2
        y_pos = Inches(2.0)
    else:
        box_width = Inches(2.0)
        box_height = Inches(1.5)
        gap = Inches(0.2)
        start_x = Inches(0.6)
        y_pos = Inches(1.8)
    
    for i, (label, desc) in enumerate(steps):
        x = start_x + i * (box_width + gap)
        
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_pos, box_width, box_height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        box.line.color.rgb = colors_list[i % len(colors_list)]
        box.line.width = Pt(4)
        
        label_box = slide.shapes.add_textbox(x + Inches(0.05), y_pos + Inches(0.1), box_width - Inches(0.1), Inches(0.5))
        tf = label_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = colors_list[i % len(colors_list)]
        p.alignment = PP_ALIGN.CENTER
        
        desc_box = slide.shapes.add_textbox(x + Inches(0.08), y_pos + Inches(0.55), box_width - Inches(0.16), box_height - Inches(0.65))
        tf = desc_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.alignment = PP_ALIGN.CENTER
        
        if i < n - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_width + Inches(0.02), y_pos + box_height/2 - Inches(0.15), Inches(0.2), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
            arrow.line.fill.background()
    
    if note:
        note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.3), Inches(12), Inches(0.8))
        tf = note_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"💡 {note}"
        p.font.size = Pt(14)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_flowchart_slide_core(prs, title, steps, note=None):
    """Core engine flowchart - shows the main processing pipeline"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    colors = [RGBColor(0x3B, 0x82, 0xF6), RGBColor(0x10, 0xB9, 0x81), RGBColor(0xF5, 0x9E, 0x0B), RGBColor(0xEF, 0x44, 0x44), RGBColor(0x8B, 0x5C, 0xF6), RGBColor(0x06, 0xB6, 0xD4)]
    
    n = len(steps)
    cols = 3
    rows = (n + cols - 1) // cols
    box_width = Inches(3.8)
    box_height = Inches(2.0)
    gap_x = Inches(0.4)
    gap_y = Inches(0.4)
    start_x = Inches(0.7)
    start_y = Inches(1.4)
    
    for i, (label, desc) in enumerate(steps):
        col = i % cols
        row = i // cols
        x = start_x + col * (box_width + gap_x)
        y = start_y + row * (box_height + gap_y)
        
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_width, box_height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        box.line.color.rgb = colors[i % len(colors)]
        box.line.width = Pt(3)
        
        num_circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + box_width/2 - Inches(0.25), y - Inches(0.2), Inches(0.5), Inches(0.5))
        num_circle.fill.solid()
        num_circle.fill.fore_color.rgb = colors[i % len(colors)]
        num_circle.line.fill.background()
        
        num_box = slide.shapes.add_textbox(x + box_width/2 - Inches(0.25), y - Inches(0.12), Inches(0.5), Inches(0.4))
        tf = num_box.text_frame
        p = tf.paragraphs[0]
        p.text = str(i + 1)
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = PP_ALIGN.CENTER
        
        label_box = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.15), box_width - Inches(0.2), Inches(0.5))
        tf = label_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = colors[i % len(colors)]
        p.alignment = PP_ALIGN.CENTER
        
        desc_box = slide.shapes.add_textbox(x + Inches(0.15), y + Inches(0.6), box_width - Inches(0.3), box_height - Inches(0.75))
        tf = desc_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.alignment = PP_ALIGN.CENTER
        
        if i < n - 1 and col < cols - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_width + Inches(0.03), y + box_height/2 - Inches(0.12), Inches(0.15), Inches(0.24))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
            arrow.line.fill.background()
        elif i < n - 1 and col == cols - 1 and row < rows - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, x + box_width/2 - Inches(0.12), y + box_height + Inches(0.03), Inches(0.24), Inches(0.15))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
            arrow.line.fill.background()
    
    if note:
        note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.5), Inches(12), Inches(0.8))
        tf = note_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"💡 {note}"
        p.font.size = Pt(13)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_flowchart_slide_technical(prs, title, layers, note=None):
    """Technical flowchart with detailed component names"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    bg.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    colors = [RGBColor(0x3B, 0x82, 0xF6), RGBColor(0x10, 0xB9, 0x81), RGBColor(0xF5, 0x9E, 0x0B), RGBColor(0xEF, 0x44, 0x44), RGBColor(0x8B, 0x5C, 0xF6), RGBColor(0x06, 0xB6, 0xD4), RGBColor(0xEC, 0x48, 0x99)]
    
    n = len(layers)
    box_width = Inches(1.75)
    box_height = Inches(4.5)
    gap = Inches(0.12)
    start_x = (prs.slide_width - n * box_width - (n - 1) * gap) / 2
    y_pos = Inches(1.1)
    
    for i, (stage, components) in enumerate(layers):
        x = start_x + i * (box_width + gap)
        
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y_pos, box_width, box_height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        box.line.color.rgb = colors[i % len(colors)]
        box.line.width = Pt(2)
        
        stage_box = slide.shapes.add_textbox(x, y_pos + Inches(0.1), box_width, Inches(0.5))
        tf = stage_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = stage
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = colors[i % len(colors)]
        p.alignment = PP_ALIGN.CENTER
        
        comp_box = slide.shapes.add_textbox(x + Inches(0.08), y_pos + Inches(0.6), box_width - Inches(0.16), box_height - Inches(0.8))
        tf = comp_box.text_frame
        tf.word_wrap = True
        
        for j, comp in enumerate(components):
            if j == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = f"• {comp}"
            p.font.size = Pt(10)
            p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
            p.space_after = Pt(6)
            p.alignment = PP_ALIGN.LEFT
        
        if i < n - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_width + Inches(0.01), y_pos + box_height/2 - Inches(0.12), Inches(0.12), Inches(0.24))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(0x4A, 0x55, 0x6A)
            arrow.line.fill.background()
    
    if note:
        note_box = slide.shapes.add_textbox(Inches(0.7), Inches(6.0), Inches(12), Inches(0.8))
        tf = note_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"⚙️ {note}"
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    return slide

# ==========================================
# SLIDE 13: Flowchart 1 - Simple (Normie-friendly)
# ==========================================
add_flowchart_slide_simple(prs,
    "Flowchart Sederhana: Dari Data ke Klasifikasi Risiko",
    [
        ("📄 Data CSV", "1 juta+ transaksi\nPengirim, Penerima,\nNominal, Waktu"),
        ("🧹 Bersihkan", "Hapus data rusak\nUbah teks jadi angka\nHitung profil akun"),
        ("🕸️ Graph", "Titik = Akun Bank\nGaris = Transfer Uang\n52K titik, 196K garis"),
        ("🧠 AI GAT", "Belajar pola jaringan\nWawancara tetangga\nFocal Loss + Early Stop"),
        ("🎯 Skor Risiko", "Probabilitas 0-100%\nLow / Moderate\nHigh / Critical")
    ],
    [RGBColor(0x3B, 0x82, 0xF6), RGBColor(0x10, 0xB9, 0x81), RGBColor(0xF5, 0x9E, 0x0B), RGBColor(0xEF, 0x44, 0x44), RGBColor(0x8B, 0x5C, 0xF6)],
    "Seperti: Sortir struk bank → buat profil tiap orang → gambar peta jaringan → detektif AI menilai → kasus diurutkan risiko")

# ==========================================
# SLIDE 14: Flowchart 2 - Core Engine
# ==========================================
add_flowchart_slide_core(prs,
    "Core Engine: Alur Pemrosesan Utama",
    [
        ("Data Ingestion", "CSV Loader\nsort_values(datetime)\ndropna(critical)\n1M+ rows → clean"),
        ("Feature Engineering", "LabelEncoder\n(Payment_currency,\nLocation, Type)\nStandardScaler\nNode features (7D)\nEdge features (8D)"),
        ("Graph Builder", "node_map {account: id}\nedge_index [src, dst]\ntemporal split\n70/15/15 train/val/test"),
        ("GAT Model", "GATConv Layer 1\n(7 → 256, 4 heads)\nGATConv Layer 2\n(256 → 64, 1 head)\nEdge MLP (8 → 32)\nClassifier (160 → 1)"),
        ("Focal Loss", "alpha=0.75\ngamma=2.0\nOversample minority\nBalanced per epoch\nEarlyStop(patience=40)"),
        ("Inference", "Sigmoid(logit)\nProbability 0-1\nThreshold=0.90\nRisk Category\nSave to MongoDB")
    ],
    "Arsitektur pipa pemrosesan: data masuk → fitur dihitung → graph dibangun → model GAT dilatih → hasil disimpan → siap untuk prediksi real-time")

# ==========================================
# SLIDE 15: Flowchart 3 - Technical Deep Dive
# ==========================================
add_flowchart_slide_technical(prs,
    "Technical Flowchart: Komponen & Teknologi Detail",
    [
        ("INPUT LAYER", ["SAML-D.csv", "1,014,934 rows", "10 columns", "DateTime parsed", "Temporal sorted"]),
        ("PREPROCESS", ["pandas.read_csv", "LabelEncoder.fit", "compute_temporal_weight", "build_node_features", "StandardScaler", "PyG Data object"]),
        ("GAT ENCODER", ["GATConv(in=7)", "4 heads × 64", "concat → 256", "BatchNorm1d", "ELU + Dropout(0.3)", "GATConv(in=256)", "1 head × 64", "BatchNorm1d", "Node Embeddings"]),
        ("EDGE PROCESS", ["edge_attr (8D)", "Linear(8→32)", "ELU + Dropout", "Edge Projection"]),
        ("CLASSIFIER", ["src_emb [64]", "dst_emb [64]", "edge_proj [32]", "Concat [160]", "Linear(160→64)", "Linear(64→32)", "Linear(32→1)", "Sigmoid"]),
        ("OUTPUT", ["Probability", "Risk Category", "MongoDB Insert", "REST API", "React Dashboard", "Recharts Viz"]),
        ("STACK", ["PyTorch 2.5", "PyG 2.6", "sklearn", "FastAPI", "Motor", "MongoDB", "React 18", "Recharts"])
    ],
    "Stack: Python backend + PyTorch Geometric untuk GNN + FastAPI untuk API + React frontend + MongoDB untuk persistence")

def add_visualization_slide_data_to_graph(prs):
    """Visualize how one transaction becomes part of a graph network"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Visualisasi: Dari Satu Transaksi ke Graph Network"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    # LEFT: CSV Row representation
    csv_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.3), Inches(3.5), Inches(2.8))
    csv_box.fill.solid()
    csv_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    csv_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    csv_box.line.width = Pt(3)
    
    csv_title = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(3.3), Inches(0.4))
    tf = csv_title.text_frame
    p = tf.paragraphs[0]
    p.text = "📄 Satu Baris CSV"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    p.alignment = PP_ALIGN.CENTER
    
    csv_content = slide.shapes.add_textbox(Inches(0.7), Inches(1.85), Inches(3.1), Inches(2.1))
    tf = csv_content.text_frame
    tf.word_wrap = True
    lines = [
        "Sender: 8724731955",
        "Receiver: 2769355426", 
        "Amount: $5,000.00",
        "Currency: US Dollar",
        "Payment: Wire Transfer",
        "Date: 2022-09-10",
        "Time: 04:17:31",
        "Is_Laundering: 0/1"
    ]
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(4)
    
    # Arrow 1
    arrow1 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.1), Inches(2.4), Inches(0.6), Inches(0.3))
    arrow1.fill.solid()
    arrow1.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    arrow1.line.fill.background()
    
    # MIDDLE: Feature extraction
    feat_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.3), Inches(3.5), Inches(2.8))
    feat_box.fill.solid()
    feat_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    feat_box.line.color.rgb = RGBColor(0x10, 0xB9, 0x81)
    feat_box.line.width = Pt(3)
    
    feat_title = slide.shapes.add_textbox(Inches(4.9), Inches(1.4), Inches(3.3), Inches(0.4))
    tf = feat_title.text_frame
    p = tf.paragraphs[0]
    p.text = "⚙️ Ekstraksi Fitur"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)
    p.alignment = PP_ALIGN.CENTER
    
    feat_content = slide.shapes.add_textbox(Inches(5.0), Inches(1.85), Inches(3.1), Inches(2.1))
    tf = feat_content.text_frame
    tf.word_wrap = True
    lines2 = [
        "Node (Akun 8724):",
        "  total_sent = $50K",
        "  tx_count = 12",
        "  foreign_ratio = 0.3",
        "",
        "Edge (Transfer):",
        "  amount = 5,000",
        "  temporal_w = 0.42",
        "  currency_enc = 0",
        "  payment_enc = 1"
    ]
    for i, line in enumerate(lines2):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(3)
    
    # Arrow 2
    arrow2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.4), Inches(2.4), Inches(0.6), Inches(0.3))
    arrow2.fill.solid()
    arrow2.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    arrow2.line.fill.background()
    
    # RIGHT: Graph visualization
    graph_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.1), Inches(1.3), Inches(3.7), Inches(2.8))
    graph_box.fill.solid()
    graph_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    graph_box.line.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
    graph_box.line.width = Pt(3)
    
    graph_title = slide.shapes.add_textbox(Inches(9.2), Inches(1.4), Inches(3.5), Inches(0.4))
    tf = graph_title.text_frame
    p = tf.paragraphs[0]
    p.text = "🕸️ Graph Network"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
    p.alignment = PP_ALIGN.CENTER
    
    # Draw simple nodes and edges visually
    # Node A (sender)
    node_a = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.8), Inches(2.0), Inches(0.6), Inches(0.6))
    node_a.fill.solid()
    node_a.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    node_a.line.fill.background()
    
    node_a_text = slide.shapes.add_textbox(Inches(9.8), Inches(2.05), Inches(0.6), Inches(0.5))
    tf = node_a_text.text_frame
    p = tf.paragraphs[0]
    p.text = "A"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Node B (receiver)
    node_b = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11.8), Inches(2.0), Inches(0.6), Inches(0.6))
    node_b.fill.solid()
    node_b.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
    node_b.line.fill.background()
    
    node_b_text = slide.shapes.add_textbox(Inches(11.8), Inches(2.05), Inches(0.6), Inches(0.5))
    tf = node_b_text.text_frame
    p = tf.paragraphs[0]
    p.text = "B"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Edge arrow between nodes
    edge_arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(10.45), Inches(2.18), Inches(1.3), Inches(0.24))
    edge_arrow.fill.solid()
    edge_arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    edge_arrow.line.fill.background()
    
    # Edge label
    edge_label = slide.shapes.add_textbox(Inches(10.3), Inches(2.5), Inches(1.5), Inches(0.3))
    tf = edge_label.text_frame
    p = tf.paragraphs[0]
    p.text = "$5K | Wire"
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    p.alignment = PP_ALIGN.CENTER
    
    # More nodes around
    node_c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.5), Inches(3.0), Inches(0.4), Inches(0.4))
    node_c.fill.solid()
    node_c.fill.fore_color.rgb = RGBColor(0x10, 0xB9, 0x81)
    node_c.line.fill.background()
    
    node_d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11.5), Inches(3.0), Inches(0.4), Inches(0.4))
    node_d.fill.solid()
    node_d.fill.fore_color.rgb = RGBColor(0x10, 0xB9, 0x81)
    node_d.line.fill.background()
    
    # Small edges
    se1 = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, Inches(9.7), Inches(2.55), Inches(0.3), Inches(0.5))
    se1.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    se1.line.width = Pt(1)
    
    se2 = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, Inches(11.4), Inches(2.55), Inches(0.3), Inches(0.5))
    se2.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    se2.line.width = Pt(1)
    
    graph_caption = slide.shapes.add_textbox(Inches(9.2), Inches(3.5), Inches(3.5), Inches(0.5))
    tf = graph_caption.text_frame
    p = tf.paragraphs[0]
    p.text = "A = Node (Akun Pengirim)\nB = Node (Akun Penerima)\n→ = Edge (Transfer $5K)"
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    p.alignment = PP_ALIGN.CENTER
    
    # BOTTOM: Explanation
    explain_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.4), Inches(12.333), Inches(2.6))
    explain_box.fill.solid()
    explain_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    explain_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    explain_box.line.width = Pt(2)
    
    explain_title = slide.shapes.add_textbox(Inches(0.7), Inches(4.5), Inches(12), Inches(0.4))
    tf = explain_title.text_frame
    p = tf.paragraphs[0]
    p.text = "Penjelasan: Satu Baris CSV Menjadi Bagian dari Graph"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    explain_content = slide.shapes.add_textbox(Inches(0.8), Inches(4.9), Inches(11.8), Inches(2.0))
    tf = explain_content.text_frame
    tf.word_wrap = True
    explains = [
        "1. Satu baris CSV = SATU transfer uang dari Akun A ke Akun B",
        "2. Akun A dan Akun B menjadi NODE (titik) dalam graph — masing-masing punya 7 angka profil",
        "3. Transfer $5K menjadi EDGE (garis berarah) dari A → B — punya 8 angka detail",
        "4. Lakukan ini untuk SEMUA 1 juta+ transaksi → jaringan besar dengan 52.000 node & 196.000 edge",
        "5. Model AI melihat BUKAN transaksi ini saja, tapi juga 'tetangga-tetangga' yang terhubung dengannya"
    ]
    for i, txt in enumerate(explains):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(8)
    
    return slide


def add_visualization_slide_attention(prs):
    """Visualize GAT attention mechanism"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Visualisasi: Perhitungan Attention (GAT)"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    # Central node
    center = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(5.8), Inches(2.2), Inches(1.0), Inches(1.0))
    center.fill.solid()
    center.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
    center.line.fill.background()
    
    center_text = slide.shapes.add_textbox(Inches(5.8), Inches(2.45), Inches(1.0), Inches(0.5))
    tf = center_text.text_frame
    p = tf.paragraphs[0]
    p.text = "Akun\n601"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Neighbor nodes - now with delta time included
    neighbors = [
        (Inches(3.5), Inches(1.2), "501", "$15K", "30 min", "0.4%"),
        (Inches(3.5), Inches(2.6), "502", "$25K", "25 min", "0.9%"),
        (Inches(7.8), Inches(1.2), "701", "$19.5K", "0 min", "18.9%"),
        (Inches(7.8), Inches(2.6), "702", "$19.5K", "5 min", "3.1%"),
        (Inches(5.8), Inches(3.8), "SELF", "$79K", "0 min", "76.5%"),
    ]
    
    for nx, ny, label, amount, dt, alpha in neighbors:
        # Node
        node = slide.shapes.add_shape(MSO_SHAPE.OVAL, nx, ny, Inches(0.7), Inches(0.7))
        node.fill.solid()
        if label == "SELF":
            node.fill.fore_color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
        else:
            node.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
        node.line.fill.background()
        
        nt = slide.shapes.add_textbox(nx, ny + Inches(0.15), Inches(0.7), Inches(0.4))
        tf = nt.text_frame
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = PP_ALIGN.CENTER
        
        # Connection line to center
        if label == "SELF":
            line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, nx + Inches(0.35), ny - Inches(0.35), Inches(0.7), Inches(0.7))
        elif nx < Inches(5.8):
            line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, nx + Inches(0.7), ny + Inches(0.35), Inches(1.3), Inches(0.35))
        else:
            line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, nx - Inches(1.3), ny + Inches(0.35), Inches(1.3), Inches(0.35))
        line.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        line.line.width = Pt(2)
        
        # Label with amount, delta time, and alpha
        alpha_box = slide.shapes.add_textbox(nx - Inches(0.4), ny + Inches(0.75), Inches(1.5), Inches(0.5))
        tf = alpha_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{amount} | Δt={dt}"
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.alignment = PP_ALIGN.CENTER
        
        p = tf.add_paragraph()
        p.text = f"α = {alpha}"
        p.font.size = Pt(11)
        p.font.bold = True
        if label == "701":
            p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)  # red for high attention
        elif label == "SELF":
            p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)  # yellow for self
        else:
            p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)  # blue for normal
        p.alignment = PP_ALIGN.CENTER
    
    # Formula box
    formula_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.3), Inches(3.0), Inches(1.8))
    formula_box.fill.solid()
    formula_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    formula_box.line.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
    formula_box.line.width = Pt(2)
    
    ft = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(2.8), Inches(1.8))
    tf = ft.text_frame
    tf.word_wrap = True
    lines = [
        "Rumus Attention Mentah:",
        "e = Amount / (Δt + 1)",
        "",
        "Contoh Perhitungan:",
        "e(701) = 19,500/(0+1)",
        "       = 19,500 ← tinggi!",
        "e(702) = 19,500/(5+1)",
        "       = 3,250",
        "e(501) = 15,000/(30+1)",
        "       = 484",
        "",
        "α = softmax(e)",
        "Σe = 103,195",
        "α(701) = 19,500/103,195",
        "       = 18.9%"
    ]
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(10)
        if i in [0, 3, 11]:
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
        elif "tinggi" in line:
            p.font.bold = True
            p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)
        else:
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(2)
    
    # Result box
    result_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.8), Inches(1.3), Inches(3.0), Inches(1.8))
    result_box.fill.solid()
    result_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    result_box.line.color.rgb = RGBColor(0xEF, 0x44, 0x44)
    result_box.line.width = Pt(2)
    
    rt = slide.shapes.add_textbox(Inches(9.9), Inches(1.4), Inches(2.8), Inches(1.6))
    tf = rt.text_frame
    tf.word_wrap = True
    lines2 = [
        "Agregasi Bobot:",
        "h' = Σ(αi × hi)",
        "",
        "h' = 6.50",
        "",
        "Sigmoid:",
        "P = 1/(1+e^(-6.5))",
        "P = 0.998",
        "",
        "→ CRITICAL ⚫"
    ]
    for i, line in enumerate(lines2):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        if i in [0, 5]:
            p.font.bold = True
            p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)
        elif i == 9:
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)
        else:
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(3)
    
    # Bottom explanation
    explain_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.5), Inches(12.333), Inches(2.5))
    explain_box.fill.solid()
    explain_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    explain_box.line.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
    explain_box.line.width = Pt(2)
    
    et = slide.shapes.add_textbox(Inches(0.7), Inches(4.55), Inches(12), Inches(2.3))
    tf = et.text_frame
    tf.word_wrap = True
    explains = [
        "Cara Kerja Graph Attention Network (GAT) — dengan Komponen Waktu:",
        "",
        "1. Setiap akun (node) melihat semua tetangganya — 'Siapa yang transfer ke/dari saya?'",
        "2. Hitung 'Skor Atensi Mentah' (e) = Amount / (Δt + 1):",
        "     • Transaksi BESAR dalam WAKTU SINGKAT = skor TINGGI (mis: 701 → e=19,500)",
        "     • Transaksi sama besar tapi lama = skor RENDAH (mis: 501 → e=484)",
        "3. Softmax: ubah skor jadi persentase (α) total 100% — SELF dapat 76.5%, 701 dapat 18.9%",
        "4. Agregasi: gabungkan fitur semua tetangga dengan bobot α → hasil embedding h' = 6.50",
        "5. Sigmoid: ubah h' jadi probabilitas 0-100% → P = 99.8% → kategori ⚫ CRITICAL",
        "",
        "🔑 Kenapa Δt penting? Tanpa waktu, transaksi $19.5K (0 menit) dan $15K (30 menit) dianggap sama. Dengan Δt, yang lebih URGENT mendapat perhatian lebih BESAR!"
    ]
    for i, txt in enumerate(explains):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(12)
        if i == 0:
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        elif "Δt penting" in txt:
            p.font.bold = True
            p.font.italic = True
            p.font.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
        else:
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(4)
    
    return slide


def add_visualization_slide_probability(prs):
    """Visualize probability to risk category conversion"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    bg.line.fill.background()
    
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(0x06, 0xB6, 0xD4)
    header.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Visualisasi: Dari Probabilitas ke Kategori Risiko"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    # Probability bar
    bar_y = Inches(1.4)
    bar_width = Inches(11.5)
    bar_height = Inches(0.6)
    bar_x = Inches(0.9)
    
    # Background bar
    bg_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bar_x, bar_y, bar_width, bar_height)
    bg_bar.fill.solid()
    bg_bar.fill.fore_color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    bg_bar.line.fill.background()
    
    # Green section (0-30%)
    green = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bar_x, bar_y, Inches(3.45), bar_height)
    green.fill.solid()
    green.fill.fore_color.rgb = RGBColor(0x10, 0xB9, 0x81)
    green.line.fill.background()
    
    # Yellow section (30-70%)
    yellow = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bar_x + Inches(3.45), bar_y, Inches(4.6), bar_height)
    yellow.fill.solid()
    yellow.fill.fore_color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
    yellow.line.fill.background()
    
    # Red section (70-90%)
    red = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bar_x + Inches(8.05), bar_y, Inches(2.3), bar_height)
    red.fill.solid()
    red.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
    red.line.fill.background()
    
    # Black section (90-100%)
    black = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bar_x + Inches(10.35), bar_y, Inches(1.15), bar_height)
    black.fill.solid()
    black.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    black.line.fill.background()
    
    # Labels on bar
    labels = [("0%", 0), ("30%", 3.45), ("70%", 8.05), ("90%", 10.35), ("100%", 11.5)]
    for text, offset in labels:
        lb = slide.shapes.add_textbox(bar_x + Inches(offset) - Inches(0.15), bar_y + Inches(0.65), Inches(0.5), Inches(0.3))
        tf = lb.text_frame
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    # Category labels below bar
    cats = [
        ("LOW", 1.725, 0x10, 0xB9, 0x81, "Transaksi wajar\nTidak perlu tindakan"),
        ("MODERATE", 5.75, 0xF5, 0x9E, 0x0B, "Pantau\nMungkin mencurigakan"),
        ("HIGH", 9.2, 0xEF, 0x44, 0x44, "Berisiko tinggi\nPerlu investigasi"),
        ("CRITICAL", 10.925, 0x1E, 0x29, 0x3B, "Hampir pasti\nPencucian uang!"),
    ]
    
    for cat, offset, r, g, b, desc in cats:
        cb = slide.shapes.add_textbox(bar_x + Inches(offset) - Inches(1.0), bar_y + Inches(1.0), Inches(2.0), Inches(0.4))
        tf = cb.text_frame
        p = tf.paragraphs[0]
        p.text = cat
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(r, g, b)
        p.alignment = PP_ALIGN.CENTER
        
        db = slide.shapes.add_textbox(bar_x + Inches(offset) - Inches(1.2), bar_y + Inches(1.4), Inches(2.4), Inches(0.6))
        tf = db.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.alignment = PP_ALIGN.CENTER
    
    # Examples
    examples_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(3.6), Inches(12.333), Inches(3.2))
    examples_box.fill.solid()
    examples_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    examples_box.line.color.rgb = RGBColor(0x06, 0xB6, 0xD4)
    examples_box.line.width = Pt(2)
    
    et = slide.shapes.add_textbox(Inches(0.7), Inches(3.7), Inches(12), Inches(3.0))
    tf = et.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Contoh Kasus Real:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(12)
    
    examples = [
        ("$500 Grocery Wire (Domestic, Bank Transfer)", "🟢 LOW", "2%", "0x10, 0xB9, 0x81"),
        ("$10K Cash Deposit (Same-day multiple deposits)", "🟡 MODERATE", "45%", "0xF5, 0x9E, 0x0B"),
        ("$50K Cross-border Crypto (High velocity)", "🔴 HIGH", "82%", "0xEF, 0x44, 0x44"),
        ("$5K Round-trip via 10 accounts in 1 hour (Mixer pattern)", "⚫ CRITICAL", "99.8%", "0x1E, 0x29, 0x3B"),
    ]
    
    for desc, badge, prob, color_str in examples:
        p = tf.add_paragraph()
        p.text = f"{badge}  {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(8)
        
        # Add probability on next line indented
        p = tf.add_paragraph()
        p.text = f"     → Probabilitas: {prob}"
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
        p.space_after = Pt(10)
    
    # Bottom note
    note = slide.shapes.add_textbox(Inches(0.7), Inches(6.7), Inches(12), Inches(0.5))
    tf = note.text_frame
    p = tf.paragraphs[0]
    p.text = "💡 Threshold optimal (0.90) dipilih dari validasi — memaksimalkan F1 Score. Di bawah 0.90 = Normal, di atas = Laundering."
    p.font.size = Pt(12)
    p.font.italic = True
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    p.alignment = PP_ALIGN.CENTER
    
    return slide


# ==========================================
# SLIDE 16: Visualization - Data to Graph
# ==========================================
add_visualization_slide_data_to_graph(prs)

# ==========================================
# SLIDE 17: Visualization - GAT Attention
# ==========================================
add_visualization_slide_attention(prs)

# ==========================================
# SLIDE 18: Visualization - Probability to Category
# ==========================================
add_visualization_slide_probability(prs)

# Save
output_path = r"D:\Python\money-laundering-detection\AML_Guard_Project_Overview_v4.pptx"
prs.save(output_path)
print(f"PowerPoint saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
