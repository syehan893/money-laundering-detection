#!/usr/bin/env python3
"""
Generate a formal academic seminar presentation for "Seminar Hasil"
Tema: Deteksi Pencucian Uang dengan Graph Neural Network (GNN)
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import nsmap
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_footer(slide, text="Universitas Indraprasta PGRI | Fakultas Teknik dan Ilmu Komputer | Teknik Informatika"):
    """Add formal footer to slide"""
    footer = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.333), Inches(0.3))
    tf = footer.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    p.alignment = PP_ALIGN.CENTER

def add_slide_number(slide, num, total):
    """Add slide number"""
    box = slide.shapes.add_textbox(Inches(12.0), Inches(7.0), Inches(0.8), Inches(0.3))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = f"{num}/{total}"
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    p.alignment = PP_ALIGN.RIGHT

# =============================
# SLIDE 1: TITLE SLIDE
# =============================
slide_layout = prs.slide_layouts[6]  # blank
slide = prs.slides.add_slide(slide_layout)

# Background
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
bg.line.fill.background()

# University header
univ_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.333), Inches(0.5))
tf = univ_box.text_frame
p = tf.paragraphs[0]
p.text = "UNIVERSITAS INDRAPRASTA PGRI"
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
p.alignment = PP_ALIGN.CENTER

faculty_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.7), Inches(12.333), Inches(0.4))
tf = faculty_box.text_frame
p = tf.paragraphs[0]
p.text = "Fakultas Teknik dan Ilmu Komputer - Program Studi Teknik Informatika"
p.font.size = Pt(14)
p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
p.alignment = PP_ALIGN.CENTER

# Accent line
line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.5), Inches(2.3), Inches(4.333), Inches(0.03))
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
line.line.fill.background()

# Main title
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.6), Inches(12.333), Inches(1.2))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "SISTEM DETEKSI PENCUCIAN UANG"
p.font.size = Pt(40)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "BERBASIS GRAPH NEURAL NETWORK (GNN)"
p2.font.size = Pt(32)
p2.font.bold = True
p2.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
p2.alignment = PP_ALIGN.CENTER

# Subtitle
sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(0.5))
tf = sub_box.text_frame
p = tf.paragraphs[0]
p.text = "Studi Kasus: Dataset Transaksi AML (SAML-D)"
p.font.size = Pt(18)
p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
p.alignment = PP_ALIGN.CENTER

# Author
author_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.3), Inches(12.333), Inches(0.6))
tf = author_box.text_frame
p = tf.paragraphs[0]
p.text = "Oleh: Muhammad Syehan"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "NIM: 202143501629"
p2.font.size = Pt(16)
p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
p2.alignment = PP_ALIGN.CENTER

# Footer
add_footer(slide, "Seminar Hasil Riset | 2026")

print("Slide 1: Title created")

# =============================
# SLIDE 2: AGENDA
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

# Header bar
header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Agenda Presentasi"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

agenda_items = [
    ("01", "Latar Belakang dan Rumusan Masalah", "Motivasi, identifikasi masalah, dan tujuan penelitian"),
    ("02", "Tinjauan Pustaka", "State-of-the-art dan teori-teori pendukung"),
    ("03", "Metodologi Penelitian", "Dataset, arsitektur GNN, tahapan implementasi"),
    ("04", "Hasil dan Pembahasan", "Evaluasi model, visualisasi, analisis performa"),
    ("05", "Implementasi Sistem", "Dashboard AML Guard dan fitur-fitur utama"),
    ("06", "Kesimpulan dan Saran", "Ringkasan kontribusi dan rekomendasi masa depan"),
]

y_start = Inches(1.3)
for i, (num, title, desc) in enumerate(agenda_items):
    y = y_start + i * Inches(0.95)
    
    # Number circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), y, Inches(0.5), Inches(0.5))
    circle.fill.solid()
    circle.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    circle.line.fill.background()
    
    num_box = slide.shapes.add_textbox(Inches(0.8), y + Inches(0.08), Inches(0.5), Inches(0.35))
    tf = num_box.text_frame
    p = tf.paragraphs[0]
    p.text = num
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1.5), y, Inches(10), Inches(0.35))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    # Description
    desc_box = slide.shapes.add_textbox(Inches(1.5), y + Inches(0.35), Inches(10), Inches(0.3))
    tf = desc_box.text_frame
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)

add_footer(slide)
add_slide_number(slide, 2, 20)

print("Slide 2: Agenda created")

# =============================
# SLIDE 3: LATAR BELAKANG
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "01  Latar Belakang"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

content = [
    ("Pencucian uang (Money Laundering) merupakan ancaman serius bagi stabilitas ekonomi global:", True, False),
    ("• Global Financial Integrity memperkirakan kerugian ekonomi akibat pencucian uang mencapai US$ 1-2 triliun per tahun", False, False),
    ("• Indonesia menempati peringkat 8 dalam Financial Secrecy Index 2022", False, False),
    ("• PPATK (Pusat Pelaporan dan Analisis Transaksi Keuangan) menangani ribuan kasus setiap tahun", False, False),
    ("", False, False),
    ("Tantangan utama dalam deteksi pencucian uang:", True, False),
    ("• Volume transaksi yang sangat besar dan kompleks", False, False),
    ("• Pola pencucian uang terus berevolusi dan menyamar dalam transaksi normal", False, False),
    ("• Sistem rule-based konvensional mudah diakali dengan teknik structuring (smurfing)", False, False),
    ("• Analisis manual tidak mampu mengatasi big data transaksi keuangan", False, False),
    ("", False, False),
    ("Graph Neural Network (GNN) menawarkan pendekatan baru:", True, False),
    ("• Memodelkan transaksi sebagai graph (jaringan) yang merepresentasikan hubungan antar akun", False, False),
    ("• Mampu menangkap pola jaringan kompleks (Fan-In, Fan-Out, Mixer, Round-Tripping)", False, False),
    ("• Attention mechanism memungkinkan fokus pada transaksi yang paling mencurigakan", False, False),
]

y_pos = Inches(1.2)
for text, is_bold, is_indent in content:
    if text == "":
        y_pos += Inches(0.1)
        continue
    box = slide.shapes.add_textbox(Inches(0.7), y_pos, Inches(12), Inches(0.35))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(14)
    if is_bold:
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    else:
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)
    y_pos += Inches(0.28)

add_footer(slide)
add_slide_number(slide, 3, 20)

print("Slide 3: Background created")

# =============================
# SLIDE 4: RUMUSAN MASALAH
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Rumusan Masalah"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

problems = [
    ("RQ1", "Bagaimana merepresentasikan data transaksi keuangan dalam bentuk graph untuk mendeteksi pola pencucian uang?"),
    ("RQ2", "Bagaimana merancang arsitektur Graph Neural Network yang efektif untuk klasifikasi transaksi mencurigakan?"),
    ("RQ3", "Bagaimana mengatasi masalah class imbalance (99:1) pada dataset transaksi AML?"),
    ("RQ4", "Bagaimana mengimplementasikan sistem deteksi secara real-time dengan dashboard interaktif?"),
]

y_pos = Inches(1.4)
for i, (code, text) in enumerate(problems):
    # Box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), y_pos, Inches(12), Inches(1.0))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    box.line.width = Pt(2)
    
    # Code
    code_box = slide.shapes.add_textbox(Inches(0.9), y_pos + Inches(0.1), Inches(1.0), Inches(0.3))
    tf = code_box.text_frame
    p = tf.paragraphs[0]
    p.text = code
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    
    # Text
    text_box = slide.shapes.add_textbox(Inches(1.9), y_pos + Inches(0.1), Inches(10.5), Inches(0.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    y_pos += Inches(1.15)

add_footer(slide)
add_slide_number(slide, 4, 20)

print("Slide 4: Research Questions created")

# =============================
# SLIDE 5: TUJUAN PENELITIAN
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Tujuan Penelitian"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

objectives = [
    ("1", "Mengembangkan representasi graph dari data transaksi keuangan yang mempertimbangkan fitur temporal dan atribut transaksi"),
    ("2", "Merancang dan mengimplementasikan model Graph Attention Network (GAT) dengan Focal Loss untuk klasifikasi edge transaksi"),
    ("3", "Mengevaluasi performa model menggunakan metrik Precision, Recall, F1-Score, dan Confusion Matrix pada data test yang belum pernah dilihat model"),
    ("4", "Membangun sistem dashboard interaktif (AML Guard) untuk visualisasi hasil deteksi dan prediksi real-time"),
]

y_pos = Inches(1.4)
for num, text in objectives:
    # Number circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), y_pos, Inches(0.6), Inches(0.6))
    circle.fill.solid()
    circle.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
    circle.line.fill.background()
    
    num_box = slide.shapes.add_textbox(Inches(1.0), y_pos + Inches(0.1), Inches(0.6), Inches(0.4))
    tf = num_box.text_frame
    p = tf.paragraphs[0]
    p.text = num
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Text
    text_box = slide.shapes.add_textbox(Inches(1.8), y_pos, Inches(10.5), Inches(0.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    y_pos += Inches(1.15)

add_footer(slide)
add_slide_number(slide, 5, 20)

print("Slide 5: Objectives created")

# =============================
# SLIDE 6: TINJAUAN PUSTAKA
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "02  Tinjauan Pustaka"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Left column: GNN & Attention
left_title = slide.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(5.8), Inches(0.4))
tf = left_title.text_frame
p = tf.paragraphs[0]
p.text = "Graph Neural Network (GNN)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

refs_left = [
    "• Kipf & Welling (2017): Graph Convolutional Network (GCN) untuk semi-supervised classification",
    "• Velickovic et al. (2018): Graph Attention Network (GAT) dengan attention mechanism",
    "• Hamilton et al. (2017): GraphSAGE untuk inductive representation learning",
    "• Wu et al. (2021): Comprehensive survey GNN dan aplikasinya",
]

y_pos = Inches(1.65)
for ref in refs_left:
    box = slide.shapes.add_textbox(Inches(0.9), y_pos, Inches(5.6), Inches(0.3))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = ref
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(6)
    y_pos += Inches(0.28)

# Right column: AML Detection
right_title = slide.shapes.add_textbox(Inches(6.9), Inches(1.2), Inches(5.8), Inches(0.4))
tf = right_title.text_frame
p = tf.paragraphs[0]
p.text = "Deteksi Pencucian Uang dengan AI"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)

refs_right = [
    "• Weber et al. (2019): Scalable Graph Neural Networks untuk deteksi fraud",
    "• Alarab et al. (2020): Graph Neural Network untuk Anti-Money Laundering",
    "• Chen et al. (2022): Heterogeneous GNN untuk deteksi transaksi mencurigakan",
    "• Savage et al. (2016): Detection of money laundering using network analysis",
]

y_pos = Inches(1.65)
for ref in refs_right:
    box = slide.shapes.add_textbox(Inches(7.1), y_pos, Inches(5.6), Inches(0.3))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = ref
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(6)
    y_pos += Inches(0.28)

# Bottom: Research gap
line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(3.5), Inches(12), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
line.line.fill.background()

gap_title = slide.shapes.add_textbox(Inches(0.7), Inches(3.7), Inches(12), Inches(0.4))
tf = gap_title.text_frame
p = tf.paragraphs[0]
p.text = "Research Gap"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)

gap_text = slide.shapes.add_textbox(Inches(0.9), Inches(4.1), Inches(11.8), Inches(1.2))
tf = gap_text.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("• Mayoritas penelitian menggunakan pendekatan supervised learning konvensional yang tidak memanfaatkan struktur jaringan transaksi. "
          "GNN mampu menangkap hubungan antar akun yang terlewatkan oleh metode tabular.\n\n"
          "• Perlu kombinasi GNN dengan mekanisme attention dan Focal Loss untuk mengatasi class imbalance yang ekstrem (99:1) pada dataset AML.\n\n"
          "• Belum ada sistem end-to-end yang mengintegrasikan model GNN dengan dashboard real-time untuk investigasi oleh analis bank.")
p.font.size = Pt(14)
p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

add_footer(slide)
add_slide_number(slide, 6, 20)

print("Slide 6: Literature Review created")

# =============================
# SLIDE 7: KERANGKA KONSEPTUAL
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Kerangka Konseptual"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Flow diagram
stages = [
    ("Input Layer", "Dataset\nSAML-D\n1 juta+ transaksi", RGBColor(0x3B, 0x82, 0xF6)),
    ("Preprocessing", "Feature Engineering\nGraph Construction\nTemporal Split", RGBColor(0x10, 0xB9, 0x81)),
    ("GAT Model", "Graph Attention\nEdge Classification\nFocal Loss", RGBColor(0xF5, 0x9E, 0x0B)),
    ("Output", "Risk Score\n0-100%\nKategori Risiko", RGBColor(0xEF, 0x44, 0x44)),
    ("Dashboard", "AML Guard\nVisualisasi\nPrediksi Real-time", RGBColor(0x8B, 0x5C, 0xF6)),
]

n = len(stages)
box_w = Inches(2.2)
box_h = Inches(1.8)
gap = Inches(0.25)
start_x = (prs.slide_width - n * box_w - (n - 1) * gap) / 2
y = Inches(2.0)

for i, (label, content, color) in enumerate(stages):
    x = start_x + i * (box_w + gap)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = color
    box.line.width = Pt(3)
    
    label_box = slide.shapes.add_textbox(x, y + Inches(0.1), box_w, Inches(0.3))
    tf = label_box.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER
    
    content_box = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.4), box_w - Inches(0.2), box_h - Inches(0.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = content
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER
    
    if i < n - 1:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_w + Inches(0.02), y + box_h/2 - Inches(0.12), Inches(0.2), Inches(0.24))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        arrow.line.fill.background()

# Analogy below
analogy = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.2), Inches(12), Inches(1.8))
analogy.fill.solid()
analogy.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
analogy.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
analogy.line.width = Pt(1)

analogy_title = slide.shapes.add_textbox(Inches(0.9), Inches(4.3), Inches(11.6), Inches(0.3))
tf = analogy_title.text_frame
p = tf.paragraphs[0]
p.text = "Analogi Sederhana"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

analogy_text = slide.shapes.add_textbox(Inches(0.9), Inches(4.6), Inches(11.6), Inches(1.3))
tf = analogy_text.text_frame
tf.word_wrap = True
lines = [
    "Seperti seorang detektif yang menangkap jaringan kriminal:",
    "1. Mengumpulkan semua bukti (data transaksi) → membuat peta hubungan antar tersangka (graph)",
    "2. Menginterogasi setiap tersangka DAN tetangganya untuk memahami jaringan (GAT Attention)",
    "3. Memberi nilai kecurigaan pada setiap hubungan berdasarkan bukti (risk score)",
    "4. Menyajikan hasil investigasi di papan pengungkit untuk tim kepolisian (dashboard)",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

add_footer(slide)
add_slide_number(slide, 7, 20)

print("Slide 7: Conceptual Framework created")

# =============================
# SLIDE 8: DATASET
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "03  Dataset dan Preprocessing"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Dataset stats box
stats_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(1.2), Inches(5.8), Inches(3.0))
stats_box.fill.solid()
stats_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
stats_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
stats_box.line.width = Pt(2)

stats_title = slide.shapes.add_textbox(Inches(0.9), Inches(1.3), Inches(5.4), Inches(0.4))
tf = stats_title.text_frame
p = tf.paragraphs[0]
p.text = "Dataset: AML-Synthetic (SAML-D)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

stats_content = slide.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(5.4), Inches(2.4))
tf = stats_content.text_frame
tf.word_wrap = True
lines = [
    "• Total transaksi: 1,014,934 baris",
    "• Akun unik (nodes): 52,000+",
    "• Transaksi unik (edges): 196,000+",
    "• Periode: September 2022 - September 2024",
    "• Sumber: Bank transaksi internasional",
    "",
    "Atribut utama:",
    "  - Sender/Receiver Account ID",
    "  - Amount, Currency, Payment Type",
    "  - Sender/Receiver Bank Location",
    "  - Date & Time, Is_Laundering (label)",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(3)

# Class distribution pie-like representation
chart_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.2), Inches(5.8), Inches(3.0))
chart_box.fill.solid()
chart_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
chart_box.line.color.rgb = RGBColor(0xEF, 0x44, 0x44)
chart_box.line.width = Pt(2)

chart_title = slide.shapes.add_textbox(Inches(6.9), Inches(1.3), Inches(5.4), Inches(0.4))
tf = chart_title.text_frame
p = tf.paragraphs[0]
p.text = "Distribusi Kelas (Imbalanced)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)

# Visual bars
# Normal bar
normal_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.0), Inches(1.9), Inches(4.5), Inches(0.4))
normal_bar.fill.solid()
normal_bar.fill.fore_color.rgb = RGBColor(0x10, 0xB9, 0x81)
normal_bar.line.fill.background()

normal_label = slide.shapes.add_textbox(Inches(7.0), Inches(2.35), Inches(4.5), Inches(0.25))
tf = normal_label.text_frame
p = tf.paragraphs[0]
p.text = "Normal: 99.3% (~1,006,000 transaksi)"
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)
p.alignment = PP_ALIGN.CENTER

# Fraud bar (very small)
fraud_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.0), Inches(2.7), Inches(0.45), Inches(0.4))
fraud_bar.fill.solid()
fraud_bar.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
fraud_bar.line.fill.background()

fraud_label = slide.shapes.add_textbox(Inches(7.0), Inches(3.15), Inches(4.5), Inches(0.25))
tf = fraud_label.text_frame
p = tf.paragraphs[0]
p.text = "Laundering: 0.7% (~8,000 transaksi)"
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)
p.alignment = PP_ALIGN.CENTER

# Note
note = slide.shapes.add_textbox(Inches(6.9), Inches(3.6), Inches(5.4), Inches(0.5))
tf = note.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "⚠️ Challenge: Data sangat tidak seimbang. Tanpa penanganan khusus, model akan cenderung mengklasifikasikan SEMUA transaksi sebagai Normal dengan akurasi tinggi tetapi gagal mendeteksi fraud."
p.font.size = Pt(11)
p.font.italic = True
p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)

# Temporal split info
split_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.4), Inches(12), Inches(2.0))
split_box.fill.solid()
split_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
split_box.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
split_box.line.width = Pt(1)

split_title = slide.shapes.add_textbox(Inches(0.9), Inches(4.5), Inches(11.6), Inches(0.3))
tf = split_title.text_frame
p = tf.paragraphs[0]
p.text = "Temporal Data Split"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

split_desc = slide.shapes.add_textbox(Inches(0.9), Inches(4.85), Inches(11.6), Inches(1.4))
tf = split_desc.text_frame
tf.word_wrap = True
lines = [
    "Data diurutkan berdasarkan waktu (chronological order) untuk mencegah data leakage:",
    "• Train (70%): Transaksi terlama → untuk melatih model belajar pola historis",
    "• Validation (15%): Transaksi tengah → untuk tuning hyperparameter dan early stopping",
    "• Test (15%): Transaksi terbaru → untuk evaluasi akhir, mensimulasikan deteksi di masa depan",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

add_footer(slide)
add_slide_number(slide, 8, 20)

print("Slide 8: Dataset created")

# =============================
# SLIDE 9: GRAPH CONSTRUCTION
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Graph Construction: Dari Transaksi ke Jaringan"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Left: Node features
node_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(1.2), Inches(5.8), Inches(2.8))
node_box.fill.solid()
node_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
node_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
node_box.line.width = Pt(2)

node_title = slide.shapes.add_textbox(Inches(0.9), Inches(1.3), Inches(5.4), Inches(0.35))
tf = node_title.text_frame
p = tf.paragraphs[0]
p.text = "Node Features (Akun Bank) - 7 dimensi"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

node_content = slide.shapes.add_textbox(Inches(0.9), Inches(1.65), Inches(5.4), Inches(2.2))
tf = node_content.text_frame
tf.word_wrap = True
lines = [
    "1. total_sent: Total nominal dikirim",
    "2. total_received: Total nominal diterima",
    "3. tx_count_sent: Jumlah transaksi keluar",
    "4. tx_count_received: Jumlah transaksi masuk",
    "5. unique_partners: Jumlah rekening unik terhubung",
    "6. foreign_currency_ratio: Rasio transaksi valas",
    "7. cross_border_ratio: Rasio transaksi lintas negara",
    "",
    "Dihitung hanya dari data TRAIN untuk mencegah leakage",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(3)

# Right: Edge features
edge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.2), Inches(5.8), Inches(2.8))
edge_box.fill.solid()
edge_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
edge_box.line.color.rgb = RGBColor(0xEF, 0x44, 0x44)
edge_box.line.width = Pt(2)

edge_title = slide.shapes.add_textbox(Inches(6.9), Inches(1.3), Inches(5.4), Inches(0.35))
tf = edge_title.text_frame
p = tf.paragraphs[0]
p.text = "Edge Features (Transaksi) - 8 dimensi"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)

edge_content = slide.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.4), Inches(2.2))
tf = edge_content.text_frame
tf.word_wrap = True
lines = [
    "1. amount: Nominal transaksi",
    "2. temporal_weight: Urgensi berdasarkan waktu",
    "3. payment_currency: Mata uang pembayaran (encoded)",
    "4. received_currency: Mata uang penerima (encoded)",
    "5. sender_bank_location: Lokasi bank pengirim",
    "6. receiver_bank_location: Lokasi bank penerima",
    "7. payment_type: Jenis pembayaran (encoded)",
    "8. temporal_feature: Menit sejak transaksi pertama",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(3)

# Bottom: Formula
formula_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.2), Inches(12), Inches(1.8))
formula_box.fill.solid()
formula_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
formula_box.line.color.rgb = RGBColor(0x10, 0xB9, 0x81)
formula_box.line.width = Pt(2)

formula_title = slide.shapes.add_textbox(Inches(0.9), Inches(4.3), Inches(11.6), Inches(0.3))
tf = formula_title.text_frame
p = tf.paragraphs[0]
p.text = "Temporal Weight (Komponen Waktu)"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)

formula_text = slide.shapes.add_textbox(Inches(0.9), Inches(4.65), Inches(11.6), Inches(1.2))
tf = formula_text.text_frame
tf.word_wrap = True
lines = [
    "Rumus:  temporal_weight = Amount / (delta_time + 1)",
    "",
    "• Transaksi besar dalam waktu singkat → weight tinggi → lebih mencurigakan",
    "• Contoh: $19,500 dengan delta_time = 0 menit → weight = 19,500 (CRITICAL)",
    "• Contoh: $15,000 dengan delta_time = 30 menit → weight = 484 (normal)",
    "• StandardScaler digunakan untuk normalisasi agar semua fitur comparable",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    if i == 0:
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)
    else:
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

add_footer(slide)
add_slide_number(slide, 9, 20)

print("Slide 9: Graph Construction created")

# =============================
# SLIDE 10: ARSITEKTUR GAT
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Arsitektur Model: Graph Attention Network (GAT)"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Architecture blocks
blocks = [
    ("Input", "Node Features\n(7-D)\nEdge Features\n(8-D)", RGBColor(0x3B, 0x82, 0xF6)),
    ("GAT Layer 1", "GATConv\nin=7, out=256\nheads=4, dropout=0.3\nBatchNorm + ELU", RGBColor(0x10, 0xB9, 0x81)),
    ("GAT Layer 2", "GATConv\nin=256, out=64\nheads=1, dropout=0.3\nBatchNorm + ELU", RGBColor(0x10, 0xB9, 0x81)),
    ("Edge\nTransformer", "Linear(8→32)\nELU + Dropout\nEdge Projection", RGBColor(0xF5, 0x9E, 0x0B)),
    ("Edge\nClassifier", "Concat[64+64+32]\nLinear(160→64)\nLinear(64→32)\nLinear(32→1)", RGBColor(0xEF, 0x44, 0x44)),
    ("Output", "Sigmoid\nProbability\n0 → 1", RGBColor(0x8B, 0x5C, 0xF6)),
]

n = len(blocks)
box_w = Inches(1.9)
box_h = Inches(2.4)
gap = Inches(0.15)
start_x = Inches(0.6)
y = Inches(1.3)

for i, (label, content, color) in enumerate(blocks):
    x = start_x + i * (box_w + gap)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = color
    box.line.width = Pt(2)
    
    label_box = slide.shapes.add_textbox(x, y + Inches(0.05), box_w, Inches(0.4))
    tf = label_box.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER
    
    content_box = slide.shapes.add_textbox(x + Inches(0.05), y + Inches(0.4), box_w - Inches(0.1), box_h - Inches(0.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = content
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER
    
    if i < n - 1:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_w + Inches(0.01), y + box_h/2 - Inches(0.1), Inches(0.12), Inches(0.2))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        arrow.line.fill.background()

# Detail box below
info_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.0), Inches(12.133), Inches(2.2))
info_box.fill.solid()
info_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
info_box.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
info_box.line.width = Pt(1)

info_title = slide.shapes.add_textbox(Inches(0.8), Inches(4.1), Inches(11.7), Inches(0.3))
tf = info_title.text_frame
p = tf.paragraphs[0]
p.text = "Detail Arsitektur dan Hyperparameter"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

# Two columns
left_info = slide.shapes.add_textbox(Inches(0.8), Inches(4.45), Inches(5.8), Inches(1.6))
tf = left_info.text_frame
tf.word_wrap = True
lines = [
    "Optimizer: Adam (lr=0.0005, weight_decay=5e-4)",
    "Loss Function: Focal Loss (alpha=0.75, gamma=2.0)",
    "Batch: Full-batch (graph inductive)",
    "Dropout: 0.3 untuk regularisasi",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(6)

right_info = slide.shapes.add_textbox(Inches(6.8), Inches(4.45), Inches(5.8), Inches(1.6))
tf = right_info.text_frame
tf.word_wrap = True
lines = [
    "Epoch: Maksimal 150, Early Stopping (patience=40)",
    "Metric Monitoring: Validation F1-Score",
    "Oversampling: Balanced sampling per epoch",
    "Hardware: GPU RTX 3060 (6GB VRAM)",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(6)

add_footer(slide)
add_slide_number(slide, 10, 20)

print("Slide 10: GAT Architecture created")

# =============================
# SLIDE 11: ATTENTION MECHANISM
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Graph Attention: Perhitungan Bobot Attention"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Center node visualization
center_x = Inches(6.5)
center_y = Inches(2.2)

# Central node (Akun 601)
center = slide.shapes.add_shape(MSO_SHAPE.OVAL, center_x - Inches(0.4), center_y, Inches(0.8), Inches(0.8))
center.fill.solid()
center.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
center.line.fill.background()

center_text = slide.shapes.add_textbox(center_x - Inches(0.4), center_y + Inches(0.2), Inches(0.8), Inches(0.4))
tf = center_text.text_frame
p = tf.paragraphs[0]
p.text = "601"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.alignment = PP_ALIGN.CENTER

# Neighbors
neighbors = [
    (center_x - Inches(2.2), center_y - Inches(0.6), "501", "$15K", "30 min", "484", "0.4%", RGBColor(0x3B, 0x82, 0xF6)),
    (center_x - Inches(2.2), center_y + Inches(0.8), "502", "$25K", "25 min", "962", "0.9%", RGBColor(0x3B, 0x82, 0xF6)),
    (center_x + Inches(1.8), center_y - Inches(0.6), "701", "$19.5K", "0 min", "19,500", "18.9%", RGBColor(0xEF, 0x44, 0x44)),
    (center_x + Inches(1.8), center_y + Inches(0.8), "702", "$19.5K", "5 min", "3,250", "3.1%", RGBColor(0x3B, 0x82, 0xF6)),
]

for nx, ny, label, amount, dt, e_val, alpha, color in neighbors:
    node = slide.shapes.add_shape(MSO_SHAPE.OVAL, nx, ny, Inches(0.6), Inches(0.6))
    node.fill.solid()
    node.fill.fore_color.rgb = color
    node.line.fill.background()
    
    nt = slide.shapes.add_textbox(nx, ny + Inches(0.15), Inches(0.6), Inches(0.3))
    tf = nt.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Edge line
    if nx < center_x:
        line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, nx + Inches(0.6), ny + Inches(0.3), Inches(1.3), Inches(0.3))
    else:
        line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, center_x + Inches(0.4), center_y + Inches(0.4), Inches(1.0), Inches(0.3))
    line.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    line.line.width = Pt(2)
    
    # Label
    info = slide.shapes.add_textbox(nx - Inches(0.3), ny + Inches(0.65), Inches(1.5), Inches(0.6))
    tf = info.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{amount}\nΔt={dt}\ne={e_val}\nα={alpha}"
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER

# Self loop
self_text = slide.shapes.add_textbox(center_x - Inches(0.5), center_y + Inches(0.9), Inches(1.0), Inches(0.5))
tf = self_text.text_frame
p = tf.paragraphs[0]
p.text = "SELF\nα=76.5%"
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
p.alignment = PP_ALIGN.CENTER

# Formula box
formula_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.5), Inches(12), Inches(2.2))
formula_box.fill.solid()
formula_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
formula_box.line.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)
formula_box.line.width = Pt(2)

formula_title = slide.shapes.add_textbox(Inches(0.9), Inches(4.6), Inches(11.6), Inches(0.3))
tf = formula_title.text_frame
p = tf.paragraphs[0]
p.text = "Rumus Perhitungan Attention"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x8B, 0x5C, 0xF6)

formula_left = slide.shapes.add_textbox(Inches(0.9), Inches(4.95), Inches(5.8), Inches(1.5))
tf = formula_left.text_frame
tf.word_wrap = True
lines = [
    "1. Attention Mentah (Edge Weight):",
    "   e_ij = Amount / (delta_time + 1)",
    "",
    "2. Softmax Normalization:",
    "   α_ij = softmax(e_ij) = exp(e_ij) / Σ exp(e_ik)",
    "",
    "3. Agregasi Bobot:",
    "   h'_i = Σ(α_ij × W × h_j)",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

formula_right = slide.shapes.add_textbox(Inches(6.8), Inches(4.95), Inches(5.8), Inches(1.5))
tf = formula_right.text_frame
tf.word_wrap = True
lines = [
    "Kenapa Δt penting?",
    "",
    "• Tanpa waktu: $19.5K (0 min) dan $15K (30 min)",
    "  dianggap SAMA — padahal yang satu URGENT",
    "",
    "• Dengan Δt: transaksi URGENT mendapat",
    "  attention 40x lebih tinggi (19,500 vs 484)",
    "",
    "→ Model fokus pada transaksi yang Paling Mencurigakan",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    if i == 0:
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)
    else:
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

add_footer(slide)
add_slide_number(slide, 11, 20)

print("Slide 11: Attention Mechanism created")

# =============================
# SLIDE 12: HASIL EVALUASI
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "04  Hasil Evaluasi Model"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Metrics boxes (4 KPI boxes)
metrics = [
    ("Precision", "99.73%", "Dari 100 yang di-flag, 99 benar fraud", RGBColor(0x3B, 0x82, 0xF6)),
    ("Recall", "99.58%", "Dari semua fraud, 99.6% berhasil tertangkap", RGBColor(0x10, 0xB9, 0x81)),
    ("F1-Score", "99.66%", "Keseimbangan sempurna Precision & Recall", RGBColor(0xF5, 0x9E, 0x0B)),
    ("AUC-ROC", ">99%", "Kemampuan diskriminasi model sangat tinggi", RGBColor(0x8B, 0x5C, 0xF6)),
]

for i, (label, value, desc, color) in enumerate(metrics):
    x = Inches(0.7) + i * Inches(3.2)
    y = Inches(1.2)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(2.9), Inches(1.6))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = color
    box.line.width = Pt(3)
    
    label_box = slide.shapes.add_textbox(x, y + Inches(0.1), Inches(2.9), Inches(0.3))
    tf = label_box.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER
    
    value_box = slide.shapes.add_textbox(x, y + Inches(0.4), Inches(2.9), Inches(0.5))
    tf = value_box.text_frame
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER
    
    desc_box = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.9), Inches(2.7), Inches(0.6))
    tf = desc_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    p.alignment = PP_ALIGN.CENTER

# Confusion Matrix
matrix_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(3.1), Inches(6), Inches(3.0))
matrix_box.fill.solid()
matrix_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
matrix_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
matrix_box.line.width = Pt(2)

matrix_title = slide.shapes.add_textbox(Inches(0.9), Inches(3.2), Inches(5.6), Inches(0.3))
tf = matrix_title.text_frame
p = tf.paragraphs[0]
p.text = "Confusion Matrix (Test Set)"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

# Matrix cells
cells = [
    ("", "Pred: Normal", "Pred: Laundering"),
    ("Actual: Normal", "429,447\n(True Negative)", "187\n(False Positive)"),
    ("Actual: Laundering", "296\n(False Negative)", "70,070\n(True Positive)"),
]

cell_w = Inches(2.3)
cell_h = Inches(0.8)
for row, row_data in enumerate(cells):
    for col, text in enumerate(row_data):
        cx = Inches(0.9) + col * cell_w
        cy = Inches(3.6) + row * cell_h
        
        cell = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, cy, cell_w - Inches(0.05), cell_h - Inches(0.05))
        
        if row == 0 or col == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
        elif row == 1 and col == 1:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xDC, 0xFC, 0xE7)
        elif row == 2 and col == 2:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xDC, 0xFC, 0xE7)
        elif row == 1 and col == 2:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xE2, 0xE2)
        elif row == 2 and col == 1:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xE2, 0xE2)
        
        cell.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        
        cell_text = slide.shapes.add_textbox(cx, cy + Inches(0.15), cell_w - Inches(0.05), Inches(0.5))
        tf = cell_text.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        p.alignment = PP_ALIGN.CENTER

# Classification Report
report_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(3.1), Inches(5.8), Inches(3.0))
report_box.fill.solid()
report_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
report_box.line.color.rgb = RGBColor(0x10, 0xB9, 0x81)
report_box.line.width = Pt(2)

report_title = slide.shapes.add_textbox(Inches(7.1), Inches(3.2), Inches(5.4), Inches(0.3))
tf = report_title.text_frame
p = tf.paragraphs[0]
p.text = "Classification Report"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)

report_content = slide.shapes.add_textbox(Inches(7.1), Inches(3.6), Inches(5.4), Inches(2.3))
tf = report_content.text_frame
tf.word_wrap = True
lines = [
    "                    Precision    Recall    F1-Score    Support",
    "Normal              99.93%      99.96%     99.94%     429,634",
    "Laundering          99.73%      99.58%     99.66%      70,366",
    "",
    "Accuracy:                                99.91%     500,000",
    "Macro Avg:           99.83%      99.77%     99.80%     500,000",
    "Weighted Avg:        99.91%      99.91%     99.91%     500,000",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.font.name = "Courier New"
    p.space_after = Pt(2)

add_footer(slide)
add_slide_number(slide, 12, 20)

print("Slide 12: Results created")

# =============================
# SLIDE 13: POLA AML TERDETEKSI
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "5 Pola Pencucian Uang yang Terdeteksi"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

patterns = [
    ("Fan-Out", "Satu sumber → Banyak tujuan", "Pecah dana ke banyak rekening\nuntuk menghindari threshold", RGBColor(0x3B, 0x82, 0xF6)),
    ("Fan-In", "Banyak sumber → Satu tujuan", "Konsolidasi dana dari banyak\nsumber ke satu akun", RGBColor(0x10, 0xB9, 0x81)),
    ("Scatter-Gather\n(Smurfing)", "Banyak sumber → Perantara →\nBanyak tujuan", "Structuring melalui money mule\nuntuk menyamarkan jejak", RGBColor(0xF5, 0x9E, 0x0B)),
    ("Gather-Scatter\n(Mixer)", "Konsolidasi → Distribusi ulang", "Akun 601: kumpulkan dari A,B,C\nlalu sebar ke D,E,F", RGBColor(0xEF, 0x44, 0x44)),
    ("Simple Cycle\n(Round-Tripping)", "Uang berputar balik ke asal", "A→B→C→A untuk membuat\njejak transaksi palsu", RGBColor(0x8B, 0x5C, 0xF6)),
]

n = len(patterns)
box_w = Inches(2.4)
box_h = Inches(2.4)
gap = Inches(0.2)
start_x = (prs.slide_width - n * box_w - (n - 1) * gap) / 2
y = Inches(1.3)

for i, (title, subtitle, desc, color) in enumerate(patterns):
    x = start_x + i * (box_w + gap)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = color
    box.line.width = Pt(3)
    
    title_box = slide.shapes.add_textbox(x, y + Inches(0.1), box_w, Inches(0.5))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER
    
    sub_box = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.55), box_w - Inches(0.2), Inches(0.35))
    tf = sub_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
    p.alignment = PP_ALIGN.CENTER
    
    desc_box = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.9), box_w - Inches(0.2), Inches(1.3))
    tf = desc_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER

# Bottom note
note = slide.shapes.add_textbox(Inches(0.7), Inches(4.0), Inches(12), Inches(0.5))
tf = note.text_frame
p = tf.paragraphs[0]
p.text = "💡 Semua pola ini terdeteksi karena GNN melihat STRUKTUR JARINGAN keseluruhan, bukan hanya transaksi individual"
p.font.size = Pt(13)
p.font.italic = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
p.alignment = PP_ALIGN.CENTER

add_footer(slide)
add_slide_number(slide, 13, 20)

print("Slide 13: AML Patterns created")

# =============================
# SLIDE 14: IMPLEMENTASI - DASHBOARD
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "05  Implementasi: AML Guard Dashboard"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

features = [
    ("🏠 Dashboard", "Ringkasan total akun, transaksi, flagged accounts, dan F1-Score model. Pie chart distribusi risiko.", RGBColor(0x3B, 0x82, 0xF6)),
    ("📋 Accounts List", "Filter berdasarkan kategori risiko (Low/Moderate/High/Critical). Search by account ID. Sorting.", RGBColor(0x10, 0xB9, 0x81)),
    ("🔍 Account Detail", "Profil akun + riwayat transaksi + visualisasi graph jaringan koneksi (nodes & edges).", RGBColor(0xF5, 0x9E, 0x0B)),
    ("🔮 Predict", "Input transaksi baru → instant risk score dengan breakdown fitur yang mempengaruhi prediksi.", RGBColor(0xEF, 0x44, 0x44)),
    ("📊 Model Performance", "Classification report, confusion matrix, training history (loss & F1 over epochs).", RGBColor(0x8B, 0x5C, 0xF6)),
]

y_pos = Inches(1.3)
for icon_title, desc, color in features:
    # Icon box
    icon_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), y_pos, Inches(0.5), Inches(0.5))
    icon_box.fill.solid()
    icon_box.fill.fore_color.rgb = color
    icon_box.line.fill.background()
    
    # Title
    title_b = slide.shapes.add_textbox(Inches(1.3), y_pos, Inches(3.5), Inches(0.35))
    tf = title_b.text_frame
    p = tf.paragraphs[0]
    p.text = icon_title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = color
    
    # Description
    desc_b = slide.shapes.add_textbox(Inches(1.3), y_pos + Inches(0.3), Inches(11.2), Inches(0.4))
    tf = desc_b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    y_pos += Inches(0.9)

# Tech stack
stack_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.5), Inches(12), Inches(1.2))
stack_box.fill.solid()
stack_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
stack_box.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
stack_box.line.width = Pt(1)

stack_title = slide.shapes.add_textbox(Inches(0.9), Inches(5.6), Inches(11.6), Inches(0.3))
tf = stack_title.text_frame
p = tf.paragraphs[0]
p.text = "Tech Stack"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

stack_text = slide.shapes.add_textbox(Inches(0.9), Inches(5.9), Inches(11.6), Inches(0.6))
tf = stack_text.text_frame
p = tf.paragraphs[0]
p.text = "Backend: Python 3.11 + PyTorch 2.5 + PyTorch Geometric 2.6 + FastAPI + Motor (async MongoDB)"
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
p2 = tf.add_paragraph()
p2.text = "Frontend: React 18 + Recharts + Tailwind CSS  |  Database: MongoDB  |  Deployment: Docker-ready"
p2.font.size = Pt(12)
p2.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

add_footer(slide)
add_slide_number(slide, 14, 20)

print("Slide 14: Implementation created")

# =============================
# SLIDE 15: ALUR DATA KE KLASIFIKASI
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "End-to-End: Dari Data ke Kategori Risiko"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Horizontal flow
flow_steps = [
    ("📄 CSV", "1M+\ntransaksi", RGBColor(0x3B, 0x82, 0xF6)),
    ("🧹 Clean", "Sort, Encode,\nScale", RGBColor(0x10, 0xB9, 0x81)),
    ("📊 Features", "Node (7D)\nEdge (8D)", RGBColor(0x10, 0xB9, 0x81)),
    ("🕸️ Graph", "52K nodes\n196K edges", RGBColor(0xF5, 0x9E, 0x0B)),
    ("🧠 GAT", "Attention\nClassifier", RGBColor(0xEF, 0x44, 0x44)),
    ("🎯 Score", "0-100%\nProb", RGBColor(0x8B, 0x5C, 0xF6)),
    ("💾 DB", "MongoDB\nStorage", RGBColor(0x06, 0xB6, 0xD4)),
    ("🖥️ UI", "AML Guard\nDashboard", RGBColor(0x8B, 0x5C, 0xF6)),
]

n = len(flow_steps)
box_w = Inches(1.4)
box_h = Inches(1.4)
gap = Inches(0.08)
start_x = Inches(0.5)
y = Inches(1.4)

for i, (icon, text, color) in enumerate(flow_steps):
    x = start_x + i * (box_w + gap)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = color
    box.line.width = Pt(2)
    
    text_box = slide.shapes.add_textbox(x, y + Inches(0.15), box_w, box_h - Inches(0.3))
    tf = text_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.alignment = PP_ALIGN.CENTER
    
    if i < n - 1:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_w + Inches(0.01), y + box_h/2 - Inches(0.08), Inches(0.1), Inches(0.16))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        arrow.line.fill.background()

# Risk categories bar
bar_y = Inches(3.2)
bar_h = Inches(0.5)

# Low
low = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), bar_y, Inches(3.45), bar_h)
low.fill.solid()
low.fill.fore_color.rgb = RGBColor(0x10, 0xB9, 0x81)
low.line.fill.background()
low_text = slide.shapes.add_textbox(Inches(0.7), bar_y + bar_h + Inches(0.05), Inches(3.45), Inches(0.25))
tf = low_text.text_frame
p = tf.paragraphs[0]
p.text = "🟢 LOW (<30%)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)
p.alignment = PP_ALIGN.CENTER

# Moderate
mod = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7) + Inches(3.45), bar_y, Inches(4.6), bar_h)
mod.fill.solid()
mod.fill.fore_color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
mod.line.fill.background()
mod_text = slide.shapes.add_textbox(Inches(0.7) + Inches(3.45), bar_y + bar_h + Inches(0.05), Inches(4.6), Inches(0.25))
tf = mod_text.text_frame
p = tf.paragraphs[0]
p.text = "🟡 MODERATE (30-70%)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
p.alignment = PP_ALIGN.CENTER

# High
high = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7) + Inches(8.05), bar_y, Inches(2.3), bar_h)
high.fill.solid()
high.fill.fore_color.rgb = RGBColor(0xEF, 0x44, 0x44)
high.line.fill.background()
high_text = slide.shapes.add_textbox(Inches(0.7) + Inches(8.05), bar_y + bar_h + Inches(0.05), Inches(2.3), Inches(0.25))
tf = high_text.text_frame
p = tf.paragraphs[0]
p.text = "🔴 HIGH (70-90%)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)
p.alignment = PP_ALIGN.CENTER

# Critical
crit = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7) + Inches(10.35), bar_y, Inches(1.15), bar_h)
crit.fill.solid()
crit.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
crit.line.fill.background()
crit_text = slide.shapes.add_textbox(Inches(0.7) + Inches(10.35), bar_y + bar_h + Inches(0.05), Inches(1.15), Inches(0.25))
tf = crit_text.text_frame
p = tf.paragraphs[0]
p.text = "⚫ CRIT (>90%)"
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
p.alignment = PP_ALIGN.CENTER

# Probability scale
scale = slide.shapes.add_textbox(Inches(0.7), bar_y - Inches(0.3), Inches(12), Inches(0.25))
tf = scale.text_frame
p = tf.paragraphs[0]
p.text = "Probabilitas Pencucian Uang: 0% ------------------------------------------------------------------------------------------------- 100%"
p.font.size = Pt(10)
p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
p.alignment = PP_ALIGN.CENTER

# Description below
flow_desc = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.3), Inches(12), Inches(2.0))
flow_desc.fill.solid()
flow_desc.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
flow_desc.line.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
flow_desc.line.width = Pt(1)

fd_title = slide.shapes.add_textbox(Inches(0.9), Inches(4.4), Inches(11.6), Inches(0.3))
tf = fd_title.text_frame
p = tf.paragraphs[0]
p.text = "Analogi Proses Deteksi"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

fd_text = slide.shapes.add_textbox(Inches(0.9), Inches(4.75), Inches(11.6), Inches(1.4))
tf = fd_text.text_frame
tf.word_wrap = True
lines = [
    "Seperti proses investigasi kepolisian:",
    "1. Mengumpulkan semua bukti (1 juta+ struk transaksi)",
    "2. Membersihkan dan mengorganisir bukti (preprocessing)",
    "3. Membuat peta hubungan antar tersangka (graph network)",
    "4. Menginterogasi setiap tersangka DAN tetangganya (GAT attention)",
    "5. Memberi nilai kecurigaan 0-100% berdasarkan bukti (sigmoid probability)",
    "6. Mengkategorikan kasus: Low → Moderate → High → Critical (risk classification)",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

add_footer(slide)
add_slide_number(slide, 15, 20)

print("Slide 15: Data to Classification Flow created")

# =============================
# SLIDE 16: PERHITUNGAN MANUAL (Account 601)
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Contoh Perhitungan Manual: Akun 601 (Mixer Pattern)"
p.font.size = Pt(24)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Step 1
s1_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(1.1), Inches(3.8), Inches(2.2))
s1_box.fill.solid()
s1_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
s1_box.line.color.rgb = RGBColor(0x3B, 0x82, 0xF6)
s1_box.line.width = Pt(2)

s1_title = slide.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(3.4), Inches(0.3))
tf = s1_title.text_frame
p = tf.paragraphs[0]
p.text = "1. Data Transaksi"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

s1_text = slide.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(3.4), Inches(1.7))
tf = s1_text.text_frame
tf.word_wrap = True
lines = [
    "501 → 601: $15,000\n   (30 menit lalu)",
    "502 → 601: $25,000\n   (25 menit lalu)",
    "601 → 701: $19,500\n   (0 menit - sekarang)",
    "601 → 702: $19,500\n   (5 menit lalu)",
    "Volume total: $79,000",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(2)

# Step 2
s2_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.7), Inches(1.1), Inches(3.8), Inches(2.2))
s2_box.fill.solid()
s2_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
s2_box.line.color.rgb = RGBColor(0x10, 0xB9, 0x81)
s2_box.line.width = Pt(2)

s2_title = slide.shapes.add_textbox(Inches(4.9), Inches(1.2), Inches(3.4), Inches(0.3))
tf = s2_title.text_frame
p = tf.paragraphs[0]
p.text = "2. Attention Mentah (e)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0x10, 0xB9, 0x81)

s2_text = slide.shapes.add_textbox(Inches(4.9), Inches(1.5), Inches(3.4), Inches(1.7))
tf = s2_text.text_frame
tf.word_wrap = True
lines = [
    "e = Amount / (Δt + 1)",
    "",
    "e(501) = 15,000/31 = 484",
    "e(502) = 25,000/26 = 962",
    "e(701) = 19,500/1 = 19,500",
    "e(702) = 19,500/6 = 3,250",
    "e(Self) = 79,000/1 = 79,000",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(2)

# Step 3
s3_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.7), Inches(1.1), Inches(3.8), Inches(2.2))
s3_box.fill.solid()
s3_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
s3_box.line.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)
s3_box.line.width = Pt(2)

s3_title = slide.shapes.add_textbox(Inches(8.9), Inches(1.2), Inches(3.4), Inches(0.3))
tf = s3_title.text_frame
p = tf.paragraphs[0]
p.text = "3. Softmax → Agregasi"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(0xF5, 0x9E, 0x0B)

s3_text = slide.shapes.add_textbox(Inches(8.9), Inches(1.5), Inches(3.4), Inches(1.7))
tf = s3_text.text_frame
tf.word_wrap = True
lines = [
    "Σe = 103,195",
    "",
    "α(501) = 0.4%",
    "α(502) = 0.9%",
    "α(701) = 18.9%",
    "α(702) = 3.1%",
    "α(Self) = 76.5%",
    "",
    "h' = Σ(α × hi) = 6.50",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(2)

# Result box
result_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(3.5), Inches(12), Inches(1.4))
result_box.fill.solid()
result_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
result_box.line.color.rgb = RGBColor(0xEF, 0x44, 0x44)
result_box.line.width = Pt(3)

result_title = slide.shapes.add_textbox(Inches(0.9), Inches(3.6), Inches(11.6), Inches(0.3))
tf = result_title.text_frame
p = tf.paragraphs[0]
p.text = "Hasil Klasifikasi"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0xEF, 0x44, 0x44)

result_text = slide.shapes.add_textbox(Inches(0.9), Inches(3.95), Inches(11.6), Inches(0.8))
tf = result_text.text_frame
tf.word_wrap = True
lines = [
    "Sigmoid: P = 1 / (1 + e^(-6.50)) = 0.998",
    "",
    "Probabilitas = 99.8%  →  Kategori: ⚫ CRITICAL",
    "Pola Terdeteksi: Gather-Scatter (Mixer) — Dana masuk dari banyak sumber, langsung disebar ke banyak tujuan dalam waktu singkat",
]
for i, line in enumerate(lines):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)

# Bottom note
note = slide.shapes.add_textbox(Inches(0.7), Inches(5.1), Inches(12), Inches(0.6))
tf = note.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "💡 Tanpa komponen waktu (Δt), transaksi $19,500 (0 menit) dan $15,000 (30 menit) akan dianggap SAMA. Dengan temporal weight, yang URGENT mendapat perhatian 40x lebih BESAR."
p.font.size = Pt(11)
p.font.italic = True
p.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6)

add_footer(slide)
add_slide_number(slide, 16, 20)

print("Slide 16: Manual Calculation created")

# =============================
# SLIDE 17: KESIMPULAN
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "06  Kesimpulan"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

conclusions = [
    ("1", "GNN efektif untuk AML", "Graph Attention Network mampu menangkap pola jaringan kompleks (Fan-In, Fan-Out, Mixer, Round-Tripping) yang tidak terdeteksi oleh metode tabular konvensional.", RGBColor(0x3B, 0x82, 0xF6)),
    ("2", "Performa sangat tinggi", "Precision 99.73%, Recall 99.58%, F1-Score 99.66% pada test set yang belum pernah dilihat model — membuktikan kemampuan generalisasi yang kuat.", RGBColor(0x10, 0xB9, 0x81)),
    ("3", "Temporal weight penting", "Komponen waktu (Δt) memungkinkan model membedakan transaksi urgent vs normal dengan perbedaan skor atensi hingga 40x.", RGBColor(0xF5, 0x9E, 0x0B)),
    ("4", "Sistem end-to-end", "Dashboard AML Guard menyediakan visualisasi real-time, prediksi instant, dan graph network untuk membantu analis bank dalam investigasi.", RGBColor(0xEF, 0x44, 0x44)),
]

y_pos = Inches(1.3)
for num, title, desc, color in conclusions:
    # Circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), y_pos, Inches(0.5), Inches(0.5))
    circle.fill.solid()
    circle.fill.fore_color.rgb = color
    circle.line.fill.background()
    
    num_box = slide.shapes.add_textbox(Inches(0.8), y_pos + Inches(0.08), Inches(0.5), Inches(0.35))
    tf = num_box.text_frame
    p = tf.paragraphs[0]
    p.text = num
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p.alignment = PP_ALIGN.CENTER
    
    # Title
    title_b = slide.shapes.add_textbox(Inches(1.4), y_pos, Inches(11), Inches(0.3))
    tf = title_b.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = color
    
    # Desc
    desc_b = slide.shapes.add_textbox(Inches(1.4), y_pos + Inches(0.3), Inches(11), Inches(0.6))
    tf = desc_b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    y_pos += Inches(1.1)

add_footer(slide)
add_slide_number(slide, 17, 20)

print("Slide 17: Conclusion created")

# =============================
# SLIDE 18: SARAN
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Saran untuk Pengembangan Masa Depan"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

suggestions = [
    "1. Integrasi dengan sistem Core Banking: Hubungkan model dengan data transaksi real-time dari bank untuk deteksi instant.",
    "2. Dynamic Graph: Kembangkan model yang mampu memproses graph yang berkembang secara dinamis (streaming transactions).",
    "3. Explainable AI: Tingkatkan interpretabilitas dengan attention visualization yang lebih detail untuk setiap prediksi.",
    "4. Multi-modal Features: Tambahkan fitur non-struktural (profil customer, geographic risk, PEP status) untuk enrichment.",
    "5. Federated Learning: Latih model di multiple bank tanpa memindahkan data mentah, menjaga privasi nasabah (Privacy-Preserving).",
    "6. Adversarial Robustness: Uji ketahanan model terhadap adaptive attacks — kriminal yang sengaja memanipulasi pola transaksi.",
]

y_pos = Inches(1.3)
for text in suggestions:
    box = slide.shapes.add_textbox(Inches(0.8), y_pos, Inches(12), Inches(0.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(8)
    y_pos += Inches(0.55)

add_footer(slide)
add_slide_number(slide, 18, 20)

print("Slide 18: Future Work created")

# =============================
# SLIDE 19: TERIMA KASIH
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
bg.line.fill.background()

# Accent line
line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.5), Inches(2.8), Inches(4.333), Inches(0.03))
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(0x3B, 0x82, 0xF6)
line.line.fill.background()

thanks = slide.shapes.add_textbox(Inches(0.5), Inches(3.0), Inches(12.333), Inches(1.0))
tf = thanks.text_frame
p = tf.paragraphs[0]
p.text = "Terima Kasih"
p.font.size = Pt(54)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.alignment = PP_ALIGN.CENTER

sub = slide.shapes.add_textbox(Inches(0.5), Inches(4.1), Inches(12.333), Inches(0.5))
tf = sub.text_frame
p = tf.paragraphs[0]
p.text = "Diskusi dan Pertanyaan"
p.font.size = Pt(24)
p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
p.alignment = PP_ALIGN.CENTER

contact = slide.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(12.333), Inches(0.8))
tf = contact.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Muhammad Syehan | NIM: 202143501629"
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
p.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "Program Studi Teknik Informatika | Universitas Indraprasta PGRI"
p2.font.size = Pt(14)
p2.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
p2.alignment = PP_ALIGN.CENTER

p3 = tf.add_paragraph()
p3.text = "Email: muhammadsyehan@email.com | GitHub: github.com/syehan/aml-guard"
p3.font.size = Pt(12)
p3.font.color.rgb = RGBColor(0x64, 0x70, 0x8B)
p3.alignment = PP_ALIGN.CENTER

add_footer(slide, "Seminar Hasil Riset | 2026")
add_slide_number(slide, 19, 20)

print("Slide 19: Thank You created")

# =============================
# SLIDE 20: DAFTAR PUSTAKA (Selected)
# =============================
slide = prs.slides.add_slide(slide_layout)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
bg.line.fill.background()

header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.8))
header.fill.solid()
header.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
header.line.fill.background()

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.5))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "Daftar Pustaka (Selected)"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

references = [
    "[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.",
    "[2] Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.",
    "[3] Hamilton, W. L., et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.",
    "[4] Weber, M., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks. IEEE.",
    "[5] Alarab, I., & Prakoonwit, S. (2020). Graph Neural Network for Fraud Detection via Aggregation. SN Computer Science.",
    "[6] Chen, J., et al. (2022). Heterogeneous Graph Neural Network for Anti-Money Laundering. ACM TKDD.",
    "[7] Savage, D., et al. (2016). Detection of Money Laundering: A Network Approach. IEEE.",
    "[8] Wu, Z., et al. (2021). A Comprehensive Survey on Graph Neural Networks. IEEE TNNLS.",
    "[9] Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.",
    "[10] FATF. (2023). Guidance for a Risk-Based Approach to Virtual Assets and Virtual Asset Service Providers.",
]

y_pos = Inches(1.2)
for ref in references:
    box = slide.shapes.add_textbox(Inches(0.8), y_pos, Inches(12), Inches(0.35))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = ref
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p.space_after = Pt(4)
    y_pos += Inches(0.32)

add_footer(slide)
add_slide_number(slide, 20, 20)

print("Slide 20: References created")

# =============================
# SAVE
# =============================
output_path = r"D:\Python\money-laundering-detection\Seminar_Hasil_AML_Guard.pptx"
prs.save(output_path)
print(f"\n✅ PowerPoint Seminar Hasil saved to: {output_path}")
print(f"📊 Total slides: {len(prs.slides)}")
