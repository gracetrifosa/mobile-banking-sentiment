# ============================
# MOBILE BANKING SENTIMENT APP
# ============================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from wordcloud import WordCloud   # untuk word cloud

st.set_page_config(page_title="Mobile Banking Sentiment", layout="wide")

# ============================
# CUSTOM CSS – Sidebar ala Splash (Navy Version)
# ============================

st.markdown("""
<style>

/* ==== SIDEBAR CONTAINER ==== */
[data-testid="stSidebar"] {
    background-color: #1E3A8A !important;
    padding: 40px 30px !important;
}

/* ==== SIDEBAR TITLE ==== */
.sidebar-title {
    font-size: 34px;
    font-weight: 900;
    color: white;
    margin-bottom: 35px;
}

/* === TEKS MENU RADIO DI SIDEBAR JADI PUTIH === */
[data-testid="stSidebar"] div[role="radiogroup"] * {
    color: white !important;
    fill: white !important;
}

div[role="radiogroup"] > label span {
    color: white !important;
}

[data-testid="stSidebar"] label {
    color: white !important;
}

/* ==== RADIO BULLET ==== */
[data-testid="stSidebar"] input[type="radio"] {
    transform: scale(1.3);
    accent-color: #FACC15 !important;
}

/* ==== SPACING ANTAR MENU ==== */
div[role="radiogroup"] > label {
    margin-bottom: 18px !important;
}

/* ==== GARIS PEMBATAS ==== */
.sidebar-line {
    height: 1px;
    background-color: rgba(255,255,255,0.3);
    margin: 30px 0;
}

.main {background-color: #F8FAFC;}

.title {
    color: white;
    padding: 20px;
    background-color:#1E3A8A;
    text-align:center;
    border-radius:10px;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================
# LOAD DATA
# ============================

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    df = df.rename(columns={
        "app_name": "platform",
        "review": "review",
        "label": "sentiment",
        "at": "date",
        "score": "score",
        "stemming": "stemming"
    })

    df["platform"] = df["platform"].replace({
        "BCA Mobile": "BCA Mobile",
        "BRImo": "BRImo",
        "brimo": "BRImo",
        "BRIMO": "BRImo"
    })

    df["sentiment"] = df["sentiment"].replace({
        "positif": "positive",
        "negatif": "negative",
        "Positif": "positive",
        "Negatif": "negative",
        "POSITIVE": "positive",
        "NEGATIVE": "negative"
    })

    df = df[df["sentiment"].isin(["positive", "negative"])]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()

# ============================
# SIDEBAR MENU
# ============================

st.sidebar.markdown("<div class='sidebar-title'>Menu</div>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    ["Home", "Dashboard", "Data", "About"]
)

st.sidebar.markdown("<div class='sidebar-line'></div>", unsafe_allow_html=True)


# ============================
# HOME PAGE – LANDING PAGE
# ============================

if menu == "Home":

    st.markdown("""
    <style>

    /* HERO SECTION */
    .hero {
        width: 100%;
        background-color: #1E3A8A;
        padding: 80px 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 900;
    }

    .hero-sub {
        font-size: 18px;
        opacity: 0.95;
        margin-top: 8px;
    }

    .hero-btn {
        margin-top: 22px;
        display: inline-block;
        background-color: #FACC15;
        color: black;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 700;
        text-decoration: none;
    }

    /* SECTION TITLES */
    .section-title {
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        color: #1E3A8A;
        margin-bottom: 8px;
    }

    .section-sub {
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
        color: #475569;
    }

    /* CARD SMALL */
    .card3 {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        min-height: 200px;
        text-align: center;
    }

    .card3 h3 {
        color: #1E3A8A;
        font-size: 20px;
        margin-bottom: 10px;
        font-weight: 800;
    }

    </style>
    """, unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Analisis Sentimen BCA Mobile & BRImo</div>
        <div class="hero-sub">Menggali persepsi pengguna berdasarkan ulasan Google PlayStore</div>
        <a class="hero-btn" href="#metodologi">Pelajari Lebih Lanjut</a>
    </div>
    """, unsafe_allow_html=True)

    # ============================
    # METRIC (CENTERED)
    # ============================

    total = len(df)
    bca = len(df[df.platform == "BCA Mobile"])
    bri = len(df[df.platform == "BRImo"])

    colL, colA, colB, colC, colR = st.columns([3, 2, 2, 2, 1])

    colA.metric("Total Review", total)
    colB.metric("BCA Mobile", bca)
    colC.metric("BRImo", bri)

    # ============================
    # METODOLOGI SECTION
    # ============================

    st.markdown("""
    <div style="margin-top: 40px;" id="metodologi"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">Metodologi dan Tujuan Proyek</div>
    <div class="section-sub">Analisis sentimen digunakan untuk memahami persepsi masyarakat terhadap aplikasi mobile banking.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card3">
            <h3>Analisis Sentimen</h3>
            <p>Mengklasifikasikan komentar pengguna menjadi positif dan negatif.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card3">
            <h3>Pemanfaatan Data</h3>
            <p>Mengolah data ulasan untuk melihat proporsi sentimen dan kualitas layanan.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card3">
            <h3>Rekomendasi</h3>
            <p>Memberikan rekomendasi berdasarkan hasil analisis data.</p>
        </div>
        """, unsafe_allow_html=True)

    # ============================
    # INFO SECTION
    # ============================

    st.markdown("""
    <div class="section-title" style="margin-top:40px;">Apa itu BCA Mobile & BRImo?</div>
    """, unsafe_allow_html=True)

    html_info = """
    <style>
    .info-box {
        background: white;
        padding: 45px;
        border-radius: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        text-align: center;
    }

    .info-text {

    
    }

    .info-subtitle {
        font-size: 24px;
        color: #1E3A8A;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .info-list {
        font-size: 19px;
        line-height: 1.9;
        display: inline-block;
        text-align: left;
        font-weight: 500;
    }
    </style>

    <div class="info-box">

        <div class="info-text">
            <b>BCA Mobile</b> dan <b>BRImo</b> adalah aplikasi mobile banking untuk transaksi digital seperti pembayaran, transfer, cek saldo, dan pengelolaan keuangan.
        </div>

        <div class="info-subtitle">Bagaimana Cara Kerjanya?</div>

        <ol class="info-list">
            <li><b>Pengumpulan Data</b> – Mengambil komentar dari Google PlayStore.</li>
            <li><b>Analisis Sentimen</b> – Mengklasifikasikan komentar menjadi positif dan negatif.</li>
            <li><b>Rekomendasi</b> – Menyusun rekomendasi berdasarkan hasil analisis.</li>
        </ol>

    </div>
    """

    components.html(html_info, height=550)



# ============================
# DASHBOARD PAGE
# ============================

elif menu == "Dashboard":

    st.markdown("<h1 class='title'>Sentiment Dashboard</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    pilih_app = col1.selectbox("Pilih Aplikasi", ["All", "BCA Mobile", "BRImo"])
    pilih_sentimen = col2.selectbox("Pilih Sentimen", ["All", "positive", "negative"])

    data_dash = df.copy()
    if pilih_app != "All":
        data_dash = data_dash[data_dash.platform == pilih_app]
    if pilih_sentimen != "All":
        data_dash = data_dash[data_dash.sentiment == pilih_sentimen]

    st.markdown("###  Sentiment Summary")
    c1, c2 = st.columns(2)
    c1.metric("Positive", len(data_dash[data_dash.sentiment == "positive"]))
    c2.metric("Negative", len(data_dash[data_dash.sentiment == "negative"]))

    st.markdown("###  Sentiment Distribution")
    if len(data_dash) > 0:
        pie = px.pie(
            data_dash,
            names="sentiment",
            color="sentiment",
            color_discrete_map={"positive": "#1E3A8A", "negative": "#3B82F6"}
        )

        pie.update_layout(
            legend=dict(x=0.75, y=0.5, font=dict(size=14))
        )

        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

        # ============================
    # COMMENT LENGTH + WORD CLOUD (SIDE BY SIDE, SAME SIZE)
    # ============================

    if len(data_dash) > 0:
        col_len, col_wc = st.columns(2)

        # --- KIRI: Comment Length Distribution ---
        with col_len:
            st.markdown("### Comment Length Distribution")
            data_dash["length"] = data_dash["review"].astype(str).apply(len)

            fig = px.histogram(
                data_dash,
                x="length",
                nbins=50,
                color="sentiment",
                color_discrete_map={"positive": "#1E3A8A", "negative": "#3B82F6"}
            )

            fig.update_layout(
                legend=dict(x=0.92, y=0.95, font=dict(size=12)),
                margin=dict(l=10, r=10, t=30, b=20),
                height=400        # tinggi plot histogram
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- KANAN: Word Cloud ---
        with col_wc:
            st.markdown("### Word Cloud – Most Frequent Words")

            all_text = " ".join(data_dash["review"].astype(str).tolist())

            wc = WordCloud(
                width=800,        # lebar kanvas
                height=600,       # tinggi kanvas = 400 (sama spt histogram)
                background_color="white",
                colormap="Blues",
                max_words=20
            ).generate(all_text)

            # ukuran figur matplotlib, pakai rasio sama
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")

            st.pyplot(fig_wc, use_container_width=True)

    else:
        st.info("Tidak ada data untuk ditampilkan / membuat word cloud.")



# ============================
# DATA PAGE
# ============================

elif menu == "Data":

    st.markdown("<h1 class='title'>Sentiment Data</h1>", unsafe_allow_html=True)
    st.markdown("### View Data")

    table = df[["platform", "review", "sentiment", "stemming"]]
    table = table.rename(columns={
        "platform": "Aplikasi",
        "review": "Review",
        "sentiment": "Sentimen",
        "stemming": "Cleaned Text"
    })

    st.dataframe(table, use_container_width=True, height=600)



# ============================
# ABOUT PAGE
# ============================

elif menu == "About":

    st.markdown("<h1 class='title'>Tentang Aplikasi</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='font-size:16px; line-height:1.7;'>
       <h3> Deskripsi Aplikasi</h3>
<p>
Aplikasi ini dirancang untuk menganalisis sentimen publik terhadap layanan mobile banking 
<b>BCA Mobile</b> dan <b>BRImo</b>, berdasarkan ulasan pengguna dan ditampilkan dalam bentuk visualisasi grafik yang mudah dipahami.
</p>

<h3> Fitur Utama</h3>
<ul>
    <li>Analisis komentar positif & negatif</li>
    <li>Perbandingan sentimen antar aplikasi</li>
    <li>Visualisasi grafik interaktif</li>
    <li>Tampilan data komentar yang telah diproses</li>
</ul>

<h3>‍ Pengembang</h3>
<p>
Aplikasi ini dikembangkan oleh Grace Trifosa Sagala (NIM 825220125) sebagai bagian dari Tugas Akhir di Universitas Tarumanagara. Tujuannya adalah memberikan pemahaman yang lebih mendalam mengenai opini pengguna terhadap layanan mobile banking melalui analisis sentimen yang terstruktur dan mudah dipahami.</b>.
</p>

<h3> Kontak</h3>
Jika Anda memiliki pertanyaan, saran, atau masukan, silakan hubungi melalui email: grace.825220125@stu.untar.ac.id
    </div>
    """, unsafe_allow_html=True)
