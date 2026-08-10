import streamlit as st
 
# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Motion Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# =========================
# CSS (combined, with card/hover styling)
# =========================
st.markdown("""
<style>
 
:root {
    --accent: #2DD4BF;
    --accent-dim: rgba(45, 212, 191, 0.15);
}
 
body, .stApp {
    background-color: #0b0b0b;
    color: white;
}
 
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
 
header {
    visibility: hidden;
}
 
h1 {
    color: white !important;
    font-weight: 700 !important;
}
 
h1, h2, h3 {
    letter-spacing: 0.2px;
}
 
.app-subtitle {
    color: #9CA3AF;
    font-size: 1.05rem;
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
}
 
.hero-wrap img {
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
 
/* Card containers (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #141414;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease, transform 0.15s ease;
}
 
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--accent) !important;
    transform: translateY(-2px);
}
 
/* Buttons */
div.stButton > button {
    background-color: var(--accent);
    color: #0b0b0b;
    border: none;
    border-radius: 10px;
    height: 44px;
    width: 100%;
    font-size: 15px;
    font-weight: 600;
    transition: background-color 0.15s ease, opacity 0.15s ease;
}
 
div.stButton > button:hover {
    background-color: #5eead4;
    color: #0b0b0b;
    opacity: 0.95;
}
 
/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #141414;
    border: 1px dashed #3a3a3a;
    border-radius: 12px;
    padding: 0.75rem;
}
 
[data-testid="stFileUploader"] button {
    color: black !important;
}
 
[data-testid="stFileUploader"] button * {
    color: black !important;
    fill: black !important;
}
 
[data-testid="stFileUploaderDropzone"] {
    background-color: #e8e8e8 !important;
    border-radius: 10px;
    color: black !important;
}
 
[data-testid="stFileUploaderDropzone"] * {
    color: black !important;
}
 
[data-testid="stFileUploader"] button:hover {
    background-color: #bbbbbb !important;
    color: black !important;
}
 
/* Download Button */
[data-testid="stDownloadButton"] button {
    color: black !important;
    background-color: white !important;
    border-radius: 10px !important;
}
 
[data-testid="stDownloadButton"] button:hover {
    background-color: #dddddd !important;
}
 
[data-testid="stDownloadButton"] button * {
    color: black !important;
}
 
[data-testid="stDownloadButton"] button p {
    color: black !important;
}
 
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222222;
}
 
section[data-testid="stSidebar"] * {
    color: white !important;
}
 
hr {
    border-color: #2a2a2a !important;
}
 
</style>
""", unsafe_allow_html=True)
 
# =========================
# Header
# =========================
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
 
with logo_col:
    try:
        st.image("images/PA_log-cutout.png", width=110)
    except Exception:
        pass
 
with title_col:
    st.title("Motion Analysis")
    st.markdown(
        '<div class="app-subtitle">動作解析ダッシュボード — アップロードしたデータから各種モーション解析を実行します</div>',
        unsafe_allow_html=True
    )
 
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
try:
    st.image(
        "images/julien-tromeur-vW6pA7b98BQ-unsplash.jpg",
        use_container_width=True
    )
except Exception:
    pass
st.markdown('</div>', unsafe_allow_html=True)
 
st.write("")
 
# =========================
# File Upload
# =========================
st.subheader("Upload Motion Data")
 
uploaded_file = st.file_uploader(
    "Upload CSV / Excel file",
    type=["csv", "xlsx"]
)
 
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.session_state["uploaded_file"] = uploaded_file
 
st.write("")
 
# =========================
# Main Tabs
# =========================
tab_analysis, tab_guide = st.tabs(["Analysis", "User Guide"])
 
# =========================
# Sidebar: PDF Download
# =========================
try:
    with open("MotionAnalysis_UserGuide.pdf", "rb") as pdf_file:
        st.sidebar.download_button(
            label="📄 Download User Guide",
            data=pdf_file.read(),
            file_name="MotionAnalysis_UserGuide.pdf",
            mime="application/pdf"
        )
except FileNotFoundError:
    st.sidebar.warning("User Guide PDF not found.")
 
st.sidebar.divider()
 
# =========================
# Analysis Tab
# =========================
with tab_analysis:
 
    st.subheader("Analysis Pages")
 
    analyses = [
        {
            "icon": "",
            "title": "Squat Analysis（スクワット分析）",
            "desc": "スクワット動作のフェーズ・可動域・左右対称性を解析します。",
            "page": "pages/squat.py",
            "key": "btn_squat"
        },
        {
            "icon": "",
            "title": "Sit-Stand Analysis（立ち座り分析）",
            "desc": "立ち座り動作の重心移動とタイミングを解析します。",
            "page": "pages/sit_stand.py",
            "key": "btn_sit_stand"
        },
        {
            "icon": "",
            "title": "Single Sit-Stand Analysis（片脚立ち座り分析）",
            "desc": "片脚立ち座り動作の安定性とバランスを評価します。",
            "page": "pages/singly_sit_stan.py",
            "key": "btn_single_sit_stand"
        },
        {
            "icon": "",
            "title": "Arm Flexion Analysis（肩関節屈曲分析）",
            "desc": "肩関節屈曲動作の可動域と動作パターンを解析します。",
            "page": "pages/arm_flexion.py",
            "key": "btn_arm_flexion"
        },
        {
            "icon": "",
            "title": "Gait Analysis（歩行分析）",
            "desc": "歩行動作の周期性・左右対称性・関節角度を解析します。",
            "page": "pages/gait.py",
            "key": "btn_gait"
        },
    ]
 
    cols_per_row = 3
    for row_start in range(0, len(analyses), cols_per_row):
        row_items = analyses[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
 
        for col, item in zip(cols, row_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {item['icon']} {item['title']}")
                    st.caption(item["desc"])
                    if st.button("開く", key=item["key"], use_container_width=True):
                        st.switch_page(item["page"])
 
# =========================
# User Guide Tab
# =========================
with tab_guide:
 
    st.header("Motion Analysis User Guide")
    st.caption("Analysis manual and interpretation guide")
 
    try:
        with open("MotionAnalysis_UserGuide.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download User Guide",
                data=pdf_file.read(),
                file_name="MotionAnalysis_UserGuide.pdf",
                mime="application/pdf"
            )
 
        st.success("User Guide is available for download.")
 
        st.markdown("""
 
### ユーザーガイド内容
 
**1. フェーズ解析**
- 動作の区分（フェーズ）と動作フェーズの判定方法
 
**2. フェーズ検出グラフ**
- 検出された各動作フェーズの時系列表示
 
**3. 関節時系列解析**
- 動作中の各関節角度の変化を時系列で評価
 
**4. フェーズ別ROM比較**
- 各フェーズにおける左右関節可動域（ROM）の比較
 
**5. 左右対称性解析**
- 左右の動作差（非対称性）の評価
 
**6. 健常者ROM比較**
- 健常者の基準値と測定結果の比較
 
**7. ムーブメントダッシュボード**
- 動作指標・パフォーマンス指標の総合サマリー
 
---
 
📄 **上の「Download User Guide」ボタンからPDFマニュアルをダウンロードしてご利用ください。**
 
""")
 
    except FileNotFoundError:
        st.error("MotionAnalysis_UserGuide.pdf が見つかりません。")
