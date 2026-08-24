import streamlit as st
from i18n import t, language_switcher
 
# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Motion Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
language_switcher()
 
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
 
h1, h2, h3, h4, h5, h6, p, li, span, label, small, div {
    color: white !important;
}
 
h1 {
    font-weight: 700 !important;
}
 
h1, h2, h3 {
    letter-spacing: 0.2px;
}
 
.app-subtitle {
    color: #9CA3AF !important;
    font-size: 1.05rem;
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
}
 
.hero-wrap img {
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
 
/* Card containers (st.container(border=True)) — 分析カード自身（＝
   マーカーを直接含み、かつ内側に別のボーダー付きブロックを
   持たない、一番内側のブロック）だけにホバー効果を適用する。
   これにより「解析ページ」全体や「Motion Analysis」ページ全体の
   ような外側の親ブロックが、中のカードにカーソルを合わせただけで
   一緒に反応してしまう問題を防ぐ。 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.analysis-card-marker):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .analysis-card-marker)) {
    background-color: #141414;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease, transform 0.15s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.analysis-card-marker):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .analysis-card-marker)):hover {
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
        f'<div class="app-subtitle">{t("app.subtitle")}</div>',
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
st.subheader(t("app.upload_header"))
 
uploaded_file = st.file_uploader(
    t("app.upload_label"),
    type=["csv", "xlsx"]
)
 
if uploaded_file is not None:
    st.success(t("app.upload_success"))
    st.session_state["uploaded_file"] = uploaded_file
 
st.write("")
 
# =========================
# Main Tabs
# =========================
tab_analysis, tab_guide = st.tabs([t("app.tab_analysis"), t("app.tab_guide")])
 
# =========================
# Sidebar: PDF Download
# =========================
try:
    with open("MotionAnalysis_UserGuide.pdf", "rb") as pdf_file:
        st.sidebar.download_button(
            label=t("app.sidebar_download_guide"),
            data=pdf_file.read(),
            file_name="MotionAnalysis_UserGuide.pdf",
            mime="application/pdf",
            key="sidebar_download_guide"
        )
except FileNotFoundError:
    st.sidebar.warning(t("app.sidebar_guide_missing"))
 
st.sidebar.divider()
 
# =========================
# Analysis Tab
# =========================
with tab_analysis:
 
    st.subheader(t("app.analysis_pages_header"))
 
    analyses = [
        {
            "icon": "",
            "title": t("app.squat_title"),
            "desc": t("app.squat_desc"),
            "page": "pages/squat.py",
            "key": "btn_squat"
        },
        {
            "icon": "",
            "title": t("app.sit_stand_title"),
            "desc": t("app.sit_stand_desc"),
            "page": "pages/sit_stand.py",
            "key": "btn_sit_stand"
        },
        {
            "icon": "",
            "title": t("app.single_sit_stand_title"),
            "desc": t("app.single_sit_stand_desc"),
            "page": "pages/singly_sit_stan.py",
            "key": "btn_single_sit_stand"
        },
        {
            "icon": "",
            "title": t("app.arm_flexion_title"),
            "desc": t("app.arm_flexion_desc"),
            "page": "pages/arm_flexion.py",
            "key": "btn_arm_flexion"
        },
        {
            "icon": "",
            "title": t("app.gait_title"),
            "desc": t("app.gait_desc"),
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
                    st.markdown('<span class="analysis-card-marker"></span>', unsafe_allow_html=True)
                    st.markdown(f"#### {item['icon']} {item['title']}")
                    st.caption(item["desc"])
                    if st.button(t("app.open_button"), key=item["key"], use_container_width=True):
                        st.switch_page(item["page"])
 
# =========================
# User Guide Tab
# =========================
with tab_guide:
 
    st.header(t("app.guide_header"))
    st.caption(t("app.guide_caption"))
 
    try:
        with open("MotionAnalysis_UserGuide.pdf", "rb") as pdf_file:
            st.download_button(
                label=t("app.sidebar_download_guide"),
                data=pdf_file.read(),
                file_name="MotionAnalysis_UserGuide.pdf",
                mime="application/pdf",
                key="tab_download_guide"
            )
 
        st.success(t("app.guide_success"))
 
        st.markdown(t("app.guide_content"))
 
    except FileNotFoundError:
        st.error(t("app.guide_error"))
