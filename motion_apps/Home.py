import streamlit as st

st.set_page_config(page_title="Motion Analysis", layout="wide")

# =========================
# Dark Theme CSS
# (same theme as the analysis pages, kept consistent)
# =========================
st.markdown(
    """
    <style>

    .stApp{
        background:#0E1117;
        color:white;
    }

    section[data-testid="stSidebar"]{
        background:#111827;
    }

    section[data-testid="stSidebar"] *{
        color:white;
    }

    h1,h2,h3,h4,h5,h6,p,span,div{
        color:white;
    }

    .block-container{
        padding-top:2rem;
    }

    header[data-testid="stHeader"]{
        background:transparent;
    }

    [data-testid="stDecoration"]{
        display:none;
    }

    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button,
    .stDownloadButton button,
    .stButton button{
        background-color:#1F2937;
        color:white !important;
        border:1px solid #3B82F6;
        border-radius:8px;
        font-weight:600;
    }

    div[data-testid="stDownloadButton"] button *,
    div[data-testid="stButton"] button *{
        color:white !important;
    }

    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stButton"] button:hover{
        background-color:#3B82F6;
        border-color:#60A5FA;
        color:white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Motion Analysis")

st.caption(
    "OpenCapで撮影した動作データ（.xlsx）をアップロードすると、左のメニューから各動作の解析ページを利用できます。"
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "動作データをアップロード（.xlsx）",
    type=["xlsx"]
)

if uploaded_file is not None:

    st.session_state["uploaded_file"] = uploaded_file

    st.success(f"アップロード完了：{uploaded_file.name}")

    st.info("左のサイドバーから解析したい動作ページを選択してください。")

else:

    st.warning("ファイルをアップロードすると、各解析ページが利用できるようになります。")

st.markdown("---")

st.subheader("解析メニュー")

st.markdown("""
- **Squat Analysis** — スクワット動作のフェーズ検出・可動域・左右差・動作スコア
- **Sit to Stand Analysis** — 座位-立位動作の解析
- **Single Sit to Stand Analysis** — 片脚座位-立位動作の解析
- **Arm Flexion Analysis** — 肩関節挙上動作の解析
- **Gait Analysis** — 歩行動作の解析
""")
