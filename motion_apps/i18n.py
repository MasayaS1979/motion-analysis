"""
Simple English / Japanese switcher shared by every page.
 
Usage in each page:
 
    from i18n import t, language_switcher
    language_switcher()          # draws the 🌐 selector in the sidebar
    st.title(t("squat.title"))   # looks text up for the current language
 
The selected language is stored in st.session_state["lang"], which is
shared across all pages of the multipage app for the duration of the
browser session, so switching language on one page keeps it switched
everywhere else.
"""
 
import streamlit as st
 
DEFAULT_LANG = "ja"
LANGUAGES = {"ja": "日本語", "en": "English"}
 
 
def init_language():
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANG
 
 
def get_lang() -> str:
    init_language()
    return st.session_state["lang"]
 
 
def language_switcher(location: str = "sidebar"):
    """Render the language radio button and keep session_state in sync."""
    init_language()
    target = st.sidebar if location == "sidebar" else st
 
    keys = list(LANGUAGES.keys())
    current = st.session_state["lang"]
    default_index = keys.index(current) if current in keys else 0
 
    choice = target.radio(
        "🌐 Language / 言語",
        keys,
        index=default_index,
        format_func=lambda k: LANGUAGES[k],
        key="lang_switcher_radio",
        horizontal=True,
    )
    st.session_state["lang"] = choice
    return choice
 
 
def t(key: str, **kwargs) -> str:
    """Translate `key` for the current language. Falls back to Japanese,
    then to the raw key itself if nothing is found."""
    init_language()
    lang = st.session_state["lang"]
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get(DEFAULT_LANG, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
 
 
TRANSLATIONS = {}
 
# =========================================================================
# app.py
# =========================================================================
TRANSLATIONS.update({
    "app.subtitle": {
        "ja": "動作解析ダッシュボード — アップロードしたデータから各種モーション解析を実行します",
        "en": "Motion analysis dashboard — run a variety of motion analyses on your uploaded data",
    },
    "app.upload_header": {
        "ja": "モーションデータのアップロード",
        "en": "Upload Motion Data",
    },
    "app.upload_label": {
        "ja": "CSV / Excelファイルをアップロード",
        "en": "Upload CSV / Excel file",
    },
    "app.upload_success": {
        "ja": "ファイルのアップロードが完了しました！",
        "en": "File uploaded successfully!",
    },
    "app.tab_analysis": {"ja": "解析", "en": "Analysis"},
    "app.tab_guide": {"ja": "ユーザーガイド", "en": "User Guide"},
    "app.sidebar_download_guide": {
        "ja": "📄 ユーザーガイドをダウンロード",
        "en": "📄 Download User Guide",
    },
    "app.sidebar_guide_missing": {
        "ja": "ユーザーガイドPDFが見つかりません。",
        "en": "User Guide PDF not found.",
    },
    "app.analysis_pages_header": {"ja": "解析ページ", "en": "Analysis Pages"},
    "app.open_button": {"ja": "開く", "en": "Open"},
    "app.squat_title": {"ja": "Squat Analysis（スクワット分析）", "en": "Squat Analysis"},
    "app.squat_desc": {
        "ja": "スクワット動作のフェーズ・可動域・左右対称性を解析します。",
        "en": "Analyzes the phases, range of motion, and left-right symmetry of squat movements.",
    },
    "app.sit_stand_title": {"ja": "Sit-Stand Analysis（立ち座り分析）", "en": "Sit-Stand Analysis"},
    "app.sit_stand_desc": {
        "ja": "立ち座り動作の重心移動とタイミングを解析します。",
        "en": "Analyzes the center-of-mass movement and timing of sit-to-stand movements.",
    },
    "app.single_sit_stand_title": {
        "ja": "Single Sit-Stand Analysis（片脚立ち座り分析）",
        "en": "Single Sit-Stand Analysis",
    },
    "app.single_sit_stand_desc": {
        "ja": "片脚立ち座り動作の安定性とバランスを評価します。",
        "en": "Evaluates the stability and balance of single-leg sit-to-stand movements.",
    },
    "app.arm_flexion_title": {"ja": "Arm Flexion Analysis（肩関節屈曲分析）", "en": "Arm Flexion Analysis"},
    "app.arm_flexion_desc": {
        "ja": "肩関節屈曲動作の可動域と動作パターンを解析します。",
        "en": "Analyzes the range of motion and movement pattern of shoulder flexion.",
    },
    "app.gait_title": {"ja": "Gait Analysis（歩行分析）", "en": "Gait Analysis"},
    "app.gait_desc": {
        "ja": "歩行動作の周期性・左右対称性・関節角度を解析します。",
        "en": "Analyzes the cyclicality, left-right symmetry, and joint angles of gait.",
    },
    "app.guide_header": {"ja": "Motion Analysis ユーザーガイド", "en": "Motion Analysis User Guide"},
    "app.guide_caption": {
        "ja": "解析マニュアルおよび解釈ガイド",
        "en": "Analysis manual and interpretation guide",
    },
    "app.guide_success": {
        "ja": "ユーザーガイドをダウンロードいただけます。",
        "en": "User Guide is available for download.",
    },
    "app.guide_error": {
        "ja": "MotionAnalysis_UserGuide.pdf が見つかりません。",
        "en": "MotionAnalysis_UserGuide.pdf could not be found.",
    },
    "app.guide_content": {
        "ja": """
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
 
""",
        "en": """
### What's in the User Guide
 
**1. Phase Analysis**
- How movement is divided into phases and how each phase is detected
 
**2. Phase Detection Graph**
- A time-series view of the detected movement phases
 
**3. Joint Time-Series Analysis**
- Evaluation of how each joint angle changes over time during the movement
 
**4. ROM Comparison by Phase**
- Comparison of left/right joint range of motion (ROM) within each phase
 
**5. Left-Right Symmetry Analysis**
- Evaluation of left-right differences (asymmetry) in the movement
 
**6. Healthy ROM Comparison**
- Comparison of your measured results against healthy reference values
 
**7. Movement Dashboard**
- An overall summary of movement and performance metrics
 
---
 
📄 **Use the "Download User Guide" button above to download the PDF manual.**
 
""",
    },
})
 
