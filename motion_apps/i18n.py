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
 
# =========================================================================
# Shared strings used by more than one analysis page
# =========================================================================
TRANSLATIONS.update({
    "common.upload_warning": {
        "ja": "ホームページからファイルをアップロードしてください",
        "en": "Please upload a file from the Home page",
    },
    "common.phase_metric_explanation": {
        "ja": """
**各指標の説明**
 
- **Min**：各フェーズにおける最小値
- **Max**：各フェーズにおける最大値
- **Mean**：各フェーズにおける平均値
- **Std**：各フェーズにおける標準偏差
- **ROM**：Range of Motion（Max − Min）
""",
        "en": """
**What each metric means**
 
- **Min**: the minimum value within each phase
- **Max**: the maximum value within each phase
- **Mean**: the average value within each phase
- **Std**: the standard deviation within each phase
- **ROM**: Range of Motion (Max − Min)
""",
    },
    "common.phase_summary_caption": {
        "ja": "各フェーズにおける各関節の最小値、最大値、平均値、標準偏差、可動域（ROM）を算出します。",
        "en": "Calculates the minimum, maximum, mean, standard deviation, and range of motion (ROM) of each joint within each phase.",
    },
    "common.metrics_expander_label": {"ja": "📖 指標の説明を見る", "en": "📖 View metric descriptions"},
    "common.joint_rom_expander_label": {"ja": "📖 Joint ROMとは", "en": "📖 What is Joint ROM?"},
    "common.joint_asymmetry_expander_label": {"ja": "📖 Joint Asymmetryとは", "en": "📖 What is Joint Asymmetry?"},
    "common.feature_expander_label": {"ja": "📖 特徴量の説明を見る", "en": "📖 View feature descriptions"},
    "common.score_expander_label": {"ja": "📖 スコアの算出方法を見る", "en": "📖 How the score is calculated"},
    "common.joint_time_series_caption": {
        "ja": "各関節運動の時系列変化を表示します。",
        "en": "Shows the time-series change of each joint's movement.",
    },
    "common.healthy_rom_caption": {
        "ja": "正常可動域（Healthy ROM）との比較を行います。",
        "en": "Compares the measured range of motion against the healthy reference range (Healthy ROM).",
    },
    "common.difference_pct_caption": {
        "ja": "Difference% = Subject ROM と Healthy ROM中央値との差",
        "en": "Difference% = the difference between the Subject's ROM and the midpoint of the Healthy ROM",
    },
    "common.seconds_after": {
        "ja": "動作開始から約 {sec} 秒後",
        "en": "About {sec} sec after the movement started",
    },
    "common.lumbar_placeholder_default": {
        "ja": "左の「Lumbar」→「Extension」にチェックを入れると表示されます",
        "en": "Check \"Lumbar\" → \"Extension\" on the left to display this chart",
    },
    "common.asymmetry_scale_note": {
        "ja": "左右の関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。",
        "en": "The left-right difference in joint range of motion (ROM), normalized by the larger side's ROM and shown as a percentage.",
    },
    "common.symmetry_caption": {
        "ja": "左右関節ROMの左右差を各Phaseごとに評価します。",
        "en": "Evaluates the left-right joint ROM difference for each phase.",
    },
})
 
# =========================================================================
# pages/squat.py
# =========================================================================
TRANSLATIONS.update({
    "squat.upload_warning": {"ja": "ホームページからファイルをアップロードしてください", "en": "Please upload a file from the Home page"},
    "squat.phase_caption": {
        "ja": "スクワット動作中の各フェーズ（Standing・Descending・Bottom・Ascending）を時系列で可視化したグラフです。",
        "en": "A time-series chart visualizing each phase of the squat movement (Standing, Descending, Bottom, Ascending).",
    },
    "squat.event_info_caption": {
        "ja": "下の数値は、動作データ全体の中で「最下点（Bottom）」「立位（Standing）」と判定された**フレーム番号**（何コマ目か）です。OpenCapのサンプリングレートは60Hzのため、フレーム番号 ÷ 60 で動作開始からの経過秒数に変換できます。",
        "en": "The numbers below are the **frame numbers** identified as \"lowest point (Bottom)\" and \"standing (Standing)\" within the full movement data. Since OpenCap's sampling rate is 60Hz, dividing the frame number by 60 converts it to elapsed seconds since the movement started.",
    },
    "squat.bottom_metric_label": {"ja": "Bottom（最下点のフレーム番号）", "en": "Bottom (lowest-point frame number)"},
    "squat.standing_metric_label": {"ja": "Standing（立位のフレーム番号）", "en": "Standing (standing frame number)"},
    "squat.dashboard_title": {"ja": "Squat Dashboard", "en": "Squat Dashboard"},
    "squat.dashboard_caption": {"ja": "スクワット動作の主要指標を表示します", "en": "Shows the key metrics of the squat movement"},
    "squat.metrics_table": {
        "ja": """
| 指標 | 説明 |
|---|---|
| **Max Knee Flexion** | 膝関節の最大屈曲角度（左右のうち大きい方） |
| **Max Hip Flexion** | 股関節の最大屈曲角度（左右のうち大きい方） |
| **Max Ankle Dorsiflexion** | 足関節の最大背屈角度（左右のうち大きい方） |
| **Lumbar Compensation** | 腰椎伸展の変化量。股関節・足関節の可動性不足を補う代償動作の可能性を示唆 |
| **Pelvic Compensation** | 骨盤傾斜の変化量。骨盤制御能力の指標 |
| **Overall Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""",
        "en": """
| Metric | Description |
|---|---|
| **Max Knee Flexion** | Maximum knee flexion angle (whichever side is larger) |
| **Max Hip Flexion** | Maximum hip flexion angle (whichever side is larger) |
| **Max Ankle Dorsiflexion** | Maximum ankle dorsiflexion angle (whichever side is larger) |
| **Lumbar Compensation** | Change in lumbar extension. May indicate a compensation strategy for limited hip/ankle mobility |
| **Pelvic Compensation** | Change in pelvic tilt. An indicator of pelvic control ability |
| **Overall Deviation** | Average deviation rate from the healthy range of motion (Healthy ROM) |
""",
    },
    "squat.checkbox_instruction": {
        "ja": "💡 左のチェックボックスで、表示する関節・骨盤・腰椎の指標を選択できます。",
        "en": "💡 Use the checkboxes on the left to choose which joint, pelvis, and lumbar metrics to display.",
    },
    "squat.joint_rom_content": {
        "ja": """
関節が動作中にどれだけ動いたかを示す指標です。
 
**ROM = 最大関節角度 − 最小関節角度**
 
| 関節 | 算出元 |
|---|---|
| **Hip** | 股関節屈曲角度 |
| **Knee** | 膝関節屈曲角度 |
| **Ankle** | 足関節背屈角度 |
""",
        "en": """
A metric showing how much each joint moved during the movement.
 
**ROM = Maximum joint angle − Minimum joint angle**
 
| Joint | Source |
|---|---|
| **Hip** | Hip flexion angle |
| **Knee** | Knee flexion angle |
| **Ankle** | Ankle dorsiflexion angle |
""",
    },
    "squat.joint_asymmetry_content": {
        "ja": """
左右の関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。
 
- **15%以下** — 左右対称。均等な下肢運動パターン
- **15%超** — 左右差あり。筋力差・可動性差・荷重偏位・代償動作の可能性
""",
        "en": """
The left-right difference in joint range of motion (ROM), normalized by the larger side's ROM and shown as a percentage.
 
- **15% or less** — Symmetric. An even lower-limb movement pattern
- **Over 15%** — Asymmetric. May indicate a strength difference, mobility difference, weight-bearing shift, or compensation
""",
    },
    "squat.feature_caption": {
        "ja": "関節運動・骨盤制御・体幹代償動作・左右差から動作パターンを評価する特徴量です。",
        "en": "Features that evaluate the movement pattern based on joint motion, pelvic control, trunk compensation, and left-right differences.",
    },
    "squat.feature_table": {
        "ja": """
| 特徴量 | 説明 |
|---|---|
| **Squat Depth** | 骨盤の上下移動量（pelvis_ty の最大値−最小値） |
| **Pelvic Stability** | 骨盤左右傾斜（pelvis_list）の標準偏差。小さいほど安定 |
| **Lumbar Compensation** | 腰椎伸展（lumbar_extension）の最大値。大きいほど代償動作の可能性 |
| **Hip Asymmetry** | 左右股関節のROM差（%）。大きいほど左右差あり |
| **Knee Asymmetry** | 左右膝関節のROM差（%）。大きいほど左右差あり |
| **Ankle Asymmetry** | 左右足関節のROM差（%）。大きいほど左右差あり |
""",
        "en": """
| Feature | Description |
|---|---|
| **Squat Depth** | Vertical pelvis displacement (max − min of pelvis_ty) |
| **Pelvic Stability** | Standard deviation of pelvic obliquity (pelvis_list). Smaller is more stable |
| **Lumbar Compensation** | Maximum lumbar extension (lumbar_extension). Larger may indicate compensation |
| **Hip Asymmetry** | Left-right hip ROM difference (%). Larger indicates greater asymmetry |
| **Knee Asymmetry** | Left-right knee ROM difference (%). Larger indicates greater asymmetry |
| **Ankle Asymmetry** | Left-right ankle ROM difference (%). Larger indicates greater asymmetry |
""",
    },
    "squat.score_caption": {
        "ja": "左右対称性・骨盤安定性・体幹代償動作・関節可動性の4要素から算出する100点満点の総合スコアです。",
        "en": "An overall score out of 100, calculated from four components: left-right symmetry, pelvic stability, trunk compensation, and joint mobility.",
    },
    "squat.score_content": {
        "ja": """
**Overall Score = Symmetry×0.35 + Stability×0.30 + Compensation×0.25 + Mobility×0.10**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Symmetry Score** | 35% | 股関節・膝関節・足関節の左右差（Asymmetry）の平均値 |
| **Stability Score** | 30% | 骨盤左右傾斜（pelvis_list）の変動性 |
| **Compensation Score** | 25% | 腰椎伸展（lumbar_extension）の最大値 |
| **Mobility Score** | 10% | 骨盤移動量（Squat Depth） |
 
- **高スコア** — 安定した左右対称な動作パターン
- **低スコア** — 左右差・骨盤制御低下・代償動作・可動性低下の可能性
""",
        "en": """
**Overall Score = Symmetry×0.35 + Stability×0.30 + Compensation×0.25 + Mobility×0.10**
 
| Component | Weight | Source |
|---|---|---|
| **Symmetry Score** | 35% | Average left-right difference (Asymmetry) across hip, knee, and ankle |
| **Stability Score** | 30% | Variability of pelvic obliquity (pelvis_list) |
| **Compensation Score** | 25% | Maximum lumbar extension (lumbar_extension) |
| **Mobility Score** | 10% | Pelvic displacement (Squat Depth) |
 
- **High score** — A stable, symmetric movement pattern
- **Low score** — May indicate asymmetry, reduced pelvic control, compensation, or reduced mobility
""",
    },
})
 
# =========================================================================
# pages/sit_stand.py
# =========================================================================
TRANSLATIONS.update({
    "sit_stand.phase_caption": {
        "ja": "立ち-座位動作中の各フェーズ（Standing・Descending・Bottom・Ascending）を時系列で可視化したグラフです。",
        "en": "A time-series chart visualizing each phase of the sit-to-stand movement (Standing, Descending, Bottom, Ascending).",
    },
    "sit_stand.event_info_caption": {
        "ja": "下の数値は、動作データ全体の中で「最下点（Bottom）」「立位（Standing）」と判定された**フレーム番号**（何コマ目か）です。OpenCapのサンプリングレートは60Hzのため、フレーム番号 ÷ 60 で動作開始からの経過秒数に変換できます。",
        "en": "The numbers below are the **frame numbers** identified as \"lowest point (Bottom)\" and \"standing (Standing)\" within the full movement data. Since OpenCap's sampling rate is 60Hz, dividing the frame number by 60 converts it to elapsed seconds since the movement started.",
    },
    "sit_stand.bottom_metric_label": {"ja": "Bottom（最下点のフレーム番号）", "en": "Bottom (lowest-point frame number)"},
    "sit_stand.standing_metric_label": {"ja": "Standing（立位のフレーム番号）", "en": "Standing (standing frame number)"},
    "sit_stand.dashboard_caption": {"ja": "Sit-to-Stand動作の主要指標を表示します", "en": "Shows the key metrics of the sit-to-stand movement"},
    "sit_stand.metrics_table": {
        "ja": """
| 指標 | 説明 |
|---|---|
| **Max Knee Flexion** | 立ち座り動作中の膝関節最大屈曲角度（左右のうち大きい方） |
| **Max Hip Flexion** | 立ち上がり開始時の股関節最大屈曲角度（左右のうち大きい方） |
| **Max Ankle Motion** | 動作中の足関節角度変化量（左右のうち大きい方） |
| **Lumbar Compensation** | 腰椎伸展の変化量。股関節・足関節の可動性不足を補う代償動作の可能性 |
| **Pelvis Compensation** | 骨盤前後傾の変化量。骨盤制御能力の指標 |
| **ROM Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""",
        "en": """
| Metric | Description |
|---|---|
| **Max Knee Flexion** | Maximum knee flexion angle during sit-to-stand (whichever side is larger) |
| **Max Hip Flexion** | Maximum hip flexion angle at the start of standing up (whichever side is larger) |
| **Max Ankle Motion** | Change in ankle angle during the movement (whichever side is larger) |
| **Lumbar Compensation** | Change in lumbar extension. May indicate a compensation strategy for limited hip/ankle mobility |
| **Pelvis Compensation** | Change in pelvic tilt. An indicator of pelvic control ability |
| **ROM Deviation** | Average deviation rate from the healthy range of motion (Healthy ROM) |
""",
    },
    "sit_stand.checkbox_instruction": {
        "ja": "💡 左のチェックボックスで、表示する関節・骨盤・腰椎の指標を選択できます。",
        "en": "💡 Use the checkboxes on the left to choose which joint, pelvis, and lumbar metrics to display.",
    },
    "sit_stand.joint_rom_content": {
        "ja": """
関節が動作中にどれだけ動いたかを示す指標です。
 
**ROM = 最大角度 − 最小角度**
 
| 関節 | 評価内容 |
|---|---|
| **Hip** | 股関節屈曲角度。立ち上がり時の体幹前傾戦略を評価 |
| **Knee** | 膝関節屈曲量。立ち上がりに必要な下肢運動を評価 |
| **Ankle** | 足関節運動。足部による重心移動能力を評価 |
""",
        "en": """
A metric showing how much each joint moved during the movement.
 
**ROM = Maximum angle − Minimum angle**
 
| Joint | What it evaluates |
|---|---|
| **Hip** | Hip flexion angle. Evaluates the trunk forward-lean strategy when standing up |
| **Knee** | Amount of knee flexion. Evaluates the lower-limb motion required to stand up |
| **Ankle** | Ankle motion. Evaluates the foot's ability to shift the center of mass |
""",
    },
    "sit_stand.joint_asymmetry_content": {
        "ja": """
左右の関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。
 
- **15%以下** — 比較的対称な運動パターン
- **15%超** — 左右荷重差・筋力差・可動性差・代償動作の可能性
""",
        "en": """
The left-right difference in joint range of motion (ROM), normalized by the larger side's ROM and shown as a percentage.
 
- **15% or less** — A relatively symmetric movement pattern
- **Over 15%** — May indicate a left-right weight-bearing difference, strength difference, mobility difference, or compensation
""",
    },
    "sit_stand.feature_caption": {
        "ja": "身体移動・姿勢制御・代償動作・左右差からSit-to-Stand動作を評価する特徴量です。",
        "en": "Features that evaluate the sit-to-stand movement based on body displacement, postural control, compensation, and left-right differences.",
    },
    "sit_stand.feature_table": {
        "ja": """
| 特徴量 | 説明 |
|---|---|
| **Seat-Off Height** | 骨盤の垂直移動量（立ち上がりの深さ） |
| **Pelvic Shift** | 骨盤の左右・前後移動量の最大値 |
| **Lumbar Compensation** | 腰椎伸展の変化量。大きいほど代償動作の可能性 |
| **Hip Asymmetry** | 左右股関節のROM差（%） |
| **Knee Asymmetry** | 左右膝関節のROM差（%） |
| **Ankle Asymmetry** | 左右足関節のROM差（%） |
""",
        "en": """
| Feature | Description |
|---|---|
| **Seat-Off Height** | Vertical pelvis displacement (depth of standing up) |
| **Pelvic Shift** | Maximum left-right / front-back pelvis displacement |
| **Lumbar Compensation** | Change in lumbar extension. Larger may indicate compensation |
| **Hip Asymmetry** | Left-right hip ROM difference (%) |
| **Knee Asymmetry** | Left-right knee ROM difference (%) |
| **Ankle Asymmetry** | Left-right ankle ROM difference (%) |
""",
    },
    "sit_stand.score_caption": {
        "ja": "左右対称性・姿勢安定性・代償動作・身体移動能力の4要素から算出する100点満点の総合スコアです。",
        "en": "An overall score out of 100, calculated from four components: left-right symmetry, postural stability, compensation, and body displacement ability.",
    },
    "sit_stand.score_content": {
        "ja": """
**Overall Score = Symmetry×0.30 + Stability×0.30 + Compensation×0.20 + Mobility×0.20**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Symmetry Score** | 30% | 股関節・膝関節・足関節の左右差（Asymmetry）の平均値 |
| **Stability Score** | 30% | 骨盤の左右・前後移動量（Pelvic Shift） |
| **Compensation Score** | 20% | 腰椎伸展の変化量（Lumbar Compensation） |
| **Mobility Score** | 20% | 骨盤垂直移動量（Seat-Off Height） |
 
- **高スコア** — 安定した効率的なSit-to-Stand動作
- **低スコア** — 左右差・姿勢制御低下・代償動作・身体移動能力低下の可能性
""",
        "en": """
**Overall Score = Symmetry×0.30 + Stability×0.30 + Compensation×0.20 + Mobility×0.20**
 
| Component | Weight | Source |
|---|---|---|
| **Symmetry Score** | 30% | Average left-right difference (Asymmetry) across hip, knee, and ankle |
| **Stability Score** | 30% | Left-right / front-back pelvis displacement (Pelvic Shift) |
| **Compensation Score** | 20% | Change in lumbar extension (Lumbar Compensation) |
| **Mobility Score** | 20% | Vertical pelvis displacement (Seat-Off Height) |
 
- **High score** — A stable, efficient sit-to-stand movement
- **Low score** — May indicate asymmetry, reduced postural control, compensation, or reduced body displacement ability
""",
    },
})
 
# =========================================================================
# pages/singly_sit_stan.py
# =========================================================================
TRANSLATIONS.update({
    "single_sit_stand.side_auto_caption": {
        "ja": "Right合計ROM={right} / Left合計ROM={left} を比較し、運動量の大きい側を自動的に「{side}」データとして判定しています。",
        "en": "Comparing total ROM (Right={right} / Left={left}), the side with the greater range of motion is automatically selected as the \"{side}\" data.",
    },
    "single_sit_stand.analyzed_side_label": {"ja": "解析対象側", "en": "Analyzed Side"},
    "single_sit_stand.phase_caption": {
        "ja": "片足立ち-座位動作中の各フェーズ（Sitting・Rising・Standing・Lowering）を時系列で可視化したグラフです。",
        "en": "A time-series chart visualizing each phase of the single-leg sit-to-stand movement (Sitting, Rising, Standing, Lowering).",
    },
    "single_sit_stand.event_info_caption": {
        "ja": "下の数値は、動作データ全体の中で「Sitting（着座位）」「Standing（立位）」と判定された**フレーム番号**（何コマ目か）です。OpenCapのサンプリングレートは60Hzのため、フレーム番号 ÷ 60 で動作開始からの経過秒数に変換できます。",
        "en": "The numbers below are the **frame numbers** identified as \"Sitting\" and \"Standing\" within the full movement data. Since OpenCap's sampling rate is 60Hz, dividing the frame number by 60 converts it to elapsed seconds since the movement started.",
    },
    "single_sit_stand.sitting_metric_label": {"ja": "Sitting（着座位のフレーム番号）", "en": "Sitting (sitting frame number)"},
    "single_sit_stand.standing_metric_label": {"ja": "Standing（立位のフレーム番号）", "en": "Standing (standing frame number)"},
    "single_sit_stand.dashboard_caption": {"ja": "Sit-to-Stand動作の主要指標を表示します", "en": "Shows the key metrics of the sit-to-stand movement"},
    "single_sit_stand.metrics_table": {
        "ja": """
| 指標 | 説明 |
|---|---|
| **Max Knee Flexion** | 解析対象側の膝関節最大屈曲角度 |
| **Max Hip Flexion** | 立ち上がり開始時の股関節最大屈曲角度 |
| **Max Ankle Motion** | 動作中の足関節角度変化量 |
| **Lumbar Compensation** | 腰椎伸展の変化量。股関節・足関節の可動性不足を補う代償動作の可能性 |
| **Pelvis Tilt Compensation** | 骨盤前後傾の変化量。骨盤制御能力の指標 |
| **ROM Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""",
        "en": """
| Metric | Description |
|---|---|
| **Max Knee Flexion** | Maximum knee flexion angle of the analyzed side |
| **Max Hip Flexion** | Maximum hip flexion angle at the start of standing up |
| **Max Ankle Motion** | Change in ankle angle during the movement |
| **Lumbar Compensation** | Change in lumbar extension. May indicate a compensation strategy for limited hip/ankle mobility |
| **Pelvis Tilt Compensation** | Change in pelvic tilt. An indicator of pelvic control ability |
| **ROM Deviation** | Average deviation rate from the healthy range of motion (Healthy ROM) |
""",
    },
    "single_sit_stand.checkbox_instruction": {
        "ja": "💡 左のチェックボックスで、表示する関節・骨盤・腰椎の指標を選択できます。",
        "en": "💡 Use the checkboxes on the left to choose which joint, pelvis, and lumbar metrics to display.",
    },
    "single_sit_stand.joint_rom_content": {
        "ja": """
関節が動作中にどれだけ動いたかを示す指標です。
 
**ROM = 最大角度 − 最小角度**
 
| 関節 | 評価内容 |
|---|---|
| **Hip** | 股関節屈曲角度。立ち上がり時の体幹前傾戦略を評価 |
| **Knee** | 膝関節屈曲量。立ち上がりに必要な下肢運動を評価 |
| **Ankle** | 足関節運動。足部による重心移動能力を評価 |
""",
        "en": """
A metric showing how much each joint moved during the movement.
 
**ROM = Maximum angle − Minimum angle**
 
| Joint | What it evaluates |
|---|---|
| **Hip** | Hip flexion angle. Evaluates the trunk forward-lean strategy when standing up |
| **Knee** | Amount of knee flexion. Evaluates the lower-limb motion required to stand up |
| **Ankle** | Ankle motion. Evaluates the foot's ability to shift the center of mass |
""",
    },
    "single_sit_stand.feature_caption": {
        "ja": "身体移動・姿勢制御・代償動作・関節可動域からSingle Sit-to-Stand動作を評価する特徴量です。",
        "en": "Features that evaluate the single sit-to-stand movement based on body displacement, postural control, compensation, and joint range of motion.",
    },
    "single_sit_stand.feature_table": {
        "ja": """
| 特徴量 | 説明 |
|---|---|
| **Seat-Off Height** | 骨盤の垂直移動量（立ち上がりの深さ） |
| **Pelvic Shift** | 骨盤の左右・前後移動量の最大値 |
| **Lumbar Compensation** | 腰椎伸展の変化量。大きいほど代償動作の可能性 |
| **Hip ROM** | 股関節可動域（最大角度−最小角度） |
| **Knee ROM** | 膝関節可動域（最大角度−最小角度） |
| **Ankle ROM** | 足関節可動域（最大角度−最小角度） |
""",
        "en": """
| Feature | Description |
|---|---|
| **Seat-Off Height** | Vertical pelvis displacement (depth of standing up) |
| **Pelvic Shift** | Maximum left-right / front-back pelvis displacement |
| **Lumbar Compensation** | Change in lumbar extension. Larger may indicate compensation |
| **Hip ROM** | Hip range of motion (max angle − min angle) |
| **Knee ROM** | Knee range of motion (max angle − min angle) |
| **Ankle ROM** | Ankle range of motion (max angle − min angle) |
""",
    },
    "single_sit_stand.score_content": {
        "ja": """
**Overall Score = Mobility×0.40 + Stability×0.40 + Compensation×0.20**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Mobility Score** | 40% | Hip/Knee/AnkleのROMを健常基準値と比較したスコアの平均 |
| **Stability Score** | 40% | 骨盤前後傾（Tilt）・左右傾斜（Obliquity）の変化量 |
| **Compensation Score** | 20% | 腰椎伸展（Lumbar Extension）の変化量 |
 
- **高スコア** — 安定した効率的なSingle Sit-to-Stand動作
- **低スコア** — 可動域不足・姿勢制御低下・代償動作の可能性
""",
        "en": """
**Overall Score = Mobility×0.40 + Stability×0.40 + Compensation×0.20**
 
| Component | Weight | Source |
|---|---|---|
| **Mobility Score** | 40% | Average of the Hip/Knee/Ankle ROM scores compared against healthy reference values |
| **Stability Score** | 40% | Change in pelvic front-back tilt (Tilt) and left-right obliquity (Obliquity) |
| **Compensation Score** | 20% | Change in lumbar extension (Lumbar Extension) |
 
- **High score** — A stable, efficient single sit-to-stand movement
- **Low score** — May indicate reduced range of motion, reduced postural control, or compensation
""",
    },
})
 
# =========================================================================
# pages/arm_flexion.py
# =========================================================================
TRANSLATIONS.update({
    "arm_flexion.phase_caption": {
        "ja": "肩関節挙上動作（Start・Raising・Top・Lowering）の各フェーズを時系列で表示します。",
        "en": "Shows each phase of the shoulder-raise movement (Start, Raising, Top, Lowering) over time.",
    },
    "arm_flexion.event_info_caption": {
        "ja": "下の数値は、動作データ全体の中で「Start（挙上開始位置）」「Stop（挙上完了・Top到達位置）」と判定された**フレーム番号**（何コマ目か）です。OpenCapのサンプリングレートは60Hzのため、フレーム番号 ÷ 60 で動作開始からの経過秒数に変換できます。",
        "en": "The numbers below are the **frame numbers** identified as \"Start\" (beginning of the raise) and \"Stop\" (raise complete / Top reached) within the full movement data. Since OpenCap's sampling rate is 60Hz, dividing the frame number by 60 converts it to elapsed seconds since the movement started.",
    },
    "arm_flexion.start_metric_label": {"ja": "Start（挙上開始のフレーム番号）", "en": "Start (raise-start frame number)"},
    "arm_flexion.stop_metric_label": {"ja": "Stop（挙上完了のフレーム番号）", "en": "Stop (raise-complete frame number)"},
    "arm_flexion.movement_analysis_caption": {
        "ja": "肩関節挙上および体幹・骨盤運動の時系列変化を表示します。",
        "en": "Shows the time-series change of shoulder raise and trunk/pelvis motion.",
    },
    "arm_flexion.pelvic_rotation_rom_caption": {
        "ja": "Pelvic Rotation ROM（この動作中の骨盤回旋の可動域）: {value}°",
        "en": "Pelvic Rotation ROM (the range of pelvic rotation during this movement): {value}°",
    },
    "arm_flexion.symmetry_caption": {
        "ja": "肩関節挙上動作における左右肩関節ROM差を各Phaseごとに評価します。",
        "en": "Evaluates the left-right shoulder ROM difference in the shoulder-raise movement, for each phase.",
    },
    "arm_flexion.healthy_rom_caption": {
        "ja": "肩関節挙上動作における正常可動域（Healthy ROM）との比較を行います。",
        "en": "Compares the shoulder-raise movement against the healthy reference range of motion (Healthy ROM).",
    },
    "arm_flexion.dashboard_caption": {"ja": "肩関節挙上動作の主要指標を表示します", "en": "Shows the key metrics of the shoulder-raise movement"},
    "arm_flexion.metrics_table": {
        "ja": """
| 指標 | 説明 |
|---|---|
| **Max Shoulder Flexion** | 肩関節挙上動作中の最大肩屈曲角度（左右のうち大きい方） |
| **Lumbar Compensation** | 肩を挙上する際に生じる腰椎伸展の変化量。肩の可動域不足を補う代償動作を評価 |
| **Pelvis Compensation** | 肩挙上動作中の骨盤傾斜変化量。骨盤の姿勢制御・下半身からの代償動作を評価 |
| **ROM Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""",
        "en": """
| Metric | Description |
|---|---|
| **Max Shoulder Flexion** | Maximum shoulder flexion angle during the raise (whichever side is larger) |
| **Lumbar Compensation** | Change in lumbar extension that occurs while raising the shoulder. Evaluates compensation for limited shoulder mobility |
| **Pelvis Compensation** | Change in pelvic tilt during the shoulder raise. Evaluates pelvic postural control / lower-body compensation |
| **ROM Deviation** | Average deviation rate from the healthy range of motion (Healthy ROM) |
""",
    },
    "arm_flexion.checkbox_instruction": {
        "ja": "💡 左のチェックボックスで、表示する肩関節・体幹・骨盤の指標を選択できます。",
        "en": "💡 Use the checkboxes on the left to choose which shoulder, trunk, and pelvis metrics to display.",
    },
    "arm_flexion.lumbar_placeholder": {
        "ja": "左の「Trunk」→「Lumbar Extension」にチェックを入れると表示されます",
        "en": "Check \"Trunk\" → \"Lumbar Extension\" on the left to display this chart",
    },
    "arm_flexion.joint_rom_content": {
        "ja": """
関節が動作中にどれだけ動いたかを示す指標です。
 
**ROM = 最大角度 − 最小角度**
 
| 部位 | 評価内容 |
|---|---|
| **Right Shoulder** | 右肩屈曲角度（arm_flex_r）。肩挙上動作における右肩の可動量 |
| **Left Shoulder** | 左肩屈曲角度（arm_flex_l）。左右肩関節の運動量を比較 |
| **Lumbar** | 腰椎伸展角度（lumbar_extension）。肩挙上時の体幹代償動作を評価 |
| **Pelvis Tilt** | 骨盤傾斜角度（pelvis_tilt）。肩挙上中の姿勢制御を評価 |
""",
        "en": """
A metric showing how much each joint moved during the movement.
 
**ROM = Maximum angle − Minimum angle**
 
| Segment | What it evaluates |
|---|---|
| **Right Shoulder** | Right shoulder flexion angle (arm_flex_r). The right shoulder's range of motion during the raise |
| **Left Shoulder** | Left shoulder flexion angle (arm_flex_l). Compares left-right shoulder range of motion |
| **Lumbar** | Lumbar extension angle (lumbar_extension). Evaluates trunk compensation during the raise |
| **Pelvis Tilt** | Pelvic tilt angle (pelvis_tilt). Evaluates postural control during the raise |
""",
    },
    "arm_flexion.joint_asymmetry_content": {
        "ja": """
左右の肩関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。
 
- **15%以下** — 左右対称。バランスの良い肩挙上パターン
- **15%超** — 左右差あり。肩関節可動域制限・筋力差・体幹代償動作の可能性
""",
        "en": """
The left-right difference in shoulder range of motion (ROM), normalized by the larger side's ROM and shown as a percentage.
 
- **15% or less** — Symmetric. A well-balanced shoulder-raise pattern
- **Over 15%** — Asymmetric. May indicate limited shoulder mobility, a strength difference, or trunk compensation
""",
    },
    "arm_flexion.feature_caption": {
        "ja": "肩関節可動域・体幹/骨盤代償動作・左右差から肩挙上動作を評価する特徴量です。",
        "en": "Features that evaluate the shoulder-raise movement based on shoulder range of motion, trunk/pelvis compensation, and left-right differences.",
    },
    "arm_flexion.feature_table": {
        "ja": """
| 特徴量 | 説明 |
|---|---|
| **Shoulder ROM** | 左右肩関節ROMの平均値（右肩・左肩それぞれの最大角度と最小角度の差から算出） |
| **Lumbar Compensation** | 肩挙上動作中に生じる腰椎伸展角度の変化量（ROM） |
| **Pelvis Tilt Compensation** | 肩挙上動作中に生じる骨盤傾斜角度の変化量（ROM） |
| **Shoulder Asymmetry** | 右肩と左肩ROMの左右差（%） |
""",
        "en": """
| Feature | Description |
|---|---|
| **Shoulder ROM** | Average of left and right shoulder ROM (each computed as max angle − min angle) |
| **Lumbar Compensation** | Change in lumbar extension angle (ROM) that occurs during the shoulder raise |
| **Pelvis Tilt Compensation** | Change in pelvic tilt angle (ROM) that occurs during the shoulder raise |
| **Shoulder Asymmetry** | Left-right difference between right and left shoulder ROM (%) |
""",
    },
    "arm_flexion.score_caption": {
        "ja": "肩関節の可動性・左右対称性・体幹/骨盤の代償動作を総合評価する100点満点のスコアです。",
        "en": "An overall score out of 100 that evaluates shoulder mobility, left-right symmetry, and trunk/pelvis compensation.",
    },
    "arm_flexion.score_content": {
        "ja": """
**Overall Score = Symmetry×0.30 + Mobility×0.40 + Lumbar×0.15 + Pelvis×0.15**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Symmetry Score** | 30% | 肩関節左右差（Shoulder Asymmetry）。差が小さいほど高スコア |
| **Mobility Score** | 40% | 肩関節ROMを180°基準で評価。可動域が大きいほど高スコア |
| **Lumbar Score** | 15% | 肩挙上時の腰椎伸展による代償動作。伸展が大きいほど低スコア |
| **Pelvis Score** | 15% | 肩挙上時の骨盤傾斜変化。動きが少ないほど高スコア |
""",
        "en": """
**Overall Score = Symmetry×0.30 + Mobility×0.40 + Lumbar×0.15 + Pelvis×0.15**
 
| Component | Weight | Source |
|---|---|---|
| **Symmetry Score** | 30% | Left-right shoulder difference (Shoulder Asymmetry). Smaller difference gives a higher score |
| **Mobility Score** | 40% | Shoulder ROM evaluated against a 180° reference. Larger ROM gives a higher score |
| **Lumbar Score** | 15% | Compensation via lumbar extension during the raise. Larger extension gives a lower score |
| **Pelvis Score** | 15% | Pelvic tilt change during the raise. Less movement gives a higher score |
""",
    },
})
 
# =========================================================================
# pages/gait.py
# =========================================================================
TRANSLATIONS.update({
    "gait.phase_caption": {
        "ja": "右脚・左脚の歩行フェーズ（Heel Strike・Mid Stance・Toe Off・Swing）を時系列で可視化したグラフです。",
        "en": "A time-series chart visualizing the right and left leg gait phases (Heel Strike, Mid Stance, Toe Off, Swing).",
    },
    "gait.metrics_caption": {
        "ja": "**Cadence** は1分間あたりの歩数（歩/分）で、歩行のリズム・速さを表す指標です。**Step Time** は1歩（あるHeel Strikeから次のHeel Strikeまで）に要する平均時間（秒）です。",
        "en": "**Cadence** is the number of steps per minute, a measure of gait rhythm and speed. **Step Time** is the average time (in seconds) for one step, from one Heel Strike to the next.",
    },
    "gait.cadence_caption": {"ja": "1分間あたりの歩数", "en": "Steps per minute"},
    "gait.step_time_caption": {"ja": "1歩あたりの平均時間", "en": "Average time per step"},
    "gait.movement_analysis_caption": {
        "ja": "各関節運動および骨盤運動の時系列変化を表示します。",
        "en": "Shows the time-series change of each joint's and the pelvis's motion.",
    },
    "gait.symmetry_caption": {
        "ja": "左右関節ROMの左右差を各歩行フェーズごとに評価します。",
        "en": "Evaluates the left-right joint ROM difference for each gait phase.",
    },
    "gait.dashboard_caption": {"ja": "歩行動作の主要指標を一覧表示します。", "en": "Lists the key metrics of the gait movement."},
    "gait.metrics_table": {
        "ja": """
| 指標 | 説明 |
|---|---|
| **Cadence** | 1分間あたりの歩数（steps/min） |
| **Hip ROM** | 股関節可動域（全フレーム中の最大屈曲角度−最小屈曲角度） |
| **Knee ROM** | 膝関節可動域 |
| **Ankle ROM** | 足関節可動域（底屈・背屈） |
| **Pelvic Tilt ROM** | 骨盤前後傾の可動域 |
| **Pelvic Rotation ROM** | 骨盤回旋の可動域 |
| **Pelvic Obliquity ROM** | 骨盤左右傾斜（Pelvic List）の可動域 |
| **Pelvic ML Stability** | 骨盤左右方向の安定性（pelvis_txの標準偏差） |
| **Lumbar Extension ROM** | 腰椎伸展の可動域 |
| **ROM Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""",
        "en": """
| Metric | Description |
|---|---|
| **Cadence** | Steps per minute (steps/min) |
| **Hip ROM** | Hip range of motion (max flexion − min flexion across all frames) |
| **Knee ROM** | Knee range of motion |
| **Ankle ROM** | Ankle range of motion (plantarflexion/dorsiflexion) |
| **Pelvic Tilt ROM** | Range of pelvic front-back tilt |
| **Pelvic Rotation ROM** | Range of pelvic rotation |
| **Pelvic Obliquity ROM** | Range of pelvic left-right obliquity (Pelvic List) |
| **Pelvic ML Stability** | Left-right pelvic stability (standard deviation of pelvis_tx) |
| **Lumbar Extension ROM** | Range of lumbar extension |
| **ROM Deviation** | Average deviation rate from the healthy range of motion (Healthy ROM) |
""",
    },
    "gait.checkbox_instruction": {
        "ja": "💡 左のチェックボックスで、表示する下肢関節・骨盤・腰椎の指標を選択できます。",
        "en": "💡 Use the checkboxes on the left to choose which lower-limb, pelvis, and lumbar metrics to display.",
    },
    "gait.joint_rom_content": {
        "ja": """
歩行周期（Gait Cycle）全体において各関節がどの程度動いたかを示す指標です。
 
**ROM = 最大関節角度 − 最小関節角度**
 
| 関節 | 評価内容 |
|---|---|
| **Hip** | 歩行中の股関節屈曲・伸展運動の可動範囲 |
| **Knee** | 歩行中の膝関節屈曲・伸展運動の可動範囲 |
| **Ankle** | 歩行中の足関節運動（底屈・背屈）の可動範囲 |
""",
        "en": """
A metric showing how much each joint moved over the entire gait cycle.
 
**ROM = Maximum joint angle − Minimum joint angle**
 
| Joint | What it evaluates |
|---|---|
| **Hip** | Range of hip flexion/extension motion during gait |
| **Knee** | Range of knee flexion/extension motion during gait |
| **Ankle** | Range of ankle motion (plantarflexion/dorsiflexion) during gait |
""",
    },
    "gait.joint_asymmetry_content": {
        "ja": """
左右の関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。
 
- **15%以下** — 左右対称。バランスの取れた歩行パターン
- **15%超** — 左右差あり。筋力差・可動域制限・荷重偏位・歩行時の代償動作の可能性
""",
        "en": """
The left-right difference in joint range of motion (ROM), normalized by the larger side's ROM and shown as a percentage.
 
- **15% or less** — Symmetric. A well-balanced gait pattern
- **Over 15%** — Asymmetric. May indicate a strength difference, limited range of motion, weight-bearing shift, or compensation during gait
""",
    },
    "gait.feature_caption": {
        "ja": "歩行リズム・時間的パラメータ・骨盤制御能力から歩行動作を評価する特徴量です。",
        "en": "Features that evaluate the gait movement based on rhythm, temporal parameters, and pelvic control ability.",
    },
    "gait.feature_table": {
        "ja": """
| 特徴量 | 説明 |
|---|---|
| **Cadence** | 1分間あたりの歩数（steps/min） |
| **Step Time** | 1歩に要する時間（sec） |
| **Pelvic Stability** | 歩行中の骨盤側方傾斜（pelvis_list）の変動性（標準偏差） |
| **Pelvic Rotation Variability** | 歩行中の骨盤回旋（pelvis_rotation）の変動性 |
| **Hip / Knee / Ankle ROM** | 各関節の可動域 |
| **Lumbar Extension ROM** | 歩行中の腰椎伸展運動範囲 |
""",
        "en": """
| Feature | Description |
|---|---|
| **Cadence** | Steps per minute (steps/min) |
| **Step Time** | Time required per step (sec) |
| **Pelvic Stability** | Variability (standard deviation) of pelvic lateral tilt (pelvis_list) during gait |
| **Pelvic Rotation Variability** | Variability of pelvic rotation (pelvis_rotation) during gait |
| **Hip / Knee / Ankle ROM** | Range of motion of each joint |
| **Lumbar Extension ROM** | Range of lumbar extension motion during gait |
""",
    },
    "gait.score_caption": {
        "ja": "左右対称性・歩行リズム・骨盤安定性・体幹代償動作・関節可動性の5要素から算出する100点満点の総合スコアです。",
        "en": "An overall score out of 100, calculated from five components: left-right symmetry, gait rhythm, pelvic stability, trunk compensation, and joint mobility.",
    },
    "gait.score_content": {
        "ja": """
**Overall Score = Symmetry×0.25 + Cadence×0.15 + Pelvic ML×0.20 + Lumbar Extension×0.15 + Mobility×0.25**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Symmetry Score** | 25% | 股関節・膝関節・足関節の左右ROM差の平均値 |
| **Cadence Score** | 15% | 基準Cadence（110 steps/min）との差 |
| **Pelvic ML Stability Score** | 20% | 骨盤左右傾斜（pelvis_list）の変動性 |
| **Lumbar Extension Score** | 15% | 腰椎伸展量が基準値（10°）からどれだけ離れているか |
| **Mobility Score** | 25% | 股・膝・足関節ROMの総合的な可動性 |
 
**スコアの目安**
 
- **90〜100** — Excellent
- **70〜89** — Good
- **70未満** — 歩行能力や運動機能の低下が示唆される
""",
        "en": """
**Overall Score = Symmetry×0.25 + Cadence×0.15 + Pelvic ML×0.20 + Lumbar Extension×0.15 + Mobility×0.25**
 
| Component | Weight | Source |
|---|---|---|
| **Symmetry Score** | 25% | Average of left-right ROM difference across hip, knee, and ankle |
| **Cadence Score** | 15% | Difference from the reference cadence (110 steps/min) |
| **Pelvic ML Stability Score** | 20% | Variability of pelvic left-right tilt (pelvis_list) |
| **Lumbar Extension Score** | 15% | How far the lumbar extension amount is from the reference value (10°) |
| **Mobility Score** | 25% | Overall mobility of hip, knee, and ankle ROM |
 
**Score guide**
 
- **90–100** — Excellent
- **70–89** — Good
- **Below 70** — May indicate reduced walking ability or motor function
""",
    },
})
 
