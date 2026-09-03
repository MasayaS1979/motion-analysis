import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from i18n import t, language_switcher
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image
from reportlab.lib.enums import TA_CENTER
from xml.sax.saxutils import escape
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
pdfmetrics.registerFontFamily(
    "HeiseiKakuGo-W5",
    normal="HeiseiKakuGo-W5", bold="HeiseiKakuGo-W5",
    italic="HeiseiKakuGo-W5", boldItalic="HeiseiKakuGo-W5"
)
def fig_to_rl_image(fig, width_cm=16):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    fig_w, fig_h = fig.get_size_inches()
    plt.close(fig)
    img_width = width_cm * cm
    img_height = img_width * (fig_h / fig_w)
    return Image(buf, width=img_width, height=img_height)
 
st.set_page_config(page_title="Single Sit-to-Stand Analysis", layout="wide")
 
language_switcher()
 
# =========================
# Dark Theme CSS
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
        padding-top:1rem;
    }
 
    header[data-testid="stHeader"]{
        background:transparent;
    }
 
    [data-testid="stDecoration"]{
        display:none;
    }
 
    /* -------------------------
       Buttons (Download CSV / Excel / PDF etc.)
       The global div/span/p color:white rule above makes the
       label text white, but Streamlit's default button keeps a
       light background — that combination makes the button
       text invisible. Give buttons an explicit dark background
       and border so the white label is readable.
    ------------------------- */
 
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
 
st.title("Single Sit-to-Stand Analysis")
 
uploaded_file = st.session_state.get("uploaded_file")
 
if uploaded_file is None:
    st.warning(t("common.upload_warning"))
    st.stop()
 
# =========================
# Phase Detection
# =========================
df = pd.read_excel(uploaded_file, header=10)
 
signal = df["pelvis_ty"]
 
signal_smooth = (
    signal.rolling(window=5, center=True)
    .mean()
    .bfill()
    .ffill()
)
 
velocity = signal_smooth.diff()
 
phase_order = ["Sitting", "Rising", "Standing", "Lowering"]
 
phases = []
velocity_threshold = 0.002
 
pelvis_min = signal_smooth.min()
pelvis_max = signal_smooth.max()
pelvis_range = pelvis_max - pelvis_min
 
sitting_threshold = pelvis_min + pelvis_range * 0.20
standing_threshold = pelvis_max - pelvis_range * 0.20
 
# -------------------------
# Sitting / Standing Event
# (mirrors the Bottom/Standing markers used on the other pages)
# -------------------------
 
sitting_idx = signal_smooth.idxmin()
standing_idx = signal_smooth.iloc[sitting_idx:].idxmax()
 
for p, v in zip(signal_smooth, velocity):
 
    if pd.isna(v):
        phase = "Sitting"
 
    # Sitting：最下点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    # → まだ動いている区間（Rising/Loweringの途中）をここに含めない
    elif p <= sitting_threshold and abs(v) < velocity_threshold:
        phase = "Sitting"
 
    # Standing：最高点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    # → Rising/Loweringの通過点をStandingに含めない
    elif p >= standing_threshold and abs(v) < velocity_threshold:
        phase = "Standing"
 
    # まだ動いている場合は、位置に関係なく速度方向で判定
    elif v > velocity_threshold:
        phase = "Rising"
 
    elif v < -velocity_threshold:
        phase = "Lowering"
 
    else:
        phase = (
            phases[-1]
            if len(phases) > 0
            else "Sitting"
        )
 
    phases.append(phase)
 
# -------------------------
# Keep only the single contiguous cluster of Sitting / Standing
# that is nearest to the detected event (sitting_idx / standing_idx).
#
# The position+velocity classification above can label more than one
# separated "quiet" cluster as Sitting or Standing (e.g. a brief stall
# partway through the motion, or noise near the threshold boundary).
# Averaging max/min ROM across two unrelated quiet clusters inflates
# Sitting_ROM / Standing_ROM well beyond what a single steady posture
# would produce — which then feeds directly into the Right_ROM /
# Left_ROM values shown in Symmetry Analysis. Here we keep only the
# run that contains (or is closest to) the true event frame, and fold
# every other same-label run back into whatever phase preceded it.
# -------------------------
 
def keep_nearest_cluster(phase_list, target_label, target_idx):
    phase_list = list(phase_list)
    runs = []
    run_start = None
    for i, p in enumerate(phase_list):
        if p == target_label:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(phase_list) - 1))
    if len(runs) <= 1:
        return phase_list

    def distance_to_target(run):
        start, end = run
        if start <= target_idx <= end:
            return 0
        return min(abs(target_idx - start), abs(target_idx - end))

    keep_run = min(runs, key=distance_to_target)
    for start, end in runs:
        if (start, end) == keep_run:
            continue
        if start > 0:
            fallback_label = phase_list[start - 1]
        elif end + 1 < len(phase_list):
            fallback_label = phase_list[end + 1]
        else:
            fallback_label = target_label
        for i in range(start, end + 1):
            phase_list[i] = fallback_label
    return phase_list
 
phases = keep_nearest_cluster(phases, "Sitting", sitting_idx)
phases = keep_nearest_cluster(phases, "Standing", standing_idx)
 
df["Phase"] = phases
df_phase = df.copy()
 
# =========================
# Phase Summary
# =========================
numeric_columns = df_phase.select_dtypes(include=np.number).columns
 
summary_rows = []
 
for variable in numeric_columns:
 
    row = {"Variable": variable}
 
    for phase in phase_order:
 
        phase_df = df_phase[df_phase["Phase"] == phase]
 
        if len(phase_df) > 0:
 
            row[f"{phase}_Min"] = round(phase_df[variable].min(), 4)
            row[f"{phase}_Max"] = round(phase_df[variable].max(), 4)
            row[f"{phase}_Mean"] = round(phase_df[variable].mean(), 4)
            row[f"{phase}_Std"] = round(phase_df[variable].std(), 4)
            row[f"{phase}_ROM"] = round(
                phase_df[variable].max() - phase_df[variable].min(),
                4
            )
 
        else:
 
            # A phase may legitimately have zero frames (e.g. a test
            # that ends at Standing never has a Lowering phase). Fill
            # with NaN so every row keeps the same set of columns and
            # downstream column selection never raises a KeyError.
            row[f"{phase}_Min"] = np.nan
            row[f"{phase}_Max"] = np.nan
            row[f"{phase}_Mean"] = np.nan
            row[f"{phase}_Std"] = np.nan
            row[f"{phase}_ROM"] = np.nan
 
    summary_rows.append(row)
 
phase_summary_df = pd.DataFrame(summary_rows)
 
# =========================
# Healthy ROM
# =========================
HEALTHY_ROM = {
    "hip_flexion_r":{"min":60.0,"max":100.0},
    "hip_flexion_l":{"min":60.0,"max":100.0},
    "knee_angle_r":{"min":70.0,"max":110.0},
    "knee_angle_l":{"min":70.0,"max":110.0},
    "ankle_angle_r":{"min":15.0,"max":30.0},
    "ankle_angle_l":{"min":15.0,"max":30.0},
    "pelvis_ty":{"min":0.05,"max":0.35},
    # NOTE: pelvis_tilt / pelvis_rotation / lumbar_extension are trunk
    # "compensation" signals rather than a primary joint ROM — smaller
    # excursion is generally considered better sit-to-stand form. The
    # 0-10° band below is a placeholder threshold for "acceptable
    # compensation" and should be reviewed/adjusted against your own
    # clinical reference rather than treated as an established
    # normative range.
    "pelvis_tilt":{"min":0.0,"max":10.0},
    "pelvis_rotation":{"min":0.0,"max":10.0},
    "lumbar_extension":{"min":0.0,"max":10.0}
}
 
healthy_rom_df = pd.DataFrame([
    {
        "Variable": variable,
        "Min": values["min"],
        "Max": values["max"]
    }
    for variable, values in HEALTHY_ROM.items()
])
 
def calculate_subject_rom(df):
 
    rom_dict = {}
 
    for variable in HEALTHY_ROM.keys():
 
        if variable in df.columns:
 
            rom_dict[variable] = (
                df[variable].max()
                -
                df[variable].min()
            )
 
    return rom_dict
 
def compare_subject_to_healthy(
    df_subject,
    healthy_rom_df
):
 
    subject_roms = calculate_subject_rom(df_subject)
 
    results = []
 
    for variable, subject_rom in subject_roms.items():
 
        healthy_row = healthy_rom_df[
            healthy_rom_df["Variable"] == variable
        ]
 
        if len(healthy_row) == 0:
            continue
 
        healthy_min = healthy_row["Min"].iloc[0]
        healthy_max = healthy_row["Max"].iloc[0]
 
        healthy_mid = (
            healthy_min + healthy_max
        ) / 2
 
        difference_percent = (
            (subject_rom - healthy_mid)
            / healthy_mid
            * 100
        )
 
        is_outside = not (
            healthy_min
            <= subject_rom
            <= healthy_max
        )
 
        results.append({
            "Variable": variable,
            "Subject_ROM": round(subject_rom,2),
            "Healthy_Min": healthy_min,
            "Healthy_Max": healthy_max,
            "ROM_Difference_%": round(difference_percent,2),
            "Out_of_Range": is_outside
        })
 
    return pd.DataFrame(results)
 
comparison_df = compare_subject_to_healthy(
    df_phase,
    healthy_rom_df
)
 
# =========================
# Auto Analysis Side Detection
# =========================
#
# Ground truth: the stance (working) leg performs the large knee
# extension needed to stand up, so its Knee ROM is normally the single
# most reliable indicator of which side was tested — the free/lifted
# leg is usually held still (small ROM) or moves in a non-functional
# way once raised. Summing Hip+Knee+Ankle ROM (the previous approach)
# let a noisy Hip or Ankle signal on the free leg outweigh a clean
# Knee signal on the stance leg, which could flip the decision.
# Knee ROM is therefore used as the primary signal; Hip+Knee+Ankle is
# only used as a tiebreaker when the Knee ROM difference is small.
# =========================
 
knee_rom_r = (
    df_phase["knee_angle_r"].max()
    -
    df_phase["knee_angle_r"].min()
)
 
knee_rom_l = (
    df_phase["knee_angle_l"].max()
    -
    df_phase["knee_angle_l"].min()
)
 
hip_rom_r = (
    df_phase["hip_flexion_r"].max()
    -
    df_phase["hip_flexion_r"].min()
)
 
hip_rom_l = (
    df_phase["hip_flexion_l"].max()
    -
    df_phase["hip_flexion_l"].min()
)
 
ankle_rom_r = (
    df_phase["ankle_angle_r"].max()
    -
    df_phase["ankle_angle_r"].min()
)
 
ankle_rom_l = (
    df_phase["ankle_angle_l"].max()
    -
    df_phase["ankle_angle_l"].min()
)
 
# -------------------------
# Majority vote across Hip / Knee / Ankle
#
# Real data showed that a single noisy joint (often Ankle) can have a
# much larger ROM on the non-tested leg than either Knee-only or a
# straight Hip+Knee+Ankle sum can tolerate — e.g. Hip and Knee both
# pointed to Left as the tested side, but a large Ankle ROM on the
# Right leg was enough to flip a simple sum in Right's favor. Voting
# per-joint and taking whichever side wins 2 of 3 joints is robust to
# exactly this failure mode: one outlier joint can no longer overrule
# two joints that agree.
# -------------------------
 
votes_right = 0
votes_left = 0
 
if hip_rom_r > hip_rom_l:
    votes_right += 1
elif hip_rom_l > hip_rom_r:
    votes_left += 1
 
if knee_rom_r > knee_rom_l:
    votes_right += 1
elif knee_rom_l > knee_rom_r:
    votes_left += 1
 
if ankle_rom_r > ankle_rom_l:
    votes_right += 1
elif ankle_rom_l > ankle_rom_r:
    votes_left += 1
 
right_score = hip_rom_r + knee_rom_r + ankle_rom_r
left_score = hip_rom_l + knee_rom_l + ankle_rom_l
 
if votes_right > votes_left:
    auto_side = "Right"
elif votes_left > votes_right:
    auto_side = "Left"
else:
    # Only possible with an exact tie on one joint (rare) — fall back
    # to the combined Hip+Knee+Ankle ROM.
    auto_side = "Right" if right_score > left_score else "Left"
 
ANALYSIS_SIDE = auto_side
 
if ANALYSIS_SIDE == "Right":
    HIP = "hip_flexion_r"
    KNEE = "knee_angle_r"
    ANKLE = "ankle_angle_r"
else:
    HIP = "hip_flexion_l"
    KNEE = "knee_angle_l"
    ANKLE = "ankle_angle_l"
 
# =========================
# Filter Analysis Side Only
# (shared by the Healthy ROM Comparison tab and the Clinical Report tab
# so both report on the same, analyzed side)
#
# pelvis_ty / pelvis_tilt / pelvis_rotation / lumbar_extension are not
# left/right variants — they describe the trunk/pelvis as a whole, so
# they are always kept regardless of which leg was analyzed.
# =========================
 
NON_SIDE_SPECIFIC_VARIABLES = [
    "pelvis_ty",
    "pelvis_tilt",
    "pelvis_rotation",
    "lumbar_extension"
]
 
if ANALYSIS_SIDE == "Right":
 
    comparison_display_df = comparison_df[
        comparison_df["Variable"].str.endswith("_r")
        |
        comparison_df["Variable"].isin(NON_SIDE_SPECIFIC_VARIABLES)
    ]
 
else:
 
    comparison_display_df = comparison_df[
        comparison_df["Variable"].str.endswith("_l")
        |
        comparison_df["Variable"].isin(NON_SIDE_SPECIFIC_VARIABLES)
    ]
 
# =========================
# Analysis Side Badge
# (large, high-visibility indicator so it's immediately clear whether
# a Left- or Right-side Single Sit-to-Stand recording was uploaded)
# =========================
 
badge_color = "#1E90FF" if ANALYSIS_SIDE == "Right" else "#FF8C00"
 
st.markdown(
    f"""
    <div style="
        background-color:{badge_color};
        color:white;
        padding:16px 24px;
        border-radius:12px;
        font-size:26px;
        font-weight:800;
        text-align:center;
        letter-spacing:1px;
        margin-bottom:24px;
        box-shadow:0 2px 8px rgba(0,0,0,0.4);
    ">
        🦵 {t("single_sit_stand.analyzed_side_label")}: {ANALYSIS_SIDE.upper()}
    </div>
    """,
    unsafe_allow_html=True
)
 
st.caption(
    t(
        "single_sit_stand.side_auto_caption",
        right=round(right_score, 1),
        left=round(left_score, 1),
        side=ANALYSIS_SIDE,
    )
)
 
# =========================
# Tabs
# =========================
 
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Phase Analysis", "Movement Analysis", "Healthy ROM Comparison",
    "Clinical Report", "Raw Data", "Dashboard", "Movement Score", "PDF Report", "Client Report"
])
 
# =========================
# Tab1
# =========================
 
with tab1:
 
    st.subheader(
        "Phase Detection Plot"
    )
 
    st.caption(t("single_sit_stand.phase_caption"))
 
    colors_phase = {
        "Sitting": "red",
        "Rising": "limegreen",
        "Standing": "dodgerblue",
        "Lowering": "orange"
    }
 
    fig, ax = plt.subplots(
        figsize=(15,6)
    )
 
    # =========================
    # Dark Theme
    # =========================
 
    fig.patch.set_facecolor(
        "black"
    )
 
    ax.set_facecolor(
        "black"
    )
 
    # 軸・目盛
    ax.tick_params(
        colors="white"
    )
 
    ax.xaxis.label.set_color(
        "white"
    )
 
    ax.yaxis.label.set_color(
        "white"
    )
 
    ax.title.set_color(
        "white"
    )
 
    # 軸線
    for spine in ax.spines.values():
 
        spine.set_color(
            "white"
        )
 
    # Grid
    ax.grid(
 
        True,
 
        color="white",
 
        alpha=0.3
 
    )
 
    # =========================
    # Pelvis Trajectory
    # =========================
 
    ax.plot(
        df_phase.index,
        df_phase["pelvis_ty"],
        color="white",
        linewidth=2,
        label="Pelvis TY"
    )
 
    for phase in phase_order:
 
        idx = (
            df_phase["Phase"]
            ==
            phase
        )
 
        ax.scatter(
 
            df_phase.index[idx],
 
            df_phase["pelvis_ty"][idx],
 
            c=colors_phase[phase],
 
            s=10,
 
            label=phase
 
        )
 
    # -------------------------
    # Event Markers
    # -------------------------
 
    ax.axvline(sitting_idx, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(standing_idx, color="dodgerblue", linestyle="--", linewidth=1.5)
 
    ax.text(
        sitting_idx,
        signal_smooth.iloc[sitting_idx],
        "Sitting",
        color="red",
        fontsize=9
    )
 
    ax.text(
        standing_idx,
        signal_smooth.iloc[standing_idx],
        "Standing",
        color="dodgerblue",
        fontsize=9
    )
 
    ax.set_title(
        "Single Sit-to-Stand Phase Detection"
    )
 
    ax.set_xlabel(
        "Frame"
    )
 
    ax.set_ylabel(
        "Pelvis Vertical Position (m)"
    )
 
    # =========================
    # Legend
    # =========================
 
    legend = ax.legend()
 
    for text in legend.get_texts():
 
        text.set_color(
            "white"
        )
 
    legend.get_frame().set_facecolor(
        "black"
    )
 
    legend.get_frame().set_edgecolor(
        "white"
    )
 
    st.pyplot(
        fig
    )
 
    # -------------------------
    # Event Information
    # -------------------------
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.metric("Sitting", sitting_idx)
 
    with col2:
        st.metric("Standing", standing_idx)
 
    # ==========================================================
    # Phase Summary Table
    # ==========================================================
 
    st.subheader("Phase Summary Table")
 
    st.caption(t("common.phase_summary_caption"))
 
    st.markdown(t("common.phase_metric_explanation"))
 
    st.dataframe(
        phase_summary_df,
        use_container_width=True
    )
 
    # ==========================================================
    # Excel Download
    # ==========================================================
 
    excel_buffer = BytesIO()
 
    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:
 
        phase_summary_df.to_excel(
            writer,
            sheet_name="Phase Summary",
            index=False
        )
 
        # Sitting
        sitting_df = phase_summary_df[
            [
                "Variable",
                "Sitting_Min",
                "Sitting_Max",
                "Sitting_Mean",
                "Sitting_Std",
                "Sitting_ROM"
            ]
        ]
 
        sitting_df.to_excel(
            writer,
            sheet_name="Sitting",
            index=False
        )
 
        # Rising
        rising_df = phase_summary_df[
            [
                "Variable",
                "Rising_Min",
                "Rising_Max",
                "Rising_Mean",
                "Rising_Std",
                "Rising_ROM"
            ]
        ]
 
        rising_df.to_excel(
            writer,
            sheet_name="Rising",
            index=False
        )
 
        # Standing
        standing_df = phase_summary_df[
            [
                "Variable",
                "Standing_Min",
                "Standing_Max",
                "Standing_Mean",
                "Standing_Std",
                "Standing_ROM"
            ]
        ]
 
        standing_df.to_excel(
            writer,
            sheet_name="Standing",
            index=False
        )
 
        # Lowering
        lowering_df = phase_summary_df[
            [
                "Variable",
                "Lowering_Min",
                "Lowering_Max",
                "Lowering_Mean",
                "Lowering_Std",
                "Lowering_ROM"
            ]
        ]
 
        lowering_df.to_excel(
            writer,
            sheet_name="Lowering",
            index=False
        )
 
    st.download_button(
        "📥 Download Excel",
        data=excel_buffer.getvalue(),
        file_name="Phase_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
    # ==========================================================
    # PDF Download
    # ==========================================================
 
    pdf_buffer = BytesIO()
 
    doc = SimpleDocTemplate(pdf_buffer)
 
    table_data = [
        phase_summary_df.columns.tolist()
    ] + phase_summary_df.values.tolist()
 
    table = Table(table_data)
 
    table.setStyle(TableStyle([
 
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
 
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
 
        ('GRID',(0,0),(-1,-1),0.5,colors.black),
 
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
 
        ('FONTSIZE',(0,0),(-1,-1),8),
 
        ('BOTTOMPADDING',(0,0),(-1,0),8),
 
    ]))
 
    doc.build([table])
 
    st.download_button(
        "📄 Download PDF",
        data=pdf_buffer.getvalue(),
        file_name="Phase_Summary.pdf",
        mime="application/pdf"
    )
 
    # ==========================================================
    # Phase Statistics
    # ==========================================================
 
    st.markdown("---")
    st.subheader("Phase Statistics")
 
    tab_sitting, tab_rising, tab_standing, tab_lowering = st.tabs(
        [
            "Sitting",
            "Rising",
            "Standing",
            "Lowering"
        ]
    )
 
    with tab_sitting:
 
        st.caption("Sitting Phase")
 
        st.dataframe(
            sitting_df,
            use_container_width=True
        )
 
    with tab_rising:
 
        st.caption("Rising Phase")
 
        st.dataframe(
            rising_df,
            use_container_width=True
        )
 
    with tab_standing:
 
        st.caption("Standing Phase")
 
        st.dataframe(
            standing_df,
            use_container_width=True
        )
 
    with tab_lowering:
 
        st.caption("Lowering Phase")
 
        st.dataframe(
            lowering_df,
            use_container_width=True
        )
 
# =========================
# Tab2
# =========================
 
with tab2:
 
    st.subheader("Joint Time Series")
 
    st.caption(t("common.joint_time_series_caption"))
 
    # OpenCap sampling rate (60 Hz)
    time = np.arange(len(df_phase)) / 60
 
    fig, ax = plt.subplots(
        8,
        1,
        figsize=(12,27)
    )
 
    # =========================
    # Dark Theme
    # =========================
 
    fig.patch.set_facecolor("black")
 
    for a in ax:
 
        a.set_facecolor("black")
 
        a.tick_params(
            colors="white"
        )
 
        a.title.set_color(
            "white"
        )
 
        a.xaxis.label.set_color(
            "white"
        )
 
        a.yaxis.label.set_color(
            "white"
        )
 
        for spine in a.spines.values():
 
            spine.set_color(
                "white"
            )
 
        a.grid(
            True,
            color="white",
            alpha=0.3
        )
 
    # =====================
    # Knee
    # =====================
 
    ax[0].plot(
        time,
        df_phase[KNEE],
        linewidth=2,
        label=f"{ANALYSIS_SIDE} Knee"
    )
 
    ax[0].set_title(
        "Knee Angle"
    )
 
    ax[0].set_xlabel(
        "Time (s)"
    )
 
    ax[0].set_ylabel(
        "Angle (deg)"
    )
 
    legend = ax[0].legend()
 
    legend.get_frame().set_facecolor(
        "black"
    )
 
    legend.get_frame().set_edgecolor(
        "white"
    )
 
    for text in legend.get_texts():
 
        text.set_color(
            "white"
        )
 
    # =====================
    # Hip
    # =====================
 
    ax[1].plot(
        time,
        df_phase[HIP],
        linewidth=2,
        label=f"{ANALYSIS_SIDE} Hip"
    )
 
    ax[1].set_title(
        "Hip Flexion"
    )
 
    ax[1].set_xlabel(
        "Time (s)"
    )
 
    ax[1].set_ylabel(
        "Angle (deg)"
    )
 
    legend = ax[1].legend()
 
    legend.get_frame().set_facecolor(
        "black"
    )
 
    legend.get_frame().set_edgecolor(
        "white"
    )
 
    for text in legend.get_texts():
 
        text.set_color(
            "white"
        )
 
    # =====================
    # Ankle
    # =====================
 
    ax[2].plot(
        time,
        df_phase[ANKLE],
        linewidth=2,
        label=f"{ANALYSIS_SIDE} Ankle"
    )
 
    ax[2].set_title(
        "Ankle Angle"
    )
 
    ax[2].set_xlabel(
        "Time (s)"
    )
 
    ax[2].set_ylabel(
        "Angle (deg)"
    )
 
    legend = ax[2].legend()
 
    legend.get_frame().set_facecolor(
        "black"
    )
 
    legend.get_frame().set_edgecolor(
        "white"
    )
 
    for text in legend.get_texts():
 
        text.set_color(
            "white"
        )
 
    # =====================
    # Pelvic Tilt
    # =====================
 
    ax[3].plot(
        time,
        df_phase["pelvis_tilt"],
        linewidth=2,
        color="cyan"
    )
 
    ax[3].set_title(
        "Pelvic Tilt"
    )
 
    ax[3].set_xlabel(
        "Time (s)"
    )
 
    ax[3].set_ylabel(
        "Angle (deg)"
    )
 
    # =====================
    # Pelvic TX
    # =====================
 
    ax[4].plot(
        time,
        df_phase["pelvis_tx"],
        linewidth=2,
        color="orange"
    )
 
    ax[4].set_title(
        "Pelvic Right-Left Position"
    )
 
    ax[4].set_xlabel(
        "Time (s)"
    )
 
    ax[4].set_ylabel(
        "Position (m)"
    )
 
    # =====================
    # Pelvic TY
    # =====================
 
    ax[5].plot(
        time,
        df_phase["pelvis_ty"],
        linewidth=2,
        color="lime"
    )
 
    ax[5].set_title(
        "Pelvic Height"
    )
 
    ax[5].set_xlabel(
        "Time (s)"
    )
 
    ax[5].set_ylabel(
        "Position (m)"
    )
 
    # =====================
    # Pelvic Rotation
    # =====================
 
    ax[6].plot(
        time,
        df_phase["pelvis_rotation"],
        linewidth=2,
        color="yellow"
    )
 
    ax[6].set_title(
        "Pelvic Rotation"
    )
 
    ax[6].set_xlabel(
        "Time (s)"
    )
 
    ax[6].set_ylabel(
        "Angle (deg)"
    )
 
    # =====================
    # Lumbar Extension
    # =====================
 
    ax[7].plot(
        time,
        df_phase["lumbar_extension"],
        linewidth=2,
        color="magenta"
    )
 
    ax[7].set_title(
        "Lumbar Extension"
    )
 
    ax[7].set_xlabel(
        "Time (s)"
    )
 
    ax[7].set_ylabel(
        "Angle (deg)"
    )
 
    plt.tight_layout()
 
    st.pyplot(
        fig
    )
 
# =========================
# Tab3 - Healthy ROM Comparison
# =========================
 
with tab3:
 
    st.subheader(
        "Healthy ROM Comparison"
    )
 
    st.caption(t("common.healthy_rom_caption"))
 
    st.caption(t("common.difference_pct_caption"))
 
    # =========================
    # Plotly Table
    # =========================
 
    fig = go.Figure(
 
        data=[
 
            go.Table(
 
                columnwidth=[
                    120,
                    120,
                    120,
                    120
                ],
 
                header=dict(
 
                    values=list(
                        comparison_display_df.columns
                    ),
 
                    fill_color="black",
 
                    font=dict(
                        color="white",
                        size=18
                    ),
 
                    align="center",
 
                    line=dict(
                        color="white",
                        width=1
                    )
 
                ),
 
                cells=dict(
 
                    values=[
                        comparison_display_df[col]
                        for col in comparison_display_df.columns
                    ],
 
                    fill_color="black",
 
                    font=dict(
                        color="white",
                        size=16
                    ),
 
                    align="center",
 
                    height=35,
 
                    line=dict(
                        color="white",
                        width=1
                    )
 
                )
 
            )
 
        ]
 
    )
 
    fig.update_layout(
 
        paper_bgcolor="black",
 
        plot_bgcolor="black",
 
        height=300,
 
        margin=dict(
 
            l=10,
            r=10,
            t=10,
            b=10
 
        )
 
    )
 
    st.plotly_chart(
 
        fig,
 
        use_container_width=True
 
    )
 
    # =========================
    # Difference Bar Plot
    # =========================
 
    bar_colors = comparison_display_df[
 
        "Out_of_Range"
 
    ].map({
 
        True:"red",
 
        False:"royalblue"
 
    })
 
    fig, ax = plt.subplots(
 
        figsize=(10,5)
 
    )
 
    fig.patch.set_facecolor(
        "black"
    )
 
    ax.set_facecolor(
        "black"
    )
 
    ax.tick_params(
        colors="white"
    )
 
    ax.title.set_color(
        "white"
    )
 
    ax.xaxis.label.set_color(
        "white"
    )
 
    ax.yaxis.label.set_color(
        "white"
    )
 
    for spine in ax.spines.values():
 
        spine.set_color(
            "white"
        )
 
    ax.grid(
 
        True,
 
        color="white",
 
        alpha=0.3
 
    )
 
    ax.bar(
 
        comparison_display_df["Variable"],
 
        comparison_display_df["ROM_Difference_%"],
 
        color=bar_colors
 
    )
 
    ax.axhline(
 
        0,
 
        linestyle="--",
 
        color="white"
 
    )
 
    ax.set_ylabel(
 
        "Difference (%)"
 
    )
 
    ax.set_title(
 
        "Healthy ROM Comparison"
 
    )
 
    plt.xticks(
 
        rotation=45,
 
        color="white"
 
    )
 
    st.pyplot(
        fig
    )
 
# =========================
# Tab4 - Clinical Report
# =========================
 
with tab4:
 
    findings = []
 
    for _, row in comparison_display_df.iterrows():
 
        if row["Out_of_Range"]:
 
            findings.append(
                f"{row['Variable']} ROM outside healthy range."
            )
 
    if len(findings) == 0:
 
        st.success(
            "No major abnormalities detected."
        )
 
    else:
 
        for item in findings:
 
            st.write("•", item)
 
# =========================
# Tab5 - Raw Data
# =========================
 
with tab5:
 
    st.subheader(
        "Single Sit-Stand Raw Data"
    )
 
    st.dataframe(
        df_phase,
        use_container_width=True
    )
 
    # =========================
    # CSV Download
    # =========================
 
    csv = df_phase.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )
 
    st.download_button(
        "Download CSV",
        csv,
        "single_sit_stand_raw_data.csv",
        "text/csv"
    )
 
    # =========================
    # Excel Download
    # =========================
 
    output = io.BytesIO()
 
    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
 
        df_phase.to_excel(
            writer,
            index=False,
            sheet_name="RawData"
        )
 
    st.download_button(
        "Download Excel",
        output.getvalue(),
        "single_sit_stand_raw_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
# =========================
# Tab6 - Dashboard
# =========================
 
with tab6:
 
    st.title(
        "Sit-to-Stand Dashboard"
    )
 
    st.caption(t("single_sit_stand.dashboard_caption"))
 
    # =========================
    # KPI
    # Max Hip/Knee/Ankle are shown for BOTH sides here (unlike the
    # rest of the page, which reports only the analyzed ANALYSIS_SIDE)
    # since the free/non-tested leg's motion is still often useful to
    # see side-by-side on the dashboard overview.
    # =========================
 
    max_hip_r = df_phase["hip_flexion_r"].max()
    max_hip_l = df_phase["hip_flexion_l"].max()
 
    max_knee_r = df_phase["knee_angle_r"].max()
    max_knee_l = df_phase["knee_angle_l"].max()
 
    max_ankle_r = df_phase["ankle_angle_r"].max()
    max_ankle_l = df_phase["ankle_angle_l"].max()
 
    # =========================
    # Compensation Metrics
    # =========================
 
    # Lumbar Extension Compensation
    lumbar_compensation = round(
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min(),
        1
    )
 
    # Pelvis Tilt Compensation
    # 骨盤前後傾変化量
    pelvis_tilt_compensation = round(
        df_phase["pelvis_tilt"].max()
        -
        df_phase["pelvis_tilt"].min(),
        1
    )
 
    # Pelvis Rotation Compensation
    pelvis_rotation_compensation = round(
        df_phase["pelvis_rotation"].max()
        -
        df_phase["pelvis_rotation"].min(),
        1
    )
 
    # =========================
    # Key Metrics
    # =========================
 
    st.subheader(
        "Key Metrics"
    )
 
    with st.expander(t("common.metrics_expander_label")):
 
        st.markdown(t("single_sit_stand.metrics_table"))
 
    # =========================
    # KPI Display
    # =========================
 
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5, row1_col6 = st.columns(6)
 
    row1_col1.metric(
        "Max Hip Flexion (R)",
        f"{max_hip_r:.1f}°"
    )
 
    row1_col2.metric(
        "Max Hip Flexion (L)",
        f"{max_hip_l:.1f}°"
    )
 
    row1_col3.metric(
        "Max Knee Flexion (R)",
        f"{max_knee_r:.1f}°"
    )
 
    row1_col4.metric(
        "Max Knee Flexion (L)",
        f"{max_knee_l:.1f}°"
    )
 
    row1_col5.metric(
        "Max Ankle Motion (R)",
        f"{max_ankle_r:.1f}°"
    )
 
    row1_col6.metric(
        "Max Ankle Motion (L)",
        f"{max_ankle_l:.1f}°"
    )
 
    row2_col1, row2_col2, row2_col3 = st.columns(3)
 
    row2_col1.metric(
        "Lumbar Compensation",
        f"{lumbar_compensation:.1f}°"
    )
 
    row2_col2.metric(
        "Pelvis Tilt Compensation",
        f"{pelvis_tilt_compensation:.1f}°"
    )
 
    row2_col3.metric(
        "Pelvic Rotation",
        f"{pelvis_rotation_compensation:.1f}°"
    )
 
    # =========================
    # Interactive Motion Viewer
    # =========================
    st.subheader(
        "Interactive Motion Viewer"
    )
 
    st.caption(t("single_sit_stand.checkbox_instruction"))
 
    left_col, right_col = st.columns(
        [1.2,4]
    )
 
    # ======================================
    # Left Panel
    # ======================================
 
    with left_col:
 
        st.markdown("### Lower Limb")
 
        show_hip = st.checkbox(
            f"{ANALYSIS_SIDE} Hip",
            value=True
        )
 
        show_knee = st.checkbox(
            f"{ANALYSIS_SIDE} Knee",
            value=True
        )
 
        show_ankle = st.checkbox(
            f"{ANALYSIS_SIDE} Ankle",
            value=False
        )
 
        st.markdown("---")
 
        st.markdown("### Pelvis")
 
        show_tilt = st.checkbox("Tilt")
        show_obliquity = st.checkbox("Obliquity")
        show_rotation = st.checkbox("Rotation")
        show_ml = st.checkbox("Medial-Lateral Deviation")
        show_vertical = st.checkbox("Vertical Displacement")
        show_ap = st.checkbox("Anterior-Posterior Deviation")
 
        st.markdown("---")
 
        st.markdown("### Lumbar")
 
        show_lumbar = st.checkbox(
            "Extension"
        )
 
    # ======================================
    # Right Panel
    # ======================================
 
    with right_col:
 
        time = np.arange(len(df_phase)) / 60
 
        fig, ax = plt.subplots(
            figsize=(15,6),
            facecolor="black"
        )
 
        ax.set_facecolor("black")
 
        ax.tick_params(colors="white")
 
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
 
        for spine in ax.spines.values():
            spine.set_color("white")
 
        ax.grid(
            color="white",
            alpha=0.25
        )
 
        # =========================
        # Lower Limb
        # =========================
 
        if show_hip:
 
            ax.plot(
                time,
                df_phase[HIP],
                label=f"{ANALYSIS_SIDE} Hip",
                linewidth=2
            )
 
        if show_knee:
 
            ax.plot(
                time,
                df_phase[KNEE],
                label=f"{ANALYSIS_SIDE} Knee",
                linewidth=2
            )
 
        if show_ankle:
 
            ax.plot(
                time,
                df_phase[ANKLE],
                label=f"{ANALYSIS_SIDE} Ankle",
                linewidth=2
            )
 
        # =========================
        # Lumbar
        # =========================
 
        if show_lumbar:
 
            ax.plot(
                time,
                df_phase["lumbar_extension"],
                label="Lumbar Extension",
                linewidth=2
            )
 
        ax.set_title(
            f"{ANALYSIS_SIDE} Sit-to-Stand Joint Motion"
        )
 
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
 
        legend = ax.legend(
            loc="upper right",
            ncol=2
        )
 
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
 
        for text in legend.get_texts():
            text.set_color("white")
 
        st.pyplot(fig)
 
        # =========================
        # Pelvic Motion
        # =========================
 
        st.subheader(
            "Sit-to-Stand Pelvic Motion"
        )
 
        fig2, ax2 = plt.subplots(
            figsize=(15,5),
            facecolor="black"
        )
 
        ax2.set_facecolor("black")
        ax2.tick_params(colors="white")
 
        ax2.title.set_color("white")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
 
        for spine in ax2.spines.values():
            spine.set_color("white")
 
        ax2.grid(
            color="white",
            alpha=0.25
        )
 
        if show_tilt:
 
            ax2.plot(
                time,
                df_phase["pelvis_tilt"],
                label="Tilt",
                linewidth=2
            )
 
        if show_obliquity:
 
            ax2.plot(
                time,
                df_phase["pelvis_list"],
                label="Obliquity",
                linewidth=2
            )
 
        if show_rotation:
 
            ax2.plot(
                time,
                df_phase["pelvis_rotation"],
                label="Rotation",
                linewidth=2
            )
 
        if show_ml:
 
            ax2.plot(
                time,
                df_phase["pelvis_tx"] * 1000,
                label="Medial-Lateral (mm)",
                linewidth=2
            )
 
        if show_vertical:
 
            ax2.plot(
                time,
                df_phase["pelvis_ty"] * 1000,
                label="Vertical (mm)",
                linewidth=2
            )
 
        if show_ap:
 
            ax2.plot(
                time,
                df_phase["pelvis_tz"] * 1000,
                label="Anterior-Posterior (mm)",
                linewidth=2
            )
 
        # Lumbar Extension is grouped visually under the Pelvis
        # section in the left panel, but was previously not drawn
        # anywhere on this chart — only on the "Joint Motion" chart
        # above. Draw it here too so checking "Extension" is
        # reflected in the Pelvic Motion plot itself.
        if show_lumbar:
 
            ax2.plot(
                time,
                df_phase["lumbar_extension"],
                label="Lumbar Extension",
                linewidth=2,
                linestyle="--"
            )
 
        ax2.set_title(
            "Sit-to-Stand Pelvic Motion"
        )
 
        ax2.set_xlabel(
            "Time (s)"
        )
 
        ax2.set_ylabel(
            "Angle / Translation"
        )
 
        handles, labels = ax2.get_legend_handles_labels()
 
        if handles:
 
            legend2 = ax2.legend()
 
            legend2.get_frame().set_facecolor(
                "black"
            )
 
            legend2.get_frame().set_edgecolor(
                "white"
            )
 
            for text in legend2.get_texts():
 
                text.set_color(
                    "white"
                )
 
        st.pyplot(
            fig2
        )
 
    # =========================
    # Joint ROM Summary
    # =========================
 
    st.subheader(
        "Joint ROM Summary"
    )
 
    with st.expander(t("common.joint_rom_expander_label")):
 
        st.markdown(t("single_sit_stand.joint_rom_content"))
 
    rom_joints = {
 
        "Hip": HIP,
 
        "Knee": KNEE,
 
        "Ankle": ANKLE
 
        }
 
    rom_values = []
 
    for variable in rom_joints.values():
 
        rom = (
 
            df_phase[variable].max()
 
            -
 
            df_phase[variable].min()
 
        )
 
        rom_values.append(
            round(
                rom,
                1
            )
        )
 
    fig, ax = plt.subplots(
        figsize=(8,4),
        facecolor="black"
    )
 
    ax.set_facecolor(
        "black"
    )
 
    ax.tick_params(
        colors="white"
    )
 
    ax.title.set_color(
        "white"
    )
 
    ax.xaxis.label.set_color(
        "white"
    )
 
    ax.yaxis.label.set_color(
        "white"
    )
 
    for spine in ax.spines.values():
 
        spine.set_color(
            "white"
        )
 
    ax.grid(
        color="white",
        alpha=0.25,
        axis="y"
    )
 
    bars = ax.bar(
        list(rom_joints.keys()),
        rom_values
    )
 
    for bar,value in zip(
        bars,
        rom_values
    ):
 
        ax.text(
            bar.get_x()
            +
            bar.get_width()/2,
 
            value,
 
            f"{value:.1f}°",
 
            ha="center",
 
            va="bottom",
 
            color="white"
 
        )
 
    ax.set_ylabel(
        "ROM (deg)"
    )
 
    ax.set_title(
        "Joint ROM"
    )
 
    st.pyplot(
        fig
    )
 
# =========================
# Tab7 - Movement Features & Score
# =========================
 
with tab7:
 
    # =========================
    # Movement Features
    # =========================
 
    st.subheader(
        "Movement Features"
    )
 
    st.caption(t("single_sit_stand.feature_caption"))
 
    with st.expander(t("common.feature_expander_label")):
 
        st.markdown(t("single_sit_stand.feature_table"))
 
    seat_off_height = (
 
        pelvis_max -
 
        pelvis_min
 
    )
 
    pelvic_shift = (
    df_phase["pelvis_tx"].max()
    -
    df_phase["pelvis_tx"].min()
    )
 
    trunk_compensation = round(
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min(),
        2
    )
 
    hip_rom = df_phase[HIP].max() - df_phase[HIP].min()
 
    knee_rom = df_phase[KNEE].max() - df_phase[KNEE].min()
 
    ankle_rom = df_phase[ANKLE].max() - df_phase[ANKLE].min()
 
    feature_df = pd.DataFrame(
 
        {
 
            "Feature":[
 
                "Seat-Off Height",
 
                "Pelvic Shift",
 
                "Lumbar Compensation",
 
                "Hip ROM",
 
                "Knee ROM",
 
                "Ankle ROM"
 
            ],
 
            "Value":[
 
                round(
                    seat_off_height,
                    3
                ),
 
                round(
                    pelvic_shift,
                    3
                ),
 
                round(
                    trunk_compensation,
                    2
                ),
 
                f"{hip_rom:.1f}°",
 
                f"{knee_rom:.1f}°",
 
                f"{ankle_rom:.1f}°"
 
            ]
 
        }
 
    )
 
    # =========================
    # Plotly Feature Table
    # =========================
 
    fig = go.Figure(
 
        data=[
 
            go.Table(
 
                columnwidth=[
 
                    250,
                    120
 
                ],
 
                header=dict(
 
                    values=[
 
                        "Feature",
                        "Value"
 
                    ],
 
                    fill_color="black",
 
                    font=dict(
 
                        color="white",
 
                        size=16
 
                    ),
 
                    align="center",
 
                    line=dict(
 
                        color="white",
 
                        width=1
 
                    )
 
                ),
 
                cells=dict(
 
                    values=[
 
                        feature_df["Feature"],
 
                        feature_df["Value"]
 
                    ],
 
                    fill_color="black",
 
                    font=dict(
 
                        color="white",
 
                        size=15
 
                    ),
 
                    align="center",
 
                    height=45,
 
                    line=dict(
 
                        color="white",
 
                        width=1
 
                    )
 
                )
 
            )
 
        ]
 
    )
 
    fig.update_layout(
 
        height=250,
 
        margin=dict(
 
            l=10,
 
            r=10,
 
            t=10,
 
            b=10
 
        ),
 
        paper_bgcolor="black",
 
        plot_bgcolor="black"
 
    )
 
    st.plotly_chart(
 
        fig,
 
        use_container_width=True
 
    )
 
    # =========================
    # Movement Score
    # =========================
 
    st.subheader(
        "Movement Score"
    )
 
    # =========================
    # Mobility Score
    # ROM Normality Based
    #
    # FIX (see chat): this used to compare the subject's ROM against
    # a single hardcoded "target" value (healthy_hip_rom = 40, etc.)
    # that was totally disconnected from the HEALTHY_ROM min/max
    # ranges already defined above and used on the Healthy ROM
    # Comparison tab. Any subject ROM that was a completely normal
    # value under HEALTHY_ROM could still score near 0 here, because
    # the old formula penalized ANY distance from the single target,
    # including exceeding it (more mobility isn't necessarily bad).
    #
    # Now: reuse HEALTHY_ROM as the single source of truth. Full
    # score (100) when the subject's ROM falls inside [min, max];
    # otherwise the score decreases in proportion to how far outside
    # the range it is, relative to the width of the healthy range.
    # =========================
 
    def calculate_rom_score(
        subject_rom,
        healthy_min,
        healthy_max
    ):
 
        if healthy_min <= subject_rom <= healthy_max:
            return 100.0
 
        healthy_span = healthy_max - healthy_min
 
        if subject_rom < healthy_min:
            deviation = healthy_min - subject_rom
        else:
            deviation = subject_rom - healthy_max
 
        score = (
 
            100
            -
            (
                deviation
                /
                healthy_span
                *
                100
            )
 
        )
 
        return max(
            0,
            round(
                score,
                1
            )
        )
 
    # Healthy ROM Reference
    # Reuses the same HEALTHY_ROM dict defined near the top of the
    # file (single source of truth — matches the Healthy ROM
    # Comparison tab) instead of separate, disconnected single-point
    # targets.
 
    hip_score = calculate_rom_score(
        hip_rom,
        HEALTHY_ROM[HIP]["min"],
        HEALTHY_ROM[HIP]["max"]
    )
 
    knee_score = calculate_rom_score(
        knee_rom,
        HEALTHY_ROM[KNEE]["min"],
        HEALTHY_ROM[KNEE]["max"]
    )
 
    ankle_score = calculate_rom_score(
        ankle_rom,
        HEALTHY_ROM[ANKLE]["min"],
        HEALTHY_ROM[ANKLE]["max"]
    )
 
    mobility_score = round(
 
        (
            hip_score
            +
            knee_score
            +
            ankle_score
        )
        /
        3,
 
        1
 
    )
 
    # =========================
    # Pelvic Control Variables
    # =========================
 
    pelvis_tilt_change = (
 
        df_phase["pelvis_tilt"].max()
        -
        df_phase["pelvis_tilt"].min()
 
    )
 
    pelvis_list_change = (
 
        df_phase["pelvis_list"].max()
        -
        df_phase["pelvis_list"].min()
 
    )
 
    # =========================
    # Stability Score
    #
    # FIX (see chat): the previous "* 2" penalty factor was an
    # untuned placeholder — realistic pelvic tilt/list excursion
    # during a normal sit-to-stand can easily be 15-30°, which the
    # old factor alone could crush to a 40-70 score even for a clean
    # rep. Softened to "* 1" here as a less punitive default; treat
    # this constant as something to calibrate against your own
    # collected subject data (e.g. the mean/SD of pelvic_instability
    # across a healthy reference sample) rather than a validated
    # clinical threshold.
    # =========================
 
    PELVIC_INSTABILITY_PENALTY_FACTOR = 1
 
    pelvic_instability = (abs(pelvis_tilt_change) + abs(pelvis_list_change))
    stability_score = round(max(0, 100 - pelvic_instability * PELVIC_INSTABILITY_PENALTY_FACTOR), 1)
    
 
    # =========================
    # Compensation Score
    #
    # FIX (see chat): same issue and same fix as Stability Score
    # above — the "* 2" factor was an untuned placeholder that overly
    # punished normal-scale lumbar extension changes.
    # =========================
 
    LUMBAR_COMPENSATION_PENALTY_FACTOR = 1
 
    lumbar_change = (
 
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min()
 
    )
 
    compensation_score = round(
 
        max(
            0,
            100 - lumbar_change * LUMBAR_COMPENSATION_PENALTY_FACTOR
        ),
 
        1
 
    )
 
    # =========================
    # Overall Score
    # =========================
 
    overall_score = round(
 
        (
            mobility_score * 0.40
            +
            stability_score * 0.40
            +
            compensation_score * 0.20
        ),
 
        1
 
    )
 
    # =========================
    # Display
    # =========================
 
    with st.expander(t("common.score_expander_label")):
 
        st.markdown(t("single_sit_stand.score_content"))
 
    score_df = pd.DataFrame({
 
        "Component":[
 
           "Mobility Score",
           "Stability Score",
           "Compensation Score",
           "Overall Score"
 
        ],
 
        "Score":[
 
            mobility_score,
            stability_score,
            compensation_score,
            overall_score
 
        ]
 
    })
 
    st.dataframe(
        score_df,
        use_container_width=True
    )
 
    st.metric(
 
         "Overall Score",
 
        f"{overall_score}/100"
 
    )
with tab8:
    lang_choice = st.radio(
        "レポート言語 / Report Language",
        ["日本語", "English"],
        horizontal=True,
        key="pdf_lang_single"
    )
    lang_code = "ja" if lang_choice == "日本語" else "en"
    UI_LABELS = {
        "ja": {
            "header": "📄 PDFレポート",
            "subject_name": "対象者名",
            "exam_date": "測定日",
            "examiner": "検者",
            "comment_label": "総合評価 (Clinical Impression)",
            "auto_generate_button": "🪄 コメントを自動生成（下書き）",
            "auto_generate_caption": "実測値をもとに下書きコメントを自動生成します。内容を確認・編集してからPDFを生成してください。",
            "generate_button": "PDFレポートを生成",
            "download_label": "📥 PDFレポートをダウンロード",
            "success_message": "PDFレポートを生成しました。上のボタンからダウンロードしてください。",
        },
        "en": {
            "header": "📄 PDF Report",
            "subject_name": "Subject Name",
            "exam_date": "Exam Date",
            "examiner": "Examiner",
            "comment_label": "Clinical Impression",
            "auto_generate_button": "🪄 Auto-generate Comment (Draft)",
            "auto_generate_caption": "Generates a draft comment from the measured values. Please review and edit before generating the PDF.",
            "generate_button": "Generate PDF Report",
            "download_label": "📥 Download PDF Report",
            "success_message": "PDF report generated. Use the button above to download.",
        },
    }
    UI = UI_LABELS[lang_code]
    def generate_single_sit_stand_auto_comment(
        lang_code, analysis_side, overall_score,
        hip_score, knee_score, ankle_score,
        lumbar_compensation, pelvis_tilt_compensation, pelvis_rotation_compensation,
        comparison_display_df, phase_summary_df, phase_order, hip_var, knee_var, ankle_var
    ):
        def _resolve_rom_columns(dframe):
            cols = list(dframe.columns)
            def find(*keywords):
                for c in cols:
                    cl = str(c).lower()
                    if all(k in cl for k in keywords):
                        return c
                return None
            oor_col = find("out", "range") or find("range")
            value_col = None
            for c in cols:
                if c == oor_col:
                    continue
                cl = str(c).lower()
                if any(k in cl for k in ["subject", "value", "measured", "result"]):
                    value_col = c
                    break
            return value_col, oor_col
        _, oor_col = _resolve_rom_columns(comparison_display_df)
        out_of_range_vars = []
        if oor_col:
            out_of_range_vars = [
                str(row.get("Variable", idx))
                for idx, row in comparison_display_df.iterrows()
                if bool(row.get(oor_col, False))
            ]
        low_mobility = [
            (label, score)
            for label, score in [("Hip", hip_score), ("Knee", knee_score), ("Ankle", ankle_score)]
            if score < 60
        ]
        # ---- Phase Statistics由来の所見 ----
        phase_stats_variables = {
            "Hip": hip_var,
            "Knee": knee_var,
            "Ankle": ankle_var,
            "Pelvic Tilt": "pelvis_tilt",
            "Pelvic Rotation": "pelvis_rotation",
            "Lumbar Extension": "lumbar_extension",
        }
        # 1) Sitting / Standing フェーズでの姿勢安定性（Stdが大きい = 動揺あり）
        # ※ しきい値2.0°は仮の基準です。臨床基準に合わせて調整してください。
        STATIC_PHASES = ["Sitting", "Standing"]
        STD_THRESHOLD = 2.0
        instability_flags = []
        for stat_label, variable in phase_stats_variables.items():
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            for phase in STATIC_PHASES:
                std_v = var_row[f"{phase}_Std"].iloc[0]
                if pd.notna(std_v) and std_v > STD_THRESHOLD:
                    instability_flags.append((stat_label, phase, std_v))
        # 2) 各関節でROMが最大となるフェーズ（主要な可動局面）
        joint_only_variables = {
            "Hip": hip_var,
            "Knee": knee_var,
            "Ankle": ankle_var,
        }
        dominant_phase_per_joint = {}
        for joint_name, variable in joint_only_variables.items():
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            rom_by_phase = {}
            for phase in phase_order:
                rom_by_phase[phase] = var_row[f"{phase}_ROM"].iloc[0]
            dominant_phase_per_joint[joint_name] = max(rom_by_phase, key=rom_by_phase.get)
        # 3) Rising（立ち上がり）と Lowering（着座）のROM差
        # （Squat/Sit-Standの Descending/Ascending に相当する局面比較）
        # ※ しきい値15%は左右対称性と同じ基準を仮採用しています。
        ecc_con_flags = []
        for joint_name, variable in joint_only_variables.items():
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            rising_val = var_row["Rising_ROM"].iloc[0]
            lowering_val = var_row["Lowering_ROM"].iloc[0]
            if pd.isna(rising_val) or pd.isna(lowering_val):
                continue
            if max(rising_val, lowering_val) == 0:
                continue
            diff_pct = abs(rising_val - lowering_val) / max(rising_val, lowering_val) * 100
            if diff_pct > 15:
                ecc_con_flags.append((joint_name, rising_val, lowering_val, diff_pct))
        if lang_code == "ja":
            side_label = "右" if analysis_side == "Right" else "左"
            if overall_score >= 80:
                score_line = f"本測定（{side_label}側）の総合スコアは{overall_score}/100と良好で、動作全体のパフォーマンスに大きな問題は見られません。"
            elif overall_score >= 60:
                score_line = f"本測定（{side_label}側）の総合スコアは{overall_score}/100であり、動作の一部に改善の余地が見られます。"
            else:
                score_line = f"本測定（{side_label}側）の総合スコアは{overall_score}/100であり、動作パターン全体に注意が必要な所見が複数見られます。"
            lines = [score_line]
            if low_mobility:
                mobility_text = "、".join(f"{label}（スコア{score:.0f}）" for label, score in low_mobility)
                lines.append(
                    f"可動域については、{mobility_text}が健常参考値と比べて低下しており、可動性の制限が疑われます。"
                )
            else:
                lines.append("可動域については、股関節・膝関節・足関節ともに健常参考値と比べて概ね良好な範囲に収まっていました。")
            if out_of_range_vars:
                range_text = "、".join(out_of_range_vars)
                lines.append(f"健常可動域との比較では、{range_text}が基準範囲外となっていました。")
            compensation_notes = []
            if lumbar_compensation > 10:
                compensation_notes.append(f"腰椎伸展の代償動作（{lumbar_compensation:.1f}°）")
            if pelvis_tilt_compensation > 10:
                compensation_notes.append(f"骨盤前後傾の代償動作（{pelvis_tilt_compensation:.1f}°）")
            if pelvis_rotation_compensation > 10:
                compensation_notes.append(f"骨盤回旋の代償動作（{pelvis_rotation_compensation:.1f}°）")
            if compensation_notes:
                lines.append(
                    "体幹・骨盤の代償動作として、" + "、".join(compensation_notes) +
                    "が観察されており、動作制御の代償パターンとして注意が必要です。"
                )
            if instability_flags:
                instab_text = "、".join(f"{phase}フェーズの{label}" for label, phase, std_v in instability_flags)
                lines.append(
                    f"静止保持局面の安定性については、{instab_text}で標準偏差が大きく、"
                    "保持中の姿勢動揺（不安定性）が疑われます。"
                )
            else:
                lines.append("静止保持局面（Sitting / Standing）の姿勢安定性については、標準偏差の観点から顕著な動揺は見られませんでした。")
            if dominant_phase_per_joint:
                dominant_text = "、".join(f"{joint}は{phase}フェーズ" for joint, phase in dominant_phase_per_joint.items())
                lines.append(f"各関節の主要な可動局面は、{dominant_text}で最大のROMを示しています。")
            if ecc_con_flags:
                ecc_text = "、".join(
                    f"{joint}（立ち上がり {rising:.1f}° ／ 着座 {lowering:.1f}°、差 {diff:.1f}%）"
                    for joint, rising, lowering, diff in ecc_con_flags
                )
                lines.append(
                    f"立ち上がり局面（Rising）と着座局面（Lowering）の可動域を比較すると、{ecc_text}で15%を超える差が見られ、"
                    "立ち上がりと着座の間で動作制御パターンに違いがある可能性があります。"
                )
            else:
                lines.append("立ち上がり局面と着座局面のROMを比較すると、いずれの関節も15%以内の差に収まっており、局面間で大きな制御の差は見られませんでした。")
            lines.append("以上は実測値からの自動生成による下書きです。臨床所見・触診所見と合わせて内容をご確認のうえ、必要に応じて修正してください。")
            return "\n".join(lines)
        else:
            if overall_score >= 80:
                score_line = f"The overall score for this {analysis_side}-side trial is {overall_score}/100, indicating generally good movement performance with no major concerns."
            elif overall_score >= 60:
                score_line = f"The overall score for this {analysis_side}-side trial is {overall_score}/100, indicating some areas of the movement pattern that could be improved."
            else:
                score_line = f"The overall score for this {analysis_side}-side trial is {overall_score}/100, indicating several findings across the movement pattern that warrant attention."
            lines = [score_line]
            if low_mobility:
                mobility_text = ", ".join(f"{label} (score {score:.0f})" for label, score in low_mobility)
                lines.append(
                    f"Regarding range of motion, {mobility_text} fell below the healthy reference, suggesting reduced mobility."
                )
            else:
                lines.append("Regarding range of motion, the hip, knee, and ankle all remained within a generally healthy reference range.")
            if out_of_range_vars:
                range_text = ", ".join(out_of_range_vars)
                lines.append(f"Compared to the healthy ROM reference, {range_text} fell outside the reference range.")
            compensation_notes = []
            if lumbar_compensation > 10:
                compensation_notes.append(f"lumbar extension compensation ({lumbar_compensation:.1f}°)")
            if pelvis_tilt_compensation > 10:
                compensation_notes.append(f"pelvic tilt compensation ({pelvis_tilt_compensation:.1f}°)")
            if pelvis_rotation_compensation > 10:
                compensation_notes.append(f"pelvic rotation compensation ({pelvis_rotation_compensation:.1f}°)")
            if compensation_notes:
                lines.append(
                    "Trunk/pelvic compensation was observed, including " + ", ".join(compensation_notes) +
                    ", which should be noted as a compensatory movement pattern."
                )
            if instability_flags:
                instab_text = ", ".join(f"{label} during {phase}" for label, phase, std_v in instability_flags)
                lines.append(
                    f"Regarding stability during static hold phases, elevated standard deviation was observed for {instab_text}, "
                    "suggesting possible postural instability during the hold."
                )
            else:
                lines.append("Regarding stability during the static hold phases (Sitting / Standing), no notable instability was observed based on standard deviation.")
            if dominant_phase_per_joint:
                dominant_text = ", ".join(f"{joint} peaks during {phase}" for joint, phase in dominant_phase_per_joint.items())
                lines.append(f"The dominant phase of motion for each joint was as follows: {dominant_text}.")
            if ecc_con_flags:
                ecc_text = ", ".join(
                    f"{joint} (Rising {rising:.1f}° / Lowering {lowering:.1f}°, diff {diff:.1f}%)"
                    for joint, rising, lowering, diff in ecc_con_flags
                )
                lines.append(
                    f"Comparing the Rising and Lowering phases, a difference exceeding 15% was observed for {ecc_text}, "
                    "suggesting a possible difference in motor control between standing up and sitting down."
                )
            else:
                lines.append("Comparing the Rising and Lowering phases, all joints remained within a 15% difference, with no major difference in control between phases.")
            lines.append("This draft was auto-generated from the measured values. Please review it alongside clinical examination and palpation findings, and edit as needed.")
            return "\n".join(lines)
    st.header(UI["header"])
    col1, col2, col3 = st.columns(3)
    with col1:
        subject_name = st.text_input(UI["subject_name"], key="pdf_subject_name_single")
    with col2:
        exam_date = st.text_input(UI["exam_date"], key="pdf_exam_date_single")
    with col3:
        examiner_name = st.text_input(UI["examiner"], key="pdf_examiner_name_single")
    if st.button(UI["auto_generate_button"], key="pdf_auto_generate_button_single"):
        st.session_state["pdf_clinical_comment_single"] = generate_single_sit_stand_auto_comment(
            lang_code, ANALYSIS_SIDE, overall_score,
            hip_score, knee_score, ankle_score,
            lumbar_compensation, pelvis_tilt_compensation, pelvis_rotation_compensation,
            comparison_display_df, phase_summary_df, phase_order, HIP, KNEE, ANKLE
        )
    st.caption(UI["auto_generate_caption"])
    clinical_comment = st.text_area(
        UI["comment_label"],
        height=150,
        key="pdf_clinical_comment_single"
    )
    if st.button(UI["generate_button"], key="pdf_generate_button_single"):
        LABELS = {
            "ja": {
                "title": "Single Sit-to-Stand Analysis 臨床レポート",
                "subject_line": "対象者名: {name}　測定日: {date}　検者: {examiner}",
                "analysis_side_label": "分析側",
                "phase_detection_heading": "Single Sit-to-Stand 位相検出",
                "key_metrics_heading": "主要指標",
                "metric_col": "指標", "value_col": "値",
                "max_hip_r_row": "最大股関節屈曲（右）(°)",
                "max_hip_l_row": "最大股関節屈曲（左）(°)",
                "max_knee_r_row": "最大膝関節屈曲（右）(°)",
                "max_knee_l_row": "最大膝関節屈曲（左）(°)",
                "max_ankle_r_row": "最大足関節可動（右）(°)",
                "max_ankle_l_row": "最大足関節可動（左）(°)",
                "lumbar_comp_row": "腰椎代償",
                "pelvis_tilt_comp_row": "骨盤前後傾代償",
                "pelvis_rot_comp_row": "骨盤回旋",
                "joint_rom_summary_heading": "関節可動域サマリー（試技全体）",
                "joint_col": "関節", "rom_col": "可動域 (°)",
                "phase_stats_heading": "フェーズ別統計（Min/Max/Mean/Std/ROM）",
                "phase_label": "フェーズ",
                "ps_variable_col": "項目",
                "ps_min_col": "最小",
                "ps_max_col": "最大",
                "ps_mean_col": "平均",
                "ps_std_col": "標準偏差",
                "ps_rom_col": "ROM (°)",
                "joint_rom_analysis_heading": "関節可動域分析（分析側: {side}）",
                "healthy_ref_col": "健常参考値 (°)", "jra_score_col": "スコア",
                "healthy_rom_heading": "健常可動域比較",
                "variable_col": "変数", "subject_value_col": "測定値",
                "healthy_min_col": "健常最小", "healthy_max_col": "健常最大", "oor_col": "範囲外",
                "no_rom_columns": "（ROM比較の列を自動検出できませんでした。利用可能な列: {cols}）",
                "clinical_findings_heading": "臨床所見",
                "no_findings": "重大な異常所見は検出されませんでした。",
                "range_finding": "{var} が健常参考範囲外です。",
                "low_score_finding": "{label}の可動域スコア（{score:.0f}）は健常参考値と比べて可動性の低下を示しています。",
                "clinical_impression_heading": "総合評価 (Clinical Impression)",
                "no_comment": "(記入なし)",
                "overall_score_label": "総合スコア: {s} / 100",
                "movement_score_heading": "動作スコア内訳",
                "component_col": "項目", "msb_score_col": "スコア",
                "additional_features_heading": "追加特徴量",
                "feature_col": "特徴量", "value_col2": "値",
            },
            "en": {
                "title": "Single Sit-to-Stand Analysis Clinical Report",
                "subject_line": "Subject Name: {name}  Exam Date: {date}  Examiner: {examiner}",
                "analysis_side_label": "Analysis Side",
                "phase_detection_heading": "Single Sit-to-Stand Phase Detection",
                "key_metrics_heading": "Key Metrics",
                "metric_col": "Metric", "value_col": "Value",
                "max_hip_r_row": "Max Hip Flexion (R) (°)",
                "max_hip_l_row": "Max Hip Flexion (L) (°)",
                "max_knee_r_row": "Max Knee Flexion (R) (°)",
                "max_knee_l_row": "Max Knee Flexion (L) (°)",
                "max_ankle_r_row": "Max Ankle Motion (R) (°)",
                "max_ankle_l_row": "Max Ankle Motion (L) (°)",
                "lumbar_comp_row": "Lumbar Compensation",
                "pelvis_tilt_comp_row": "Pelvis Tilt Compensation",
                "pelvis_rot_comp_row": "Pelvic Rotation",
                "joint_rom_summary_heading": "Joint ROM Summary (Whole Trial)",
                "joint_col": "Joint", "rom_col": "ROM (°)",
                "phase_stats_heading": "Phase Statistics (Min/Max/Mean/Std/ROM)",
                "phase_label": "Phase",
                "ps_variable_col": "Variable",
                "ps_min_col": "Min",
                "ps_max_col": "Max",
                "ps_mean_col": "Mean",
                "ps_std_col": "Std",
                "ps_rom_col": "ROM (deg)",
                "joint_rom_analysis_heading": "Joint ROM Analysis (Analysis Side: {side})",
                "healthy_ref_col": "Healthy Reference (°)", "jra_score_col": "Score",
                "healthy_rom_heading": "Healthy ROM Comparison",
                "variable_col": "Variable", "subject_value_col": "Subject Value",
                "healthy_min_col": "Healthy Min", "healthy_max_col": "Healthy Max", "oor_col": "Out of Range",
                "no_rom_columns": "(Could not detect ROM comparison columns automatically. Available columns: {cols})",
                "clinical_findings_heading": "Clinical Findings",
                "no_findings": "No significant abnormal findings detected.",
                "range_finding": "{var} is outside the healthy reference range.",
                "low_score_finding": "{label} ROM score ({score:.0f}) indicates reduced mobility relative to healthy reference.",
                "clinical_impression_heading": "Clinical Impression",
                "no_comment": "(No comment entered)",
                "overall_score_label": "OVERALL SCORE: {s} / 100",
                "movement_score_heading": "Movement Score Breakdown",
                "component_col": "Component", "msb_score_col": "Score",
                "additional_features_heading": "Additional Features",
                "feature_col": "Feature", "value_col2": "Value",
            },
        }
        LB = LABELS[lang_code]
        buf_pdf = BytesIO()
        doc = SimpleDocTemplate(buf_pdf, pagesize=A4,
                                 topMargin=1.5*cm, bottomMargin=1.5*cm,
                                 leftMargin=1.5*cm, rightMargin=1.5*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleJP", parent=styles["Title"],
            fontName="HeiseiKakuGo-W5", fontSize=18, alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            "HeadingJP", parent=styles["Heading2"],
            fontName="HeiseiKakuGo-W5", fontSize=13, spaceBefore=12, spaceAfter=6
        )
        normal_style = ParagraphStyle(
            "NormalJP", parent=styles["Normal"],
            fontName="HeiseiKakuGo-W5", fontSize=10, leading=14
        )
        badge_style = ParagraphStyle(
            "BadgeJP", parent=styles["Normal"],
            fontName="HeiseiKakuGo-W5", fontSize=11, alignment=TA_CENTER,
            textColor=colors.white,
            backColor=colors.HexColor("#1565C0") if ANALYSIS_SIDE == "Right" else colors.HexColor("#C2185B"),
            borderPadding=6
        )
        # ---- Overall Score用ハイライトスタイル（コンパクト版） ----
        if overall_score >= 80:
            score_color = colors.HexColor("#2E7D32")
        elif overall_score >= 60:
            score_color = colors.HexColor("#F9A825")
        else:
            score_color = colors.HexColor("#C62828")
        score_style = ParagraphStyle(
            "ScoreStyle", parent=styles["Normal"],
            fontName="HeiseiKakuGo-W5", fontSize=16, leading=20, alignment=TA_CENTER,
            textColor=colors.white, backColor=score_color,
            borderPadding=6, spaceBefore=6, spaceAfter=8
        )
        TABLE_STYLE = TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ])
        elements = []
        elements.append(Paragraph(LB["title"], title_style))
        elements.append(Spacer(1, 0.5*cm))
        subject_line = LB["subject_line"].format(
            name=escape(subject_name), date=escape(exam_date), examiner=escape(examiner_name)
        )
        elements.append(Paragraph(subject_line, normal_style))
        elements.append(Spacer(1, 0.3*cm))
        elements.append(Paragraph(f"{LB['analysis_side_label']}: {ANALYSIS_SIDE}", badge_style))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Phase Detection Plot ----
        # 注意: matplotlibにCJKフォントが無いため、グラフ内のタイトル/軸ラベル/凡例は
        # 言語選択に関わらず固定の英語表記にしています。
        elements.append(Paragraph(LB["phase_detection_heading"], heading_style))
        fig_phase_pdf, ax_phase_pdf = plt.subplots(figsize=(10, 4))
        colors_phase_pdf = {"Sitting": "red", "Rising": "limegreen", "Standing": "dodgerblue", "Lowering": "orange"}
        ax_phase_pdf.plot(df.index, signal, color="black", linewidth=1, alpha=0.4)
        for phase in phase_order:
            idx = df_phase["Phase"] == phase
            ax_phase_pdf.scatter(
                df_phase.index[idx],
                signal[idx],
                c=colors_phase_pdf.get(phase, "gray"),
                s=8,
                label=phase
            )
        ax_phase_pdf.axvline(sitting_idx, color="red", linestyle="--", linewidth=1)
        ax_phase_pdf.axvline(standing_idx, color="dodgerblue", linestyle="--", linewidth=1)
        ax_phase_pdf.set_title("Phase Detection Plot")
        ax_phase_pdf.set_xlabel("Frame")
        ax_phase_pdf.set_ylabel("Pelvis Vertical Position (m)")
        ax_phase_pdf.legend(loc="upper right", fontsize=8)
        ax_phase_pdf.grid(alpha=0.3)
        elements.append(fig_to_rl_image(fig_phase_pdf, width_cm=16))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Key Metrics ----
        elements.append(Paragraph(LB["key_metrics_heading"], heading_style))
        key_metrics_data = [
            [LB["metric_col"], LB["value_col"]],
            [LB["max_hip_r_row"], f"{max_hip_r:.1f}"],
            [LB["max_hip_l_row"], f"{max_hip_l:.1f}"],
            [LB["max_knee_r_row"], f"{max_knee_r:.1f}"],
            [LB["max_knee_l_row"], f"{max_knee_l:.1f}"],
            [LB["max_ankle_r_row"], f"{max_ankle_r:.1f}"],
            [LB["max_ankle_l_row"], f"{max_ankle_l:.1f}"],
            [LB["lumbar_comp_row"], f"{lumbar_compensation:.2f}"],
            [LB["pelvis_tilt_comp_row"], f"{pelvis_tilt_compensation:.2f}"],
            [LB["pelvis_rot_comp_row"], f"{pelvis_rotation_compensation:.2f}"],
        ]
        key_metrics_table = Table(key_metrics_data, colWidths=[9*cm, 6*cm])
        key_metrics_table.setStyle(TABLE_STYLE)
        elements.append(key_metrics_table)
        elements.append(Spacer(1, 0.5*cm))
        # ---- Joint ROM Summary (試技全体でのROM。tab6のJoint ROM Summaryと同じ計算) ----
        elements.append(Paragraph(LB["joint_rom_summary_heading"], heading_style))
        rom_joints_pdf = {"Hip": HIP, "Knee": KNEE, "Ankle": ANKLE}
        rom_summary_rows = [[LB["joint_col"], LB["rom_col"]]]
        rom_summary_values = []
        for joint_name, variable in rom_joints_pdf.items():
            rom_val = df_phase[variable].max() - df_phase[variable].min()
            rom_summary_values.append(rom_val)
            rom_summary_rows.append([joint_name, f"{rom_val:.1f}"])
        rom_summary_table = Table(rom_summary_rows, hAlign="LEFT")
        rom_summary_table.setStyle(TABLE_STYLE)
        elements.append(rom_summary_table)
        elements.append(Spacer(1, 0.3*cm))
        fig_rom_summary_pdf, ax_rom_summary_pdf = plt.subplots(figsize=(6, 3))
        ax_rom_summary_pdf.bar(list(rom_joints_pdf.keys()), rom_summary_values, color="royalblue")
        for i, v in enumerate(rom_summary_values):
            ax_rom_summary_pdf.text(i, v, f"{v:.1f}°", ha="center", va="bottom", fontsize=8)
        ax_rom_summary_pdf.set_ylabel("ROM (deg)")
        ax_rom_summary_pdf.set_title("Joint ROM Summary (Whole Trial)")
        ax_rom_summary_pdf.grid(alpha=0.3, axis="y")
        elements.append(fig_to_rl_image(fig_rom_summary_pdf, width_cm=11))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Phase Statistics (Min/Max/Mean/Std/ROM per Phase) ----
        elements.append(Paragraph(LB["phase_stats_heading"], heading_style))
        side_tag = "R" if ANALYSIS_SIDE == "Right" else "L"
        phase_stats_variables = {
            f"Hip ({side_tag})": HIP,
            f"Knee ({side_tag})": KNEE,
            f"Ankle ({side_tag})": ANKLE,
            "Pelvic Tilt": "pelvis_tilt",
            "Pelvic Rotation": "pelvis_rotation",
            "Lumbar Extension": "lumbar_extension",
        }
        for phase in phase_order:
            elements.append(Paragraph(
                f"{LB['phase_label']}: {phase}",
                normal_style
            ))
            phase_stats_rows = [[
                LB["ps_variable_col"], LB["ps_min_col"], LB["ps_max_col"],
                LB["ps_mean_col"], LB["ps_std_col"], LB["ps_rom_col"]
            ]]
            for stat_label, variable in phase_stats_variables.items():
                var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
                if len(var_row) == 0:
                    continue
                min_v = var_row[f"{phase}_Min"].iloc[0]
                max_v = var_row[f"{phase}_Max"].iloc[0]
                mean_v = var_row[f"{phase}_Mean"].iloc[0]
                std_v = var_row[f"{phase}_Std"].iloc[0]
                rom_v = var_row[f"{phase}_ROM"].iloc[0]
                phase_stats_rows.append([
                    stat_label,
                    f"{min_v:.1f}" if pd.notna(min_v) else "-",
                    f"{max_v:.1f}" if pd.notna(max_v) else "-",
                    f"{mean_v:.1f}" if pd.notna(mean_v) else "-",
                    f"{std_v:.1f}" if pd.notna(std_v) else "-",
                    f"{rom_v:.1f}" if pd.notna(rom_v) else "-",
                ])
            phase_stats_table = Table(phase_stats_rows, hAlign="LEFT")
            phase_stats_table.setStyle(TABLE_STYLE)
            elements.append(phase_stats_table)
            elements.append(Spacer(1, 0.3*cm))
        elements.append(Spacer(1, 0.2*cm))
        # ---- Joint ROM Analysis (healthy reference + score つき、既存のロジック) ----
        elements.append(Paragraph(LB["joint_rom_analysis_heading"].format(side=ANALYSIS_SIDE), heading_style))
        rom_analysis_data = [
            [LB["joint_col"], LB["rom_col"], LB["healthy_ref_col"], LB["jra_score_col"]],
            ["Hip", f"{hip_rom:.1f}", f"{HEALTHY_ROM[HIP]['min']:.0f}-{HEALTHY_ROM[HIP]['max']:.0f}", f"{hip_score:.0f}"],
            ["Knee", f"{knee_rom:.1f}", f"{HEALTHY_ROM[KNEE]['min']:.0f}-{HEALTHY_ROM[KNEE]['max']:.0f}", f"{knee_score:.0f}"],
            ["Ankle", f"{ankle_rom:.1f}", f"{HEALTHY_ROM[ANKLE]['min']:.0f}-{HEALTHY_ROM[ANKLE]['max']:.0f}", f"{ankle_score:.0f}"],
        ]
        rom_analysis_table = Table(rom_analysis_data, colWidths=[4*cm, 4*cm, 5*cm, 3*cm])
        rom_analysis_table.setStyle(TABLE_STYLE)
        elements.append(rom_analysis_table)
        elements.append(Spacer(1, 0.3*cm))
        fig_rom_pdf, ax_rom_pdf = plt.subplots(figsize=(8, 4))
        joint_labels = ["Hip", "Knee", "Ankle"]
        subject_values = [hip_rom, knee_rom, ankle_rom]
        healthy_mid_values = [
            (HEALTHY_ROM[HIP]["min"] + HEALTHY_ROM[HIP]["max"]) / 2,
            (HEALTHY_ROM[KNEE]["min"] + HEALTHY_ROM[KNEE]["max"]) / 2,
            (HEALTHY_ROM[ANKLE]["min"] + HEALTHY_ROM[ANKLE]["max"]) / 2,
        ]
        x_pos = range(len(joint_labels))
        ax_rom_pdf.bar([x - 0.2 for x in x_pos], subject_values, width=0.4, label="Subject", color="dodgerblue")
        ax_rom_pdf.bar([x + 0.2 for x in x_pos], healthy_mid_values, width=0.4, label="Healthy Reference (mid)", color="lightgray")
        ax_rom_pdf.set_xticks(list(x_pos))
        ax_rom_pdf.set_xticklabels(joint_labels)
        ax_rom_pdf.set_ylabel("ROM (°)")
        ax_rom_pdf.legend(fontsize=8)
        elements.append(fig_to_rl_image(fig_rom_pdf, width_cm=14))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Healthy ROM Comparison (comparison_display_df, 列名は自動検出) ----
        elements.append(Paragraph(LB["healthy_rom_heading"], heading_style))
        def _resolve_rom_columns(dframe):
            cols = list(dframe.columns)
            def find(*keywords):
                for c in cols:
                    cl = str(c).lower()
                    if all(k in cl for k in keywords):
                        return c
                return None
            oor_col = find("out", "range") or find("range")
            min_col = find("min")
            max_col = find("max")
            value_col = None
            for c in cols:
                if c in (oor_col, min_col, max_col):
                    continue
                cl = str(c).lower()
                if any(k in cl for k in ["subject", "value", "measured", "result"]):
                    value_col = c
                    break
            if value_col is None:
                for c in cols:
                    if c in (oor_col, min_col, max_col):
                        continue
                    if pd.api.types.is_numeric_dtype(dframe[c]):
                        value_col = c
                        break
            return value_col, min_col, max_col, oor_col
        VALUE_COL, MIN_COL, MAX_COL, OOR_COL = _resolve_rom_columns(comparison_display_df)
        if VALUE_COL is None:
            elements.append(Paragraph(
                LB["no_rom_columns"].format(cols=list(comparison_display_df.columns)),
                normal_style
            ))
        else:
            rom_table_data = [[
                LB["variable_col"], LB["subject_value_col"],
                LB["healthy_min_col"], LB["healthy_max_col"], LB["oor_col"]
            ]]
            for idx, row in comparison_display_df.iterrows():
                rom_table_data.append([
                    str(row.get("Variable", idx)),
                    f"{row[VALUE_COL]:.2f}" if VALUE_COL else "",
                    f"{row[MIN_COL]:.2f}" if MIN_COL else "",
                    f"{row[MAX_COL]:.2f}" if MAX_COL else "",
                    "⚠️" if (OOR_COL and bool(row.get(OOR_COL, False))) else "",
                ])
            rom_table = Table(rom_table_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm, 2.5*cm])
            rom_table.setStyle(TABLE_STYLE)
            elements.append(rom_table)
            elements.append(Spacer(1, 0.3*cm))
            if OOR_COL:
                bar_colors = ["crimson" if bool(v) else "seagreen" for v in comparison_display_df[OOR_COL]]
            else:
                bar_colors = "seagreen"
            fig_healthy_pdf, ax_healthy_pdf = plt.subplots(figsize=(10, 4))
            ax_healthy_pdf.bar(comparison_display_df.index.astype(str), comparison_display_df[VALUE_COL], color=bar_colors)
            ax_healthy_pdf.set_ylabel("Value")
            ax_healthy_pdf.tick_params(axis="x", rotation=45)
            elements.append(fig_to_rl_image(fig_healthy_pdf, width_cm=16))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Clinical Findings ----
        elements.append(Paragraph(LB["clinical_findings_heading"], heading_style))
        findings = []
        if OOR_COL:
            for idx, row in comparison_display_df.iterrows():
                if bool(row.get(OOR_COL, False)):
                    findings.append(LB["range_finding"].format(var=row.get("Variable", idx)))
        for label, score in [("Hip", hip_score), ("Knee", knee_score), ("Ankle", ankle_score)]:
            if score < 60:
                findings.append(LB["low_score_finding"].format(label=label, score=score))
        findings_text = "<br/>".join(findings) if findings else LB["no_findings"]
        elements.append(Paragraph(findings_text, normal_style))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Clinical Impression ----
        elements.append(Paragraph(LB["clinical_impression_heading"], heading_style))
        comment_text = escape(clinical_comment).replace("\n", "<br/>") if clinical_comment else LB["no_comment"]
        elements.append(Paragraph(comment_text, normal_style))
        elements.append(Spacer(1, 0.5*cm))
        # ---- Overall Score (highlighted, コンパクト) ----
        elements.append(Paragraph(LB["overall_score_label"].format(s=overall_score), score_style))
        elements.append(Spacer(1, 0.3*cm))
        # ---- Movement Score Breakdown ----
        elements.append(Paragraph(LB["movement_score_heading"], heading_style))
        score_table_data = [[LB["component_col"], LB["msb_score_col"]]] + [[str(c) for c in row] for row in score_df.values.tolist()]
        score_table = Table(score_table_data, colWidths=[9*cm, 6*cm])
        score_table.setStyle(TABLE_STYLE)
        elements.append(score_table)
        elements.append(Spacer(1, 0.5*cm))
        # ---- Additional Features ----
        elements.append(Paragraph(LB["additional_features_heading"], heading_style))
        feature_table_data = [[LB["feature_col"], LB["value_col2"]]]
        for _, row in feature_df.iterrows():
            feature_table_data.append([str(row["Feature"]), str(row["Value"])])
        feature_table = Table(feature_table_data, colWidths=[9*cm, 6*cm])
        feature_table.setStyle(TABLE_STYLE)
        elements.append(feature_table)
        doc.build(elements)
        buf_pdf.seek(0)
        st.download_button(
            label=UI["download_label"],
            data=buf_pdf,
            file_name="Single_SitStand_Clinical_Report.pdf",
            mime="application/pdf",
            key="pdf_download_button_single"
        )
        st.success(UI["success_message"])
 
with tab9:
 
    client_lang_choice = st.radio(
        "レポート言語 / Report Language",
        ["日本語", "English"],
        horizontal=True,
        key="client_lang_radio"
    )
    client_lang_code = "ja" if client_lang_choice == "日本語" else "en"
 
    CLIENT_UI = {
        "ja": {
            "header": "クライアント向けレポート",
            "caption": "専門用語を減らし、スコアと図解を中心にした、対象者本人にそのまま渡せるレポートです。",
            "subject_name": "対象者名",
            "exam_date": "測定日",
            "examiner": "検者",
            "side_label": "分析側",
            "comment_heading": "総合コメント",
            "comment_label": "対象者向けの総合コメントを記入してください（PDFに反映されます）",
            "auto_generate_button": "🪄 コメントを自動生成（下書き）",
            "auto_generate_caption": "実測値をもとに、専門用語を減らした下書きコメントを自動生成します。内容を確認・編集してからPDFを生成してください。",
            "generate_button": "📄 クライアント向けレポートを生成",
            "download_label": "📥 クライアント向けレポートをダウンロード",
            "success_message": "クライアント向けレポートを生成しました。上のボタンからダウンロードしてください。",
        },
        "en": {
            "header": "Client Report",
            "caption": "A plain-language report with scores and visuals, ready to hand directly to the client.",
            "subject_name": "Subject Name",
            "exam_date": "Exam Date",
            "examiner": "Examiner",
            "side_label": "Analysis Side",
            "comment_heading": "Comment",
            "comment_label": "Enter an overall comment for the client (included in the PDF)",
            "auto_generate_button": "🪄 Auto-generate Comment (Draft)",
            "auto_generate_caption": "Generates a plain-language draft comment from the measured values. Please review and edit before generating the PDF.",
            "generate_button": "📄 Generate Client Report",
            "download_label": "📥 Download Client Report",
            "success_message": "Client report generated. Use the button above to download it.",
        },
    }
    CUI = CLIENT_UI[client_lang_code]
 
    st.subheader(CUI["header"])
    st.caption(CUI["caption"])
 
    client_col1, client_col2, client_col3 = st.columns(3)
    with client_col1:
        client_subject_name = st.text_input(
            CUI["subject_name"], value="", key="client_subject_name_input"
        )
    with client_col2:
        client_exam_date = st.text_input(
            CUI["exam_date"], value="", key="client_exam_date_input"
        )
    with client_col3:
        client_examiner_name = st.text_input(
            CUI["examiner"], value="", key="client_examiner_name_input"
        )
 
    JOINT_LABEL = {
        "ja": {
            "hip_flexion_r": "股関節（右）", "hip_flexion_l": "股関節（左）",
            "knee_angle_r": "ひざ（右）", "knee_angle_l": "ひざ（左）",
            "ankle_angle_r": "足首（右）", "ankle_angle_l": "足首（左）",
            "pelvis_ty": "座り立ちの深さ（骨盤の上下動）",
            "pelvis_tilt": "骨盤の前後の傾き", "pelvis_rotation": "骨盤の左右の回旋",
            "lumbar_extension": "腰の反り（腰椎伸展）",
        },
        "en": {
            "hip_flexion_r": "Hip (Right)", "hip_flexion_l": "Hip (Left)",
            "knee_angle_r": "Knee (Right)", "knee_angle_l": "Knee (Left)",
            "ankle_angle_r": "Ankle (Right)", "ankle_angle_l": "Ankle (Left)",
            "pelvis_ty": "Sit-to-Stand Depth (Pelvic Vertical Displacement)",
            "pelvis_tilt": "Pelvic Tilt", "pelvis_rotation": "Pelvic Rotation",
            "lumbar_extension": "Lumbar Extension",
        },
    }
 
    def client_tier(score, lang_code):
        if score >= 80:
            hex_c, bg_hex, ja, en = "#2E7D32", "#E8F5E9", "良好", "Good"
        elif score >= 60:
            hex_c, bg_hex, ja, en = "#B8860B", "#FFF8E1", "この調子で", "Keep it up"
        else:
            hex_c, bg_hex, ja, en = "#C62828", "#FFEBEE", "サポートが必要", "Needs support"
        label = ja if lang_code == "ja" else en
        return hex_c, bg_hex, label
 
    def make_client_gauge(score, color_hex, width_cm=8.6):
        gfig, gax = plt.subplots(figsize=(4.6, 2.6), subplot_kw={"aspect": "equal"})
        theta_bg = np.linspace(180, 0, 200)
        r_outer, r_inner = 1.0, 0.72
        x_out = r_outer * np.cos(np.radians(theta_bg))
        y_out = r_outer * np.sin(np.radians(theta_bg))
        x_in = r_inner * np.cos(np.radians(theta_bg[::-1]))
        y_in = r_inner * np.sin(np.radians(theta_bg[::-1]))
        gax.fill(np.concatenate([x_out, x_in]), np.concatenate([y_out, y_in]), color="#E0E0E0")
        theta_end = 180 - (max(0, min(100, score)) / 100) * 180
        theta_score = np.linspace(180, theta_end, 200)
        xo = r_outer * np.cos(np.radians(theta_score))
        yo = r_outer * np.sin(np.radians(theta_score))
        xi = r_inner * np.cos(np.radians(theta_score[::-1]))
        yi = r_inner * np.sin(np.radians(theta_score[::-1]))
        gax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=color_hex)
        gax.text(0, -0.05, f"{score:.0f}", ha="center", va="center", fontsize=38, fontweight="bold", color=color_hex)
        gax.text(0, -0.38, "/ 100", ha="center", va="center", fontsize=12, color="#78909C")
        gax.set_xlim(-1.15, 1.15)
        gax.set_ylim(-0.5, 1.15)
        gax.axis("off")
        return fig_to_rl_image(gfig, width_cm=width_cm)
 
    def make_client_score_bars(score_items):
        bfig, bax = plt.subplots(figsize=(8.6, 0.62 * len(score_items) + 0.6))
        y_pos = np.arange(len(score_items))[::-1]
        labels = [lbl for lbl, _ in score_items]
        values = [val for _, val in score_items]
        bax.barh(y_pos, [100] * len(values), height=0.5, color="#ECEFF1", zorder=1)
        for y, v in zip(y_pos, values):
            c_hex, _, _ = client_tier(v, "en")
            bax.barh(y, max(0, min(100, v)), height=0.5, color=c_hex, zorder=2)
            bax.text(min(100, max(0, v)) + 2, y, f"{v:.0f}", va="center", ha="left", fontsize=11, fontweight="bold", color="#263238")
        for x in (60, 80):
            bax.axvline(x, color="#B0BEC5", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        bax.set_yticks(y_pos)
        bax.set_yticklabels(labels, fontsize=11, color="#263238")
        bax.set_xlim(0, 112)
        bax.set_xticks([])
        for spine in bax.spines.values():
            spine.set_visible(False)
        bax.tick_params(left=False)
        bfig.tight_layout()
        return fig_to_rl_image(bfig, width_cm=15.5)
 
    def make_client_phase_timeline(phase_boxes):
        # phase_boxes: [(english_title, color_hex), ...] — English-only inside the
        # raster image on purpose. matplotlib's default font has no CJK glyphs and we
        # don't want a dependency on a Japanese font being installed on the server,
        # so Japanese labels/notes are placed as native PDF text just below this
        # image (see the small table right after) instead of inside the chart.
        n = len(phase_boxes)
        tfig, tax = plt.subplots(figsize=(15.5 / 2.2, 1.25))
        box_w, gap = 3.0, 0.6
        total_w = n * box_w + (n - 1) * gap
        x0 = -total_w / 2
        for i, (title, color_hex) in enumerate(phase_boxes):
            x = x0 + i * (box_w + gap)
            tax.add_patch(plt.Rectangle((x, 0), box_w, 1.0, facecolor=color_hex, alpha=0.15, edgecolor=color_hex, linewidth=1.6))
            tax.text(x + box_w / 2, 0.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#263238")
            if i < n - 1:
                tax.annotate("", xy=(x + box_w + gap - 0.05, 0.5), xytext=(x + box_w + 0.05, 0.5),
                             arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=1.4))
        tax.set_xlim(x0 - 0.3, x0 + total_w + 0.3)
        tax.set_ylim(-0.15, 1.15)
        tax.axis("off")
        tfig.tight_layout()
        return fig_to_rl_image(tfig, width_cm=16)
 
    def generate_client_auto_comment(
        lang_code, overall_score, mobility_score, stability_score, compensation_score,
        lumbar_compensation, pelvis_tilt_compensation, pelvis_rotation_compensation
    ):
        # tab8のgenerate_single_sit_stand_auto_commentの平易版。専門用語を避け、対象者本人が読んでも
        # 分かる言葉で下書きコメントを組み立てる。実測値ベースで動的に生成される。
        # このアプリは片脚での測定のため、左右差（symmetry）のスコア・所見は扱わない。
        if lang_code == "ja":
            if overall_score >= 80:
                lines = [f"今回の総合スコアは{overall_score:.0f}/100で、とても良い状態です。"]
            elif overall_score >= 60:
                lines = [f"今回の総合スコアは{overall_score:.0f}/100でした。全体的には悪くありませんが、いくつか気をつけたい点があります。"]
            else:
                lines = [f"今回の総合スコアは{overall_score:.0f}/100でした。いくつか改善していきたいポイントが見つかりました。"]
 
            if mobility_score >= 80:
                lines.append("座り立ちの深さ（可動域）はしっかり出せています。")
            else:
                lines.append("座り立ちの深さ（可動域）には、まだ伸びしろがあります。")
 
            if stability_score >= 80 and compensation_score >= 80:
                lines.append("動作中の姿勢も安定しており、腰や骨盤への負担も少なめです。")
            else:
                notes = []
                if stability_score < 80:
                    notes.append("座る・立ち上がる動作中に骨盤が少し揺れやすい")
                if compensation_score < 80:
                    notes.append("座る・立ち上がる際に腰や骨盤が反りやすい")
                if notes:
                    lines.append("、また".join(notes) + "傾向が見られました。")
 
            lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。")
            return "\n".join(lines)
        else:
            if overall_score >= 80:
                lines = [f"This check scored {overall_score:.0f}/100 overall — a great result."]
            elif overall_score >= 60:
                lines = [f"This check scored {overall_score:.0f}/100 overall. Things look reasonably good, with a few points worth keeping an eye on."]
            else:
                lines = [f"This check scored {overall_score:.0f}/100 overall. A few areas stood out that are worth working on."]
 
            if mobility_score >= 80:
                lines.append("Sit-to-stand depth (mobility) looks solid.")
            else:
                lines.append("There's room to improve sit-to-stand depth (mobility).")
 
            if stability_score >= 80 and compensation_score >= 80:
                lines.append("Posture stayed steady throughout, with little strain on the lower back or pelvis.")
            else:
                notes = []
                if stability_score < 80:
                    notes.append("some pelvic wobble while sitting down and standing up")
                if compensation_score < 80:
                    notes.append("a tendency for the lower back/pelvis to arch while standing up")
                if notes:
                    lines.append("We noticed " + " and ".join(notes) + ".")
 
            lines.append("Try working through the recommended actions below at a comfortable pace before the next check.")
            return "\n".join(lines)
 
    st.markdown(f"#### {CUI['comment_heading']}")
    if "client_report_comment" not in st.session_state:
        st.session_state["client_report_comment"] = ""
    if st.button(CUI["auto_generate_button"], key="client_report_auto_comment_btn"):
        st.session_state["client_report_comment"] = generate_client_auto_comment(
            client_lang_code, overall_score, mobility_score, stability_score, compensation_score,
            lumbar_compensation, pelvis_tilt_compensation, pelvis_rotation_compensation
        )
    st.caption(CUI["auto_generate_caption"])
    client_comment = st.text_area(
        CUI["comment_label"],
        key="client_report_comment",
        height=150
    )
 
    if st.button(CUI["generate_button"], key="client_report_generate_btn"):
 
        from reportlab.platypus import PageBreak, HRFlowable
 
        # ---- 動的な所見の収集（実測値ベース。片脚測定のため左右差は扱わない） ----
        out_of_range_rows = comparison_display_df[comparison_display_df["Out_of_Range"]]
 
        concern_flags = {
            "range": len(out_of_range_rows) > 0,
            "compensation": (lumbar_compensation > 10) or (pelvis_tilt_compensation > 10) or (pelvis_rotation_compensation > 10),
            "stability": stability_score < 60,
        }
 
        JL = JOINT_LABEL[client_lang_code]
 
        concern_items = []
        if concern_flags["compensation"]:
            if client_lang_code == "ja":
                concern_items.append((
                    "腰・骨盤が反りやすい" if lumbar_compensation > 10 else "骨盤が傾き／回旋しやすい",
                    f"座る・立ち上がる動きの中で、腰や骨盤が大きく動く場面が見られました"
                    f"(腰の反り {lumbar_compensation:.1f}°、骨盤の前後の傾き {pelvis_tilt_compensation:.1f}°、"
                    f"骨盤の回旋 {pelvis_rotation_compensation:.1f}°)。この状態が続くと、腰まわりへの負担が"
                    "蓄積しやすくなります。"
                ))
            else:
                concern_items.append((
                    "Lower back / pelvis compensation",
                    f"Noticeable lumbar and pelvic movement was observed while sitting and standing "
                    f"(lumbar extension {lumbar_compensation:.1f}°, pelvic tilt {pelvis_tilt_compensation:.1f}°, "
                    f"pelvic rotation {pelvis_rotation_compensation:.1f}°). Over time this can add strain around the lower back."
                ))
        if concern_flags["range"]:
            if client_lang_code == "ja":
                range_text = "・".join(JL.get(v, v) for v in out_of_range_rows["Variable"].tolist())
                concern_items.append((
                    "可動域が基準の範囲外",
                    f"{range_text}が、一般的な健常範囲の外にありました。可動域の制限、"
                    "またはやや動きすぎている可能性があります。"
                ))
            else:
                range_text = ", ".join(JOINT_LABEL["en"].get(v, v) for v in out_of_range_rows["Variable"].tolist())
                concern_items.append((
                    "Range of motion outside reference",
                    f"{range_text} fell outside the typical healthy range, suggesting possible "
                    "restricted or excessive range of motion."
                ))
        if concern_flags["stability"]:
            if client_lang_code == "ja":
                concern_items.append((
                    "動作中のグラつき",
                    "座る・立ち上がる動作の中で、骨盤が横に揺れる場面がありました。"
                    "体幹やお尻まわりの筋力を使って、じっと支える意識をすると安定しやすくなります。"
                ))
            else:
                concern_items.append((
                    "Movement wobble",
                    "Some side-to-side pelvic movement was observed while sitting and standing. "
                    "Engaging the core and hip muscles to hold steady can help improve stability."
                ))
        if len(concern_items) == 0:
            if client_lang_code == "ja":
                concern_items.append(("良い状態です", "今回のチェックでは、特に大きな気になるポイントはありませんでした。この調子を維持しましょう。"))
            else:
                concern_items.append(("Looking good", "No major concerns were found in this check. Keep up the good work."))
 
        # ---- アクション（気になるポイントに応じて選択、最大3件） ----
        action_pool_ja = [
            ("compensation", "毎日 10回", "お腹に軽く力を入れたまま、ゆっくり立ち座り",
             "腰を反らさず、体幹で支える感覚をつかむ練習になります。"),
            ("stability", "毎日 左右10秒ずつ", "片足立ち（つかまってOK）",
             "バランス感覚を養い、動作中のグラつきを減らします。"),
            ("range", "毎日 5回", "痛みのない範囲で、ゆっくり深く座る練習",
             "無理のない範囲で可動域を広げていく練習になります。"),
            ("default", "毎日 5秒×3回", "椅子に深く座り、背すじを伸ばしてキープ",
             "姿勢を保つための筋力を養います。"),
        ]
        action_pool_en = [
            ("compensation", "Daily x10", "Slow sit-to-stands with gentle core engagement",
             "Builds the habit of supporting the movement with your core instead of arching your back."),
            ("stability", "Daily 10s/side", "Single-leg balance (holding on is fine)",
             "Builds balance and reduces wobble during movement."),
            ("range", "Daily x5", "Slow, pain-free deep sitting practice",
             "Gradually improves range of motion within a comfortable limit."),
            ("default", "Daily 5s x3", "Sit tall in a chair and hold your posture",
             "Builds the postural strength needed to hold a good position."),
        ]
        action_pool = action_pool_ja if client_lang_code == "ja" else action_pool_en
        selected_actions = [a for a in action_pool if a[0] != "default" and concern_flags.get(a[0], False)]
        if len(selected_actions) == 0:
            selected_actions = [a for a in action_pool if a[0] == "default"]
        selected_actions = (selected_actions + [a for a in action_pool if a[0] == "default"])[:3]
 
        # ---- フェーズ別の一言メモ（実測値ベース。Sitting→Rising→Standing→Loweringの順） ----
        STATIC_PHASES_C = ["Sitting", "Standing"]
        STD_THRESHOLD_C = 2.0
        phase_std_flag = {p: False for p in STATIC_PHASES_C}
        for variable in ["pelvis_tilt", "pelvis_rotation", "lumbar_extension", "pelvis_list"]:
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            for phase in STATIC_PHASES_C:
                std_v = var_row[f"{phase}_Std"].iloc[0]
                if pd.notna(std_v) and std_v > STD_THRESHOLD_C:
                    phase_std_flag[phase] = True
 
        # 図(画像)側は英語のみ。日本語ラベル・所見はネイティブPDFテキスト(下の表)で表示する。
        phase_titles_en = ["① Sitting", "② Rising", "③ Standing", "④ Lowering"]
        phase_colors = [
            "#2E7D32" if not phase_std_flag["Sitting"] else "#B8860B",
            "#C62828" if lumbar_compensation > 10 else "#2E7D32",
            "#B8860B" if phase_std_flag["Standing"] else "#2E7D32",
            "#B8860B" if (pelvis_tilt_compensation > 10 or pelvis_rotation_compensation > 10) else "#2E7D32",
        ]
        phase_boxes = list(zip(phase_titles_en, phase_colors))
 
        if client_lang_code == "ja":
            phase_titles_local = ["① 座っている", "② 立ち上がる", "③ 立っている", "④ 座る"]
            phase_notes_local = [
                "座位は安定しています" if not phase_std_flag["Sitting"] else "座位でやや不安定な様子がありました",
                "腰が反りやすい傾向があります" if lumbar_compensation > 10 else "スムーズに立ち上がれています",
                "立位は安定しています" if not phase_std_flag["Standing"] else "立位でやや不安定な様子がありました",
                "骨盤の傾き・回旋が大きめです" if (pelvis_tilt_compensation > 10 or pelvis_rotation_compensation > 10) else "安定して座れています",
            ]
        else:
            phase_titles_local = phase_titles_en
            phase_notes_local = [
                "Stable" if not phase_std_flag["Sitting"] else "Slightly unstable",
                "Back tends to arch" if lumbar_compensation > 10 else "Smooth rise",
                "Stable" if not phase_std_flag["Standing"] else "Slightly unstable",
                "Pelvis tilts/rotates more than usual" if (pelvis_tilt_compensation > 10 or pelvis_rotation_compensation > 10) else "Stable, controlled descent",
            ]
 
        # ---- スタイル ----
        c_styles = getSampleStyleSheet()
        c_title_style = ParagraphStyle("CTitle", parent=c_styles["Title"], fontName="HeiseiKakuGo-W5", fontSize=21, alignment=TA_CENTER, textColor=colors.white, spaceAfter=2)
        c_subtitle_style = ParagraphStyle("CSubtitle", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, alignment=TA_CENTER, textColor=colors.white)
        c_section_style = ParagraphStyle("CSection", parent=c_styles["Heading2"], fontName="HeiseiKakuGo-W5", fontSize=13.5, textColor=colors.HexColor("#263238"), spaceBefore=9, spaceAfter=5)
        c_lead_style = ParagraphStyle("CLead", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, leading=16, textColor=colors.HexColor("#263238"), alignment=TA_CENTER, spaceAfter=4)
        c_body_style = ParagraphStyle("CBody", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.6, leading=14.5, textColor=colors.HexColor("#263238"))
        c_small_muted_style = ParagraphStyle("CSmallMuted", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8.3, leading=12.5, textColor=colors.HexColor("#607D8B"))
        c_card_title_style = ParagraphStyle("CCardTitle", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, textColor=colors.HexColor("#263238"), alignment=TA_CENTER)
        c_card_status_style = ParagraphStyle("CCardStatus", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=12, alignment=TA_CENTER)
        c_score_headline_style = ParagraphStyle("CScoreHeadline", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=12.5, alignment=TA_CENTER, textColor=colors.HexColor("#263238"), spaceBefore=8)
        c_footer_style = ParagraphStyle("CFooter", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=7.8, leading=11.5, textColor=colors.HexColor("#607D8B"), alignment=TA_CENTER)
        c_page_label_style = ParagraphStyle("CPageLabel", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8, textColor=colors.HexColor("#607D8B"))
        c_action_head_style = ParagraphStyle("CActionHead", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.3, textColor=colors.white)
        c_action_body_style = ParagraphStyle("CActionBody", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.2, leading=13.5, textColor=colors.HexColor("#263238"))
        c_action_note_style = ParagraphStyle("CActionNote", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8.2, leading=12, textColor=colors.HexColor("#607D8B"))
 
        LINE_HEX_C = "#CFD8DC"
        BAND_C = colors.HexColor("#0D47A1")
        BLUE_C = colors.HexColor("#1565C0")
        BLUE_BG_C = "#E3F2FD"
 
        def c_tier_pack(score):
            return client_tier(score, client_lang_code)
 
        def make_client_card(label, score, desc):
            color_hex, bg_hex, tier_text = c_tier_pack(score)
            inner = Table(
                [
                    [Paragraph(label, c_card_title_style)],
                    [Spacer(1, 0.12 * cm)],
                    [Paragraph(f'<font color="{color_hex}"><b>{tier_text}</b></font>', c_card_status_style)],
                    [Spacer(1, 0.16 * cm)],
                    [Paragraph(desc, c_small_muted_style)],
                ],
                colWidths=[5.6 * cm]
            )
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(color_hex)),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            return inner
 
        if client_lang_code == "ja":
            card_data = [
                ("可動域", mobility_score, "座り立ちの深さは十分に出ています。" if mobility_score >= 80 else "座り立ちの深さに、もう少し伸びしろがあります。"),
                ("安定性", stability_score, "動作中の骨盤は落ち着いています。" if stability_score >= 80 else "動作中に骨盤が少し揺れる傾向があります。"),
                ("姿勢のクセ", compensation_score, "腰・骨盤の反りは少なめです。" if compensation_score >= 80 else "座る・立ち上がる動きの中で、腰や骨盤が反りやすい傾向があります。"),
            ]
        else:
            card_data = [
                ("Mobility", mobility_score, "Good sit-to-stand depth." if mobility_score >= 80 else "There is room to deepen the sit-to-stand movement."),
                ("Stability", stability_score, "The pelvis stays steady during the movement." if stability_score >= 80 else "Some pelvic wobble was observed during the movement."),
                ("Posture Habit", compensation_score, "Minimal lower-back/pelvis compensation." if compensation_score >= 80 else "The lower back and pelvis tend to arch during the movement."),
            ]
 
        elements_c = []
 
        side_label_local = ("右" if ANALYSIS_SIDE == "Right" else "左") if client_lang_code == "ja" else ANALYSIS_SIDE
 
        band = Table(
            [[Paragraph(
                "あなたの片脚立ち座りチェック結果" if client_lang_code == "ja" else "Your Single-Leg Sit-to-Stand Results",
                c_title_style
            )],
                [Paragraph(
                    (f"対象者：{client_subject_name or '-'}　｜　測定日：{client_exam_date or '-'}　｜　検者：{client_examiner_name or '-'}　｜　{CUI['side_label']}：{side_label_local}"
                     if client_lang_code == "ja" else
                     f"Subject: {client_subject_name or '-'}  |  Exam Date: {client_exam_date or '-'}  |  Examiner: {client_examiner_name or '-'}  |  {CUI['side_label']}: {side_label_local}"),
                    c_subtitle_style
                )]],
            colWidths=[19 * cm]
        )
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BAND_C),
            ("TOPPADDING", (0, 0), (0, 0), 14),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
            ("TOPPADDING", (0, 1), (0, 1), 2),
            ("BOTTOMPADDING", (0, 1), (0, 1), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements_c.append(band)
        elements_c.append(Spacer(1, 0.5 * cm))
 
        gauge_color_hex, _, gauge_tier_label = c_tier_pack(overall_score)
        gauge_img_c = make_client_gauge(overall_score, gauge_color_hex)
        gauge_table_c = Table([[gauge_img_c]], colWidths=[19 * cm])
        gauge_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(gauge_table_c)
 
        if client_lang_code == "ja":
            elements_c.append(Paragraph(f"総合評価：<b>{gauge_tier_label}</b>", c_score_headline_style))
            elements_c.append(Paragraph(
                "立ち座りの動きを、可動域・安定性・姿勢のクセの3つの視点でチェックしました。",
                c_lead_style
            ))
        else:
            elements_c.append(Paragraph(f"Overall: <b>{gauge_tier_label}</b>", c_score_headline_style))
            elements_c.append(Paragraph(
                "Your sit-to-stand movement was checked across three areas: mobility, stability, and posture habits.",
                c_lead_style
            ))
 
        elements_c.append(Spacer(1, 0.45 * cm))
        elements_c.append(Paragraph("3つのポイント" if client_lang_code == "ja" else "Three Key Areas", c_section_style))
 
        cards_row_c = Table([[make_client_card(*c) for c in card_data]], colWidths=[6.33 * cm] * 3)
        cards_row_c.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements_c.append(cards_row_c)
        elements_c.append(Spacer(1, 0.4 * cm))
 
        elements_c.append(Paragraph(
            "3つのスコアを見比べる" if client_lang_code == "ja" else "Comparing the Three Scores",
            c_section_style
        ))
        elements_c.append(Paragraph(
            ("点線は「この調子で(60点)」「良好(80点)」の目安ラインです。"
             if client_lang_code == "ja" else
             "The dotted lines mark the 60 (“keep it up”) and 80 (“good”) reference points."),
            c_body_style
        ))
        elements_c.append(Spacer(1, 0.12 * cm))
        score_bar_labels = ["Mobility", "Stability", "Posture"]
        bars_img_c = make_client_score_bars(list(zip(score_bar_labels, [mobility_score, stability_score, compensation_score])))
        bars_table_c = Table([[bars_img_c]], colWidths=[19 * cm])
        bars_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(bars_table_c)
 
        elements_c.append(Spacer(1, 0.25 * cm))
        elements_c.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(LINE_HEX_C)))
        elements_c.append(Spacer(1, 0.12 * cm))
        elements_c.append(Paragraph(
            "続きは次のページで、動きの流れとおすすめアクションをご紹介します。" if client_lang_code == "ja"
            else "Continued on the next page: movement flow and recommended actions.",
            c_page_label_style
        ))
        elements_c.append(PageBreak())
 
        elements_c.append(Paragraph("動きの流れをチェック" if client_lang_code == "ja" else "Movement Flow", c_section_style))
        elements_c.append(Paragraph(
            ("立ち座りの動作を4つの場面に分けてみると、どこで体に負担がかかりやすいかが見えてきます。"
             if client_lang_code == "ja" else
             "Breaking the movement into four phases makes it easier to see where load tends to build up."),
            c_body_style
        ))
        elements_c.append(Spacer(1, 0.15 * cm))
        timeline_img_c = make_client_phase_timeline(phase_boxes)
        timeline_table_c = Table([[timeline_img_c]], colWidths=[19 * cm])
        timeline_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(timeline_table_c)
        elements_c.append(Spacer(1, 0.15 * cm))
 
        phase_note_row_titles = [
            Paragraph(f"<font color='{phase_colors[i]}'><b>{phase_titles_local[i]}</b></font>", c_small_muted_style)
            for i in range(4)
        ]
        phase_note_row_notes = [
            Paragraph(phase_notes_local[i], c_small_muted_style)
            for i in range(4)
        ]
        phase_note_table = Table([phase_note_row_titles, phase_note_row_notes], colWidths=[4.75 * cm] * 4)
        phase_note_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ]))
        elements_c.append(phase_note_table)
        elements_c.append(Spacer(1, 0.22 * cm))
 
        elements_c.append(Paragraph("気になるポイント" if client_lang_code == "ja" else "Points to Note", c_section_style))
        for title, desc in concern_items:
            row_c = Table(
                [[Paragraph(f"<font color='#C62828'><b>●</b></font>  <b>{title}</b>", c_action_body_style),
                  Paragraph(desc, c_action_body_style)]],
                colWidths=[4.4 * cm, 14.6 * cm]
            )
            row_c.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements_c.append(row_c)
        elements_c.append(Spacer(1, 0.16 * cm))
 
        elements_c.append(Paragraph(CUI["comment_heading"], c_section_style))
        comment_display = client_comment.strip() if client_comment and client_comment.strip() else (
            "(記入なし)" if client_lang_code == "ja" else "(No comment entered)"
        )
        comment_html = escape(comment_display).replace("\n", "<br/>")
        elements_c.append(Paragraph(comment_html, c_body_style))
        elements_c.append(Spacer(1, 0.16 * cm))
 
        elements_c.append(Paragraph(
            "おすすめのアクション（今週から）" if client_lang_code == "ja" else "Recommended Actions (Starting This Week)",
            c_section_style
        ))
        action_rows_c = [[
            Paragraph("回数の目安" if client_lang_code == "ja" else "Frequency", c_action_head_style),
            Paragraph("やること" if client_lang_code == "ja" else "What to Do", c_action_head_style),
            Paragraph("ねらい" if client_lang_code == "ja" else "Why", c_action_head_style),
        ]]
        for _, freq, what, why in selected_actions:
            action_rows_c.append([freq, Paragraph(what, c_action_body_style), Paragraph(why, c_action_note_style)])
        action_table_c = Table(action_rows_c, colWidths=[3.2 * cm, 8.4 * cm, 7.4 * cm])
        action_table_c.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, 0), BLUE_C),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor(BLUE_BG_C)),
            ("TEXTCOLOR", (0, 1), (0, -1), BLUE_C),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor(LINE_HEX_C)),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(LINE_HEX_C)),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements_c.append(action_table_c)
        elements_c.append(Spacer(1, 0.16 * cm))
 
        elements_c.append(Paragraph(
            "参考：今回の測定値（くわしく知りたい方向け）" if client_lang_code == "ja" else "Reference: Measured Values (For Those Who Want Detail)",
            c_section_style
        ))
        if client_lang_code == "ja":
            detail_rows_c = [["項目", "測定値", "備考"]]
        else:
            detail_rows_c = [["Item", "Value", "Note"]]
 
        def in_range_note(var):
            row = comparison_display_df[comparison_display_df["Variable"] == var]
            if len(row) == 0:
                return "-"
            out = bool(row["Out_of_Range"].iloc[0])
            if client_lang_code == "ja":
                return "基準範囲外" if out else "健康な範囲内"
            return "Outside reference" if out else "Within reference"
 
        for var in [HIP, KNEE, ANKLE]:
            row = comparison_display_df[comparison_display_df["Variable"] == var]
            if len(row) == 0:
                continue
            val = row["Subject_ROM"].iloc[0]
            detail_rows_c.append([JOINT_LABEL[client_lang_code].get(var, var), f"{val:.1f}°", in_range_note(var)])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["lumbar_extension"], f"{lumbar_compensation:.1f}°",
            ("やや大きめ" if lumbar_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if lumbar_compensation > 10 else "Small")
        ])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["pelvis_tilt"], f"{pelvis_tilt_compensation:.1f}°",
            ("やや大きめ" if pelvis_tilt_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if pelvis_tilt_compensation > 10 else "Small")
        ])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["pelvis_rotation"], f"{pelvis_rotation_compensation:.1f}°",
            ("やや大きめ" if pelvis_rotation_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if pelvis_rotation_compensation > 10 else "Small")
        ])
 
        detail_table_c = Table(detail_rows_c, colWidths=[6 * cm, 4 * cm, 9 * cm])
        detail_table_c.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#78909C")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#607D8B")),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(LINE_HEX_C)),
            ("TOPPADDING", (0, 0), (-1, -1), 2.8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements_c.append(detail_table_c)
        elements_c.append(Spacer(1, 0.16 * cm))
 
        elements_c.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(LINE_HEX_C)))
        elements_c.append(Spacer(1, 0.12 * cm))
        elements_c.append(Paragraph(
            ("このレポートはスマートフォンの動画をもとにした簡易チェックの結果であり、医学的な診断ではありません。"
             "痛みや強い違和感がある場合は、無理をせず医療・専門家にご相談ください。")
            if client_lang_code == "ja" else
            ("This report is based on a simplified check from smartphone video and is not a medical diagnosis. "
             "If you experience pain or significant discomfort, please consult a healthcare professional."),
            c_footer_style
        ))
 
        def draw_client_bg(canvas, doc_):
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#FAFAFA"))
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.restoreState()
 
        # ページ数は「気になるポイント」の件数やコメントの長さによって変わるため、
        # 固定の「1/2」を書く代わりに、実際の総ページ数を描画時に計算してfooterに描く。
        from reportlab.pdfgen import canvas as canvas_module
 
        class _ClientReportNumberedCanvas(canvas_module.Canvas):
            def __init__(self, *args, **kwargs):
                canvas_module.Canvas.__init__(self, *args, **kwargs)
                self._saved_page_states = []
 
            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()
 
            def save(self):
                total_pages = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self._draw_totalized_page_number(total_pages)
                    canvas_module.Canvas.showPage(self)
                canvas_module.Canvas.save(self)
 
            def _draw_totalized_page_number(self, total_pages):
                self.setFont("HeiseiKakuGo-W5", 8)
                self.setFillColor(colors.HexColor("#607D8B"))
                self.drawString(1.6 * cm, 1.0 * cm, f"{self._pageNumber} / {total_pages}")
 
        client_report_buffer = BytesIO()
        client_doc = SimpleDocTemplate(
            client_report_buffer,
            pagesize=A4,
            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
            leftMargin=1.6 * cm, rightMargin=1.6 * cm
        )
        client_doc.build(
            elements_c,
            onFirstPage=draw_client_bg,
            onLaterPages=draw_client_bg,
            canvasmaker=_ClientReportNumberedCanvas
        )
 
        st.download_button(
            CUI["download_label"],
            data=client_report_buffer.getvalue(),
            file_name="Single_SitStand_Client_Report.pdf",
            mime="application/pdf",
            key="client_report_download_btn"
        )
        st.success(CUI["success_message"])
 
