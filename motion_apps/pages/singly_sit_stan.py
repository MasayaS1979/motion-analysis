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
 
    # Find every contiguous run of target_label
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
 
        return min(
            abs(target_idx - start),
            abs(target_idx - end)
        )
 
    keep_run = min(runs, key=distance_to_target)
 
    for start, end in runs:
 
        if (start, end) == keep_run:
            continue
 
        # Fold this stray cluster back into whatever phase came
        # right before it started.
        fallback_label = (
            phase_list[start - 1]
            if start > 0
            else target_label
        )
 
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
 
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Phase Analysis",
    "Movement Analysis",
    "Healthy ROM Comparison",
    "Clinical Report",
    "Raw Data",
    "Dashboard",
    "Movement Score"
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
 
        df_phase["pelvis_tx"]
        .abs()
        .max()
 
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
    # =========================
 
    def calculate_rom_score(
        subject_rom,
        healthy_rom
    ):
 
        deviation = abs(
            subject_rom - healthy_rom
        )
 
        score = (
 
            100
            -
            (
                deviation
                /
                healthy_rom
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
    # Single Sit-to-Stand
 
    healthy_hip_rom = 40
    healthy_knee_rom = 60
    healthy_ankle_rom = 30
 
    hip_score = calculate_rom_score(
        hip_rom,
        healthy_hip_rom
    )
 
    knee_score = calculate_rom_score(
        knee_rom,
        healthy_knee_rom
    )
 
    ankle_score = calculate_rom_score(
        ankle_rom,
        healthy_ankle_rom
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
    # =========================
 
    pelvic_instability = (
 
        abs(pelvis_tilt_change)
        +
        abs(pelvis_list_change)
 
    )
 
    stability_score = round(
 
        max(
            0,
            100 - pelvic_instability * 2
        ),
 
        1
 
    )
 
    # =========================
    # Compensation Score
    # =========================
 
    lumbar_change = (
 
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min()
 
    )
 
    compensation_score = round(
 
        max(
            0,
            100 - lumbar_change * 2
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
 
