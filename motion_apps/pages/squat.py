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
    normal="HeiseiKakuGo-W5",
    bold="HeiseiKakuGo-W5",
    italic="HeiseiKakuGo-W5",
    boldItalic="HeiseiKakuGo-W5"
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
 
st.set_page_config(page_title="Squat Analysis", layout="wide")
 
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
 
st.title("Squat Analysis")
 
uploaded_file = st.session_state.get("uploaded_file")
 
if uploaded_file is None:
    st.warning(t("common.upload_warning"))
    st.stop()
 
# =========================
# Squat Phase Detection
# =========================
 
df = pd.read_excel(uploaded_file, header=10)
 
signal = df["pelvis_ty"]
 
signal_smooth = (
    signal
    .rolling(window=5, center=True)
    .mean()
    .bfill()
    .ffill()
)
 
velocity = signal_smooth.diff()
 
pelvis_min = signal_smooth.min()
pelvis_max = signal_smooth.max()
pelvis_range = pelvis_max - pelvis_min
 
phase_order = [
    "Standing",
    "Descending",
    "Bottom",
    "Ascending"
]
 
display_phase_order = [
    "Standing",
    "Descending",
    "Bottom",
    "Ascending"
]
 
# -------------------------
# Bottom / Standing Event
# (for plot markers / event metrics only)
# -------------------------
 
bottom_idx = signal_smooth.idxmin()
 
standing_idx = (
    signal_smooth.iloc[bottom_idx:]
    .idxmax()
)
 
velocity_threshold = 0.002
 
bottom_threshold = pelvis_min + pelvis_range * 0.25
standing_threshold = pelvis_max - pelvis_range * 0.15
 
phases = []
 
for p, v in zip(signal_smooth, velocity):
 
    if pd.isna(v):
        phase = "Standing"
 
    # Bottom：最下点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    # → まだ動いている区間はここに含めない
    elif p <= bottom_threshold and abs(v) < velocity_threshold:
        phase = "Bottom"
 
    # Standing：最高点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    # → Ascending/Descending の通過点をStandingに含めない
    elif p >= standing_threshold and abs(v) < velocity_threshold:
        phase = "Standing"
 
    # まだ動いている場合は、位置に関係なく速度方向で判定
    elif v < -velocity_threshold:
        phase = "Descending"
 
    elif v > velocity_threshold:
        phase = "Ascending"
 
    else:
        phase = (
            phases[-1]
            if len(phases) > 0
            else "Standing"
        )
 
    phases.append(phase)
 
# -------------------------
# Keep only the single contiguous cluster of Bottom / Standing
# that is nearest to the detected event (bottom_idx / standing_idx).
#
# The position+velocity classification above can label more than one
# separated "quiet" cluster as Bottom or Standing (e.g. a brief stall
# partway through the motion, or noise near the threshold boundary).
# Averaging max/min ROM across two unrelated quiet clusters inflates
# Bottom_ROM / Standing_ROM well beyond what a single steady posture
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
 
phases = keep_nearest_cluster(phases, "Bottom", bottom_idx)
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
 
    for phase in display_phase_order:
 
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
    "hip_flexion_r": {"min": 90.0, "max": 130.0},
    "hip_flexion_l": {"min": 90.0, "max": 130.0},
    "knee_angle_r": {"min": 90.0, "max": 140.0},
    "knee_angle_l": {"min": 90.0, "max": 140.0},
    "ankle_angle_r": {"min": 15.0, "max": 35.0},
    "ankle_angle_l": {"min": 15.0, "max": 35.0},
    # NOTE: pelvis_tilt / pelvis_rotation / lumbar_extension are trunk
    # "compensation" signals rather than a primary joint ROM — smaller
    # excursion is generally considered better squat form. The 0-10°
    # band below is a placeholder threshold for "acceptable compensation"
    # and should be reviewed/adjusted against your own clinical reference
    # rather than treated as an established normative range.
    "pelvis_tilt": {"min": 0.0, "max": 10.0},
    "pelvis_rotation": {"min": 0.0, "max": 10.0},
    "lumbar_extension": {"min": 0.0, "max": 10.0}
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
            "Subject_ROM": round(subject_rom, 2),
            "Healthy_Min": healthy_min,
            "Healthy_Max": healthy_max,
            "ROM_Difference_%": round(difference_percent, 2),
            "Out_of_Range": is_outside
        })
 
    return pd.DataFrame(results)
 
comparison_df = compare_subject_to_healthy(
    df_phase,
    healthy_rom_df
)
 
# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Squat Phases",
    "Movement Analysis",
    "Symmetry Analysis",
    "Clinical Report",
    "Raw Data",
    "Dashboard",
    "Movement Score",
    "PDF Report"
])
 
# =========================
# Squat Phases
# =========================
 
with tab1:
 
    st.subheader("Phase Detection Plot")
 
    st.caption(t("squat.phase_caption"))
 
    colors_phase = {
        "Standing": "dodgerblue",
        "Descending": "orange",
        "Bottom": "red",
        "Ascending": "limegreen"
    }
 
    fig, ax = plt.subplots(
        figsize=(15, 6)
    )
 
    # =========================
    # Dark Theme
    # =========================
 
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
 
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
 
    for spine in ax.spines.values():
        spine.set_color("white")
 
    ax.grid(True, color="white", alpha=0.3)
 
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
 
        idx = df_phase["Phase"] == phase
 
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
 
    ax.axvline(bottom_idx, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(standing_idx, color="dodgerblue", linestyle="--", linewidth=1.5)
 
    ax.text(
        bottom_idx,
        signal_smooth.iloc[bottom_idx],
        "Bottom",
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
 
    ax.set_title("Phase Detection Plot")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pelvis Vertical Position (m)")
 
    legend = ax.legend()
 
    for text in legend.get_texts():
        text.set_color("white")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    st.pyplot(fig)
 
    # -------------------------
    # Event Information
    # -------------------------
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.metric(
            "Bottom",
            bottom_idx,
            help="骨盤の高さ（pelvis_ty）が最も低くなったフレーム番号（しゃがみの最下点）。"
        )
 
    with col2:
        st.metric(
            "Standing",
            standing_idx,
            help="Bottom以降で骨盤の高さが最も高くなったフレーム番号（立ち上がり完了地点）。"
        )
 
    st.caption(
        "※ Bottom / Standing の数値は動画・計測データの「フレーム番号」です。"
        "サンプリング周波数60Hzの場合、フレーム番号 ÷ 60 で経過秒数に換算できます"
        "（例：Bottom = 90 → 90 ÷ 60 = 1.5秒）。"
    )
 
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
    # Create Phase Tables
    # ==========================================================
 
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
 
    descending_df = phase_summary_df[
        [
            "Variable",
            "Descending_Min",
            "Descending_Max",
            "Descending_Mean",
            "Descending_Std",
            "Descending_ROM"
        ]
    ]
 
    bottom_df = phase_summary_df[
        [
            "Variable",
            "Bottom_Min",
            "Bottom_Max",
            "Bottom_Mean",
            "Bottom_Std",
            "Bottom_ROM"
        ]
    ]
 
    ascending_df = phase_summary_df[
        [
            "Variable",
            "Ascending_Min",
            "Ascending_Max",
            "Ascending_Mean",
            "Ascending_Std",
            "Ascending_ROM"
        ]
    ]
 
    # ==========================================================
    # Excel
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
 
        standing_df.to_excel(
            writer,
            sheet_name="Standing",
            index=False
        )
 
        descending_df.to_excel(
            writer,
            sheet_name="Descending",
            index=False
        )
 
        bottom_df.to_excel(
            writer,
            sheet_name="Bottom",
            index=False
        )
 
        ascending_df.to_excel(
            writer,
            sheet_name="Ascending",
            index=False
        )
 
    # ==========================================================
    # PDF
    # ==========================================================
 
    pdf_buffer = BytesIO()
 
    doc = SimpleDocTemplate(pdf_buffer)
 
    table_data = (
        [phase_summary_df.columns.tolist()]
        + phase_summary_df.values.tolist()
    )
 
    table = Table(table_data)
 
    table.setStyle(TableStyle([
 
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
 
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
 
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
 
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
 
        ("FONTSIZE", (0, 0), (-1, -1), 8),
 
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8)
 
    ]))
 
    doc.build([table])
 
    # ==========================================================
    # Download Buttons
    # ==========================================================
 
    col1, col2 = st.columns(2)
 
    with col1:
 
        st.download_button(
 
            "📥 Download Excel",
 
            data=excel_buffer.getvalue(),
 
            file_name="Squat_Phase_Summary.xlsx",
 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
 
        )
 
    with col2:
 
        st.download_button(
 
            "📄 Download PDF",
 
            data=pdf_buffer.getvalue(),
 
            file_name="Squat_Phase_Summary.pdf",
 
            mime="application/pdf"
 
        )
 
    # ==========================================================
    # Phase Tables
    # ==========================================================
 
    st.markdown("---")
 
    st.subheader("Phase Statistics")
 
    tab_standing, tab_descending, tab_bottom, tab_ascending = st.tabs(
        [
            "Standing",
            "Descending",
            "Bottom",
            "Ascending"
        ]
    )
 
    with tab_standing:
 
        st.dataframe(
            standing_df,
            use_container_width=True
        )
 
    with tab_descending:
 
        st.dataframe(
            descending_df,
            use_container_width=True
        )
 
    with tab_bottom:
 
        st.dataframe(
            bottom_df,
            use_container_width=True
        )
 
    with tab_ascending:
 
        st.dataframe(
            ascending_df,
            use_container_width=True
        )
 
# =========================
# Movement Analysis
# =========================
with tab2:
 
    st.subheader("Joint Time Series")
 
    st.caption(t("common.joint_time_series_caption"))
 
    # OpenCap sampling rate (60 Hz)
    time = np.arange(len(df_phase)) / 60
 
    fig, ax = plt.subplots(
        9,
        1,
        figsize=(12, 30)
    )
 
    # =========================
    # Dark Theme
    # =========================
 
    fig.patch.set_facecolor("black")
 
    for a in ax:
 
        a.set_facecolor("black")
 
        a.tick_params(colors="white")
 
        a.title.set_color("white")
        a.xaxis.label.set_color("white")
        a.yaxis.label.set_color("white")
 
        for spine in a.spines.values():
            spine.set_color("white")
 
        a.grid(True, color="white", alpha=0.3)
 
    # =====================
    # Knee
    # =====================
 
    ax[0].plot(time, df_phase["knee_angle_r"], label="Right", linewidth=2)
    ax[0].plot(time, df_phase["knee_angle_l"], label="Left", linewidth=2)
 
    ax[0].set_title("Knee Angle")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Angle (deg)")
 
    legend = ax[0].legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Hip
    # =====================
 
    ax[1].plot(time, df_phase["hip_flexion_r"], label="Right", linewidth=2)
    ax[1].plot(time, df_phase["hip_flexion_l"], label="Left", linewidth=2)
 
    ax[1].set_title("Hip Flexion")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (deg)")
 
    legend = ax[1].legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Ankle
    # =====================
 
    ax[2].plot(time, df_phase["ankle_angle_r"], label="Right", linewidth=2)
    ax[2].plot(time, df_phase["ankle_angle_l"], label="Left", linewidth=2)
 
    ax[2].set_title("Ankle Angle")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (deg)")
 
    legend = ax[2].legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Pelvic Tilt
    # =====================
 
    ax[3].plot(time, df_phase["pelvis_tilt"], linewidth=2, color="cyan")
 
    ax[3].set_title("Pelvic Tilt")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Right-Left Position
    # =====================
 
    ax[4].plot(time, df_phase["pelvis_tx"], linewidth=2, color="orange")
 
    ax[4].set_title("Pelvic Right-Left Position")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Height
    # =====================
 
    ax[5].plot(time, df_phase["pelvis_ty"], linewidth=2, color="lime")
 
    ax[5].set_title("Pelvic Height")
    ax[5].set_xlabel("Time (s)")
    ax[5].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Forward-Backward Position
    # =====================
 
    ax[6].plot(time, df_phase["pelvis_tz"], linewidth=2, color="magenta")
 
    ax[6].set_title("Pelvic Forward-Backward Position")
    ax[6].set_xlabel("Time (s)")
    ax[6].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Rotation
    # =====================
 
    ax[7].plot(time, df_phase["pelvis_rotation"], linewidth=2, color="yellow")
 
    ax[7].set_title("Pelvic Rotation")
    ax[7].set_xlabel("Time (s)")
    ax[7].set_ylabel("Angle (deg)")
 
    # =====================
    # Lumbar Extension
    # =====================
 
    ax[8].plot(time, df_phase["lumbar_extension"], linewidth=2, color="deepskyblue")
 
    ax[8].set_title("Lumbar Extension")
    ax[8].set_xlabel("Time (s)")
    ax[8].set_ylabel("Angle (deg)")
 
    plt.tight_layout()
 
    st.pyplot(fig)
 
# =========================
# Symmetry Analysis
# =========================
 
with tab3:
 
    st.subheader("Phase Symmetry")
 
    st.caption(t("common.symmetry_caption"))
 
    joints = {
 
        "Hip": ("hip_flexion_r", "hip_flexion_l"),
        "Knee": ("knee_angle_r", "knee_angle_l"),
        "Ankle": ("ankle_angle_r", "ankle_angle_l")
 
    }
 
    for joint_name, (right_var, left_var) in joints.items():
 
        st.markdown(f"## {joint_name}")
 
        right_df = phase_summary_df[
            phase_summary_df["Variable"] == right_var
        ]
 
        left_df = phase_summary_df[
            phase_summary_df["Variable"] == left_var
        ]
 
        symmetry_results = []
        rom_difference = []
 
        for phase in phase_order:
 
            right_rom = right_df[f"{phase}_ROM"].iloc[0]
            left_rom = left_df[f"{phase}_ROM"].iloc[0]
 
            if max(right_rom, left_rom) == 0:
 
                asymmetry = 0
 
            else:
 
                asymmetry = (
                    abs(right_rom - left_rom)
                    / max(right_rom, left_rom)
                    * 100
                )
 
            rom_difference.append(asymmetry)
 
            symmetry_results.append({
 
                "Phase": phase,
 
                "Right_ROM": round(right_rom, 2),
                "Left_ROM": round(left_rom, 2),
                "Asymmetry_%": round(asymmetry, 2)
 
            })
 
        symmetry_df = pd.DataFrame(symmetry_results)
 
        # =========================
        # Plotly Table
        # =========================
 
        fig = go.Figure(
 
            data=[
 
                go.Table(
 
                    columnwidth=[120, 120, 120, 120],
 
                    header=dict(
 
                        values=list(symmetry_df.columns),
 
                        fill_color="black",
 
                        font=dict(color="white", size=18),
 
                        align="center",
 
                        line=dict(color="white", width=1)
 
                    ),
 
                    cells=dict(
 
                        values=[
                            symmetry_df[col]
                            for col in symmetry_df.columns
                        ],
 
                        fill_color="black",
 
                        font=dict(color="white", size=16),
 
                        align="center",
 
                        height=35,
 
                        line=dict(color="white", width=1)
 
                    )
 
                )
 
            ]
 
        )
 
        fig.update_layout(
 
            paper_bgcolor="black",
 
            plot_bgcolor="black",
 
            height=300,
 
            margin=dict(l=10, r=10, t=10, b=10)
 
        )
 
        st.plotly_chart(fig, use_container_width=True)
 
        # =========================
        # Summary Metrics
        # =========================
 
        col1, col2 = st.columns(2)
 
        with col1:
 
            st.metric("Maximum Asymmetry", f"{max(rom_difference):.1f}%")
 
        with col2:
 
            st.metric("Average Asymmetry", f"{np.mean(rom_difference):.1f}%")
 
        # =========================
        # Bar Plot
        # =========================
 
        fig, ax = plt.subplots(figsize=(8, 4))
 
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
 
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
 
        for spine in ax.spines.values():
            spine.set_color("white")
 
        ax.grid(True, color="white", alpha=0.3)
 
        ax.bar(
            symmetry_df["Phase"],
            symmetry_df["Asymmetry_%"],
            color="royalblue"
        )
 
        ax.axhline(15, linestyle="--", color="red", label="15% Threshold")
 
        ax.set_ylabel("Asymmetry (%)")
        ax.set_title(f"{joint_name} ROM Asymmetry by Phase")
 
        legend = ax.legend()
 
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
 
        for text in legend.get_texts():
            text.set_color("white")
 
        st.pyplot(fig)
 
    # =========================
    # Healthy ROM Comparison
    # =========================
 
    st.subheader("Healthy ROM Comparison")
 
    st.caption(t("common.healthy_rom_caption"))
 
    st.caption(t("common.difference_pct_caption"))
 
    fig = go.Figure(
 
        data=[
 
            go.Table(
 
                columnwidth=[120, 120, 120, 120],
 
                header=dict(
 
                    values=list(comparison_df.columns),
 
                    fill_color="black",
 
                    font=dict(color="white", size=18),
 
                    align="center",
 
                    line=dict(color="white", width=1)
 
                ),
 
                cells=dict(
 
                    values=[
                        comparison_df[col]
                        for col in comparison_df.columns
                    ],
 
                    fill_color="black",
 
                    font=dict(color="white", size=16),
 
                    align="center",
 
                    height=35,
 
                    line=dict(color="white", width=1)
 
                )
 
            )
 
        ]
 
    )
 
    fig.update_layout(
 
        paper_bgcolor="black",
 
        plot_bgcolor="black",
 
        height=300,
 
        margin=dict(l=10, r=10, t=10, b=10)
 
    )
 
    st.plotly_chart(fig, use_container_width=True)
 
    # =========================
    # Difference Bar Plot
    # =========================
 
    bar_colors = comparison_df["Out_of_Range"].map({
        True: "red",
        False: "royalblue"
    })
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
 
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
 
    for spine in ax.spines.values():
        spine.set_color("white")
 
    ax.grid(True, color="white", alpha=0.3)
 
    ax.bar(
        comparison_df["Variable"],
        comparison_df["ROM_Difference_%"],
        color=bar_colors
    )
 
    ax.axhline(0, linestyle="--", color="white")
 
    ax.set_ylabel("Difference (%)")
    ax.set_title("Healthy ROM Comparison")
 
    plt.xticks(rotation=45, color="white")
 
    st.pyplot(fig)
 
# =========================
# Clinical Report
# =========================
with tab4:
 
    findings = []
 
    if max(rom_difference) > 15:
        findings.append(
            "Squat knee ROM asymmetry exceeds 15%."
        )
 
    for _, row in comparison_df.iterrows():
 
        if row["Out_of_Range"]:
 
            findings.append(
                f"{row['Variable']} ROM outside healthy range."
            )
 
    if len(findings) == 0:
 
        st.success("No major abnormalities detected.")
 
    else:
 
        for item in findings:
            st.write("•", item)
 
# =========================
# Raw Data
# =========================
with tab5:
 
    st.subheader("Squat Raw Data")
 
    st.dataframe(
        df_phase,
        use_container_width=True
    )
 
    # =========================
    # CSV Download
    # =========================
 
    csv = df_phase.to_csv(
        index=False
    ).encode("utf-8-sig")
 
    st.download_button(
        "Download CSV",
        csv,
        "squat_raw_data.csv",
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
        "squat_raw_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
# =========================
# Dashboard
# =========================
with tab6:
 
    st.title("Squat Dashboard")
 
    st.caption(t("squat.dashboard_caption"))
 
    # =========================
    # KPI
    # =========================
 
    max_hip_flexion_r = df_phase["hip_flexion_r"].max()
    max_hip_flexion_l = df_phase["hip_flexion_l"].max()
 
    max_knee_flexion_r = df_phase["knee_angle_r"].max()
    max_knee_flexion_l = df_phase["knee_angle_l"].max()
 
    max_ankle_flexion_r = df_phase["ankle_angle_r"].max()
    max_ankle_flexion_l = df_phase["ankle_angle_l"].max()
 
    lumbar_compensation = round(
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min(),
        1
    )
 
    pelvic_compensation = round(
        abs(
            df_phase["pelvis_tilt"].max()
            -
            df_phase["pelvis_tilt"].min()
        ),
        1
    )
 
    pelvic_rotation_rom = round(
        abs(
            df_phase["pelvis_rotation"].max()
            -
            df_phase["pelvis_rotation"].min()
        ),
        1
    )
 
    st.subheader("Key Metrics")
 
    with st.expander(t("common.metrics_expander_label")):
 
        st.markdown(t("squat.metrics_table"))
 
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5, row1_col6 = st.columns(6)
 
    row1_col1.metric("Max Hip Flexion (R)", f"{max_hip_flexion_r:.1f}°")
 
    row1_col2.metric("Max Hip Flexion (L)", f"{max_hip_flexion_l:.1f}°")
 
    row1_col3.metric("Max Knee Flexion (R)", f"{max_knee_flexion_r:.1f}°")
 
    row1_col4.metric("Max Knee Flexion (L)", f"{max_knee_flexion_l:.1f}°")
 
    row1_col5.metric("Max Ankle Dorsiflexion (R)", f"{max_ankle_flexion_r:.1f}°")
 
    row1_col6.metric("Max Ankle Dorsiflexion (L)", f"{max_ankle_flexion_l:.1f}°")
 
    row2_col1, row2_col2, row2_col3 = st.columns(3)
 
    row2_col1.metric("Lumbar Compensation", f"{lumbar_compensation:.1f}°")
 
    row2_col2.metric("Pelvic Compensation", f"{pelvic_compensation:.1f}°")
 
    row2_col3.metric("Pelvic Rotation", f"{pelvic_rotation_rom:.1f}°")
 
    # =========================
    # Interactive Motion Viewer
    # =========================
 
    st.subheader("Interactive Motion Viewer")
 
    st.caption(t("squat.checkbox_instruction"))
 
    left_col, right_col = st.columns([1.2, 4])
 
    # ======================================
    # Left Panel
    # ======================================
 
    with left_col:
 
        st.markdown("### Lower Limb")
 
        st.markdown("#### Hip")
 
        show_hip_r = st.checkbox("Right Hip", value=True)
 
        show_hip_l = st.checkbox("Left Hip", value=True)
 
        st.markdown("#### Knee")
 
        show_knee_r = st.checkbox("Right Knee", value=True)
 
        show_knee_l = st.checkbox("Left Knee", value=True)
 
        st.markdown("#### Ankle")
 
        show_ankle_r = st.checkbox("Right Ankle")
 
        show_ankle_l = st.checkbox("Left Ankle")
 
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
 
        show_lumbar = st.checkbox("Extension")
 
    # ======================================
    # Right Panel
    # ======================================
 
    with right_col:
 
        time = np.arange(len(df_phase)) / 60
 
        # =========================
        # Joint Motion
        # =========================
 
        fig, ax = plt.subplots(
            figsize=(15, 6),
            facecolor="black"
        )
 
        ax.set_facecolor("black")
 
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
 
        for spine in ax.spines.values():
            spine.set_color("white")
 
        ax.grid(color="white", alpha=0.25)
 
        if show_hip_r:
 
            ax.plot(
                time,
                df_phase["hip_flexion_r"],
                label="Right Hip",
                linewidth=2
            )
 
        if show_hip_l:
 
            ax.plot(
                time,
                df_phase["hip_flexion_l"],
                label="Left Hip",
                linewidth=2
            )
 
        if show_knee_r:
 
            ax.plot(
                time,
                df_phase["knee_angle_r"],
                label="Right Knee",
                linewidth=2
            )
 
        if show_knee_l:
 
            ax.plot(
                time,
                df_phase["knee_angle_l"],
                label="Left Knee",
                linewidth=2
            )
 
        if show_ankle_r:
 
            ax.plot(
                time,
                df_phase["ankle_angle_r"],
                label="Right Ankle",
                linewidth=2
            )
 
        if show_ankle_l:
 
            ax.plot(
                time,
                df_phase["ankle_angle_l"],
                label="Left Ankle",
                linewidth=2
            )
 
        if show_lumbar:
 
            ax.plot(
                time,
                df_phase["lumbar_extension"],
                label="Lumbar Extension",
                linewidth=2
            )
 
        ax.set_title("Squat Joint Motion")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg) / Position (m)")
 
        legend = ax.legend(loc="upper right", ncol=2, fontsize=9)
 
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
 
        for text in legend.get_texts():
            text.set_color("white")
 
        st.pyplot(fig)
 
        # ======================================
        # Pelvic Motion
        # ======================================
 
        st.subheader("Pelvic Motion")
 
        fig2, ax2 = plt.subplots(
            figsize=(15, 5),
            facecolor="black"
        )
 
        ax2.set_facecolor("black")
 
        ax2.tick_params(colors="white")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
        ax2.title.set_color("white")
 
        for spine in ax2.spines.values():
            spine.set_color("white")
 
        ax2.grid(color="white", alpha=0.25)
 
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
 
        ax2.set_title("Squat Pelvic Motion")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angle (deg) / Translation (mm)")
 
        legend2 = ax2.legend(loc="upper right", ncol=2, fontsize=9)
 
        legend2.get_frame().set_facecolor("black")
        legend2.get_frame().set_edgecolor("white")
 
        for text in legend2.get_texts():
            text.set_color("white")
 
        st.pyplot(fig2)
 
    # =========================
    # Joint ROM Summary
    # =========================
 
    st.subheader("Joint ROM Summary")
 
    with st.expander(t("common.joint_rom_expander_label")):
 
        st.markdown(t("squat.joint_rom_content"))
 
    rom_joints = {
 
        "Hip": "hip_flexion_r",
 
        "Knee": "knee_angle_r",
 
        "Ankle": "ankle_angle_r"
 
    }
 
    rom_values = []
 
    for variable in rom_joints.values():
 
        rom = (
            df_phase[variable].max()
            -
            df_phase[variable].min()
        )
 
        rom_values.append(round(rom, 1))
 
    fig, ax = plt.subplots(
        figsize=(8, 4),
        facecolor="black"
    )
 
    ax.set_facecolor("black")
 
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
 
    for spine in ax.spines.values():
        spine.set_color("white")
 
    ax.grid(color="white", alpha=0.25, axis="y")
 
    bars = ax.bar(
        list(rom_joints.keys()),
        rom_values
    )
 
    for bar, value in zip(bars, rom_values):
 
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}°",
            ha="center",
            va="bottom",
            color="white"
        )
 
    ax.set_ylabel("ROM (deg)")
    ax.set_title("Squat Joint ROM")
 
    st.pyplot(fig)
 
    # =========================
    # Joint Asymmetry
    # =========================
 
    st.subheader("Joint Asymmetry")
 
    with st.expander(t("common.joint_asymmetry_expander_label")):
 
        st.markdown(t("squat.joint_asymmetry_content"))
 
    asymmetry_joints = {
 
        "Hip": ("hip_flexion_r", "hip_flexion_l"),
        "Knee": ("knee_angle_r", "knee_angle_l"),
        "Ankle": ("ankle_angle_r", "ankle_angle_l")
 
    }
 
    asymmetry_results = {}
 
    for joint_name, (right_var, left_var) in asymmetry_joints.items():
 
        right_rom = (
            df_phase[right_var].max()
            -
            df_phase[right_var].min()
        )
 
        left_rom = (
            df_phase[left_var].max()
            -
            df_phase[left_var].min()
        )
 
        if max(right_rom, left_rom) == 0:
 
            asymmetry = 0
 
        else:
 
            asymmetry = (
                abs(right_rom - left_rom)
                / max(right_rom, left_rom)
                * 100
            )
 
        asymmetry_results[joint_name] = round(asymmetry, 2)
 
    fig, ax = plt.subplots(
        figsize=(8, 4),
        facecolor="black"
    )
 
    ax.set_facecolor("black")
 
    ax.tick_params(colors="white")
    ax.title.set_color("white")
 
    for spine in ax.spines.values():
        spine.set_color("white")
 
    ax.grid(color="white", alpha=0.25, axis="y")
 
    bars = ax.bar(
        list(asymmetry_results.keys()),
        list(asymmetry_results.values())
    )
 
    for bar, value in zip(bars, asymmetry_results.values()):
 
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color="white"
        )
 
    ax.axhline(15, color="red", linestyle="--", label="15% Threshold")
 
    ax.set_ylabel("Asymmetry (%)")
    ax.set_title("Joint Asymmetry")
 
    legend = ax.legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    st.pyplot(fig)
 
# =========================
# Movement Score
# =========================
with tab7:
 
    # =========================
    # Movement Features
    # =========================
 
    st.subheader("Movement Features")
 
    st.caption(t("squat.feature_caption"))
 
    with st.expander(t("common.feature_expander_label")):
 
        st.markdown(t("squat.feature_table"))
 
    squat_depth = pelvis_max - pelvis_min
 
    pelvis_stability = df_phase["pelvis_list"].std()
 
    trunk_compensation = df_phase["lumbar_extension"].max()
 
    hip_asymmetry = asymmetry_results["Hip"]
 
    knee_asymmetry = asymmetry_results["Knee"]
 
    ankle_asymmetry = asymmetry_results["Ankle"]
 
    feature_df = pd.DataFrame({
 
        "Feature": [
 
            "Squat Depth",
 
            "Pelvic Stability",
 
            "Lumbar Compensation",
 
            "Hip Asymmetry",
 
            "Knee Asymmetry",
 
            "Ankle Asymmetry"
 
        ],
 
        "Value": [
 
            round(squat_depth, 3),
 
            round(pelvis_stability, 2),
 
            round(trunk_compensation, 2),
 
            round(hip_asymmetry, 2),
 
            round(knee_asymmetry, 2),
 
            round(ankle_asymmetry, 2)
 
        ]
    })
 
    # =========================
    # Plotly Feature Table
    # =========================
 
    fig = go.Figure(
 
        data=[
 
            go.Table(
 
                columnwidth=[250, 120],
 
                header=dict(
 
                    values=["Feature", "Value"],
 
                    fill_color="black",
 
                    font=dict(color="white", size=16),
 
                    align="center",
 
                    line=dict(color="white", width=1)
 
                ),
 
                cells=dict(
 
                    values=[
                        feature_df["Feature"],
                        feature_df["Value"]
                    ],
 
                    fill_color="black",
 
                    height=45,
 
                    font=dict(color="white", size=15),
 
                    align="center",
 
                    line=dict(color="white", width=1)
 
                )
 
            )
 
        ]
     
 
    )
 
    fig.update_layout(
 
        height=230,
 
        margin=dict(l=10, r=10, t=10, b=5),
 
        paper_bgcolor="black",
 
        plot_bgcolor="black"
 
    )
 
    st.plotly_chart(fig, use_container_width=True)
 
    # =========================
    # Movement Score
    # =========================
 
    overall_asymmetry = (
        hip_asymmetry
        +
        knee_asymmetry
        +
        ankle_asymmetry
    ) / 3
 
    symmetry_score = max(0, 100 - overall_asymmetry)
 
    stability_score = max(0, 100 - pelvis_stability * 10)
 
    compensation_score = max(0, 100 - trunk_compensation * 2)
 
    mobility_score = min(100, squat_depth * 500)
 
    overall_score = round(
 
        (
            symmetry_score * 0.35
            +
            stability_score * 0.30
            +
            compensation_score * 0.25
            +
            mobility_score * 0.10
        ),
 
        1
    )
 
    st.subheader("Movement Score")
 
    st.caption(t("squat.score_caption"))
 
    with st.expander(t("common.score_expander_label")):
 
        st.markdown(t("squat.score_content"))
 
    st.metric(
        "Overall Score",
        f"{overall_score}/100"
    )
# =========================
# PDF Report
# =========================
with tab8:

    lang_choice = st.radio(
        "レポート言語 / Report Language",
        ["日本語", "English"],
        horizontal=True
    )

    lang_code = "ja" if lang_choice == "日本語" else "en"

    UI_LABELS = {
        "ja": {
            "header": "PDFレポート",
            "caption": "Squat Analysisの主要な結果をまとめた統合PDFレポートを生成します。",
            "subject_name": "対象者名",
            "exam_date": "測定日",
            "examiner": "検者",
            "comment_heading": "総合評価 (Clinical Impression)",
            "comment_label": "検者による総合所見・コメントを記入してください（PDFに反映されます）",
            "generate_button": "📄 PDFレポートを生成",
            "download_label": "📥 PDFレポートをダウンロード",
            "success_message": "PDFレポートを生成しました。上のボタンからダウンロードしてください。"
        },
        "en": {
            "header": "PDF Report",
            "caption": "Generate a comprehensive PDF report summarizing the key Squat Analysis results.",
            "subject_name": "Subject Name",
            "exam_date": "Exam Date",
            "examiner": "Examiner",
            "comment_heading": "Clinical Impression",
            "comment_label": "Enter the examiner's overall clinical impression / comments (included in the PDF)",
            "generate_button": "📄 Generate PDF Report",
            "download_label": "📥 Download PDF Report",
            "success_message": "PDF report generated. Use the button above to download it."
        }
    }

    UL = UI_LABELS[lang_code]

    st.subheader(UL["header"])
    st.caption(UL["caption"])

    col1, col2, col3 = st.columns(3)
    with col1:
        subject_name = st.text_input(UL["subject_name"], value="")
    with col2:
        exam_date = st.text_input(UL["exam_date"], value="")
    with col3:
        examiner_name = st.text_input(UL["examiner"], value="")

    st.markdown(f"#### {UL['comment_heading']}")
    clinical_comment = st.text_area(
        UL["comment_label"],
        value="",
        height=150
    )

    if st.button("📄 Generate PDF Report"):

        LABELS = {
            "ja": {
                "title": "Squat Analysis 臨床レポート",
                "subject_info": "対象者名: {name} ｜ 測定日: {date} ｜ 検者: {examiner}",
                "event_info": "Bottom フレーム: {b} ／ Standing フレーム: {s}",
                "phase_detection_heading": "スクワット位相検出",
                "phase_plot_title": "位相検出プロット",
                "xlabel_frame": "フレーム",
                "ylabel_pelvis": "骨盤垂直位置 (m)",
                "key_metrics_heading": "主要指標",
                "metric_col": "指標", "right_col": "右", "left_col": "左",
                "max_hip": "最大股関節屈曲 (°)",
                "max_knee": "最大膝関節屈曲 (°)",
                "max_ankle": "最大足関節背屈 (°)",
                "compensation_line": "腰椎代償: {l:.1f}°　骨盤代償: {p:.1f}°　骨盤回旋: {r:.1f}°",
                "joint_rom_heading": "関節可動域サマリー（試技全体）",
                "joint_col": "関節", "rom_col": "可動域 (°)",
                "symmetry_heading": "左右対称性分析",
                "phase_col": "フェーズ", "right_rom_col": "右ROM", "left_rom_col": "左ROM", "asym_col": "非対称性(%)",
                "max_avg_label": "{joint}　(最大: {mx:.1f}% / 平均: {avg:.1f}%)",
                "asym_by_phase_title": "{joint} フェーズ別ROM非対称性",
                "threshold_label": "15%しきい値",
                "healthy_rom_heading": "健常可動域比較",
                "difference_pct_label": "差分 (%)",
                "clinical_findings_heading": "臨床所見",
                "no_findings": "重大な異常所見は検出されませんでした。",
                "asym_finding": "{joint}の可動域非対称性が15%を超えています（{v:.1f}%）。",
                "range_finding": "{var} の可動域が健常範囲外です。",
                "clinical_impression_heading": "Clinical Impression（総合評価）",
                "no_comment": "(記入なし)",
                "movement_score_heading": "動作スコア",
                "overall_score_label": "総合スコア: {s} / 100",
                "feature_col": "特徴量", "value_col": "値",
            },
            "en": {
                "title": "Squat Analysis Clinical Report",
                "subject_info": "Subject: {name}  |  Exam Date: {date}  |  Examiner: {examiner}",
                "event_info": "Bottom frame: {b} / Standing frame: {s}",
                "phase_detection_heading": "Squat Phase Detection",
                "phase_plot_title": "Phase Detection Plot",
                "xlabel_frame": "Frame",
                "ylabel_pelvis": "Pelvis Vertical Position (m)",
                "key_metrics_heading": "Key Metrics",
                "metric_col": "Metric", "right_col": "Right", "left_col": "Left",
                "max_hip": "Max Hip Flexion (°)",
                "max_knee": "Max Knee Flexion (°)",
                "max_ankle": "Max Ankle Dorsiflexion (°)",
                "compensation_line": "Lumbar Compensation: {l:.1f}°  Pelvic Compensation: {p:.1f}°  Pelvic Rotation: {r:.1f}°",
                "joint_rom_heading": "Joint ROM Summary (Whole Trial)",
                "joint_col": "Joint", "rom_col": "ROM (deg)",
                "symmetry_heading": "Symmetry Analysis",
                "phase_col": "Phase", "right_rom_col": "Right_ROM", "left_rom_col": "Left_ROM", "asym_col": "Asymmetry_%",
                "max_avg_label": "{joint}  (Max: {mx:.1f}% / Avg: {avg:.1f}%)",
                "asym_by_phase_title": "{joint} ROM Asymmetry by Phase",
                "threshold_label": "15% Threshold",
                "healthy_rom_heading": "Healthy ROM Comparison",
                "difference_pct_label": "Difference (%)",
                "clinical_findings_heading": "Clinical Findings",
                "no_findings": "No major abnormalities detected.",
                "asym_finding": "{joint} ROM asymmetry exceeds 15% ({v:.1f}%).",
                "range_finding": "{var} ROM outside healthy range.",
                "clinical_impression_heading": "Clinical Impression",
                "no_comment": "(No comment entered)",
                "movement_score_heading": "Movement Score",
                "overall_score_label": "OVERALL SCORE: {s} / 100",
                "feature_col": "Feature", "value_col": "Value",
            },
        }
        LB = LABELS[lang_code]

        report_buffer = BytesIO()
        doc = SimpleDocTemplate(
            report_buffer,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleJP", parent=styles["Title"],
            fontName="HeiseiKakuGo-W5", fontSize=18
        )
        heading_style = ParagraphStyle(
            "HeadingJP", parent=styles["Heading2"],
            fontName="HeiseiKakuGo-W5", spaceBefore=12, spaceAfter=6
        )
        normal_style = ParagraphStyle(
            "NormalJP", parent=styles["Normal"],
            fontName="HeiseiKakuGo-W5", fontSize=9
        )
        # ---- Overall Score用ハイライトスタイル（コンパクト版） ----
        if overall_score >= 80:
            score_color = colors.HexColor("#1B7A3D")   # 緑
        elif overall_score >= 60:
            score_color = colors.HexColor("#B8860B")   # 黄土色
        else:
            score_color = colors.HexColor("#C0392B")   # 赤
        score_style = ParagraphStyle(
            "ScoreStyle",
            parent=styles["Normal"],
            fontName="HeiseiKakuGo-W5",
            fontSize=16,
            leading=20,
            textColor=colors.white,
            backColor=score_color,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=8,
            borderPadding=6
        )
        # ※ しきい値(80/60)は仮の基準です。臨床基準に合わせて調整してください。
        TABLE_STYLE = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ])
        colors_phase_pdf = {
            "Standing": "dodgerblue",
            "Descending": "orange",
            "Bottom": "red",
            "Ascending": "limegreen"
        }
        elements = []
        # ---- Title / Subject Info ----
        elements.append(Paragraph(LB["title"], title_style))
        elements.append(Spacer(1, 6))
        info_text = LB["subject_info"].format(
            name=subject_name or "-",
            date=exam_date or "-",
            examiner=examiner_name or "-"
        )
        elements.append(Paragraph(info_text, normal_style))
        elements.append(
            Paragraph(
                LB["event_info"].format(b=bottom_idx, s=standing_idx),
                normal_style
            )
        )
        elements.append(Spacer(1, 12))
        # ---- Squat Phase Detection Plot ----
        elements.append(Paragraph(LB["phase_detection_heading"], heading_style))
        pdf_phase_fig, pdf_phase_ax = plt.subplots(figsize=(10, 4))
        pdf_phase_ax.plot(
            df_phase.index, df_phase["pelvis_ty"],
            color="black", linewidth=1, alpha=0.4
        )
        for phase in phase_order:
            idx = df_phase["Phase"] == phase
            pdf_phase_ax.scatter(
                df_phase.index[idx],
                df_phase["pelvis_ty"][idx],
                c=colors_phase_pdf[phase],
                s=8,
                label=phase
            )
        pdf_phase_ax.axvline(bottom_idx, color="red", linestyle="--", linewidth=1)
        pdf_phase_ax.axvline(standing_idx, color="dodgerblue", linestyle="--", linewidth=1)
        pdf_phase_ax.set_title("Phase Detection Plot")
        pdf_phase_ax.set_xlabel("Frame")
        pdf_phase_ax.set_ylabel("Pelvis Vertical Position (m)")
        pdf_phase_ax.legend(fontsize=8)
        pdf_phase_ax.grid(alpha=0.3)
        elements.append(fig_to_rl_image(pdf_phase_fig, width_cm=16))
        elements.append(Spacer(1, 12))
        # ---- Key Metrics ----
        elements.append(Paragraph(LB["key_metrics_heading"], heading_style))
        key_metrics_data = [
            [LB["metric_col"], LB["right_col"], LB["left_col"]],
            [LB["max_hip"], f"{max_hip_flexion_r:.1f}", f"{max_hip_flexion_l:.1f}"],
            [LB["max_knee"], f"{max_knee_flexion_r:.1f}", f"{max_knee_flexion_l:.1f}"],
            [LB["max_ankle"], f"{max_ankle_flexion_r:.1f}", f"{max_ankle_flexion_l:.1f}"],
        ]
        key_metrics_table = Table(key_metrics_data, hAlign="LEFT")
        key_metrics_table.setStyle(TABLE_STYLE)
        elements.append(key_metrics_table)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            LB["compensation_line"].format(
                l=lumbar_compensation, p=pelvic_compensation, r=pelvic_rotation_rom
            ),
            normal_style
        ))
        elements.append(Spacer(1, 12))
        # ---- Joint ROM Summary (試技全体でのROM。フェーズ別ではない) ----
        elements.append(Paragraph(LB["joint_rom_heading"], heading_style))
        rom_joints_pdf = {
            "Hip": "hip_flexion_r",
            "Knee": "knee_angle_r",
            "Ankle": "ankle_angle_r",
        }
        rom_summary_rows = [[LB["joint_col"], LB["rom_col"]]]
        rom_summary_values = []
        for joint_name, variable in rom_joints_pdf.items():
            rom_val = df_phase[variable].max() - df_phase[variable].min()
            rom_summary_values.append(rom_val)
            rom_summary_rows.append([joint_name, f"{rom_val:.1f}"])
        rom_summary_table = Table(rom_summary_rows, hAlign="LEFT")
        rom_summary_table.setStyle(TABLE_STYLE)
        elements.append(rom_summary_table)
        elements.append(Spacer(1, 6))
        rom_summary_fig, rom_summary_ax = plt.subplots(figsize=(6, 3))
        rom_summary_ax.bar(list(rom_joints_pdf.keys()), rom_summary_values, color="royalblue")
        for i, v in enumerate(rom_summary_values):
            rom_summary_ax.text(i, v, f"{v:.1f}°", ha="center", va="bottom", fontsize=8)
        rom_summary_ax.set_ylabel("ROM (deg)")
        rom_summary_ax.set_title("Joint ROM Summary (Whole Trial)")
        rom_summary_ax.grid(alpha=0.3, axis="y")
        elements.append(fig_to_rl_image(rom_summary_fig, width_cm=11))
        elements.append(Spacer(1, 12))
        # ---- Symmetry Analysis (表 + グラフ) ----
        elements.append(Paragraph(LB["symmetry_heading"], heading_style))
        symmetry_joints_pdf = {
            "Hip": ("hip_flexion_r", "hip_flexion_l"),
            "Knee": ("knee_angle_r", "knee_angle_l"),
            "Ankle": ("ankle_angle_r", "ankle_angle_l"),
        }
        for joint_name, (right_var, left_var) in symmetry_joints_pdf.items():
            right_df = phase_summary_df[phase_summary_df["Variable"] == right_var]
            left_df = phase_summary_df[phase_summary_df["Variable"] == left_var]
            rows = [[LB["phase_col"], LB["right_rom_col"], LB["left_rom_col"], LB["asym_col"]]]
            asym_values = []
            for phase in phase_order:
                right_rom = right_df[f"{phase}_ROM"].iloc[0]
                left_rom = left_df[f"{phase}_ROM"].iloc[0]
                asymmetry = 0 if max(right_rom, left_rom) == 0 else (
                    abs(right_rom - left_rom) / max(right_rom, left_rom) * 100
                )
                asym_values.append(asymmetry)
                rows.append([phase, f"{right_rom:.2f}", f"{left_rom:.2f}", f"{asymmetry:.2f}"])
            elements.append(Paragraph(
                LB["max_avg_label"].format(
                    joint=joint_name, mx=max(asym_values), avg=np.mean(asym_values)
                ),
                normal_style
            ))
            joint_table = Table(rows, hAlign="LEFT")
            joint_table.setStyle(TABLE_STYLE)
            elements.append(joint_table)
            elements.append(Spacer(1, 6))
            sym_fig, sym_ax = plt.subplots(figsize=(6, 3))
            sym_ax.bar(phase_order, asym_values, color="royalblue")
            sym_ax.axhline(15, color="red", linestyle="--", linewidth=1, label="15% Threshold")
            sym_ax.set_ylabel("Asymmetry (%)")
            sym_ax.set_title(f"{joint_name} ROM Asymmetry by Phase")
            sym_ax.legend(fontsize=8)
            sym_ax.grid(alpha=0.3, axis="y")
            elements.append(fig_to_rl_image(sym_fig, width_cm=11))
            elements.append(Spacer(1, 10))
        # ---- Healthy ROM Comparison (表 + グラフ) ----
        elements.append(Paragraph(LB["healthy_rom_heading"], heading_style))
        hrom_rows = [list(comparison_df.columns)]
        for _, row in comparison_df.iterrows():
            hrom_rows.append([str(v) for v in row.tolist()])
        hrom_table = Table(hrom_rows, hAlign="LEFT")
        hrom_table.setStyle(TABLE_STYLE)
        elements.append(hrom_table)
        elements.append(Spacer(1, 8))
        hrom_fig, hrom_ax = plt.subplots(figsize=(10, 4))
        bar_colors_pdf = [
            "red" if row["Out_of_Range"] else "royalblue"
            for _, row in comparison_df.iterrows()
        ]
        hrom_ax.bar(
            comparison_df["Variable"],
            comparison_df["ROM_Difference_%"],
            color=bar_colors_pdf
        )
        hrom_ax.axhline(0, color="black", linestyle="--", linewidth=1)
        hrom_ax.set_ylabel("Difference (%)")
        hrom_ax.set_title("Healthy ROM Comparison")
        plt.setp(hrom_ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        hrom_ax.grid(alpha=0.3, axis="y")
        elements.append(fig_to_rl_image(hrom_fig, width_cm=16))
        elements.append(Spacer(1, 12))
        # ---- Clinical Findings ----
        elements.append(Paragraph(LB["clinical_findings_heading"], heading_style))
        pdf_findings = []
        for joint_name in symmetry_joints_pdf.keys():
            asym_value = asymmetry_results.get(joint_name, 0)
            if asym_value > 15:
                pdf_findings.append(LB["asym_finding"].format(joint=joint_name, v=asym_value))
        for _, row in comparison_df.iterrows():
            if row["Out_of_Range"]:
                pdf_findings.append(LB["range_finding"].format(var=row["Variable"]))
        if len(pdf_findings) == 0:
            elements.append(Paragraph(LB["no_findings"], normal_style))
        else:
            for item in pdf_findings:
                elements.append(Paragraph(f"・{item}", normal_style))
        elements.append(Spacer(1, 12))
        # ---- Clinical Impression（検者記入欄） ----
        elements.append(Paragraph(LB["clinical_impression_heading"], heading_style))
        comment_text = (
            escape(clinical_comment).replace("\n", "<br/>")
            if clinical_comment.strip()
            else LB["no_comment"]
        )
        elements.append(Paragraph(comment_text, normal_style))
        elements.append(Spacer(1, 12))
        # ---- Movement Score（ハイライト表示・コンパクト） ----
        elements.append(Paragraph(LB["movement_score_heading"], heading_style))
        elements.append(Paragraph(
            LB["overall_score_label"].format(s=overall_score),
            score_style
        ))
        feature_rows = [[LB["feature_col"], LB["value_col"]]]
        for _, row in feature_df.iterrows():
            feature_rows.append([str(row["Feature"]), str(row["Value"])])
        feature_table = Table(feature_rows, hAlign="LEFT")
        feature_table.setStyle(TABLE_STYLE)
        elements.append(feature_table)
        doc.build(elements)
        st.download_button(
            "📥 Download PDF Report",
            data=report_buffer.getvalue(),
            file_name="Squat_Clinical_Report.pdf",
            mime="application/pdf"
        )
        st.success("PDFレポートを生成しました。上のボタンからダウンロードしてください。")
