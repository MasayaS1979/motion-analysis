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
 
st.set_page_config(page_title="Gait Analysis", layout="wide")
 
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
 
st.title("Gait Analysis")
 
uploaded_file = st.session_state.get("uploaded_file")
 
if uploaded_file is None:
    st.warning(t("common.upload_warning"))
    st.stop()
 
# =========================
# Load Data
# =========================
 
df = pd.read_excel(uploaded_file, header=10)
 
# =========================
# Gait Phase Detector
# =========================
 
def gait_detector(signal):
 
    signal_smooth = (
        signal
        .rolling(window=5, center=True)
        .mean()
        .bfill()
        .ffill()
    )
 
    velocity = signal_smooth.diff()
 
    signal_min = signal_smooth.min()
    signal_max = signal_smooth.max()
 
    signal_range = signal_max - signal_min
 
    heel_threshold = signal_min + signal_range * 0.25
    toe_threshold = signal_max - signal_range * 0.25
 
    velocity_threshold = 0.05
 
    phases = []
 
    for p, v in zip(signal_smooth, velocity):
 
        if pd.isna(v):
 
            phase = "Heel Strike"
 
        elif p <= heel_threshold:
 
            if v < 0:
                phase = "Heel Strike"
            else:
                phase = "Mid Stance"
 
        elif p >= toe_threshold:
 
            if v > 0:
                phase = "Toe Off"
            else:
                phase = "Swing"
 
        else:
 
            if v > velocity_threshold:
                phase = "Toe Off"
 
            elif v < -velocity_threshold:
                phase = "Heel Strike"
 
            else:
                phase = (
                    phases[-1]
                    if len(phases) > 0
                    else "Mid Stance"
                )
 
        phases.append(phase)
 
    return phases, signal_smooth
 
# =========================
# Right / Left Gait
# =========================
 
phase_r, signal_smooth_r = gait_detector(df["ankle_angle_r"])
 
df_phase_r = df.copy()
df_phase_r["Phase"] = phase_r
 
phase_l, signal_smooth_l = gait_detector(df["ankle_angle_l"])
 
df_phase_l = df.copy()
df_phase_l["Phase"] = phase_l
 
df_phase = df_phase_r.copy()
 
phase_order = [
    "Heel Strike",
    "Mid Stance",
    "Toe Off",
    "Swing"
]
 
display_phase_order = [
    "Heel Strike",
    "Mid Stance",
    "Toe Off",
    "Swing"
]
 
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
 
    "hip_flexion_r": {"min": -10, "max": 30},
    "hip_flexion_l": {"min": -10, "max": 30},
 
    "knee_angle_r": {"min": 0, "max": 65},
    "knee_angle_l": {"min": 0, "max": 65},
 
    "ankle_angle_r": {"min": -10, "max": 20},
    "ankle_angle_l": {"min": -10, "max": 20},
 
    # NOTE: pelvis_tilt / pelvis_rotation / lumbar_extension are trunk
    # "compensation" signals rather than a primary joint ROM — smaller
    # excursion is generally considered better gait form. The 0-10°
    # band below is a placeholder threshold for "acceptable
    # compensation" and should be reviewed/adjusted against your own
    # clinical reference rather than treated as an established
    # normative range.
    "pelvis_tilt": {"min": 0, "max": 10},
    "pelvis_rotation": {"min": 0, "max": 10},
    "lumbar_extension": {"min": 0, "max": 10}
 
}
 
healthy_rom_df = pd.DataFrame([
    {
        "Variable": variable,
        "Min": values["min"],
        "Max": values["max"]
    }
    for variable, values in HEALTHY_ROM.items()
])
 
# =========================
# Gait Phase Weights
# =========================
 
PHASE_WEIGHTS = {
 
    "Heel Strike": 0.20,
    "Mid Stance": 0.40,
    "Toe Off": 0.20,
    "Swing": 0.20
 
}
 
# =========================
# Subject ROM
# =========================
 
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
 
# =========================
# Healthy Comparison
# =========================
 
def compare_gait_to_healthy(df):
 
    results = []
 
    for variable, limits in HEALTHY_ROM.items():
 
        if variable not in df.columns:
            continue
 
        subject_min = df[variable].min()
        subject_max = df[variable].max()
 
        subject_rom = subject_max - subject_min
 
        healthy_min = limits["min"]
        healthy_max = limits["max"]
 
        healthy_rom = healthy_max - healthy_min
 
        rom_difference = (
            (subject_rom - healthy_rom)
            / healthy_rom
            * 100
        )
 
        min_difference = subject_min - healthy_min
        max_difference = subject_max - healthy_max
 
        status = "Normal"
 
        if abs(rom_difference) > 20:
            status = "Abnormal"
 
        results.append({
 
            "Variable": variable,
 
            "Subject_Min": round(subject_min, 2),
            "Healthy_Min": healthy_min,
            "Min_Diff": round(min_difference, 2),
 
            "Subject_Max": round(subject_max, 2),
            "Healthy_Max": healthy_max,
            "Max_Diff": round(max_difference, 2),
 
            "Subject_ROM": round(subject_rom, 2),
            "Healthy_ROM": round(healthy_rom, 2),
            "ROM_Difference_%": round(rom_difference, 2),
 
            "Status": status
 
        })
 
    return pd.DataFrame(results)
 
comparison_df = compare_gait_to_healthy(df_phase)
 
# =========================
# Phase Weighted Symmetry
# =========================
 
def calculate_phase_weighted_symmetry(phase_summary_df):
 
    phase_scores = []
 
    joints = [
        ("hip_flexion_r", "hip_flexion_l"),
        ("knee_angle_r", "knee_angle_l"),
        ("ankle_angle_r", "ankle_angle_l")
    ]
 
    for phase, weight in PHASE_WEIGHTS.items():
 
        asymmetry_values = []
 
        for right_var, left_var in joints:
 
            right_row = phase_summary_df[
                phase_summary_df["Variable"] == right_var
            ]
 
            left_row = phase_summary_df[
                phase_summary_df["Variable"] == left_var
            ]
 
            if len(right_row) > 0 and len(left_row) > 0:
 
                right_rom = right_row[f"{phase}_ROM"].values[0]
                left_rom = left_row[f"{phase}_ROM"].values[0]
 
                if max(right_rom, left_rom) > 0:
 
                    asymmetry = (
                        abs(right_rom - left_rom)
                        / max(right_rom, left_rom)
                        * 100
                    )
 
                    asymmetry_values.append(asymmetry)
 
        if len(asymmetry_values) > 0:
 
            phase_asymmetry = np.mean(asymmetry_values)
            phase_score = 100 - phase_asymmetry
 
            phase_scores.append(phase_score * weight)
 
    return round(sum(phase_scores), 1)
 
# =========================
# Gait Metrics
# =========================
 
duration_sec = len(df_phase) / 60
 
heel_events = df_phase.index[
    (df_phase["Phase"] == "Heel Strike")
    &
    (df_phase["Phase"].shift(1) != "Heel Strike")
]
 
cadence = len(heel_events) / duration_sec * 60
 
if len(heel_events) > 1:
 
    step_time = np.mean(np.diff(heel_events)) / 60
 
else:
 
    step_time = np.nan
 
# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Gait Phases",
    "Movement Analysis",
    "Symmetry Analysis",
    "Clinical Report",
    "Raw Data",
    "Dashboard",
    "Movement Score",
    "PDF Report"
])
 
# =========================
# Gait Phases
# =========================
 
with tab1:
 
    st.subheader("Gait Phase Detection Plot")
 
    st.caption(t("gait.phase_caption"))
 
    colors_phase = {
 
        "Heel Strike": "blue",
        "Mid Stance": "orange",
        "Toe Off": "limegreen",
        "Swing": "red"
 
    }
 
    fig, ax = plt.subplots(figsize=(15, 6))
 
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
    # Ankle Signals
    # =========================
 
    ax.plot(
        signal_smooth_r,
        color="white",
        linewidth=2,
        label="Right Ankle"
    )
 
    ax.plot(
        signal_smooth_l,
        color="cyan",
        linewidth=2,
        linestyle="--",
        label="Left Ankle"
    )
 
    for phase in phase_order:
 
        idx_r = df_phase_r["Phase"] == phase
 
        ax.scatter(
            df_phase_r.index[idx_r],
            signal_smooth_r[idx_r],
            c=colors_phase[phase],
            marker="o",
            s=10,
            label=f"R {phase}"
        )
 
    for phase in phase_order:
 
        idx_l = df_phase_l["Phase"] == phase
 
        ax.scatter(
            df_phase_l.index[idx_l],
            signal_smooth_l[idx_l],
            c=colors_phase[phase],
            marker="^",
            s=10,
            label=f"L {phase}"
        )
 
    ax.set_title("Gait Phase Detection Plot")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ankle Angle (deg)")
 
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
 
    for text in legend.get_texts():
        text.set_color("white")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    plt.tight_layout()
 
    st.pyplot(fig)
 
    # -------------------------
    # Gait Metrics
    # -------------------------
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    with col1:
 
        st.metric("Cadence (steps/min)", round(cadence, 1))
 
    with col2:
 
        st.metric(
            "Step Time (sec)",
            round(step_time, 2) if not np.isnan(step_time) else "N/A"
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
 
    heel_strike_df = phase_summary_df[
        [
            "Variable",
            "Heel Strike_Min",
            "Heel Strike_Max",
            "Heel Strike_Mean",
            "Heel Strike_Std",
            "Heel Strike_ROM"
        ]
    ]
 
    mid_stance_df = phase_summary_df[
        [
            "Variable",
            "Mid Stance_Min",
            "Mid Stance_Max",
            "Mid Stance_Mean",
            "Mid Stance_Std",
            "Mid Stance_ROM"
        ]
    ]
 
    toe_off_df = phase_summary_df[
        [
            "Variable",
            "Toe Off_Min",
            "Toe Off_Max",
            "Toe Off_Mean",
            "Toe Off_Std",
            "Toe Off_ROM"
        ]
    ]
 
    swing_df = phase_summary_df[
        [
            "Variable",
            "Swing_Min",
            "Swing_Max",
            "Swing_Mean",
            "Swing_Std",
            "Swing_ROM"
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
 
        heel_strike_df.to_excel(
            writer,
            sheet_name="Heel Strike",
            index=False
        )
 
        mid_stance_df.to_excel(
            writer,
            sheet_name="Mid Stance",
            index=False
        )
 
        toe_off_df.to_excel(
            writer,
            sheet_name="Toe Off",
            index=False
        )
 
        swing_df.to_excel(
            writer,
            sheet_name="Swing",
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
 
            file_name="Gait_Phase_Summary.xlsx",
 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
 
        )
 
    with col2:
 
        st.download_button(
 
            "📄 Download PDF",
 
            data=pdf_buffer.getvalue(),
 
            file_name="Gait_Phase_Summary.pdf",
 
            mime="application/pdf"
 
        )
 
    # ==========================================================
    # Phase Tables
    # ==========================================================
 
    st.markdown("---")
 
    st.subheader("Phase Statistics")
 
    tab_hs, tab_ms, tab_to, tab_sw = st.tabs(
        [
            "Heel Strike",
            "Mid Stance",
            "Toe Off",
            "Swing"
        ]
    )
 
    with tab_hs:
 
        st.dataframe(
            heel_strike_df,
            use_container_width=True
        )
 
    with tab_ms:
 
        st.dataframe(
            mid_stance_df,
            use_container_width=True
        )
 
    with tab_to:
 
        st.dataframe(
            toe_off_df,
            use_container_width=True
        )
 
    with tab_sw:
 
        st.dataframe(
            swing_df,
            use_container_width=True
        )
 
# =========================
# Movement Analysis
# =========================
with tab2:
 
    st.subheader("Joint Time Series")
 
    st.caption(t("gait.movement_analysis_caption"))
 
    # OpenCap sampling rate (60 Hz)
    time = np.arange(len(df_phase)) / 60
 
    fig, ax = plt.subplots(
        9,
        1,
        figsize=(12, 29)
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
    # Hip Flexion
    # =====================
 
    ax[0].plot(time, df_phase["hip_flexion_r"], label="Right", linewidth=2)
    ax[0].plot(time, df_phase["hip_flexion_l"], label="Left", linewidth=2)
 
    ax[0].set_title("Hip Flexion")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Angle (deg)")
 
    legend = ax[0].legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Knee Angle
    # =====================
 
    ax[1].plot(time, df_phase["knee_angle_r"], label="Right", linewidth=2)
    ax[1].plot(time, df_phase["knee_angle_l"], label="Left", linewidth=2)
 
    ax[1].set_title("Knee Angle")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (deg)")
 
    legend = ax[1].legend()
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Ankle Angle
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
    # Pelvic Obliquity
    # =====================
 
    ax[4].plot(time, df_phase["pelvis_list"], linewidth=2, color="orange")
 
    ax[4].set_title("Pelvic Obliquity")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Rotation
    # =====================
 
    ax[5].plot(time, df_phase["pelvis_rotation"], linewidth=2, color="lime")
 
    ax[5].set_title("Pelvic Rotation")
    ax[5].set_xlabel("Time (s)")
    ax[5].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Medial-Lateral
    # =====================
 
    ax[6].plot(time, df_phase["pelvis_tx"], linewidth=2, color="yellow")
 
    ax[6].set_title("Pelvic Medial-Lateral Position")
    ax[6].set_xlabel("Time (s)")
    ax[6].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Vertical
    # =====================
 
    ax[7].plot(time, df_phase["pelvis_ty"], linewidth=2, color="deepskyblue")
 
    ax[7].set_title("Pelvic Vertical Position")
    ax[7].set_xlabel("Time (s)")
    ax[7].set_ylabel("Position (m)")
 
    # =====================
    # Lumbar Extension
    # =====================
 
    ax[8].plot(time, df_phase["lumbar_extension"], linewidth=2, color="magenta")
 
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
 
    st.caption(t("gait.symmetry_caption"))
 
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
 
        if len(right_df) == 0 or len(left_df) == 0:
            continue
 
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
        ax.set_title(f"Gait {joint_name} ROM Asymmetry by Phase")
 
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
 
    bar_colors = comparison_df["Status"].map({
        "Normal": "royalblue",
        "Abnormal": "red"
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
    ax.set_title("Gait Healthy ROM Comparison")
 
    plt.xticks(rotation=45, color="white")
 
    st.pyplot(fig)
 
# =========================
# Clinical Report
# =========================
with tab4:
 
    findings = []
 
    if cadence < 90:
 
        findings.append("Cadence below normal range.")
 
    if cadence > 130:
 
        findings.append("Cadence above normal range.")
 
    if max(rom_difference) > 15:
        findings.append(
            "Gait joint ROM asymmetry exceeds 15%."
        )
 
    for _, row in comparison_df.iterrows():
 
        if row["Status"] == "Abnormal":
 
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
 
    st.subheader("Gait Raw Data")
 
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
        "gait_raw_data.csv",
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
        "gait_raw_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
# =========================
# Dashboard
# =========================
with tab6:
 
    st.title("Gait Dashboard")
 
    st.caption(t("gait.dashboard_caption"))
 
    # =========================
    # Gait Feature Calculation
    # =========================
 
    pelvis_stability = df_phase["pelvis_list"].std()
 
    rotation_variability = df_phase["pelvis_rotation"].std()
 
    # Combined (max of L/R) ROM values — kept for the Movement Score
    # tab and the Gait Features table on tab7, which report a single
    # figure per joint rather than a left/right breakdown.
    max_hip_rom = max(
        df_phase["hip_flexion_r"].max() - df_phase["hip_flexion_r"].min(),
        df_phase["hip_flexion_l"].max() - df_phase["hip_flexion_l"].min()
    )
 
    max_knee_rom = max(
        df_phase["knee_angle_r"].max() - df_phase["knee_angle_r"].min(),
        df_phase["knee_angle_l"].max() - df_phase["knee_angle_l"].min()
    )
 
    max_ankle_rom = max(
        (
            df_phase["ankle_angle_r"].quantile(0.95)
            -
            df_phase["ankle_angle_r"].quantile(0.05)
        ),
        (
            df_phase["ankle_angle_l"].quantile(0.95)
            -
            df_phase["ankle_angle_l"].quantile(0.05)
        )
    )
 
    # Left/Right ROM values shown individually on the Dashboard Key
    # Metrics below.
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
 
    ankle_rom_r = (
        df_phase["ankle_angle_r"].quantile(0.95)
        -
        df_phase["ankle_angle_r"].quantile(0.05)
    )
 
    ankle_rom_l = (
        df_phase["ankle_angle_l"].quantile(0.95)
        -
        df_phase["ankle_angle_l"].quantile(0.05)
    )
 
    pelvis_tilt_rom = (
        df_phase["pelvis_tilt"].max()
        -
        df_phase["pelvis_tilt"].min()
    )
 
    pelvis_rotation_rom = (
        df_phase["pelvis_rotation"].max()
        -
        df_phase["pelvis_rotation"].min()
    )
 
    pelvic_obliquity_rom = (
        df_phase["pelvis_list"].max()
        -
        df_phase["pelvis_list"].min()
    )
 
    pelvis_ml_sd = df_phase["pelvis_tx"].std()
 
    lumbar_extension_rom = (
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min()
    )
 
    st.subheader("Key Metrics")
 
    with st.expander(t("common.metrics_expander_label")):
 
        st.markdown(t("gait.metrics_table"))
 
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5, row1_col6, row1_col7 = st.columns(7)
 
    row1_col1.metric("Cadence", f"{cadence:.1f}")
 
    row1_col2.metric("Hip ROM (R)", f"{hip_rom_r:.1f}°")
 
    row1_col3.metric("Hip ROM (L)", f"{hip_rom_l:.1f}°")
 
    row1_col4.metric("Knee ROM (R)", f"{knee_rom_r:.1f}°")
 
    row1_col5.metric("Knee ROM (L)", f"{knee_rom_l:.1f}°")
 
    row1_col6.metric("Ankle ROM (R)", f"{ankle_rom_r:.1f}°")
 
    row1_col7.metric("Ankle ROM (L)", f"{ankle_rom_l:.1f}°")
 
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
 
    row2_col1.metric("Pelvic Tilt", f"{pelvis_tilt_rom:.1f}°")
 
    row2_col2.metric("Pelvic Rotation", f"{pelvis_rotation_rom:.1f}°")
 
    row2_col3.metric("Pelvic Obliquity", f"{pelvic_obliquity_rom:.1f}°")
 
    row2_col4.metric("Lumbar Extension", f"{lumbar_extension_rom:.1f}°")
 
    # =========================
    # Interactive Motion Viewer
    # =========================
 
    st.subheader("Interactive Motion Viewer")
 
    st.caption(t("gait.checkbox_instruction"))
 
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
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
 
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
 
        ax.set_title("Gait Joint Motion")
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
 
        ax2.set_title("Gait Pelvic Motion")
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
 
        st.markdown(t("gait.joint_rom_content"))
 
    rom_joints = {
 
        "Hip": "hip_flexion_r",
        "Knee": "knee_angle_r",
        "Ankle": "ankle_angle_r"
 
    }
 
    rom_values = []
 
    for variable in rom_joints.values():
 
        rom = (
            df_phase[variable].quantile(0.95)
            -
            df_phase[variable].quantile(0.05)
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
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color="white"
        )
 
    ax.set_ylabel("ROM (deg)")
    ax.set_title("Gait Joint ROM")
 
    st.pyplot(fig)
 
    # =========================
    # Joint Asymmetry
    # =========================
 
    st.subheader("Joint Asymmetry")
 
    with st.expander(t("common.joint_asymmetry_expander_label")):
 
        st.markdown(t("gait.joint_asymmetry_content"))
 
    gait_asymmetry_joints = {
 
        "Hip": ("hip_flexion_r", "hip_flexion_l"),
        "Knee": ("knee_angle_r", "knee_angle_l"),
        "Ankle": ("ankle_angle_r", "ankle_angle_l")
 
    }
 
    gait_asymmetry_results = {}
 
    for joint_name, (right_var, left_var) in gait_asymmetry_joints.items():
 
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
 
        gait_asymmetry_results[joint_name] = round(asymmetry, 2)
 
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
        list(gait_asymmetry_results.keys()),
        list(gait_asymmetry_results.values())
    )
 
    for bar, value in zip(bars, gait_asymmetry_results.values()):
 
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
    ax.set_title("Gait Joint Asymmetry")
 
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
    # Gait Features
    # =========================
 
    st.subheader("Gait Features")
 
    st.caption(t("gait.feature_caption"))
 
    with st.expander(t("common.feature_expander_label")):
 
        st.markdown(t("gait.feature_table"))
 
    gait_feature_df = pd.DataFrame({
 
        "Feature": [
 
            "Cadence",
            "Step Time",
            "Pelvic Stability",
            "Pelvic Rotation Variability",
            "Hip ROM",
            "Knee ROM",
            "Ankle ROM",
            "Lumbar Extension ROM"
 
        ],
 
        "Value": [
 
            round(cadence, 2),
 
            round(step_time, 2) if not np.isnan(step_time) else "N/A",
 
            round(pelvis_stability, 2),
 
            round(rotation_variability, 2),
 
            round(max_hip_rom, 2),
 
            round(max_knee_rom, 2),
 
            round(max_ankle_rom, 2),
 
            round(lumbar_extension_rom, 2)
 
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
                        gait_feature_df["Feature"],
                        gait_feature_df["Value"]
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
 
        height=250,
 
        margin=dict(l=10, r=10, t=10, b=10),
 
        paper_bgcolor="black",
 
        plot_bgcolor="black"
 
    )
 
    st.plotly_chart(fig, use_container_width=True)
 
    # =========================
    # Gait Score
    # =========================
 
    mean_asymmetry = np.mean(list(gait_asymmetry_results.values()))
 
    symmetry_score = max(0, 100 - mean_asymmetry)
 
    cadence_score = max(0, 100 - abs(cadence - 110))
 
    pelvic_ml_score = max(0, 100 - pelvis_stability * 10)
 
    lumbar_extension_score = max(
        0,
        100 - abs(lumbar_extension_rom - 10) * 5
    )
 
    mobility_score = min(
        100,
        (max_hip_rom + max_knee_rom + max_ankle_rom) / 2
    )
 
    overall_score = round(
 
        (
            symmetry_score * 0.25
            +
            cadence_score * 0.15
            +
            pelvic_ml_score * 0.20
            +
            lumbar_extension_score * 0.15
            +
            mobility_score * 0.25
        ),
 
        1
 
    )
 
    st.subheader("Movement Score")
 
    st.caption(t("gait.score_caption"))
 
    with st.expander(t("common.score_expander_label")):
 
        st.markdown(t("gait.score_content"))
 
    st.metric(
        "Overall Score",
        f"{overall_score}/100"
    )

squat.py/sit_stand.py/arm_flexion.pyと同じ考え方でGaitにも対応させました。ただし歩行周期にはSquat/Sit-Standのような「静止保持局面」が存在しないため、自動コメントの所見はGaitの周期特性に合わせて次のように調整しています。

各フェーズ（Heel Strike/Mid Stance/Toe Off/Swing）でのStd（標準偏差）が大きい関節・フェーズ → ストライド間のばらつき（動作の一貫性低下）を示唆
各関節（Hip/Knee/Ankle）でROMが最大となるフェーズ → 主要な可動局面を記述
Mid Stance（立脚中期）とSwing（遊脚期）のROM差が15%を超える場合 → 立脚時の制御と遊脚時の下肢前方移動の間で動作パターンに差があることを示唆

Phase Statistics表はHip(R/L)、Knee(R/L)、Ankle(R/L)、Pelvic Tilt、Pelvic Rotation、Pelvic Obliquity、Lumbar Extensionを対象にしています（Pelvic Obliquityは既存のKey Metrics/自動コメントで既に扱われているため含めています）。tab8全コードです。

python
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
            "caption": "Gait Analysisの主要な結果をまとめた統合PDFレポートを生成します。",
            "subject_name": "対象者名",
            "exam_date": "測定日",
            "examiner": "検者",
            "comment_heading": "総合評価 (Clinical Impression)",
            "comment_label": "検者による総合所見・コメントを記入してください（PDFに反映されます）",
            "auto_generate_button": "🪄 コメントを自動生成（下書き）",
            "auto_generate_caption": "計測データに基づいて下書きコメントを自動生成します。内容は必ず確認・編集の上でご使用ください。",
            "generate_button": "📄 PDFレポートを生成",
            "download_label": "📥 PDFレポートをダウンロード",
            "success_message": "PDFレポートを生成しました。上のボタンからダウンロードしてください。"
        },
        "en": {
            "header": "PDF Report",
            "caption": "Generate a comprehensive PDF report summarizing the key Gait Analysis results.",
            "subject_name": "Subject Name",
            "exam_date": "Exam Date",
            "examiner": "Examiner",
            "comment_heading": "Clinical Impression",
            "comment_label": "Enter the examiner's overall clinical impression / comments (included in the PDF)",
            "auto_generate_button": "🪄 Auto-generate Comment (Draft)",
            "auto_generate_caption": "Automatically generates a draft comment based on the measured data. Please review and edit before use.",
            "generate_button": "📄 Generate PDF Report",
            "download_label": "📥 Download PDF Report",
            "success_message": "PDF report generated. Use the button above to download it."
        }
    }
    UL = UI_LABELS[lang_code]
    def generate_gait_auto_comment(
        lang_code, overall_score, gait_asymmetry_results, comparison_df,
        cadence, lumbar_extension_rom, pelvis_tilt_rom, pelvis_rotation_rom,
        pelvic_obliquity_rom, phase_summary_df, phase_order
    ):
        lines = []

        # ---- Phase Statistics由来の所見 ----
        phase_stats_variables = {
            "Hip (R)": "hip_flexion_r",
            "Hip (L)": "hip_flexion_l",
            "Knee (R)": "knee_angle_r",
            "Knee (L)": "knee_angle_l",
            "Ankle (R)": "ankle_angle_r",
            "Ankle (L)": "ankle_angle_l",
            "Pelvic Tilt": "pelvis_tilt",
            "Pelvic Rotation": "pelvis_rotation",
            "Pelvic Obliquity": "pelvis_list",
            "Lumbar Extension": "lumbar_extension",
        }
        joint_pairs = {
            "Hip": ("hip_flexion_r", "hip_flexion_l"),
            "Knee": ("knee_angle_r", "knee_angle_l"),
            "Ankle": ("ankle_angle_r", "ankle_angle_l"),
        }

        # 1) 各フェーズでのStd（標準偏差）が大きい = ストライド間のばらつきが大きい
        # 歩行にはSquat/Sit-Standのような「静止保持局面」が無いため、
        # 全フェーズを対象にばらつき（動作の再現性）として評価する。
        # ※ しきい値2.0°は仮の基準です。臨床基準に合わせて調整してください。
        STD_THRESHOLD = 2.0
        variability_flags = []
        for stat_label, variable in phase_stats_variables.items():
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            for phase in phase_order:
                std_v = var_row[f"{phase}_Std"].iloc[0]
                if pd.notna(std_v) and std_v > STD_THRESHOLD:
                    variability_flags.append((stat_label, phase, std_v))

        # 2) 各関節でROMが最大となるフェーズ（主要な可動局面）
        dominant_phase_per_joint = {}
        for joint_name, (right_var, left_var) in joint_pairs.items():
            right_row = phase_summary_df[phase_summary_df["Variable"] == right_var]
            left_row = phase_summary_df[phase_summary_df["Variable"] == left_var]
            if len(right_row) == 0 or len(left_row) == 0:
                continue
            avg_rom_by_phase = {}
            for phase in phase_order:
                r_val = right_row[f"{phase}_ROM"].iloc[0]
                l_val = left_row[f"{phase}_ROM"].iloc[0]
                avg_rom_by_phase[phase] = np.nanmean([r_val, l_val])
            dominant_phase_per_joint[joint_name] = max(avg_rom_by_phase, key=avg_rom_by_phase.get)

        # 3) Mid Stance（立脚中期）と Swing（遊脚期）のROM差
        # 立脚時の制御と遊脚時の下肢前方移動という、歩行周期における
        # 代表的な2局面を比較する（Squat/Sit-StandのDescending/Ascendingに相当）。
        # ※ しきい値15%は左右対称性と同じ基準を仮採用しています。
        stance_swing_flags = []
        for joint_name, (right_var, left_var) in joint_pairs.items():
            right_row = phase_summary_df[phase_summary_df["Variable"] == right_var]
            left_row = phase_summary_df[phase_summary_df["Variable"] == left_var]
            if len(right_row) == 0 or len(left_row) == 0:
                continue
            stance_avg = np.nanmean([
                right_row["Mid Stance_ROM"].iloc[0],
                left_row["Mid Stance_ROM"].iloc[0]
            ])
            swing_avg = np.nanmean([
                right_row["Swing_ROM"].iloc[0],
                left_row["Swing_ROM"].iloc[0]
            ])
            if max(stance_avg, swing_avg) == 0:
                continue
            diff_pct = abs(stance_avg - swing_avg) / max(stance_avg, swing_avg) * 100
            if diff_pct > 15:
                stance_swing_flags.append((joint_name, stance_avg, swing_avg, diff_pct))

        if lang_code == "ja":
            if overall_score >= 80:
                lines.append(f"総合スコアは{overall_score}点であり、良好な歩行パターンを示している。")
            elif overall_score >= 60:
                lines.append(f"総合スコアは{overall_score}点であり、軽度から中等度の歩行パターンの逸脱が見られる。")
            else:
                lines.append(f"総合スコアは{overall_score}点であり、歩行パターンに明らかな逸脱が見られる。")
            if cadence < 90:
                lines.append(f"ケイデンスは{cadence:.1f}歩/分であり、正常範囲を下回っている。")
            elif cadence > 130:
                lines.append(f"ケイデンスは{cadence:.1f}歩/分であり、正常範囲を上回っている。")
            for joint_name, asym_value in gait_asymmetry_results.items():
                if asym_value > 15:
                    lines.append(f"{joint_name}関節のROM非対称性が{asym_value:.1f}%と、15%の基準値を超えている。")
            abnormal_vars = comparison_df[comparison_df["Status"] == "Abnormal"]["Variable"].tolist()
            if len(abnormal_vars) > 0:
                lines.append("健常者ROMとの比較において、" + "、".join(str(v) for v in abnormal_vars) + "が基準範囲外であった。")
            compensation_notes = []
            if lumbar_extension_rom > 10:
                compensation_notes.append(f"腰椎伸展（{lumbar_extension_rom:.1f}°）")
            if pelvis_tilt_rom > 10:
                compensation_notes.append(f"骨盤前後傾（{pelvis_tilt_rom:.1f}°）")
            if pelvis_rotation_rom > 10:
                compensation_notes.append(f"骨盤回旋（{pelvis_rotation_rom:.1f}°）")
            if pelvic_obliquity_rom > 10:
                compensation_notes.append(f"骨盤側方傾斜（{pelvic_obliquity_rom:.1f}°）")
            if compensation_notes:
                lines.append("、".join(compensation_notes) + "の可動域が大きく、代償動作の可能性がある。")
            if variability_flags:
                var_text = "、".join(f"{phase}フェーズの{label}" for label, phase, std_v in variability_flags[:6])
                lines.append(
                    f"フェーズ別のばらつきについては、{var_text}で標準偏差が大きく、"
                    "ストライド間での動作の再現性（一貫性）低下が疑われる。"
                )
            else:
                lines.append("フェーズ別のばらつきについては、標準偏差の観点から顕著な逸脱は見られなかった。")
            if dominant_phase_per_joint:
                dominant_text = "、".join(f"{joint}は{phase}フェーズ" for joint, phase in dominant_phase_per_joint.items())
                lines.append(f"各関節の主要な可動局面は、{dominant_text}で最大のROMを示している。")
            if stance_swing_flags:
                ss_text = "、".join(
                    f"{joint}（立脚中期 {stance:.1f}° ／ 遊脚期 {swing:.1f}°、差 {diff:.1f}%）"
                    for joint, stance, swing, diff in stance_swing_flags
                )
                lines.append(
                    f"立脚中期（Mid Stance）と遊脚期（Swing）の可動域を比較すると、{ss_text}で15%を超える差が見られ、"
                    "立脚時の制御と遊脚時の下肢前方移動の間で動作パターンに違いがある可能性がある。"
                )
            else:
                lines.append("立脚中期と遊脚期のROMを比較すると、いずれの関節も15%以内の差に収まっており、局面間で大きな制御の差は見られなかった。")
            if len(lines) <= 1:
                lines.append("その他、顕著な代償動作や左右差は認められなかった。")
            lines.append("")
            lines.append("※本コメントは計測データに基づく自動生成の下書きです。実際の触診・観察所見と照らし合わせた上で、検者が内容を確認・修正してください。")
        else:
            if overall_score >= 80:
                lines.append(f"The overall score is {overall_score}, indicating a favorable gait pattern.")
            elif overall_score >= 60:
                lines.append(f"The overall score is {overall_score}, indicating mild to moderate deviation from a typical gait pattern.")
            else:
                lines.append(f"The overall score is {overall_score}, indicating a clear deviation from a typical gait pattern.")
            if cadence < 90:
                lines.append(f"Cadence is {cadence:.1f} steps/min, below the normal range.")
            elif cadence > 130:
                lines.append(f"Cadence is {cadence:.1f} steps/min, above the normal range.")
            for joint_name, asym_value in gait_asymmetry_results.items():
                if asym_value > 15:
                    lines.append(f"{joint_name} ROM asymmetry is {asym_value:.1f}%, exceeding the 15% threshold.")
            abnormal_vars = comparison_df[comparison_df["Status"] == "Abnormal"]["Variable"].tolist()
            if len(abnormal_vars) > 0:
                lines.append("Compared to healthy reference ROM, the following variables were outside the normal range: " + ", ".join(str(v) for v in abnormal_vars) + ".")
            compensation_notes = []
            if lumbar_extension_rom > 10:
                compensation_notes.append(f"lumbar extension ({lumbar_extension_rom:.1f}°)")
            if pelvis_tilt_rom > 10:
                compensation_notes.append(f"pelvic tilt ({pelvis_tilt_rom:.1f}°)")
            if pelvis_rotation_rom > 10:
                compensation_notes.append(f"pelvic rotation ({pelvis_rotation_rom:.1f}°)")
            if pelvic_obliquity_rom > 10:
                compensation_notes.append(f"pelvic obliquity ({pelvic_obliquity_rom:.1f}°)")
            if compensation_notes:
                lines.append("Increased range noted in " + ", ".join(compensation_notes) + ", suggesting possible compensatory movement.")
            if variability_flags:
                var_text = ", ".join(f"{label} during {phase}" for label, phase, std_v in variability_flags[:6])
                lines.append(
                    f"Regarding phase-by-phase variability, elevated standard deviation was observed for {var_text}, "
                    "suggesting possible reduced stride-to-stride consistency."
                )
            else:
                lines.append("Regarding phase-by-phase variability, no notable deviation was observed based on standard deviation.")
            if dominant_phase_per_joint:
                dominant_text = ", ".join(f"{joint} peaks during {phase}" for joint, phase in dominant_phase_per_joint.items())
                lines.append(f"The dominant phase of motion for each joint was as follows: {dominant_text}.")
            if stance_swing_flags:
                ss_text = ", ".join(
                    f"{joint} (Mid Stance {stance:.1f}° / Swing {swing:.1f}°, diff {diff:.1f}%)"
                    for joint, stance, swing, diff in stance_swing_flags
                )
                lines.append(
                    f"Comparing Mid Stance and Swing, a difference exceeding 15% was observed for {ss_text}, "
                    "suggesting a possible difference in motor control between stance-phase stability and swing-phase limb advancement."
                )
            else:
                lines.append("Comparing Mid Stance and Swing, all joints remained within a 15% difference, with no major difference in control between phases.")
            if len(lines) <= 1:
                lines.append("No other significant compensations or asymmetries were noted.")
            lines.append("")
            lines.append("Note: This comment is an automatically generated draft based on measured data. Please review and revise it against actual palpation and observational findings before finalizing.")
        return "\n".join(lines)
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
    if "gait_clinical_comment" not in st.session_state:
        st.session_state["gait_clinical_comment"] = ""
    if st.button(UL["auto_generate_button"]):
        st.session_state["gait_clinical_comment"] = generate_gait_auto_comment(
            lang_code, overall_score, gait_asymmetry_results, comparison_df,
            cadence, lumbar_extension_rom, pelvis_tilt_rom, pelvis_rotation_rom,
            pelvic_obliquity_rom, phase_summary_df, phase_order
        )
    st.caption(UL["auto_generate_caption"])
    clinical_comment = st.text_area(
        UL["comment_label"],
        key="gait_clinical_comment",
        height=150
    )
    if st.button(UL["generate_button"]):
        LABELS = {
            "ja": {
                "title": "歩行分析 臨床レポート",
                "subject_line": "対象者名: {name} ｜ 測定日: {date} ｜ 検者: {examiner}",
                "cadence_line_full": "ケイデンス: {cadence:.1f} 歩/分 ／ ステップ時間: {step_time:.2f} 秒",
                "cadence_line_na": "ケイデンス: {cadence:.1f} 歩/分 ／ ステップ時間: データなし",
                "phase_detection_heading": "歩行相検出",
                "key_metrics_heading": "主要指標",
                "metric_col": "指標",
                "right_col": "右",
                "left_col": "左",
                "hip_rom_row": "股関節可動域 (°)",
                "knee_rom_row": "膝関節可動域 (°)",
                "ankle_rom_row": "足関節可動域 (°)",
                "pelvis_line": "骨盤前後傾: {tilt:.1f}°　骨盤回旋: {rot:.1f}°　骨盤側方傾斜: {obl:.1f}°　腰椎伸展: {lum:.1f}°",
                "joint_rom_summary_heading": "関節可動域サマリー（試技全体）",
                "joint_col": "関節",
                "right_rom_col": "右 ROM (°)",
                "left_rom_col": "左 ROM (°)",
                "phase_stats_heading": "フェーズ別統計（Min/Max/Mean/Std/ROM）",
                "phase_label": "フェーズ",
                "ps_variable_col": "項目",
                "ps_min_col": "最小",
                "ps_max_col": "最大",
                "ps_mean_col": "平均",
                "ps_std_col": "標準偏差",
                "ps_rom_col": "ROM (°)",
                "symmetry_heading": "左右対称性分析",
                "max_avg_label": "{joint}　（最大: {max_val:.1f}% ／ 平均: {avg_val:.1f}%）",
                "phase_col": "相",
                "right_rom_col2": "右ROM",
                "left_rom_col2": "左ROM",
                "asym_col": "非対称性(%)",
                "healthy_rom_heading": "健常者ROMとの比較",
                "clinical_findings_heading": "臨床所見",
                "no_findings": "顕著な異常は検出されませんでした。",
                "cadence_low_finding": "ケイデンスが正常範囲を下回っています。",
                "cadence_high_finding": "ケイデンスが正常範囲を上回っています。",
                "asym_finding": "{joint}のROM非対称性が15%を超えています（{value:.1f}%）。",
                "range_finding": "{variable}のROMが健常範囲外です。",
                "clinical_impression_heading": "総合評価（Clinical Impression）",
                "no_comment": "(記入なし)",
                "movement_score_heading": "総合運動スコア",
                "overall_score_label": "総合スコア: {score} / 100",
                "feature_col": "項目",
                "value_col": "値"
            },
            "en": {
                "title": "Gait Analysis Clinical Report",
                "subject_line": "Subject: {name} | Exam Date: {date} | Examiner: {examiner}",
                "cadence_line_full": "Cadence: {cadence:.1f} steps/min / Step Time: {step_time:.2f} sec",
                "cadence_line_na": "Cadence: {cadence:.1f} steps/min / Step Time: N/A",
                "phase_detection_heading": "Gait Phase Detection",
                "key_metrics_heading": "Key Metrics",
                "metric_col": "Metric",
                "right_col": "Right",
                "left_col": "Left",
                "hip_rom_row": "Hip ROM (°)",
                "knee_rom_row": "Knee ROM (°)",
                "ankle_rom_row": "Ankle ROM (°)",
                "pelvis_line": "Pelvic Tilt: {tilt:.1f}°  Pelvic Rotation: {rot:.1f}°  Pelvic Obliquity: {obl:.1f}°  Lumbar Extension: {lum:.1f}°",
                "joint_rom_summary_heading": "Joint ROM Summary (Whole Trial)",
                "joint_col": "Joint",
                "right_rom_col": "Right ROM (°)",
                "left_rom_col": "Left ROM (°)",
                "phase_stats_heading": "Phase Statistics (Min/Max/Mean/Std/ROM)",
                "phase_label": "Phase",
                "ps_variable_col": "Variable",
                "ps_min_col": "Min",
                "ps_max_col": "Max",
                "ps_mean_col": "Mean",
                "ps_std_col": "Std",
                "ps_rom_col": "ROM (deg)",
                "symmetry_heading": "Symmetry Analysis",
                "max_avg_label": "{joint}  (Max: {max_val:.1f}% / Avg: {avg_val:.1f}%)",
                "phase_col": "Phase",
                "right_rom_col2": "Right_ROM",
                "left_rom_col2": "Left_ROM",
                "asym_col": "Asymmetry_%",
                "healthy_rom_heading": "Healthy ROM Comparison",
                "clinical_findings_heading": "Clinical Findings",
                "no_findings": "No major abnormalities detected.",
                "cadence_low_finding": "Cadence below normal range.",
                "cadence_high_finding": "Cadence above normal range.",
                "asym_finding": "{joint} ROM asymmetry exceeds 15% ({value:.1f}%).",
                "range_finding": "{variable} ROM outside healthy range.",
                "clinical_impression_heading": "Clinical Impression",
                "no_comment": "(No comment entered)",
                "movement_score_heading": "Movement Score",
                "overall_score_label": "OVERALL SCORE: {score} / 100",
                "feature_col": "Feature",
                "value_col": "Value"
            }
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
            spaceAfter=10,
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
            "Heel Strike": "blue",
            "Mid Stance": "orange",
            "Toe Off": "limegreen",
            "Swing": "red"
        }
        elements = []
        # ---- Title / Subject Info ----
        elements.append(Paragraph(LB["title"], title_style))
        elements.append(Spacer(1, 6))
        info_text = LB["subject_line"].format(
            name=subject_name or "-",
            date=exam_date or "-",
            examiner=examiner_name or "-"
        )
        elements.append(Paragraph(info_text, normal_style))
        if not np.isnan(step_time):
            cadence_text = LB["cadence_line_full"].format(
                cadence=cadence, step_time=step_time
            )
        else:
            cadence_text = LB["cadence_line_na"].format(cadence=cadence)
        elements.append(Paragraph(cadence_text, normal_style))
        elements.append(Spacer(1, 12))
        # ---- Gait Phase Detection Plot ----
        elements.append(Paragraph(LB["phase_detection_heading"], heading_style))
        pdf_phase_fig, pdf_phase_ax = plt.subplots(figsize=(10, 4))
        pdf_phase_ax.plot(
            signal_smooth_r, color="white", linewidth=1, alpha=0.4, label="Right Ankle"
        )
        pdf_phase_ax.plot(
            signal_smooth_l, color="cyan", linewidth=1, alpha=0.4, linestyle="--", label="Left Ankle"
        )
        for phase in phase_order:
            idx_r = df_phase_r["Phase"] == phase
            pdf_phase_ax.scatter(
                df_phase_r.index[idx_r], signal_smooth_r[idx_r],
                c=colors_phase_pdf[phase], marker="o", s=8, label=f"R {phase}"
            )
        for phase in phase_order:
            idx_l = df_phase_l["Phase"] == phase
            pdf_phase_ax.scatter(
                df_phase_l.index[idx_l], signal_smooth_l[idx_l],
                c=colors_phase_pdf[phase], marker="^", s=8, label=f"L {phase}"
            )
        pdf_phase_ax.set_title("Gait Phase Detection Plot")
        pdf_phase_ax.set_xlabel("Frame")
        pdf_phase_ax.set_ylabel("Ankle Angle (deg)")
        pdf_phase_ax.legend(fontsize=6, ncol=2)
        pdf_phase_ax.grid(alpha=0.3)
        elements.append(fig_to_rl_image(pdf_phase_fig, width_cm=16))
        elements.append(Spacer(1, 12))
        # ---- Key Metrics ----
        elements.append(Paragraph(LB["key_metrics_heading"], heading_style))
        key_metrics_data = [
            [LB["metric_col"], LB["right_col"], LB["left_col"]],
            [LB["hip_rom_row"], f"{hip_rom_r:.1f}", f"{hip_rom_l:.1f}"],
            [LB["knee_rom_row"], f"{knee_rom_r:.1f}", f"{knee_rom_l:.1f}"],
            [LB["ankle_rom_row"], f"{ankle_rom_r:.1f}", f"{ankle_rom_l:.1f}"],
        ]
        key_metrics_table = Table(key_metrics_data, hAlign="LEFT")
        key_metrics_table.setStyle(TABLE_STYLE)
        elements.append(key_metrics_table)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            LB["pelvis_line"].format(
                tilt=pelvis_tilt_rom,
                rot=pelvis_rotation_rom,
                obl=pelvic_obliquity_rom,
                lum=lumbar_extension_rom
            ),
            normal_style
        ))
        elements.append(Spacer(1, 12))
        # ---- Joint ROM Summary (Whole Trial) ----
        elements.append(Paragraph(LB["joint_rom_summary_heading"], heading_style))
        rom_joints_pdf = {
            "Hip": ("hip_flexion_r", "hip_flexion_l"),
            "Knee": ("knee_angle_r", "knee_angle_l"),
            "Ankle": ("ankle_angle_r", "ankle_angle_l"),
        }
        rom_summary_rows_pdf = [[LB["joint_col"], LB["right_rom_col"], LB["left_rom_col"]]]
        rom_r_values_pdf = []
        rom_l_values_pdf = []
        joint_names_pdf = []
        for joint_name, (right_var, left_var) in rom_joints_pdf.items():
            rom_r_whole = df_phase[right_var].max() - df_phase[right_var].min()
            rom_l_whole = df_phase[left_var].max() - df_phase[left_var].min()
            rom_r_values_pdf.append(rom_r_whole)
            rom_l_values_pdf.append(rom_l_whole)
            joint_names_pdf.append(joint_name)
            rom_summary_rows_pdf.append(
                [joint_name, f"{rom_r_whole:.1f}", f"{rom_l_whole:.1f}"]
            )
        rom_summary_table = Table(rom_summary_rows_pdf, hAlign="LEFT")
        rom_summary_table.setStyle(TABLE_STYLE)
        elements.append(rom_summary_table)
        elements.append(Spacer(1, 8))
        rom_summary_fig, rom_summary_ax = plt.subplots(figsize=(8, 4))
        x_pos = np.arange(len(joint_names_pdf))
        bar_width = 0.35
        rom_summary_ax.bar(
            x_pos - bar_width / 2, rom_r_values_pdf, bar_width,
            label="Right", color="royalblue"
        )
        rom_summary_ax.bar(
            x_pos + bar_width / 2, rom_l_values_pdf, bar_width,
            label="Left", color="orange"
        )
        rom_summary_ax.set_xticks(x_pos)
        rom_summary_ax.set_xticklabels(joint_names_pdf)
        rom_summary_ax.set_ylabel("ROM (deg)")
        rom_summary_ax.set_title("Joint ROM Summary (Whole Trial)")
        rom_summary_ax.legend(fontsize=8)
        rom_summary_ax.grid(alpha=0.3, axis="y")
        elements.append(fig_to_rl_image(rom_summary_fig, width_cm=12))
        elements.append(Spacer(1, 12))
        # ---- Phase Statistics (Min/Max/Mean/Std/ROM per Phase) ----
        elements.append(Paragraph(LB["phase_stats_heading"], heading_style))
        phase_stats_variables = {
            "Hip (R)": "hip_flexion_r",
            "Hip (L)": "hip_flexion_l",
            "Knee (R)": "knee_angle_r",
            "Knee (L)": "knee_angle_l",
            "Ankle (R)": "ankle_angle_r",
            "Ankle (L)": "ankle_angle_l",
            "Pelvic Tilt": "pelvis_tilt",
            "Pelvic Rotation": "pelvis_rotation",
            "Pelvic Obliquity": "pelvis_list",
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
            elements.append(Spacer(1, 8))
        elements.append(Spacer(1, 4))
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
            rows = [[LB["phase_col"], LB["right_rom_col2"], LB["left_rom_col2"], LB["asym_col"]]]
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
                    joint=joint_name,
                    max_val=max(asym_values),
                    avg_val=np.mean(asym_values)
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
            sym_ax.set_title(f"Gait {joint_name} ROM Asymmetry by Phase")
            sym_ax.legend(fontsize=8)
            sym_ax.grid(alpha=0.3, axis="y")
            plt.setp(sym_ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
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
            "red" if row["Status"] == "Abnormal" else "royalblue"
            for _, row in comparison_df.iterrows()
        ]
        hrom_ax.bar(
            comparison_df["Variable"], comparison_df["ROM_Difference_%"], color=bar_colors_pdf
        )
        hrom_ax.axhline(0, color="black", linestyle="--", linewidth=1)
        hrom_ax.set_ylabel("Difference (%)")
        hrom_ax.set_title("Gait Healthy ROM Comparison")
        plt.setp(hrom_ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        hrom_ax.grid(alpha=0.3, axis="y")
        elements.append(fig_to_rl_image(hrom_fig, width_cm=16))
        elements.append(Spacer(1, 12))
        # ---- Clinical Findings ----
        elements.append(Paragraph(LB["clinical_findings_heading"], heading_style))
        pdf_findings = []
        if cadence < 90:
            pdf_findings.append(LB["cadence_low_finding"])
        if cadence > 130:
            pdf_findings.append(LB["cadence_high_finding"])
        for joint_name in symmetry_joints_pdf.keys():
            asym_value = gait_asymmetry_results.get(joint_name, 0)
            if asym_value > 15:
                pdf_findings.append(
                    LB["asym_finding"].format(joint=joint_name, value=asym_value)
                )
        for _, row in comparison_df.iterrows():
            if row["Status"] == "Abnormal":
                pdf_findings.append(LB["range_finding"].format(variable=row["Variable"]))
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
        # ---- Movement Score（ハイライト表示・コンパクト版） ----
        elements.append(Paragraph(LB["movement_score_heading"], heading_style))
        elements.append(Paragraph(
            LB["overall_score_label"].format(score=overall_score),
            score_style
        ))
        feature_rows = [[LB["feature_col"], LB["value_col"]]]
        for _, row in gait_feature_df.iterrows():
            feature_rows.append([str(row["Feature"]), str(row["Value"])])
        feature_table = Table(feature_rows, hAlign="LEFT")
        feature_table.setStyle(TABLE_STYLE)
        elements.append(feature_table)
        doc.build(elements)
        st.download_button(
            UL["download_label"],
            data=report_buffer.getvalue(),
            file_name="Gait_Clinical_Report.pdf",
            mime="application/pdf"
        )
        st.success(UL["success_message"])
