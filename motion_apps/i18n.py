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
    "ankle_angle_l": {"min": -10, "max": 20}
 
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Gait Phases",
    "Movement Analysis",
    "Symmetry Analysis",
    "Clinical Report",
    "Raw Data",
    "Dashboard",
    "Movement Score"
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
        8,
        1,
        figsize=(12, 26)
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
 
    comparison_df["ROM_Difference_%"] = pd.to_numeric(
        comparison_df["ROM_Difference_%"],
        errors="coerce"
    )
 
    overall_deviation = round(
        comparison_df["ROM_Difference_%"].abs().mean(),
        1
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
 
    col1, col2, col3, col4, col5, col6 = st.columns(6)
 
    col1.metric("Cadence", f"{cadence:.1f}")
 
    col2.metric("Hip ROM", f"{max_hip_rom:.1f}°")
 
    col3.metric("Knee ROM", f"{max_knee_rom:.1f}°")
 
    col4.metric("Ankle ROM", f"{max_ankle_rom:.1f}°")
 
    col5.metric("Pelvic Tilt", f"{pelvis_tilt_rom:.1f}°")
 
    col6.metric("ROM Deviation", f"{overall_deviation:.1f}%")
 
    col7, col8, col9 = st.columns(3)
 
    col7.metric("Pelvic Rotation", f"{pelvis_rotation_rom:.1f}°")
 
    col8.metric("Pelvic Obliquity", f"{pelvic_obliquity_rom:.1f}°")
 
    col9.metric("Lumbar Extension", f"{lumbar_extension_rom:.1f}°")
 
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
 
 
