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
 
st.set_page_config(page_title="Arm Flexion Analysis", layout="wide")
 
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
 
st.title("Arm Flexion Analysis")
 
uploaded_file = st.session_state.get("uploaded_file")
 
if uploaded_file is None:
    st.warning(t("common.upload_warning"))
    st.stop()
 
# =========================
# Phase Detection
# =========================
 
df = pd.read_excel(uploaded_file, header=10)
 
if "arm_flex_l" in df.columns and "arm_flex_r" in df.columns:
    df["arm_flex_avg"] = (
        df["arm_flex_l"]
        +
        df["arm_flex_r"]
    ) / 2
 
arm_flexion = (
    df["arm_flex_l"]
    +
    df["arm_flex_r"]
) / 2
 
arm_smooth = (
    arm_flexion
    .rolling(window=5, center=True)
    .mean()
    .bfill()
    .ffill()
)
 
# =========================
# Velocity (deg/sec)
# OpenCap = 60 Hz
# =========================
 
fps = 60
 
arm_velocity = (
    arm_smooth
    .diff()
    .fillna(0)
    * fps
)
 
arm_min = arm_smooth.min()
arm_max = arm_smooth.max()
arm_range = arm_max - arm_min
 
start_threshold = arm_min + arm_range * 0.10
top_threshold = arm_max - arm_range * 0.10
 
velocity_threshold = max(
    abs(arm_velocity).max() * 0.15,
    10
)
 
phase_order = [
    "Start",
    "Raising",
    "Top",
    "Lowering"
]
 
display_phase_order = [
    "Start",
    "Raising",
    "Top",
    "Lowering"
]
 
phases = []
 
for p, v in zip(arm_smooth, arm_velocity):
 
    # 開始位置
    if p <= start_threshold:
        phase = "Start"
 
    # 挙上完了
    elif p >= top_threshold:
        phase = "Top"
 
    # 挙上中
    elif v > velocity_threshold:
        phase = "Raising"
 
    # 降下中
    elif v < -velocity_threshold:
        phase = "Lowering"
 
    # 停止している場合
    else:
        if p > (arm_min + arm_max) / 2:
            phase = "Top"
        else:
            phase = "Start"
 
    phases.append(phase)
 
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
    "arm_flex_r": {"min": 160.0, "max": 180.0},
    "arm_flex_l": {"min": 160.0, "max": 180.0},
    "lumbar_extension": {"min": 10.0, "max": 20.0},
    "pelvis_tilt": {"min": 5.0, "max": 10.0}
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Phase Analysis",
    "Movement Analysis",
    "Symmetry Analysis",
    "Clinical Report",
    "Raw Data",
    "Dashboard",
    "Movement Score"
])
 
# =========================
# Phase Analysis
# =========================
 
with tab1:
 
    st.subheader("Phase Detection Plot")
 
    st.caption(t("arm_flexion.phase_caption"))
 
    colors_phase = {
        "Start": "blue",
        "Raising": "limegreen",
        "Top": "orange",
        "Lowering": "red"
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
 
    ax.grid(
        True,
        color="white",
        alpha=0.3
    )
 
    ax.plot(
        df_phase.index,
        arm_flexion,
        color="white",
        linewidth=2,
        label="Mean Arm Flexion"
    )
 
    for phase in phase_order:
 
        idx = df_phase["Phase"] == phase
 
        ax.scatter(
            df_phase.index[idx],
            arm_flexion[idx],
            c=colors_phase[phase],
            s=10,
            label=phase
        )
 
    ax.set_title("Phase Detection Plot")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Shoulder Flexion Angle (deg)")
 
    legend = ax.legend()
 
    for text in legend.get_texts():
        text.set_color("white")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    st.pyplot(fig)
 
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
 
    start_df = phase_summary_df[
        [
            "Variable",
            "Start_Min",
            "Start_Max",
            "Start_Mean",
            "Start_Std",
            "Start_ROM"
        ]
    ]
 
    raising_df = phase_summary_df[
        [
            "Variable",
            "Raising_Min",
            "Raising_Max",
            "Raising_Mean",
            "Raising_Std",
            "Raising_ROM"
        ]
    ]
 
    top_df = phase_summary_df[
        [
            "Variable",
            "Top_Min",
            "Top_Max",
            "Top_Mean",
            "Top_Std",
            "Top_ROM"
        ]
    ]
 
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
 
        start_df.to_excel(
            writer,
            sheet_name="Start",
            index=False
        )
 
        raising_df.to_excel(
            writer,
            sheet_name="Raising",
            index=False
        )
 
        top_df.to_excel(
            writer,
            sheet_name="Top",
            index=False
        )
 
        lowering_df.to_excel(
            writer,
            sheet_name="Lowering",
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
 
            file_name="Arm_Flexion_Phase_Summary.xlsx",
 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
 
        )
 
    with col2:
 
        st.download_button(
 
            "📄 Download PDF",
 
            data=pdf_buffer.getvalue(),
 
            file_name="Arm_Flexion_Phase_Summary.pdf",
 
            mime="application/pdf"
 
        )
 
    # ==========================================================
    # Phase Tables
    # ==========================================================
 
    st.markdown("---")
 
    st.subheader("Phase Statistics")
 
    tab_start, tab_raising, tab_top, tab_lowering = st.tabs(
        [
            "Start",
            "Raising",
            "Top",
            "Lowering"
        ]
    )
 
    with tab_start:
 
        st.dataframe(
            start_df,
            use_container_width=True
        )
 
    with tab_raising:
 
        st.dataframe(
            raising_df,
            use_container_width=True
        )
 
    with tab_top:
 
        st.dataframe(
            top_df,
            use_container_width=True
        )
 
    with tab_lowering:
 
        st.dataframe(
            lowering_df,
            use_container_width=True
        )
 
# =========================
# Movement Analysis
# =========================
with tab2:
 
    st.subheader("Joint Time Series")
 
    st.caption(t("arm_flexion.movement_analysis_caption"))
 
    # OpenCap sampling rate (60 Hz)
    time = np.arange(len(df_phase)) / 60
 
    fig, ax = plt.subplots(
        7,
        1,
        figsize=(12, 24)
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
 
        a.grid(
            True,
            color="white",
            alpha=0.3
        )
 
    # =====================
    # Shoulder Flexion R/L
    # =====================
 
    ax[0].plot(
        time,
        df_phase["arm_flex_r"],
        label="Right Arm",
        linewidth=2
    )
 
    ax[0].plot(
        time,
        df_phase["arm_flex_l"],
        label="Left Arm",
        linewidth=2
    )
 
    ax[0].set_title("Shoulder Flexion")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Angle (deg)")
 
    legend = ax[0].legend(loc="upper right")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Average Shoulder Flexion
    # =====================
 
    arm_avg = (
        df_phase["arm_flex_r"]
        +
        df_phase["arm_flex_l"]
    ) / 2
 
    ax[1].plot(
        time,
        arm_avg,
        linewidth=2,
        color="cyan"
    )
 
    ax[1].set_title("Average Shoulder Flexion")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (deg)")
 
    # =====================
    # Lumbar Extension
    # =====================
 
    ax[2].plot(
        time,
        df_phase["lumbar_extension"],
        linewidth=2,
        color="magenta"
    )
 
    ax[2].set_title("Lumbar Extension")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Tilt
    # =====================
 
    ax[3].plot(
        time,
        df_phase["pelvis_tilt"],
        linewidth=2,
        color="orange"
    )
 
    ax[3].set_title("Pelvic Tilt")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Rotation
    # =====================
 
    ax[4].plot(
        time,
        df_phase["pelvis_rotation"],
        linewidth=2,
        color="lime"
    )
 
    ax[4].set_title("Pelvic Rotation")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Vertical
    # =====================
 
    ax[5].plot(
        time,
        df_phase["pelvis_ty"],
        linewidth=2,
        color="yellow"
    )
 
    ax[5].set_title("Pelvic Vertical Displacement")
    ax[5].set_xlabel("Time (s)")
    ax[5].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Anterior-Posterior
    # =====================
 
    ax[6].plot(
        time,
        df_phase["pelvis_tz"],
        linewidth=2,
        color="deepskyblue"
    )
 
    ax[6].set_title("Pelvic Anterior-Posterior Movement")
    ax[6].set_xlabel("Time (s)")
    ax[6].set_ylabel("Position (m)")
 
    plt.tight_layout()
 
    st.pyplot(fig)
 
# =========================
# Symmetry Analysis
# =========================
 
with tab3:
 
    st.subheader("Phase Symmetry")
 
    st.caption(t("arm_flexion.symmetry_caption"))
 
    joints = {
 
        "Shoulder": (
            "arm_flex_r",
            "arm_flex_l"
        )
 
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
 
            if pd.isna(right_rom) or pd.isna(left_rom):
 
                asymmetry = np.nan
 
            elif max(right_rom, left_rom) == 0:
 
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
 
                "Right_ROM": round(right_rom, 2) if not pd.isna(right_rom) else np.nan,
 
                "Left_ROM": round(left_rom, 2) if not pd.isna(left_rom) else np.nan,
 
                "Asymmetry_%": round(asymmetry, 2) if not pd.isna(asymmetry) else np.nan
 
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
 
        valid_asymmetry = [x for x in rom_difference if not pd.isna(x)]
 
        col1, col2 = st.columns(2)
 
        with col1:
 
            st.metric(
                "Maximum Asymmetry",
                f"{np.nanmax(valid_asymmetry):.1f}%" if valid_asymmetry else "N/A"
            )
 
        with col2:
 
            st.metric(
                "Average Asymmetry",
                f"{np.nanmean(valid_asymmetry):.1f}%" if valid_asymmetry else "N/A"
            )
 
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
 
        ax.axhline(
            15,
            linestyle="--",
            color="red",
            label="15% Threshold"
        )
 
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
 
    st.caption(t("arm_flexion.healthy_rom_caption"))
 
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
 
    valid_rom_difference = [
        x for x in rom_difference if not pd.isna(x)
    ]
 
    if valid_rom_difference and max(valid_rom_difference) > 15:
        findings.append(
            "Arm flexion shoulder ROM asymmetry exceeds 15%."
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
 
    st.subheader("Arm Flexion Raw Data")
 
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
        "arm_flexion_raw_data.csv",
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
        "arm_flexion_raw_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
# =========================
# Dashboard
# =========================
with tab6:
 
    st.title("Arm Flexion Dashboard")
 
    st.caption(t("arm_flexion.dashboard_caption"))
 
    # =========================
    # KPI
    # =========================
 
    max_arm_flexion = max(
        df_phase["arm_flex_r"].max(),
        df_phase["arm_flex_l"].max()
    )
 
    lumbar_compensation = round(
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min(),
        1
    )
 
    pelvis_compensation = round(
        df_phase["pelvis_tilt"].max()
        -
        df_phase["pelvis_tilt"].min(),
        1
    )
 
    comparison_df["ROM_Difference_%"] = pd.to_numeric(
        comparison_df["ROM_Difference_%"],
        errors="coerce"
    )
 
    overall_deviation = round(
        comparison_df["ROM_Difference_%"]
        .abs()
        .mean(),
        1
    )
 
    st.subheader("Key Metrics")
 
    with st.expander(t("common.metrics_expander_label")):
 
        st.markdown(t("arm_flexion.metrics_table"))
 
    col1, col2, col3, col4 = st.columns(4)
 
    col1.metric(
        "Max Shoulder Flexion",
        f"{max_arm_flexion:.1f}°"
    )
 
    col2.metric(
        "Lumbar Compensation",
        f"{lumbar_compensation:.1f}°"
    )
 
    col3.metric(
        "Pelvis Compensation",
        f"{pelvis_compensation:.1f}°"
    )
 
    col4.metric(
        "ROM Deviation",
        f"{overall_deviation:.1f}%"
    )
 
    # =========================
    # Interactive Motion Viewer
    # =========================
 
    st.subheader("Interactive Motion Viewer")
 
    st.caption(t("arm_flexion.checkbox_instruction"))
 
    left_col, right_col = st.columns([1.2, 4])
 
    # ======================================
    # Left Panel
    # ======================================
 
    with left_col:
 
        st.markdown("### Shoulder")
 
        st.markdown("#### Flexion")
 
        show_arm_r = st.checkbox("Right Shoulder", value=True)
 
        show_arm_l = st.checkbox("Left Shoulder", value=True)
 
        st.markdown("---")
 
        st.markdown("### Trunk")
 
        show_lumbar = st.checkbox("Lumbar Extension")
 
        st.markdown("---")
 
        st.markdown("### Pelvis")
 
        show_tilt = st.checkbox("Pelvic Tilt")
 
        show_obliquity = st.checkbox("Pelvic Obliquity")
 
        show_rotation = st.checkbox("Pelvic Rotation")
 
        show_ml = st.checkbox("Medial-Lateral Deviation")
 
        show_vertical = st.checkbox("Vertical Displacement")
 
        show_ap = st.checkbox("Anterior-Posterior Deviation")
 
    # ======================================
    # Right Panel
    # ======================================
 
    with right_col:
 
        time = np.arange(len(df_phase)) / 60
 
        # =========================
        # Shoulder Motion
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
 
        if show_arm_r:
 
            ax.plot(
                time,
                df_phase["arm_flex_r"],
                label="Right Shoulder",
                linewidth=2
            )
 
        if show_arm_l:
 
            ax.plot(
                time,
                df_phase["arm_flex_l"],
                label="Left Shoulder",
                linewidth=2
            )
 
        if show_lumbar:
 
            ax.plot(
                time,
                df_phase["lumbar_extension"],
                label="Lumbar Extension",
                linewidth=2
            )
 
        ax.set_title("Arm Flexion Motion")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
 
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
                label="Pelvic Tilt",
                linewidth=2
            )
 
        if show_obliquity:
 
            ax2.plot(
                time,
                df_phase["pelvis_list"],
                label="Pelvic Obliquity",
                linewidth=2
            )
 
        if show_rotation:
 
            ax2.plot(
                time,
                df_phase["pelvis_rotation"],
                label="Pelvic Rotation",
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
                label="Vertical Displacement (mm)",
                linewidth=2
            )
 
        if show_ap:
 
            ax2.plot(
                time,
                df_phase["pelvis_tz"] * 1000,
                label="Anterior-Posterior (mm)",
                linewidth=2
            )
 
        ax2.set_title("Pelvic Motion")
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
 
        st.markdown(t("arm_flexion.joint_rom_content"))
 
    rom_joints = {
 
        "Right Shoulder": "arm_flex_r",
 
        "Left Shoulder": "arm_flex_l",
 
        "Lumbar": "lumbar_extension",
 
        "Pelvis Tilt": "pelvis_tilt"
 
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
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
 
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
    ax.set_title("Arm Flexion ROM")
 
    plt.xticks(rotation=20, color="white")
 
    st.pyplot(fig)
 
    # =========================
    # Joint Asymmetry
    # =========================
 
    st.subheader("Joint Asymmetry")
 
    with st.expander(t("common.joint_asymmetry_expander_label")):
 
        st.markdown(t("arm_flexion.joint_asymmetry_content"))
 
    asymmetry_joints = {
 
        "Shoulder": (
            "arm_flex_r",
            "arm_flex_l"
        )
 
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
        figsize=(6, 4),
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
    ax.set_title("Shoulder Asymmetry")
 
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
 
    st.caption(t("arm_flexion.feature_caption"))
 
    with st.expander(t("common.feature_expander_label")):
 
        st.markdown(t("arm_flexion.feature_table"))
 
    right_rom = (
        df_phase["arm_flex_r"].max()
        -
        df_phase["arm_flex_r"].min()
    )
 
    left_rom = (
        df_phase["arm_flex_l"].max()
        -
        df_phase["arm_flex_l"].min()
    )
 
    shoulder_rom = (right_rom + left_rom) / 2
 
    shoulder_asymmetry = asymmetry_results["Shoulder"]
 
    feature_df = pd.DataFrame({
 
        "Feature": [
 
            "Shoulder ROM",
 
            "Lumbar Compensation",
 
            "Pelvis Tilt Compensation",
 
            "Shoulder Asymmetry"
 
        ],
 
        "Value": [
 
            round(shoulder_rom, 2),
 
            round(lumbar_compensation, 2),
 
            round(pelvis_compensation, 2),
 
            round(shoulder_asymmetry, 2)
 
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
 
                    font=dict(color="white", size=15),
 
                    align="center",
 
                    height=45,
 
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
    # Movement Score
    # =========================
 
    symmetry_score = max(0, 100 - shoulder_asymmetry)
 
    mobility_score = min(100, shoulder_rom / 180 * 100)
 
    lumbar_score = max(0, 100 - lumbar_compensation * 1.5)
 
    pelvis_score = max(0, 100 - pelvis_compensation * 2)
 
    overall_score = round(
 
        (
            symmetry_score * 0.30
            +
            mobility_score * 0.40
            +
            lumbar_score * 0.15
            +
            pelvis_score * 0.15
        ),
 
        1
    )
 
    st.subheader("Movement Score")
 
    st.caption(t("arm_flexion.score_caption"))
 
    with st.expander(t("common.score_expander_label")):
 
        st.markdown(t("arm_flexion.score_content"))
 
    st.metric(
        "Overall Score",
        f"{overall_score}/100"
    )
 
 
