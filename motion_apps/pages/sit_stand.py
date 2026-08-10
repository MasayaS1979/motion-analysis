import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
 
st.set_page_config(page_title="Sit_Stand Analysis", layout="wide")
 
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
 
st.title("Sit_Stand Analysis")
 
uploaded_file = st.session_state.get("uploaded_file")
 
if uploaded_file is None:
    st.warning("Please upload file from Home page")
    st.stop()
 
# =========================
# Phase Detection
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
 
pelvis_min = signal_smooth.min()
pelvis_max = signal_smooth.max()
pelvis_range = pelvis_max - pelvis_min
 
phase_order = [
    "Bottom",
    "Ascending",
    "Standing",
    "Descending"
]
 
display_phase_order = [
    "Bottom",
    "Ascending",
    "Standing",
    "Descending"
]
 
# -------------------------
# Velocity (frame-to-frame)
# -------------------------
 
velocity = signal_smooth.diff()
 
velocity_threshold = 0.002
 
bottom_threshold = pelvis_min + pelvis_range * 0.20
standing_threshold = pelvis_max - pelvis_range * 0.20
 
# -------------------------
# Bottom / Standing Event
# (used only for the plot markers / event metrics below)
# -------------------------
 
bottom_idx = signal_smooth.idxmin()
 
standing_idx = (
    signal_smooth.iloc[bottom_idx:]
    .idxmax()
)
 
# -------------------------
# Phase Classification
# 位置(しきい値)だけでなく速度も見て判定する。
# これにより、まだ動いている（Ascending/Descendingの途中の）
# フレームが Bottom/Standing に混入し、それらのROMが
# 過大評価されるのを防ぐ。
# -------------------------
 
phases = []
 
for p, v in zip(signal_smooth, velocity):
 
    if pd.isna(v):
        phase = "Bottom"
 
    # Bottom：最下点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    elif p <= bottom_threshold and abs(v) < velocity_threshold:
        phase = "Bottom"
 
    # Standing：最高点付近 かつ 速度がほぼゼロ（静止）の場合のみ
    elif p >= standing_threshold and abs(v) < velocity_threshold:
        phase = "Standing"
 
    # まだ動いている場合は、位置に関係なく速度方向で判定
    elif v > velocity_threshold:
        phase = "Ascending"
 
    elif v < -velocity_threshold:
        phase = "Descending"
 
    else:
        phase = (
            phases[-1]
            if len(phases) > 0
            else "Bottom"
        )
 
    phases.append(phase)
 
# -------------------------
# Keep only the single contiguous cluster of Bottom / Standing
# that is nearest to the detected event (bottom_idx / standing_idx).
#
# The position+velocity classification above can label more than one
# separated "quiet" cluster as Bottom or Standing (e.g. the initial
# sitting rest AND the final sitting rest after Descending, or a brief
# stall partway through the motion). Averaging max/min ROM across two
# unrelated quiet clusters inflates Bottom_ROM / Standing_ROM well
# beyond what a single steady posture would produce. Here we keep only
# the run that contains (or is closest to) the true event frame, and
# fold every other same-label run back into whatever phase preceded it.
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
        # right before it started (there is always a preceding
        # frame, since the very first frame is only ever "Bottom"
        # via the pd.isna(v) branch, which is itself part of the
        # earliest Bottom run).
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
    "hip_flexion_r":{"min":90.0,"max":110.0},
    "hip_flexion_l":{"min":90.0,"max":110.0},
    "knee_angle_r":{"min":85.0,"max":105.0},
    "knee_angle_l":{"min":85.0,"max":105.0},
    "ankle_angle_r":{"min":10.0,"max":20.0},
    "ankle_angle_l":{"min":10.0,"max":20.0}
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
 
    st.caption(
        "立ち-座位動作中の各フェーズ（Standing・Descending・Bottom・Ascending）を時系列で可視化したグラフです。"
    )
 
    colors_phase = {
        "Standing": "dodgerblue",
        "Descending": "orange",
        "Bottom": "red",
        "Ascending": "limegreen"
    }
 
    fig, ax = plt.subplots(
        figsize=(15,6)
    )
 
    # =========================
    # Dark Theme
    # =========================
 
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
 
    # 軸・目盛
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
 
    # 軸線
    for spine in ax.spines.values():
        spine.set_color("white")
 
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
 
    ax.set_title(
        "Phase Detection Plot"
    )
 
    ax.set_xlabel(
        "Frame"
    )
 
    ax.set_ylabel(
        "Pelvis Vertical Position (m)"
    )
 
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
        st.metric("Bottom", bottom_idx)
 
    with col2:
        st.metric("Standing", standing_idx)
 
    # ==========================================================
    # Phase Summary Table
    # ==========================================================
 
    st.subheader("Phase Summary Table")
 
    st.caption(
        "各フェーズにおける各関節の最小値、最大値、平均値、標準偏差、可動域（ROM）を算出します。"
    )
 
    st.markdown("""
**各指標の説明**
 
- **Min**：各フェーズにおける最小値
- **Max**：各フェーズにおける最大値
- **Mean**：各フェーズにおける平均値
- **Std**：各フェーズにおける標準偏差
- **ROM**：Range of Motion（Max − Min）
""")
 
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
 
        ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
 
        ("BACKGROUND",(0,1),(-1,-1),colors.beige),
 
        ("GRID",(0,0),(-1,-1),0.5,colors.black),
 
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
 
        ("FONTSIZE",(0,0),(-1,-1),8),
 
        ("BOTTOMPADDING",(0,0),(-1,0),8)
 
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
 
            file_name="Phase_Summary.xlsx",
 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
 
        )
 
    with col2:
 
        st.download_button(
 
            "📄 Download PDF",
 
            data=pdf_buffer.getvalue(),
 
            file_name="Phase_Summary.pdf",
 
            mime="application/pdf"
 
        )
 
    # ==========================================================
    # Phase Tables
    # ==========================================================
 
    st.markdown("---")
 
    st.subheader("Phase Statistics")
 
    tab_bottom, tab_asc, tab_stand, tab_desc = st.tabs(
        [
            "Bottom",
            "Ascending",
            "Standing",
            "Descending"
        ]
    )
 
    with tab_bottom:
 
        st.dataframe(
            bottom_df,
            use_container_width=True
        )
 
    with tab_asc:
 
        st.dataframe(
            ascending_df,
            use_container_width=True
        )
 
    with tab_stand:
 
        st.dataframe(
            standing_df,
            use_container_width=True
        )
 
    with tab_desc:
 
        st.dataframe(
            descending_df,
            use_container_width=True
        )
 
# =========================
# ROM Analysis
# =========================
with tab2:
 
    st.subheader("Joint Time Series")
 
    st.caption(
        "各関節運動の時系列変化を表示します。"
    )
 
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
 
        # Background
        a.set_facecolor("black")
 
        # Tick color
        a.tick_params(colors="white")
 
        # Labels
        a.title.set_color("white")
        a.xaxis.label.set_color("white")
        a.yaxis.label.set_color("white")
 
        # Axis line
        for spine in a.spines.values():
            spine.set_color("white")
 
        # Grid
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
        df_phase["knee_angle_r"],
        label="Right Leg",
        linewidth=2
    )
 
    ax[0].plot(
        time,
        df_phase["knee_angle_l"],
        label="Left Leg",
        linewidth=2
    )
 
    ax[0].set_title("Knee ROM")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Angle (deg)")
 
    legend = ax[0].legend(loc="upper right")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Hip
    # =====================
 
    ax[1].plot(
        time,
        df_phase["hip_flexion_r"],
        label="Right Leg",
        linewidth=2
    )
 
    ax[1].plot(
        time,
        df_phase["hip_flexion_l"],
        label="Left Leg",
        linewidth=2
    )
 
    ax[1].set_title("Hip ROM")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (deg)")
 
    legend = ax[1].legend(loc="upper right")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Ankle
    # =====================
 
    ax[2].plot(
        time,
        df_phase["ankle_angle_r"],
        label="Right Leg",
        linewidth=2
    )
 
    ax[2].plot(
        time,
        df_phase["ankle_angle_l"],
        label="Left Leg",
        linewidth=2
    )
 
    ax[2].set_title("Ankle ROM")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (deg)")
 
    legend = ax[2].legend(loc="upper right")
 
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
 
    for text in legend.get_texts():
        text.set_color("white")
 
    # =====================
    # Pelvic Tilt
    # =====================
 
    ax[3].plot(
        time,
        df_phase["pelvis_tilt"],
        linewidth=2,
        color="cyan"
    )
 
    ax[3].set_title("Pelvic Tilt")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Angle (deg)")
 
    # =====================
    # Pelvic Right-Left Position
    # =====================
 
    ax[4].plot(
        time,
        df_phase["pelvis_tx"],
        linewidth=2,
        color="orange"
    )
 
    ax[4].set_title("Pelvic Right-Left Position")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Position (m)")
 
    # =====================
    # Pelvic Height
    # =====================
 
    ax[5].plot(
        time,
        df_phase["pelvis_ty"],
        linewidth=2,
        color="lime"
    )
 
    ax[5].set_title("Pelvic Height")
    ax[5].set_xlabel("Time (s)")
    ax[5].set_ylabel("Position (m)")
 
    # =====================
    # Lumbar Extension
    # =====================
 
    ax[6].plot(
        time,
        df_phase["lumbar_extension"],
        linewidth=2,
        color="magenta"
    )
 
    ax[6].set_title("Lumbar Extension")
    ax[6].set_xlabel("Time (s)")
    ax[6].set_ylabel("Angle (deg)")
 
    plt.tight_layout()
 
    st.pyplot(fig)
 
 
# =========================
# Symmetry Analysis
# =========================
 
with tab3:
 
    st.subheader(
        "Phase Symmetry"
    )
 
    st.caption(
        "左右関節ROMの左右差を各Phaseごとに評価します。"
    )
 
    joints = {
 
        "Hip": (
            "hip_flexion_r",
            "hip_flexion_l"
        ),
 
        "Knee": (
            "knee_angle_r",
            "knee_angle_l"
        ),
 
        "Ankle": (
            "ankle_angle_r",
            "ankle_angle_l"
        )
 
    }
 
    for joint_name, (right_var, left_var) in joints.items():
 
        st.markdown(
            f"## {joint_name}"
        )
 
        right_df = phase_summary_df[
            phase_summary_df["Variable"] == right_var
        ]
 
        left_df = phase_summary_df[
            phase_summary_df["Variable"] == left_var
        ]
 
        symmetry_results = []
 
        rom_difference = []
 
        for phase in phase_order:
 
            right_rom = right_df[
                f"{phase}_ROM"
            ].iloc[0]
 
            left_rom = left_df[
                f"{phase}_ROM"
            ].iloc[0]
 
            if max(right_rom, left_rom) == 0:
 
                asymmetry = 0
 
            else:
 
                asymmetry = (
 
                    abs(
                        right_rom -
                        left_rom
                    )
 
                    /
 
                    max(
                        right_rom,
                        left_rom
                    )
 
                    * 100
 
                )
 
            rom_difference.append(
                asymmetry
            )
 
            symmetry_results.append({
 
                "Phase":
                    phase,
 
                "Right_ROM":
                    round(right_rom, 2),
 
                "Left_ROM":
                    round(left_rom, 2),
 
                "Asymmetry_%":
                    round(asymmetry, 2)
 
            })
 
        symmetry_df = pd.DataFrame(
            symmetry_results
        )
 
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
                            symmetry_df.columns
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
                            symmetry_df[col]
                            for col in symmetry_df.columns
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
        # Summary Metrics
        # =========================
 
        col1, col2 = st.columns(2)
 
        with col1:
 
            st.metric(
 
                "Maximum Asymmetry",
 
                f"{max(rom_difference):.1f}%"
 
            )
 
        with col2:
 
            st.metric(
 
                "Average Asymmetry",
 
                f"{np.mean(rom_difference):.1f}%"
 
            )
 
        # =========================
        # Bar Plot
        # =========================
 
        fig, ax = plt.subplots(
 
            figsize=(8,4)
 
        )
 
        # Dark Theme
 
        fig.patch.set_facecolor("black")
 
        ax.set_facecolor("black")
 
        ax.tick_params(
            colors="white"
        )
 
        ax.title.set_color("white")
 
        ax.xaxis.label.set_color("white")
 
        ax.yaxis.label.set_color("white")
 
        for spine in ax.spines.values():
 
            spine.set_color("white")
 
        ax.grid(
 
            True,
 
            color="white",
 
            alpha=0.3
 
        )
 
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
 
        ax.set_ylabel(
 
            "Asymmetry (%)"
 
        )
 
        ax.set_title(
 
            f"{joint_name} ROM Asymmetry by Phase"
 
        )
 
        legend = ax.legend()
 
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
 
        st.pyplot(
            fig
        )
 
    # =========================
    # Healthy ROM Comparison
    # =========================
 
    st.subheader(
        "Healthy ROM Comparison"
    )
 
    st.caption(
        "正常可動域（Healthy ROM）との比較を行います。"
    )
 
    st.caption(
        "Difference% = Subject ROM と Healthy ROM中央値との差"
    )
 
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
                        comparison_df.columns
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
                        comparison_df[col]
                        for col in comparison_df.columns
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
 
    bar_colors = comparison_df["Out_of_Range"].map({
 
        True: "red",
 
        False: "royalblue"
 
    })
 
    fig, ax = plt.subplots(
 
        figsize=(10,5)
 
    )
 
    # Dark Theme
 
    fig.patch.set_facecolor("black")
 
    ax.set_facecolor("black")
 
    ax.tick_params(
        colors="white"
    )
 
    ax.title.set_color("white")
 
    ax.xaxis.label.set_color("white")
 
    ax.yaxis.label.set_color("white")
 
    for spine in ax.spines.values():
 
        spine.set_color("white")
 
    ax.grid(
 
        True,
 
        color="white",
 
        alpha=0.3
 
    )
 
    ax.bar(
 
        comparison_df["Variable"],
 
        comparison_df["ROM_Difference_%"],
 
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
# Clinical Report
# =========================
with tab4:
 
    findings = []
 
    for _, row in comparison_df.iterrows():
 
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
# Raw Data
# =========================
with tab5:
 
    st.subheader(
        "Sit-Stand Raw Data"
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
        "sit_stand_raw_data.csv",
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
        "sit_stand_raw_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
 
# =========================
# Dashboard
# =========================
with tab6:
 
    st.title(
        "Sit-to-Stand Dashboard"
    )
 
    st.caption(
        "Sit-to-Stand動作の主要指標を表示します"
    )
 
    # =========================
    # KPI
    # =========================
 
    max_knee = max(
        df_phase["knee_angle_r"].max(),
        df_phase["knee_angle_l"].max()
    )
 
    max_hip = max(
        df_phase["hip_flexion_r"].max(),
        df_phase["hip_flexion_l"].max()
    )
 
    max_ankle = max(
        df_phase["ankle_angle_r"].max(),
        df_phase["ankle_angle_l"].max()
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
 
    st.subheader(
        "Key Metrics"
    )
 
    with st.expander("📖 指標の説明を見る"):
 
        st.markdown("""
| 指標 | 説明 |
|---|---|
| **Max Knee Flexion** | 立ち座り動作中の膝関節最大屈曲角度（左右のうち大きい方） |
| **Max Hip Flexion** | 立ち上がり開始時の股関節最大屈曲角度（左右のうち大きい方） |
| **Max Ankle Motion** | 動作中の足関節角度変化量（左右のうち大きい方） |
| **Lumbar Compensation** | 腰椎伸展の変化量。股関節・足関節の可動性不足を補う代償動作の可能性 |
| **Pelvis Compensation** | 骨盤前後傾の変化量。骨盤制御能力の指標 |
| **ROM Deviation** | 健常可動域（Healthy ROM）との平均偏差率 |
""")
 
    col1, col2, col3, col4, col5, col6 = st.columns(6)
 
    col1.metric(
        "Max Knee Flexion",
        f"{max_knee:.1f}°"
    )
 
    col2.metric(
        "Max Hip Flexion",
        f"{max_hip:.1f}°"
    )
 
    col3.metric(
        "Max Ankle Motion",
        f"{max_ankle:.1f}°"
    )
 
    col4.metric(
        "Lumbar Compensation",
        f"{lumbar_compensation:.1f}°"
    )
 
    col5.metric(
        "Pelvis Compensation",
        f"{pelvis_compensation:.1f}°"
    )
 
    col6.metric(
        "ROM Deviation",
        f"{overall_deviation:.1f}%"
    )
 
    # =========================
    # Interactive Motion Viewer
    # =========================
 
    st.subheader(
        "Interactive Motion Viewer"
    )
 
    st.caption(
        "💡 左のチェックボックスで、表示する関節・骨盤・腰椎の指標を選択できます。"
    )
 
    left_col, right_col = st.columns(
        [1.2,4]
    )
 
    # ======================================
    # Left Panel
    # ======================================
 
    with left_col:
 
        st.markdown(
            "### Lower Limb"
        )
 
        st.markdown(
            "#### Hip"
        )
 
        show_hip_r = st.checkbox(
            "Right Hip",
            value=True
        )
 
        show_hip_l = st.checkbox(
            "Left Hip",
            value=True
        )
 
        st.markdown(
            "#### Knee"
        )
 
        show_knee_r = st.checkbox(
            "Right Knee",
            value=True
        )
 
        show_knee_l = st.checkbox(
            "Left Knee",
            value=True
        )
 
        st.markdown(
            "#### Ankle"
        )
 
        show_ankle_r = st.checkbox(
            "Right Ankle"
        )
 
        show_ankle_l = st.checkbox(
            "Left Ankle"
        )
 
        st.markdown(
            "---"
        )
 
        st.markdown(
            "### Pelvis"
        )
 
        show_tilt = st.checkbox(
            "Tilt"
        )
 
        show_obliquity = st.checkbox(
            "Obliquity"
        )
 
        show_rotation = st.checkbox(
            "Rotation"
        )
 
        show_ml = st.checkbox(
            "Medial-Lateral Deviation"
        )
 
        show_vertical = st.checkbox(
            "Vertical Displacement"
        )
 
        show_ap = st.checkbox(
            "Anterior-Posterior Deviation"
        )
 
        st.markdown(
            "---"
        )
 
        st.markdown(
            "### Lumbar"
        )
 
        show_lumbar = st.checkbox(
            "Extension"
        )
 
    # ======================================
    # Right Panel
    # ======================================
 
    with right_col:
 
        time = np.arange(
            len(df_phase)
        ) / 60
 
        # =========================
        # Joint Motion
        # =========================
 
        fig, ax = plt.subplots(
            figsize=(15,6),
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
            alpha=0.25
        )
 
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
 
        ax.set_title(
            "Sit-to-Stand Joint Motion"
        )
 
        ax.set_xlabel(
            "Time (s)"
        )
 
        ax.set_ylabel(
            "Angle (deg)"
        )
 
        legend = ax.legend(
            loc="upper right",
            ncol=2
        )
 
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
 
        st.pyplot(
            fig
        )
 
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
 
        ax2.set_facecolor(
            "black"
        )
 
        ax2.tick_params(
            colors="white"
        )
 
        for spine in ax2.spines.values():
 
            spine.set_color(
                "white"
            )
 
        ax2.grid(
            color="white",
            alpha=0.25
        )
 
        if show_tilt:
 
            ax2.plot(
                time,
                df_phase["pelvis_tilt"],
                label="Tilt"
            )
 
        if show_obliquity:
 
            ax2.plot(
                time,
                df_phase["pelvis_list"],
                label="Obliquity"
            )
 
        if show_rotation:
 
            ax2.plot(
                time,
                df_phase["pelvis_rotation"],
                label="Rotation"
            )
 
        if show_ml:
 
            ax2.plot(
                time,
                df_phase["pelvis_tx"]*1000,
                label="Medial-Lateral (mm)"
            )
 
        if show_vertical:
 
            ax2.plot(
                time,
                df_phase["pelvis_ty"]*1000,
                label="Vertical (mm)"
            )
 
        if show_ap:
 
            ax2.plot(
                time,
                df_phase["pelvis_tz"]*1000,
                label="Anterior-Posterior (mm)"
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
 
        legend2 = ax2.legend()
 
        legend2.get_frame().set_facecolor(
            "black"
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
 
    with st.expander("📖 Joint ROMとは"):
 
        st.markdown("""
関節が動作中にどれだけ動いたかを示す指標です。
 
**ROM = 最大角度 − 最小角度**
 
| 関節 | 評価内容 |
|---|---|
| **Hip** | 股関節屈曲角度。立ち上がり時の体幹前傾戦略を評価 |
| **Knee** | 膝関節屈曲量。立ち上がりに必要な下肢運動を評価 |
| **Ankle** | 足関節運動。足部による重心移動能力を評価 |
""")
 
    rom_joints = {
 
        "Hip":
        "hip_flexion_r",
 
        "Knee":
        "knee_angle_r",
 
        "Ankle":
        "ankle_angle_r"
 
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
    # Joint Asymmetry
    # =========================
 
    st.subheader(
        "Joint Asymmetry"
    )
 
    with st.expander("📖 Joint Asymmetryとは"):
 
        st.markdown("""
左右の関節可動域（ROM）の差を、大きい方のROMで正規化しパーセント表示した指標です。
 
- **15%以下** — 比較的対称な運動パターン
- **15%超** — 左右荷重差・筋力差・可動性差・代償動作の可能性
""")
 
    asymmetry_joints = {
 
        "Hip":
        (
            "hip_flexion_r",
            "hip_flexion_l"
        ),
 
        "Knee":
        (
            "knee_angle_r",
            "knee_angle_l"
        ),
 
        "Ankle":
        (
            "ankle_angle_r",
            "ankle_angle_l"
        )
 
    }
 
    asymmetry_results = {}
 
    for joint_name,(right_var,left_var) in asymmetry_joints.items():
 
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
 
        if max(
            right_rom,
            left_rom
        ) == 0:
 
            asymmetry = 0
 
        else:
 
            asymmetry = (
 
                abs(
                    right_rom -
                    left_rom
                )
 
                /
 
                max(
                    right_rom,
                    left_rom
                )
 
                *
 
                100
 
            )
 
        asymmetry_results[joint_name] = round(
            asymmetry,
            2
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
 
        list(
            asymmetry_results.keys()
        ),
 
        list(
            asymmetry_results.values()
        )
 
    )
 
    for bar,value in zip(
        bars,
        asymmetry_results.values()
    ):
 
        ax.text(
 
            bar.get_x()
            +
            bar.get_width()/2,
 
            value,
 
            f"{value:.1f}%",
 
            ha="center",
 
            va="bottom",
 
            color="white"
 
        )
 
    ax.axhline(
        15,
        color="red",
        linestyle="--",
        label="15% Threshold"
    )
 
    ax.set_ylabel(
        "Asymmetry (%)"
    )
 
    ax.set_title(
        "Joint Asymmetry"
    )
 
    legend=ax.legend()
 
    legend.get_frame().set_facecolor(
        "black"
    )
 
    for text in legend.get_texts():
 
        text.set_color(
            "white"
        )
 
    st.pyplot(
        fig
    )
 
 
# =========================
# Movement Features & Score
# =========================
 
with tab7:
 
    # =========================
    # Movement Features
    # =========================
 
    st.subheader(
        "Movement Features"
    )
 
    st.caption(
        "身体移動・姿勢制御・代償動作・左右差からSit-to-Stand動作を評価する特徴量です。"
    )
 
    with st.expander("📖 特徴量の説明を見る"):
 
        st.markdown("""
| 特徴量 | 説明 |
|---|---|
| **Seat-Off Height** | 骨盤の垂直移動量（立ち上がりの深さ） |
| **Pelvic Shift** | 骨盤の左右・前後移動量の最大値 |
| **Lumbar Compensation** | 腰椎伸展の変化量。大きいほど代償動作の可能性 |
| **Hip Asymmetry** | 左右股関節のROM差（%） |
| **Knee Asymmetry** | 左右膝関節のROM差（%） |
| **Ankle Asymmetry** | 左右足関節のROM差（%） |
""")
 
    seat_off_height = (
 
        pelvis_max -
 
        pelvis_min
 
    )
 
    pelvic_shift = (
        df_phase["pelvis_tx"]
        .abs()
        .max()
    )
 
    # Lumbar compensation = ROM of lumbar extension (max - min),
    # matching the definition given above and the KPI metric.
    trunk_compensation = round(
        df_phase["lumbar_extension"].max()
        -
        df_phase["lumbar_extension"].min(),
        2
    )
 
    hip_asymmetry = asymmetry_results["Hip"]
 
    knee_asymmetry = asymmetry_results["Knee"]
 
    ankle_asymmetry = asymmetry_results["Ankle"]
 
    feature_df = pd.DataFrame({
 
        "Feature":[
 
            "Seat-Off Height",
 
            "Pelvic Shift",
 
            "Lumbar Compensation",
 
            "Hip Asymmetry",
 
            "Knee Asymmetry",
 
            "Ankle Asymmetry"
 
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
 
            round(
                hip_asymmetry,
                2
            ),
 
            round(
                knee_asymmetry,
                2
            ),
 
            round(
                ankle_asymmetry,
                2
            )
 
        ]
 
    })
 
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
 
    # Overall Symmetry Index
 
    overall_asymmetry = (
        hip_asymmetry +
        knee_asymmetry +
        ankle_asymmetry
    ) / 3
 
    # Symmetry Score
 
    symmetry_score = max(
        0,
        100 - overall_asymmetry
    )
 
    stability_score = max(
 
        0,
 
        100 -
 
        pelvic_shift * 100
 
    )
 
    compensation_score = max(
 
        0,
 
        100 -
 
        trunk_compensation * 2
 
    )
 
    mobility_score = min(
 
        100,
 
        seat_off_height * 500
 
    )
 
    overall_score = round(
 
        (
            symmetry_score * 0.30
            +
            stability_score * 0.30
            +
            compensation_score * 0.20
            +
            mobility_score * 0.20
        ),
 
        1
    )
 
    st.subheader(
 
        "Movement Score"
 
    )
 
    st.caption(
        "左右対称性・姿勢安定性・代償動作・身体移動能力の4要素から算出する100点満点の総合スコアです。"
    )
 
    with st.expander("📖 スコアの算出方法を見る"):
 
        st.markdown("""
**Overall Score = Symmetry×0.30 + Stability×0.30 + Compensation×0.20 + Mobility×0.20**
 
| 要素 | 重み | 算出元 |
|---|---|---|
| **Symmetry Score** | 30% | 股関節・膝関節・足関節の左右差（Asymmetry）の平均値 |
| **Stability Score** | 30% | 骨盤の左右・前後移動量（Pelvic Shift） |
| **Compensation Score** | 20% | 腰椎伸展の変化量（Lumbar Compensation） |
| **Mobility Score** | 20% | 骨盤垂直移動量（Seat-Off Height） |
 
- **高スコア** — 安定した効率的なSit-to-Stand動作
- **低スコア** — 左右差・姿勢制御低下・代償動作・身体移動能力低下の可能性
""")
 
    st.metric(
 
        "Overall Score",
 
        f"{overall_score}/100"
 
    )
 