"""
Sit-to-Stand 用の自動コメント生成モジュール。
Squat の squat_auto_comment.py と同じ使い方になるように作っています。

    from sit_stand_auto_comment import (
        analyze_sit_stand_details,
        generate_client_auto_comment as gen_client_comment,
    )

    sit_stand_details = analyze_sit_stand_details(
        df_phase, phase_summary_df, comparison_df, phase_order
    )
    comment = gen_client_comment(
        lang_code, overall_score, sit_stand_details,
        mobility_score, stability_score, symmetry_score, compensation_score
    )

※ しきい値（左右差15%、代償10°、静止局面のStd 2.0°、上昇/下降差15%）は
  既存コードと同じ仮の基準です。臨床基準に合わせて調整してください。
"""

import numpy as np
import pandas as pd

ASYM_THRESHOLD = 15.0          # 左右差 (%)
COMPENSATION_THRESHOLD = 10.0  # 腰・骨盤の代償 (°)
STD_THRESHOLD = 2.0            # 静止局面の動揺 (°)
PHASE_DIFF_THRESHOLD = 15.0    # 立ち上がり / 座り込みの ROM 差 (%)

JOINT_PAIRS = {
    "Hip": ("hip_flexion_r", "hip_flexion_l"),
    "Knee": ("knee_angle_r", "knee_angle_l"),
    "Ankle": ("ankle_angle_r", "ankle_angle_l"),
}

JOINT_SIMPLE = {
    "ja": {"Hip": "股関節", "Knee": "ひざ", "Ankle": "足首"},
    "en": {"Hip": "hip", "Knee": "knee", "Ankle": "ankle"},
}

VAR_LABEL = {
    "ja": {
        "hip_flexion_r": "股関節（右）", "hip_flexion_l": "股関節（左）",
        "knee_angle_r": "ひざ（右）", "knee_angle_l": "ひざ（左）",
        "ankle_angle_r": "足首（右）", "ankle_angle_l": "足首（左）",
        "pelvis_tilt": "骨盤の前後の傾き", "pelvis_rotation": "骨盤の左右の回旋",
        "lumbar_extension": "腰の反り",
    },
    "en": {
        "hip_flexion_r": "right hip", "hip_flexion_l": "left hip",
        "knee_angle_r": "right knee", "knee_angle_l": "left knee",
        "ankle_angle_r": "right ankle", "ankle_angle_l": "left ankle",
        "pelvis_tilt": "pelvic tilt", "pelvis_rotation": "pelvic rotation",
        "lumbar_extension": "lower-back arch",
    },
}

STATIC_PHASE_LABEL = {
    "ja": {"Bottom": "座っている間", "Standing": "立ち上がった後"},
    "en": {"Bottom": "while seated", "Standing": "after standing up"},
}


VAR_TO_JOINT = {
    "hip_flexion_r": "Hip", "hip_flexion_l": "Hip",
    "knee_angle_r": "Knee", "knee_angle_l": "Knee",
    "ankle_angle_r": "Ankle", "ankle_angle_l": "Ankle",
}


def _low_joints(details, lang):
    """正常範囲より小さかった関節を、左右をまとめて重複なしで返す。"""
    names = []
    for o in details["out_of_range"]:
        joint = VAR_TO_JOINT.get(o["var"])
        if o["direction"] == "low" and joint:
            name = JOINT_SIMPLE[lang][joint]
            if name not in names:
                names.append(name)
    return names


def _rom(df, col):
    if col not in df.columns:
        return np.nan
    return float(df[col].max() - df[col].min())


def _phase_value(phase_summary_df, variable, column):
    row = phase_summary_df[phase_summary_df["Variable"] == variable]
    if len(row) == 0 or column not in row.columns:
        return np.nan
    return row[column].iloc[0]


def analyze_sit_stand_details(df_phase, phase_summary_df, comparison_df, phase_order):
    """実測値から、コメント作成に使う所見をまとめて辞書で返す。"""

    # 1) 左右差（試技全体の ROM で比較）
    asymmetry = {}
    for joint, (r_var, l_var) in JOINT_PAIRS.items():
        r_rom, l_rom = _rom(df_phase, r_var), _rom(df_phase, l_var)
        if np.isnan(r_rom) or np.isnan(l_rom) or max(r_rom, l_rom) == 0:
            continue
        asymmetry[joint] = abs(r_rom - l_rom) / max(r_rom, l_rom) * 100
    asym_flags = sorted(
        [(j, v) for j, v in asymmetry.items() if v > ASYM_THRESHOLD],
        key=lambda x: x[1], reverse=True,
    )

    # 2) 正常範囲から外れた項目（小さすぎ / 大きすぎ）
    out_of_range = []
    for _, row in comparison_df.iterrows():
        if not bool(row.get("Out_of_Range", False)):
            continue
        val, lo, hi = row["Subject_ROM"], row["Healthy_Min"], row["Healthy_Max"]
        direction = "low" if val < lo else "high"
        out_of_range.append({"var": row["Variable"], "value": val, "lo": lo, "hi": hi,
                             "direction": direction})

    # 3) 腰・骨盤の代償
    compensation = {
        "lumbar_extension": _rom(df_phase, "lumbar_extension"),
        "pelvis_tilt": _rom(df_phase, "pelvis_tilt"),
        "pelvis_rotation": _rom(df_phase, "pelvis_rotation"),
    }
    compensation_flags = [k for k, v in compensation.items()
                          if pd.notna(v) and v > COMPENSATION_THRESHOLD]

    # 4) 静止局面（座位 / 立位）での骨盤・腰の動揺
    unstable_phases = []
    for phase in ["Bottom", "Standing"]:
        for var in ["pelvis_tilt", "pelvis_rotation", "pelvis_list", "lumbar_extension"]:
            std_v = _phase_value(phase_summary_df, var, f"{phase}_Std")
            if pd.notna(std_v) and std_v > STD_THRESHOLD:
                unstable_phases.append(phase)
                break

    # 5) 立ち上がり（Ascending）と座り込み（Descending）の動きの差
    phase_diff_flags = []
    for joint, (r_var, l_var) in JOINT_PAIRS.items():
        asc = np.nanmean([_phase_value(phase_summary_df, r_var, "Ascending_ROM"),
                          _phase_value(phase_summary_df, l_var, "Ascending_ROM")])
        desc = np.nanmean([_phase_value(phase_summary_df, r_var, "Descending_ROM"),
                           _phase_value(phase_summary_df, l_var, "Descending_ROM")])
        if np.isnan(asc) or np.isnan(desc) or max(asc, desc) == 0:
            continue
        diff_pct = abs(asc - desc) / max(asc, desc) * 100
        if diff_pct > PHASE_DIFF_THRESHOLD:
            phase_diff_flags.append({"joint": joint, "asc": asc, "desc": desc,
                                     "diff_pct": diff_pct,
                                     "larger": "asc" if asc > desc else "desc"})

    return {
        "asymmetry": asymmetry,
        "asym_flags": asym_flags,
        "out_of_range": out_of_range,
        "compensation": compensation,
        "compensation_flags": compensation_flags,
        "unstable_phases": unstable_phases,
        "phase_diff_flags": phase_diff_flags,
    }


def generate_client_auto_comment(lang_code, overall_score, details,
                                 mobility_score, stability_score,
                                 symmetry_score, compensation_score):
    """対象者本人が読める、専門用語を減らした下書きコメントを作る。"""
    if lang_code == "ja":
        return _comment_ja(overall_score, details, mobility_score,
                           stability_score, symmetry_score, compensation_score)
    return _comment_en(overall_score, details, mobility_score,
                       stability_score, symmetry_score, compensation_score)


def _comment_ja(overall, d, mobility, stability, symmetry, compensation):
    js, vl = JOINT_SIMPLE["ja"], VAR_LABEL["ja"]

    if overall >= 80:
        lines = [f"今回の総合スコアは{overall:.0f}/100で、とても良い状態です。"]
    elif overall >= 60:
        lines = [f"今回の総合スコアは{overall:.0f}/100でした。全体的には悪くありませんが、いくつか気をつけたい点があります。"]
    else:
        lines = [f"今回の総合スコアは{overall:.0f}/100でした。いくつか改善していきたいポイントが見つかりました。"]

    # 可動域
    if mobility >= 80:
        lines.append("立ち座りの動きの大きさ（可動域）はしっかり出せています。")
    else:
        lines.append("立ち座りの動きの大きさ（可動域）には、まだ伸びしろがあります。")
    low = _low_joints(d, "ja")
    if low:
        lines.append(f"特に{'・'.join(low)}の動きが小さめで、硬さが出ている可能性があります。")

    # 左右差
    if d["asym_flags"]:
        joint_text = "・".join(f"{js.get(j, j)}（約{v:.0f}%）" for j, v in d["asym_flags"])
        lines.append(f"{joint_text}で左右の動きに差が見られ、片側に体重をかけて立ち座りしている可能性があります。")
    elif symmetry >= 80:
        lines.append("左右の動きもよく揃っていました。")

    # 腰・骨盤の代償
    if d["compensation_flags"]:
        comp_text = "・".join(vl.get(k, k) for k in d["compensation_flags"])
        lines.append(f"立ち上がる・座る際に{comp_text}が大きく、腰まわりで動きを補っている傾向があります。")
    elif compensation >= 80:
        lines.append("腰や骨盤への負担は少なめです。")

    # 静止局面の安定性
    if d["unstable_phases"]:
        ph = "・".join(STATIC_PHASE_LABEL["ja"][p] for p in d["unstable_phases"])
        lines.append(f"{ph}に骨盤が少し揺れやすい様子がありました。")
    elif stability >= 80:
        lines.append("座っている間・立った後の姿勢も安定していました。")

    # 立ち上がりと座り込みの差
    if d["phase_diff_flags"]:
        f = max(d["phase_diff_flags"], key=lambda x: x["diff_pct"])
        lines.append(f"立ち上がる時と座る時で、{js.get(f['joint'], f['joint'])}の動き方に差が見られました。どちらの動作もゆっくり同じように行う意識を持ってみましょう。")

    lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。")
    return "\n".join(lines)


def _comment_en(overall, d, mobility, stability, symmetry, compensation):
    js, vl = JOINT_SIMPLE["en"], VAR_LABEL["en"]

    if overall >= 80:
        lines = [f"This check scored {overall:.0f}/100 overall — a great result."]
    elif overall >= 60:
        lines = [f"This check scored {overall:.0f}/100 overall. Things look reasonably good, with a few points worth keeping an eye on."]
    else:
        lines = [f"This check scored {overall:.0f}/100 overall. A few areas stood out that are worth working on."]

    if mobility >= 80:
        lines.append("Your range of movement when sitting down and standing up looks solid.")
    else:
        lines.append("There's room to increase your range of movement when sitting down and standing up.")
    low = _low_joints(d, "en")
    if low:
        joined = low[0] if len(low) == 1 else ", ".join(low[:-1]) + " and " + low[-1]
        lines.append(f"In particular, the {joined} moved less than typical, which may point to some stiffness.")

    if d["asym_flags"]:
        joint_text = ", ".join(f"{js.get(j, j)} (about {v:.0f}%)" for j, v in d["asym_flags"])
        lines.append(f"A left-right difference was seen at the {joint_text}, which may mean you lean more on one side.")
    elif symmetry >= 80:
        lines.append("The left and right sides moved very evenly.")

    if d["compensation_flags"]:
        comp_text = ", ".join(vl.get(k, k) for k in d["compensation_flags"])
        lines.append(f"There was noticeable {comp_text} while sitting down and standing up, suggesting the lower back is helping out.")
    elif compensation >= 80:
        lines.append("There was little strain on the lower back or pelvis.")

    if d["unstable_phases"]:
        ph = " and ".join(STATIC_PHASE_LABEL["en"][p] for p in d["unstable_phases"])
        lines.append(f"The pelvis tended to wobble a little {ph}.")
    elif stability >= 80:
        lines.append("Your posture stayed steady both while seated and after standing up.")

    if d["phase_diff_flags"]:
        f = max(d["phase_diff_flags"], key=lambda x: x["diff_pct"])
        lines.append(f"The {js.get(f['joint'], f['joint'])} moved differently when standing up versus sitting down. Try doing both movements slowly and in the same controlled way.")

    lines.append("Try working through the recommended actions below at a comfortable pace before the next check.")
    return "\n".join(lines)
