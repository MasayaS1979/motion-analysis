"""
arm_flexion_auto_comment.py
---------------------------------------------------------------
Arm Flexion Analysis の自動コメント（PDF Report / Client Report）用モジュール。
squat_auto_comment.py / sit_stand_auto_comment.py と同じ構成・同じ呼び出し方にそろえています。

実測データ（df_phase）から、肩関節・肘・体幹（腰部）・骨盤について
以下を読み取り、臨床向け／クライアント向けのコメント下書きを生成します。

  - 肩関節屈曲の最大角度・可動域（左右別）と正常範囲との比較
  - 左右差（試技全体／フェーズ別で最大となる局面）
  - 反復ごとの一貫性（最大挙上角度のばらつき・前半→後半の変化）
  - 左右のタイミング差（挙上が半分まで進んだ時刻のずれ）
  - 挙上と下降の可動域差（求心性／遠心性の制御差）
  - 挙上中の代償：肘の曲がり、腕の外への開き（列がある場合）
  - 腰部：開始姿勢からの反り（腕を上げたときに腰を反らせて補う動き）
  - 体幹の側屈（列がある場合）
  - 骨盤：前後傾・側方傾斜・回旋・左右移動・上下移動（つま先立ち）
  - 反復全体：テンポ（挙上／下降時間）

■ Squat / Sit-to-Stand 版との違い
  - 反復検出：骨盤の高さではなく、左右平均の肩関節屈曲角度の極大点（最大挙上）で数えます。
  - 静止局面は Start（開始位置）と Top（最大挙上）です。
  - 左右の骨盤移動は pelvis_tz（OpenSim/OpenCap の左右方向）を使います。
  - 腰椎伸展・骨盤前後傾・骨盤回旋は「動きが少ないほど良い」代償指標として扱い、
    正常範囲の下限を下回っても問題としては扱いません。

※ 本モジュール内のしきい値はすべて「仮の基準」です。
  THRESHOLDS を臨床基準に合わせて調整してください。
"""

import numpy as np
import pandas as pd

FS = 60  # サンプリング周波数（Hz）

# ---------------------------------------------------------------
# しきい値（仮）— 臨床基準に合わせて調整してください
# ---------------------------------------------------------------
THRESHOLDS = {
    "asym_pct": 15.0,           # 左右差（%）
    "ecc_con_pct": 15.0,        # 挙上／下降のROM差（%）
    "rep_cv_pct": 8.0,          # 反復ごとの最大挙上角度の変動係数（%）
    "rep_trend_deg": 8.0,       # 前半→後半の最大挙上角度の変化（°）
    "timing_lag_ms": 100.0,     # 左右の挙上タイミングのずれ（ms）
    "static_std_deg": 2.0,      # 静止局面（開始位置・最大挙上）の標準偏差（°）
    "elbow_bend_deg": 20.0,     # 挙上中の肘の曲がり（開始位置からの増加量）
    "arm_add_change_deg": 15.0, # 挙上中に腕が外へ開いた量（arm_add の変化）
    "lumbar_change_deg": 10.0,  # 腰部の開始姿勢からの反り（°）
    "trunk_bend_deg": 8.0,      # 体幹側屈の可動範囲（°）
    "pelvis_rom_deg": 10.0,     # 骨盤（前後傾・回旋）の可動範囲（°）
    "pelvis_list_deg": 5.0,     # 骨盤の側方傾斜の可動範囲（°）
    "pelvis_shift_mm": 20.0,    # 骨盤の左右移動（mm）
    "pelvis_rise_mm": 20.0,     # 骨盤の上昇（つま先立ち・伸び上がり）（mm）
    "tempo_ratio": 1.5,         # 挙上時間／下降時間の比（これ超で「下ろすのが速い」）
    "phase_rom_min_deg": 10.0,  # フェーズ別左右差を評価する最小ROM
    "lumbar_note_deg": 2.0,     # 腰部の反りを文中で言及する最小量
}

# 「動きが少ないほど良い」代償指標（正常範囲の下限を下回っても問題にしない）
COMPENSATION_VARS = {"lumbar_extension", "pelvis_tilt", "pelvis_rotation"}

SHOULDER = ("arm_flex_r", "arm_flex_l")
ELBOW = ("elbow_flex_r", "elbow_flex_l")
ARM_ADD = ("arm_add_r", "arm_add_l")

NAMES = {
    "ja": {"shoulder": "肩関節", "R": "右", "L": "左",
           "Start": "開始位置", "Raising": "挙上", "Top": "最大挙上", "Lowering": "下降"},
    "en": {"shoulder": "Shoulder", "R": "right", "L": "left",
           "Start": "Start", "Raising": "Raising", "Top": "Top", "Lowering": "Lowering"},
}


# ===============================================================
# 1. データの読み取り
# ===============================================================
def _pct_diff(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m * 100


def _detect_reps(arm_angle, fs=FS):
    """左右平均の肩関節屈曲角度の極大点（最大挙上）から反復を検出する。"""
    s = pd.Series(arm_angle).rolling(5, center=True).mean().bfill().ffill().to_numpy()
    lo, hi = np.nanmin(s), np.nanmax(s)
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return [], s

    up_line = lo + rng * 0.6
    min_gap = int(1.0 * fs)
    cand = [i for i in range(1, len(s) - 1)
            if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > up_line]
    cand.sort(key=lambda i: -s[i])
    maxima = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in maxima):
            maxima.append(i)
    maxima.sort()
    if not maxima:
        return [], s

    bounds = [0]
    for a, b in zip(maxima[:-1], maxima[1:]):
        bounds.append(a + int(np.argmin(s[a:b + 1])))
    bounds.append(len(s) - 1)

    start_line = lo + rng * 0.10
    top_line = hi - rng * 0.10
    reps = []
    for k, m in enumerate(maxima):
        start, end = bounds[k], bounds[k + 1]
        pre = np.where(s[start:m + 1] <= start_line)[0]
        raise_start = start + pre[-1] if len(pre) else start
        reach = np.where(s[raise_start:m + 1] >= top_line)[0]
        raise_end = raise_start + reach[0] if len(reach) else m
        post = np.where(s[m:end + 1] <= start_line)[0]
        lower_end = m + post[0] if len(post) else end
        leave = np.where(s[m:lower_end + 1] >= top_line)[0]
        lower_start = m + leave[-1] if len(leave) else m
        reps.append({
            "start": start, "end": end, "top": m,
            "raise_start": raise_start, "raise_end": raise_end,
            "raise_s": (raise_end - raise_start) / fs,
            "lower_s": (lower_end - lower_start) / fs,
            "peak_deg": float(s[m]),
        })
    return reps, s


def _nanmean(values):
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _phase_value(phase_summary_df, var, phase, stat):
    row = phase_summary_df[phase_summary_df["Variable"] == var]
    if len(row) == 0:
        return np.nan
    col = f"{phase}_{stat}"
    return row[col].iloc[0] if col in row.columns else np.nan


def _phase_mean(df_phase, col, phase):
    if col not in df_phase.columns:
        return np.nan
    v = df_phase.loc[df_phase["Phase"] == phase, col]
    return float(v.mean()) if len(v) else np.nan


def analyze_arm_flexion_details(df_phase, phase_summary_df, comparison_df,
                                phase_order=("Start", "Raising", "Top", "Lowering"),
                                fs=FS):
    """実測データから、肩・体幹・骨盤ごとの詳細な特徴量をまとめて返す。"""
    th = THRESHOLDS
    rv, lv = SHOULDER
    d = {"joints": {}, "reps": [], "n_reps": 0}
    if rv not in df_phase.columns or lv not in df_phase.columns:
        d.update({"pelvis": {}, "lumbar": {}, "arm_extra": {}})
        return d

    r, l = df_phase[rv].to_numpy(), df_phase[lv].to_numpy()
    reps, _ = _detect_reps((r + l) / 2, fs)
    d["reps"], d["n_reps"] = reps, len(reps)

    # ---------------- 肩関節 ----------------
    j = {
        "peak_R": float(np.nanmax(r)), "peak_L": float(np.nanmax(l)),
        "rom_R": float(np.nanmax(r) - np.nanmin(r)),
        "rom_L": float(np.nanmax(l) - np.nanmin(l)),
    }
    j["asym_pct"] = _pct_diff(j["rom_R"], j["rom_L"])
    j["larger_side"] = "R" if j["rom_R"] >= j["rom_L"] else "L"

    for side, var in (("R", rv), ("L", lv)):
        row = comparison_df[comparison_df["Variable"] == var]
        if len(row):
            rom = row["Subject_ROM"].iloc[0]
            j[f"below_{side}"] = rom < row["Healthy_Min"].iloc[0]
            j[f"above_{side}"] = rom > row["Healthy_Max"].iloc[0]

    phase_asym = {}
    for ph in phase_order:
        a = _phase_value(phase_summary_df, rv, ph, "ROM")
        b = _phase_value(phase_summary_df, lv, ph, "ROM")
        if pd.notna(a) and pd.notna(b) and max(a, b) >= th["phase_rom_min_deg"]:
            phase_asym[ph] = _pct_diff(a, b)
    if phase_asym:
        j["worst_phase"] = max(phase_asym, key=phase_asym.get)
        j["worst_phase_asym"] = phase_asym[j["worst_phase"]]

    raise_rom = _nanmean([_phase_value(phase_summary_df, v, "Raising", "ROM") for v in SHOULDER])
    lower_rom = _nanmean([_phase_value(phase_summary_df, v, "Lowering", "ROM") for v in SHOULDER])
    j["raise_rom"], j["lower_rom"] = raise_rom, lower_rom
    j["ecc_con_pct"] = (_pct_diff(raise_rom, lower_rom)
                        if pd.notna(raise_rom) and pd.notna(lower_rom) else 0.0)

    j["unstable_phases"] = []
    for ph in ("Start", "Top"):
        stds = [_phase_value(phase_summary_df, v, ph, "Std") for v in SHOULDER]
        if any(pd.notna(x) and x > th["static_std_deg"] for x in stds):
            j["unstable_phases"].append(ph)

    if reps:
        pk_r = [np.nanmax(r[p["start"]:p["end"] + 1]) for p in reps]
        pk_l = [np.nanmax(l[p["start"]:p["end"] + 1]) for p in reps]
        pk_mean = np.mean([pk_r, pk_l], axis=0)
        j["rep_peaks"] = pk_mean.tolist()
        j["rep_cv_pct"] = float(np.std(pk_mean) / np.mean(pk_mean) * 100) if np.mean(pk_mean) else 0.0
        half = max(1, len(pk_mean) // 2)
        j["rep_trend_deg"] = (float(np.mean(pk_mean[-half:]) - np.mean(pk_mean[:half]))
                              if len(pk_mean) >= 2 else 0.0)

        def _half_time(x):
            mid = (np.nanmax(x) + np.nanmin(x)) / 2
            idx = np.where(x >= mid)[0]
            return int(idx[0]) if len(idx) else None

        lags = []
        for p in reps:
            a, b = p["raise_start"], p["raise_end"]
            if b - a < 3:
                continue
            tr_, tl_ = _half_time(r[a:b + 1]), _half_time(l[a:b + 1])
            if tr_ is not None and tl_ is not None:
                lags.append(abs(tr_ - tl_) / fs * 1000)
        j["timing_lag_ms"] = float(np.mean(lags)) if lags else 0.0
    d["joints"]["shoulder"] = j

    # ---------------- 肘・腕の開き（列がある場合のみ） ----------------
    extra = {}
    if all(c in df_phase.columns for c in ELBOW):
        # 最大挙上時に、開始位置よりどれだけ肘が曲がったか（左右の大きい方）
        bends = [_phase_mean(df_phase, c, "Top") - _phase_mean(df_phase, c, "Start") for c in ELBOW]
        bends = [b for b in bends if pd.notna(b)]
        if bends:
            extra["elbow_bend_deg"] = float(max(bends))
    if all(c in df_phase.columns for c in ARM_ADD):
        # 腕が外へ開くと arm_add は負方向に変化する（OpenSim の内転が正）
        opens = [-(_phase_mean(df_phase, c, "Top") - _phase_mean(df_phase, c, "Start")) for c in ARM_ADD]
        opens = [o for o in opens if pd.notna(o)]
        if opens:
            extra["arm_open_deg"] = float(max(opens))
    if "lumbar_bending" in df_phase.columns:
        extra["trunk_bend_rom"] = float(df_phase["lumbar_bending"].max() - df_phase["lumbar_bending"].min())
    d["arm_extra"] = extra

    # ---------------- 骨盤 ----------------
    def rng(col):
        return float(df_phase[col].max() - df_phase[col].min()) if col in df_phase.columns else np.nan

    def top_minus_start(col):
        t, s = _phase_mean(df_phase, col, "Top"), _phase_mean(df_phase, col, "Start")
        return float(t - s) if pd.notna(t) and pd.notna(s) else np.nan

    pel = {
        "tilt_rom": rng("pelvis_tilt"),
        "tilt_change": top_minus_start("pelvis_tilt"),
        "list_rom": rng("pelvis_list"),
        "rot_rom": rng("pelvis_rotation"),
    }
    shift = top_minus_start("pelvis_tz")          # 左右方向（+ = 右）
    pel["shift_mm"] = shift * 1000 if pd.notna(shift) else np.nan
    pel["shift_side"] = None if pd.isna(shift) else ("R" if shift > 0 else "L")
    rise = top_minus_start("pelvis_ty")           # 上下方向（+ = 上）
    pel["rise_mm"] = rise * 1000 if pd.notna(rise) else np.nan
    pel["unstable_phases"] = [
        ph for ph in ("Start", "Top")
        if any(pd.notna(_phase_value(phase_summary_df, v, ph, "Std"))
               and _phase_value(phase_summary_df, v, ph, "Std") > th["static_std_deg"]
               for v in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"))
    ]
    d["pelvis"] = pel

    # ---------------- 腰部 ----------------
    lum = {}
    if "lumbar_extension" in df_phase.columns:
        le = df_phase["lumbar_extension"]
        base = _phase_mean(df_phase, "lumbar_extension", "Start")
        lum["rom"] = float(le.max() - le.min())
        if pd.notna(base):
            up = le[df_phase["Phase"].isin(["Raising", "Top"])]
            lum["max_ext_change"] = float(up.max() - base) if len(up) else 0.0
            lum["top_change"] = top_minus_start("lumbar_extension")
        else:
            lum["max_ext_change"], lum["top_change"] = 0.0, np.nan
        ph_rom = {ph: _phase_value(phase_summary_df, "lumbar_extension", ph, "ROM") for ph in phase_order}
        ph_rom = {k: v for k, v in ph_rom.items() if pd.notna(v)}
        lum["main_phase"] = max(ph_rom, key=ph_rom.get) if ph_rom else None
    d["lumbar"] = lum

    # ---------------- 反復全体（テンポ） ----------------
    if reps:
        raise_t = np.array([p["raise_s"] for p in reps])
        lower_t = np.array([p["lower_s"] for p in reps])
        d["raise_mean_s"] = float(raise_t.mean())
        d["lower_mean_s"] = float(lower_t.mean())
        d["tempo_ratio"] = float(raise_t.mean() / lower_t.mean()) if lower_t.mean() else np.nan
    return d


def _flag(value, key):
    return pd.notna(value) and value > THRESHOLDS[key]


# ===============================================================
# 2. 臨床向けコメント（PDF Report）
# ===============================================================
def generate_arm_flexion_auto_comment(lang_code, overall_score, details, feature_values=None):
    """専門職向けの詳細コメントを、部位ごとの見出し付きで生成する。"""
    th = THRESHOLDS
    N = NAMES[lang_code]
    ja = lang_code == "ja"
    out = []

    def head(ja_t, en_t):
        out.append("")
        out.append(f"【{ja_t}】" if ja else f"[{en_t}]")

    out.append("【総合】" if ja else "[Overall]")
    if ja:
        lvl = "動作全体は良好で、大きな問題は見られません" if overall_score >= 80 else (
            "一部に改善の余地があります" if overall_score >= 60 else "複数の注意すべき所見があります")
        out.append(f"総合スコアは{overall_score}/100で、{lvl}。解析した挙上の回数は{details['n_reps']}回です。")
    else:
        lvl = "indicating generally good performance" if overall_score >= 80 else (
            "with some room for improvement" if overall_score >= 60 else "with several findings that warrant attention")
        out.append(f"Overall score: {overall_score}/100, {lvl}. {details['n_reps']} arm-raise repetitions were analysed.")

    # ---------------- 肩関節 ----------------
    j = details["joints"].get("shoulder")
    if j:
        head("肩関節", "Shoulder")
        if ja:
            line = (f"最大屈曲は右{j['peak_R']:.1f}°／左{j['peak_L']:.1f}°、"
                    f"可動域は右{j['rom_R']:.1f}°／左{j['rom_L']:.1f}°。")
        else:
            line = (f"Peak flexion R {j['peak_R']:.1f}° / L {j['peak_L']:.1f}°; "
                    f"ROM R {j['rom_R']:.1f}° / L {j['rom_L']:.1f}°.")
        below = [N[s] for s in ("R", "L") if j.get(f"below_{s}")]
        above = [N[s] for s in ("R", "L") if j.get(f"above_{s}")]
        if ja:
            if below:
                line += f"{'・'.join(below)}で正常範囲を下回っており、肩関節屈曲の可動域制限が疑われます。"
            elif above:
                line += f"{'・'.join(above)}で正常範囲を上回っており、体幹の反りなどで角度を補っている可能性があります。"
            else:
                line += "左右とも正常範囲内です。"
        else:
            if below:
                line += f" Below the normal range on the {' and '.join(below)} side, suggesting restricted shoulder flexion."
            elif above:
                line += f" Above the normal range on the {' and '.join(above)} side, possibly boosted by trunk extension."
            else:
                line += " Both sides within the normal range."
        out.append(line)

        if j["asym_pct"] > th["asym_pct"]:
            out.append(f"左右差は{j['asym_pct']:.1f}%でしきい値（{th['asym_pct']:.0f}%）を超え、"
                       f"{N[j['larger_side']]}側の可動域が大きくなっています。" if ja else
                       f"Left-right asymmetry is {j['asym_pct']:.1f}% (above {th['asym_pct']:.0f}%), "
                       f"with greater ROM on the {N[j['larger_side']]} side.")
        elif j.get("worst_phase") and j.get("worst_phase_asym", 0) > th["asym_pct"]:
            out.append(f"試技全体の左右差は{j['asym_pct']:.1f}%と小さいものの、"
                       f"{N[j['worst_phase']]}局面に限ると{j['worst_phase_asym']:.1f}%の差があります。" if ja else
                       f"Overall asymmetry is small ({j['asym_pct']:.1f}%), but reaches {j['worst_phase_asym']:.1f}% "
                       f"in the {N[j['worst_phase']]} phase.")
        else:
            out.append(f"左右差は{j['asym_pct']:.1f}%で、しきい値内です。" if ja else
                       f"Left-right asymmetry is {j['asym_pct']:.1f}%, within threshold.")

        if j.get("timing_lag_ms", 0) > th["timing_lag_ms"]:
            out.append(f"挙上が半分まで進む時刻に左右で平均{j['timing_lag_ms']:.0f}msのずれがあり、"
                       "左右の協調性（タイミング）に差がある可能性があります。" if ja else
                       f"Mean left-right lag at mid-raise is {j['timing_lag_ms']:.0f} ms, "
                       "suggesting a possible bilateral coordination difference.")

        if "rep_cv_pct" in j:
            if j["rep_cv_pct"] > th["rep_cv_pct"]:
                out.append(f"反復ごとの最大挙上角度の変動係数は{j['rep_cv_pct']:.1f}%と大きく、動作の再現性にばらつきがあります。"
                           if ja else f"Rep-to-rep variability of peak elevation is high (CV {j['rep_cv_pct']:.1f}%).")
            if abs(j.get("rep_trend_deg", 0)) > th["rep_trend_deg"]:
                trend = j["rep_trend_deg"]
                out.append((f"前半から後半にかけて最大挙上角度が{abs(trend):.1f}°{'増加' if trend > 0 else '減少'}しており、"
                            f"{'慣れ・可動域の拡大' if trend > 0 else '疲労や挙上の浅化'}の影響が考えられます。") if ja else
                           (f"Peak elevation {'increased' if trend > 0 else 'decreased'} by {abs(trend):.1f}° from early to late reps, "
                            f"possibly reflecting {'warm-up / increasing ROM' if trend > 0 else 'fatigue or lower raises'}."))

        if j["ecc_con_pct"] > th["ecc_con_pct"]:
            out.append(f"挙上{j['raise_rom']:.1f}°／下降{j['lower_rom']:.1f}°で{j['ecc_con_pct']:.1f}%の差があり、"
                       "求心性（挙上）と遠心性（下降）の制御に違いがある可能性があります。" if ja else
                       f"Raising {j['raise_rom']:.1f}° vs Lowering {j['lower_rom']:.1f}° ({j['ecc_con_pct']:.1f}% difference), "
                       "suggesting differing concentric/eccentric control.")

        if j["unstable_phases"]:
            ph = "・".join(N[p] for p in j["unstable_phases"]) if ja else ", ".join(N[p] for p in j["unstable_phases"])
            out.append(f"{ph}局面で角度の標準偏差が大きく、保持中の動揺が見られます。" if ja else
                       f"Elevated SD during {ph}, indicating sway while holding the position.")

    # ---------------- 肘・腕の開き・体幹側屈 ----------------
    ex = details.get("arm_extra", {})
    ex_lines = []
    if pd.notna(ex.get("elbow_bend_deg", np.nan)):
        v = ex["elbow_bend_deg"]
        if v > th["elbow_bend_deg"]:
            ex_lines.append(f"最大挙上時に肘が開始位置より{v:.1f}°曲がっており、肘を曲げて挙上を補っている可能性があります。"
                            if ja else f"The elbow flexed {v:.1f}° more at the top than at start, possibly compensating for limited elevation.")
    if pd.notna(ex.get("arm_open_deg", np.nan)):
        v = ex["arm_open_deg"]
        if v > th["arm_add_change_deg"]:
            ex_lines.append(f"挙上中に腕が外側へ{v:.1f}°開いており、真正面への挙上から外れています（肩甲骨・肩関節の制限の可能性）。"
                            if ja else f"The arm drifted {v:.1f}° outward while raising, deviating from a straight forward raise.")
    if _flag(ex.get("trunk_bend_rom", np.nan), "trunk_bend_deg"):
        ex_lines.append(f"体幹の側屈が{ex['trunk_bend_rom']:.1f}°見られ、左右どちらかに体を傾けて挙上している可能性があります。"
                        if ja else f"Trunk lateral bending of {ex['trunk_bend_rom']:.1f}° suggests leaning to one side while raising.")
    if ex_lines:
        head("挙上中の代償（肘・腕・体幹）", "Compensation During the Raise (Elbow / Arm / Trunk)")
        out.extend(ex_lines)

    # ---------------- 腰部 ----------------
    lum = details.get("lumbar", {})
    if lum:
        head("腰部", "Lumbar")
        ext = lum.get("max_ext_change", 0.0)
        tc = lum.get("top_change", np.nan)
        if ja:
            txt = f"腰椎伸展の可動範囲は{lum['rom']:.1f}°。"
            if ext >= th["lumbar_note_deg"]:
                txt += f"腕を挙げる局面で開始姿勢より最大{ext:.1f}°反り"
                txt += f"、最大挙上時は開始姿勢から{tc:+.1f}°の位置です。" if pd.notna(tc) else "ました。"
            else:
                txt += "腕を挙げても腰の反りはほとんど増えていません。"
            out.append(txt)
        else:
            txt = f"Lumbar extension range {lum['rom']:.1f}°. "
            if ext >= th["lumbar_note_deg"]:
                txt += f"Up to {ext:.1f}° more extension than at start while raising"
                txt += f"; {tc:+.1f}° at the top vs start." if pd.notna(tc) else "."
            else:
                txt += "Little additional extension while raising the arms."
            out.append(txt)
        if ext > th["lumbar_change_deg"]:
            out.append("腕を挙げる際に腰を反らせて角度を補う代償（腰椎過伸展）が見られ、肩関節・胸椎の可動性低下が疑われます。"
                       if ja else "Lumbar hyperextension is used to complete the raise, suggesting limited shoulder / thoracic mobility.")
        else:
            out.append("開始姿勢からの反りはしきい値内で、腰部の代償は少ない状態です。" if ja else
                       "Extension change is within threshold; minimal lumbar compensation.")
        if lum.get("main_phase"):
            out.append(f"腰部の動きが最も大きいのは{N[lum['main_phase']]}局面です。" if ja else
                       f"Lumbar motion is greatest during the {N[lum['main_phase']]} phase.")

    # ---------------- 骨盤 ----------------
    p = details.get("pelvis", {})
    if p:
        head("骨盤", "Pelvis")
        items = []
        if pd.notna(p["tilt_rom"]):
            items.append((f"前後傾の可動範囲{p['tilt_rom']:.1f}°（最大挙上で開始位置から{p['tilt_change']:+.1f}°）" if ja else
                          f"tilt range {p['tilt_rom']:.1f}° ({p['tilt_change']:+.1f}° at top vs start)",
                          _flag(p["tilt_rom"], "pelvis_rom_deg")))
        if pd.notna(p["list_rom"]):
            items.append((f"側方傾斜{p['list_rom']:.1f}°" if ja else f"obliquity range {p['list_rom']:.1f}°",
                          _flag(p["list_rom"], "pelvis_list_deg")))
        if pd.notna(p["rot_rom"]):
            items.append((f"回旋{p['rot_rom']:.1f}°" if ja else f"rotation range {p['rot_rom']:.1f}°",
                          _flag(p["rot_rom"], "pelvis_rom_deg")))
        if pd.notna(p["shift_mm"]):
            side_txt = (N[p["shift_side"]] + "へ") if ja else f"to the {N[p['shift_side']]}"
            items.append((f"最大挙上時の左右移動{abs(p['shift_mm']):.0f}mm（{side_txt}）" if ja else
                          f"lateral shift at top {abs(p['shift_mm']):.0f} mm ({side_txt})",
                          abs(p["shift_mm"]) > th["pelvis_shift_mm"]))
        if pd.notna(p["rise_mm"]):
            items.append((f"最大挙上時の上下移動{p['rise_mm']:+.0f}mm" if ja else f"vertical change at top {p['rise_mm']:+.0f} mm",
                          p["rise_mm"] > th["pelvis_rise_mm"]))
        if items:
            sep = "、" if ja else ", "
            out.append(("計測値：" if ja else "Measured: ") + sep.join(t for t, _ in items) + ("。" if ja else "."))
            flagged = [t for t, f in items if f]
            if flagged:
                out.append(("しきい値を超えた項目：" + "、".join(t.split("（")[0] for t in flagged) +
                            "。腕の挙上に伴って骨盤が代償的に動いている可能性があります。") if ja else
                           ("Above threshold: " + ", ".join(t.split(" (")[0] for t in flagged) +
                            ". The pelvis may be moving to compensate for the arm raise."))
            else:
                out.append("いずれもしきい値内で、骨盤は比較的安定しています。" if ja else
                           "All within thresholds; the pelvis remained relatively stable.")
            if pd.notna(p["rise_mm"]) and p["rise_mm"] > th["pelvis_rise_mm"]:
                out.append("最大挙上時に骨盤が上昇しており、つま先立ちや伸び上がりで挙上を補っている可能性があります。" if ja else
                           "The pelvis rose at the top, possibly rising onto the toes to complete the raise.")
        if p["unstable_phases"]:
            ph = "・".join(N[x] for x in p["unstable_phases"]) if ja else ", ".join(N[x] for x in p["unstable_phases"])
            out.append(f"{ph}局面で骨盤角度の標準偏差が大きく、保持中の骨盤動揺が見られます。" if ja else
                       f"Elevated pelvic SD during {ph}, indicating pelvic sway while holding.")

    # ---------------- テンポ ----------------
    if details["n_reps"]:
        head("反復のテンポ", "Tempo")
        out.append(f"挙上{details['raise_mean_s']:.2f}秒／下降{details['lower_mean_s']:.2f}秒です。" if ja else
                   f"Raising {details['raise_mean_s']:.2f} s / lowering {details['lower_mean_s']:.2f} s.")
        tr = details.get("tempo_ratio")
        if pd.notna(tr) and tr > th["tempo_ratio"]:
            out.append("下降が挙上に比べて速く、腕を下ろす局面の遠心性コントロールが不十分な可能性があります。" if ja else
                       "Lowering is notably faster than raising, suggesting limited eccentric control.")
        elif pd.notna(tr) and tr < 1 / th["tempo_ratio"]:
            out.append("挙上が下降に比べて速く、反動を使って腕を振り上げている可能性があります。" if ja else
                       "Raising is notably faster than lowering, possibly swinging the arms up with momentum.")

    out.append("")
    out.append("※ 実測値からの自動生成による下書きです。しきい値は仮の基準のため、臨床所見・触診所見と合わせて"
               "内容を確認し、必要に応じて修正してください。" if ja else
               "* Auto-generated draft from measured values. Thresholds are provisional; please review alongside "
               "clinical examination findings and edit as needed.")
    return "\n".join(out)


# ===============================================================
# 3. クライアント向けコメント（Client Report）
# ===============================================================
def generate_client_auto_comment(lang_code, overall_score, details,
                                 mobility_score, symmetry_score, lumbar_score, pelvis_score):
    """専門用語を減らし、「良い点」と「気をつけたい点」を平易な言葉で伝える。"""
    th = THRESHOLDS
    ja = lang_code == "ja"
    side = {"R": "右", "L": "左"} if ja else {"R": "right", "L": "left"}
    good, care = [], []

    if ja:
        head = (f"今回の総合スコアは{overall_score:.0f}/100で、" +
                ("とても良い状態です。" if overall_score >= 80 else
                 "全体的には悪くありませんが、いくつか気をつけたい点があります。" if overall_score >= 60 else
                 "いくつか改善していきたいポイントが見つかりました。"))
    else:
        head = (f"Your overall score was {overall_score:.0f}/100 — " +
                ("a great result." if overall_score >= 80 else
                 "reasonably good, with a few points to keep an eye on." if overall_score >= 60 else
                 "we found a few areas worth working on."))

    j = details["joints"].get("shoulder")
    if j:
        below = [s for s in ("R", "L") if j.get(f"below_{s}")]
        if below:
            if len(below) == 2:
                care.append("両腕とも、目安の高さまで少し挙がりきっていません。" if ja else
                            "Both arms don't quite reach the typical height.")
            else:
                care.append(f"{side[below[0]]}腕が、目安の高さまで少し挙がりきっていません。" if ja else
                            f"Your {side[below[0]]} arm doesn't quite reach the typical height.")
        else:
            good.append("腕をしっかり高く挙げられています。" if ja else "You raised your arms to a good height.")
        if j["asym_pct"] > th["asym_pct"]:
            care.append(f"{side[j['larger_side']]}腕の方が大きく動いていて、左右で挙げやすさに差があります。" if ja else
                        f"Your {side[j['larger_side']]} arm moves more — the two sides differ in how easily they rise.")
        else:
            good.append("左右の腕をバランスよく挙げられています。" if ja else "Both arms rose evenly.")
        if j.get("rep_trend_deg", 0) < -th["rep_trend_deg"]:
            care.append("回数を重ねるにつれて、腕の挙がる高さが低くなっていました（疲れのサインかもしれません）。" if ja else
                        "Your arms rose less as the reps went on — possibly a sign of fatigue.")

    lum = details.get("lumbar", {})
    if lum:
        if lum.get("max_ext_change", 0) > th["lumbar_change_deg"]:
            care.append("腕を挙げるときに腰を反らせて高さを補う傾向があり、腰まわりに負担がかかりやすい状態です。" if ja else
                        "You tend to arch your lower back to get your arms higher, which can strain the area.")
        else:
            good.append("腕を挙げても腰が反らず、よい姿勢を保てています。" if ja else
                        "Your lower back stayed in a good position as you raised your arms.")

    p = details.get("pelvis", {})
    if p:
        pel_flags = [_flag(p.get("tilt_rom"), "pelvis_rom_deg"),
                     _flag(p.get("list_rom"), "pelvis_list_deg"),
                     _flag(p.get("rot_rom"), "pelvis_rom_deg")]
        shift = pd.notna(p.get("shift_mm")) and abs(p["shift_mm"]) > th["pelvis_shift_mm"]
        rise = _flag(p.get("rise_mm"), "pelvis_rise_mm")
        if rise:
            care.append("腕を挙げきるときに、つま先立ちや伸び上がりで高さを補っている様子があります。" if ja else
                        "You seem to rise onto your toes or stretch up to get your arms higher.")
        if shift:
            care.append(f"腕を挙げたときに、骨盤（腰の土台）が{side[p['shift_side']]}に少しずれています。" if ja else
                        f"Your pelvis shifts slightly to the {side[p['shift_side']]} when your arms are up.")
        if any(pel_flags):
            care.append("腕を挙げ下げする途中で、骨盤が傾いたりねじれたりしやすい傾向があります。" if ja else
                        "Your pelvis tends to tilt or twist as you raise and lower your arms.")
        if not (rise or shift or any(pel_flags)):
            good.append("骨盤（腰の土台）は安定していました。" if ja else "Your pelvis stayed stable.")

    ex = details.get("arm_extra", {})
    if ex.get("elbow_bend_deg", 0) > th["elbow_bend_deg"]:
        care.append("腕を挙げたときに肘が曲がりやすい傾向があります。" if ja else
                    "Your elbows tend to bend as you raise your arms.")
    if ex.get("arm_open_deg", 0) > th["arm_add_change_deg"]:
        care.append("腕を挙げる途中で、腕が外側に開きやすい傾向があります。" if ja else
                    "Your arms tend to drift outward as you raise them.")
    if _flag(ex.get("trunk_bend_rom", np.nan), "trunk_bend_deg"):
        care.append("腕を挙げるときに、体が左右どちらかに傾きやすい傾向があります。" if ja else
                    "Your body tends to lean to one side as you raise your arms.")

    tr = details.get("tempo_ratio")
    if pd.notna(tr) and tr > th["tempo_ratio"]:
        care.append("腕を下ろすスピードが速めです。ゆっくり下ろすと、肩まわりの筋力づくりにもなります。" if ja else
                    "You lower your arms quite fast — lowering slowly also builds shoulder strength.")

    lines = [head]
    sep = "" if ja else " "
    if good:
        lines.append(("◎ よくできている点：" if ja else "Strengths: ") + sep.join(good[:3]))
    if care:
        lines.append(("△ 気をつけたい点：" if ja else "Points to watch: ") + sep.join(care[:4]))
    lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。" if ja else
                 "Try the recommended actions below at a comfortable pace before your next check.")
    return "\n".join(lines)
