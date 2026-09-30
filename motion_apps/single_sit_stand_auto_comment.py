"""
single_sit_stand_auto_comment.py
---------------------------------------------------------------
Single-Leg Sit-to-Stand（片脚立ち座り）Analysis の自動コメント
（PDF Report / Client Report）用モジュール。
squat_auto_comment.py / sit_stand_auto_comment.py と同じ構成・同じ呼び出し方にそろえています。

実測データ（df_phase）から、支持脚（解析側）の股関節・膝関節・足関節、
骨盤・体幹について以下を読み取り、臨床向け／クライアント向けのコメント下書きを生成します。

  - 支持脚の最大角度・可動域と正常範囲との比較
  - 反復ごとの一貫性（ピーク角度のばらつき・前半→後半の変化）
  - 着座（Lowering）と立ち上がり（Rising）の可動域差（遠心性／求心性の制御差）
  - 体幹の前傾（立ち上がりで股関節が座位から追加で曲がる量）
  - 膝が内側に入る動き（Knee-in）の目安：股関節内転・内旋の増加量
    （hip_adduction_* / hip_rotation_* の列がある場合のみ）
  - 骨盤：反対側（浮かせている脚の側）への下制（pelvis_list）、
          前後傾・回旋、左右方向の揺れ（pelvis_tz）
  - 体幹の側屈（lumbar_bending がある場合）
  - 腰部：立位からの反り・立ち上がり中の丸まり
  - 反復全体：立ち上がりの高さ・テンポ（立ち上がり／着座時間）

■ 座標系（OpenSim / OpenCap）
  - pelvis_tx：前後（+ = 前）、pelvis_ty：上下（+ = 上）、pelvis_tz：左右（+ = 右）
  - pelvis_list：+ で右側が下がる（左側が上がる）向きとして扱っています。
    符号が逆に見える場合は PELVIS_LIST_RIGHT_DOWN_POSITIVE を False にしてください。

※ 本モジュール内のしきい値はすべて「仮の基準」です。
  THRESHOLDS を臨床基準に合わせて調整してください。
"""

import numpy as np
import pandas as pd

FS = 60  # サンプリング周波数（Hz）

# pelvis_list の符号：+ = 右側が下がる（OpenSim の右手系で X 軸まわり正回転）
PELVIS_LIST_RIGHT_DOWN_POSITIVE = True

# ---------------------------------------------------------------
# しきい値（仮）— 臨床基準に合わせて調整してください
# ---------------------------------------------------------------
THRESHOLDS = {
    "ecc_con_pct": 20.0,        # 着座／立ち上がりのROM差（%）
    "rep_cv_pct": 8.0,          # 反復ごとのピーク角度の変動係数（%）
    "rep_trend_deg": 8.0,       # 前半→後半のピーク角度の変化（°）
    "static_std_deg": 2.0,      # 静止局面（座位・立位）の標準偏差（°）
    "lean_low_deg": 10.0,       # 立ち上がり時の体幹前傾がこれ未満 → 前傾不足
    "lean_high_deg": 45.0,      # これ超 → 前傾が大きすぎる
    "ankle_df_limited": 10.0,   # 足関節背屈の最大値がこれ未満 → 制限の可能性
    "knee_in_add_deg": 8.0,     # 股関節内転の増加量（立位比）→ Knee-in の目安
    "knee_in_rot_deg": 10.0,    # 股関節内旋の増加量（立位比）→ Knee-in の目安
    "pelvic_drop_deg": 5.0,     # 反対側への骨盤下制（立位比）
    "pelvis_rom_deg": 10.0,     # 骨盤（前後傾・回旋）の可動範囲（°）
    "pelvis_sway_mm": 50.0,     # 動作中の骨盤の左右の揺れ幅（mm）
    "trunk_side_deg": 10.0,     # 体幹側屈の可動範囲（°）
    "lumbar_change_deg": 10.0,  # 腰部の立位からの変化（°）
    "lumbar_note_deg": 2.0,     # 腰部の反り／丸まりを文中で言及する最小量
    "tempo_ratio": 1.5,         # 着座時間／立ち上がり時間の比（逆数未満で「速く座る」）
}

NAMES = {
    "ja": {"hip": "股関節", "knee": "膝関節", "ankle": "足関節",
           "Right": "右", "Left": "左",
           "Sitting": "座位", "Rising": "立ち上がり", "Standing": "立位", "Lowering": "着座"},
    "en": {"hip": "Hip", "knee": "Knee", "ankle": "Ankle",
           "Right": "right", "Left": "left",
           "Sitting": "Seated", "Rising": "Rising", "Standing": "Standing", "Lowering": "Sitting down"},
}
CLIENT_NAMES_JA = {"hip": "股関節", "knee": "ひざ", "ankle": "足首"}
CLIENT_NAMES_EN = {"hip": "hip", "knee": "knee", "ankle": "ankle"}


# ===============================================================
# 1. データの読み取り
# ===============================================================
def _pct_diff(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m * 100


def _nanmean(values):
    arr = np.array([v for v in values if pd.notna(v)], dtype=float)
    return float(arr.mean()) if len(arr) else np.nan


def _detect_reps(pelvis_ty, fs=FS):
    """骨盤の高さ（pelvis_ty）の極大点（立位）から反復を検出し、各反復の区間を返す。"""
    s = pd.Series(pelvis_ty).rolling(5, center=True).mean().bfill().ffill().to_numpy()
    lo, hi = np.nanmin(s), np.nanmax(s)
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return [], s

    stand_line = lo + rng * 0.6
    min_gap = int(1.0 * fs)
    cand = [i for i in range(1, len(s) - 1)
            if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > stand_line]
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

    seat_line = lo + rng * 0.10
    top_line = hi - rng * 0.10
    reps = []
    for k, m in enumerate(maxima):
        start, end = bounds[k], bounds[k + 1]
        seg = s[start:end + 1]
        pre_seat = np.where(s[start:m + 1] <= seat_line)[0]
        rise_start = start + pre_seat[-1] if len(pre_seat) else start
        reach = np.where(s[rise_start:m + 1] >= top_line)[0]
        rise_end = rise_start + reach[0] if len(reach) else m
        post_seat = np.where(s[m:end + 1] <= seat_line)[0]
        sit_end = m + post_seat[0] if len(post_seat) else end
        leave = np.where(s[m:sit_end + 1] >= top_line)[0]
        sit_start = m + leave[-1] if len(leave) else m
        reps.append({
            "start": start, "end": end, "top": m,
            "rise_start": rise_start, "rise_end": rise_end,
            "ascent_s": (rise_end - rise_start) / fs,
            "descent_s": (sit_end - sit_start) / fs,
            "height_m": s[m] - seg.min(),
        })
    return reps, s


def _phase_value(phase_summary_df, var, phase, stat):
    row = phase_summary_df[phase_summary_df["Variable"] == var]
    if len(row) == 0:
        return np.nan
    col = f"{phase}_{stat}"
    return row[col].iloc[0] if col in row.columns else np.nan


def _side_code(analysis_side):
    s = str(analysis_side).strip().lower()
    return "l" if s.startswith("l") or s.startswith("左") else "r"


def analyze_single_sit_stand_details(df_phase, phase_summary_df, comparison_df,
                                     phase_order=("Sitting", "Rising", "Standing", "Lowering"),
                                     analysis_side="Right", fs=FS):
    """実測データから、支持脚・骨盤・体幹ごとの詳細な特徴量をまとめて返す。"""
    th = THRESHOLDS
    sc = _side_code(analysis_side)
    side_name = "Left" if sc == "l" else "Right"
    joints = {
        "hip": f"hip_flexion_{sc}",
        "knee": f"knee_angle_{sc}",
        "ankle": f"ankle_angle_{sc}",
    }
    phase = df_phase["Phase"] if "Phase" in df_phase.columns else pd.Series([None] * len(df_phase))
    reps, _ = (_detect_reps(df_phase["pelvis_ty"].to_numpy(), fs)
               if "pelvis_ty" in df_phase.columns else ([], None))
    d = {"side": side_name, "side_code": sc, "reps": reps, "n_reps": len(reps), "joints": {}}

    def mean_in(col, ph):
        if col not in df_phase.columns:
            return np.nan
        v = df_phase.loc[phase == ph, col]
        return float(v.mean()) if len(v) else np.nan

    def moving_mask():
        return phase.isin(["Rising", "Lowering"])

    # ---------------- 支持脚の関節 ----------------
    for key, var in joints.items():
        if var not in df_phase.columns:
            continue
        x = df_phase[var].to_numpy(dtype=float)
        j = {"var": var,
             "peak": float(np.nanmax(x)),
             "rom": float(np.nanmax(x) - np.nanmin(x))}
        row = comparison_df[comparison_df["Variable"] == var]
        if len(row):
            rom = row["Subject_ROM"].iloc[0]
            j["rom_cmp"] = float(rom)
            j["healthy_min"] = float(row["Healthy_Min"].iloc[0])
            j["healthy_max"] = float(row["Healthy_Max"].iloc[0])
            j["below"] = bool(rom < row["Healthy_Min"].iloc[0])
            j["above"] = bool(rom > row["Healthy_Max"].iloc[0])

        desc = _phase_value(phase_summary_df, var, "Lowering", "ROM")
        asc = _phase_value(phase_summary_df, var, "Rising", "ROM")
        j["desc_rom"], j["asc_rom"] = desc, asc
        j["ecc_con_pct"] = _pct_diff(desc, asc) if pd.notna(desc) and pd.notna(asc) else 0.0

        j["unstable_phases"] = [
            ph for ph in ("Sitting", "Standing")
            if pd.notna(_phase_value(phase_summary_df, var, ph, "Std"))
            and _phase_value(phase_summary_df, var, ph, "Std") > th["static_std_deg"]
        ]

        if reps:
            pk = np.array([np.nanmax(x[p["start"]:p["end"] + 1]) for p in reps])
            j["rep_peaks"] = pk.tolist()
            j["rep_cv_pct"] = float(np.std(pk) / np.mean(pk) * 100) if np.mean(pk) else 0.0
            half = max(1, len(pk) // 2)
            j["rep_trend_deg"] = float(np.mean(pk[-half:]) - np.mean(pk[:half])) if len(pk) >= 2 else 0.0
        d["joints"][key] = j

    # ---------------- 立ち上がり時の体幹前傾 ----------------
    hip_var = joints["hip"]
    d["forward_lean_deg"], d["lean_pattern"] = np.nan, None
    if hip_var in df_phase.columns:
        seated = mean_in(hip_var, "Sitting")
        rising = df_phase.loc[phase == "Rising", hip_var]
        if len(rising) and pd.notna(seated):
            lean = float(max(0.0, rising.max() - seated))
            d["forward_lean_deg"] = lean
            d["lean_pattern"] = ("low" if lean < th["lean_low_deg"] else
                                 "high" if lean > th["lean_high_deg"] else "normal")

    # ---------------- Knee-in（股関節の内転・内旋） ----------------
    kin = {}
    for name, col, thr in (("add", f"hip_adduction_{sc}", th["knee_in_add_deg"]),
                           ("rot", f"hip_rotation_{sc}", th["knee_in_rot_deg"])):
        if col not in df_phase.columns:
            continue
        base = mean_in(col, "Standing")
        mv = df_phase.loc[moving_mask(), col]
        if pd.isna(base) or len(mv) == 0:
            continue
        inc = float(mv.max() - base)   # + = 内転／内旋が増えた量
        kin[f"{name}_inc"] = inc
        kin[f"{name}_flag"] = inc > thr
    kin["available"] = bool(kin)
    kin["flag"] = bool(kin.get("add_flag") or kin.get("rot_flag"))
    d["knee_in"] = kin

    # ---------------- 骨盤 ----------------
    pel = {}

    def rng(col, mask=None):
        if col not in df_phase.columns:
            return np.nan
        v = df_phase[col] if mask is None else df_phase.loc[mask, col]
        return float(v.max() - v.min()) if len(v) else np.nan

    pel["tilt_rom"] = rng("pelvis_tilt")
    pel["rot_rom"] = rng("pelvis_rotation")
    pel["list_rom"] = rng("pelvis_list")

    # 反対側（浮かせている脚の側）への骨盤下制
    pel["drop_deg"] = np.nan
    if "pelvis_list" in df_phase.columns:
        base = mean_in("pelvis_list", "Standing")
        mv = df_phase.loc[moving_mask(), "pelvis_list"]
        if pd.notna(base) and len(mv):
            dev = mv - base
            # 右支持なら「左が下がる」＝右が上がる向き
            right_down_sign = 1.0 if PELVIS_LIST_RIGHT_DOWN_POSITIVE else -1.0
            contra_down_sign = -right_down_sign if sc == "r" else right_down_sign
            pel["drop_deg"] = float(max(0.0, (dev * contra_down_sign).max()))
    pel["drop_flag"] = pd.notna(pel["drop_deg"]) and pel["drop_deg"] > th["pelvic_drop_deg"]

    # 左右の揺れ（pelvis_tz：+ = 右）
    pel["sway_mm"] = rng("pelvis_tz", moving_mask()) * 1000 if "pelvis_tz" in df_phase.columns else np.nan
    if "pelvis_tz" in df_phase.columns:
        s_ = mean_in("pelvis_tz", "Standing")
        b_ = mean_in("pelvis_tz", "Sitting")
        shift = (s_ - b_) if pd.notna(s_) and pd.notna(b_) else np.nan
        pel["shift_mm"] = shift * 1000 if pd.notna(shift) else np.nan
        # 支持脚側へ寄っているか（右支持なら + 方向）
        pel["shift_toward_stance"] = (None if pd.isna(shift) else
                                      (shift > 0) if sc == "r" else (shift < 0))
    else:
        pel["shift_mm"], pel["shift_toward_stance"] = np.nan, None
    pel["sway_flag"] = pd.notna(pel["sway_mm"]) and pel["sway_mm"] > th["pelvis_sway_mm"]

    # 前後方向（pelvis_tx）：参考値
    if "pelvis_tx" in df_phase.columns:
        f_ = mean_in("pelvis_tx", "Standing") - mean_in("pelvis_tx", "Sitting")
        pel["forward_mm"] = f_ * 1000 if pd.notna(f_) else np.nan
    else:
        pel["forward_mm"] = np.nan

    pel["unstable_phases"] = [
        ph for ph in ("Sitting", "Standing")
        if any(pd.notna(_phase_value(phase_summary_df, v, ph, "Std"))
               and _phase_value(phase_summary_df, v, ph, "Std") > th["static_std_deg"]
               for v in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"))
    ]
    d["pelvis"] = pel

    # ---------------- 体幹の側屈 ----------------
    d["trunk_side_rom"] = rng("lumbar_bending", moving_mask()) if "lumbar_bending" in df_phase.columns else np.nan

    # ---------------- 腰部 ----------------
    lum = {}
    if "lumbar_extension" in df_phase.columns:
        le = df_phase["lumbar_extension"]
        base = mean_in("lumbar_extension", "Standing")
        seated = mean_in("lumbar_extension", "Sitting")
        rising = le[phase == "Rising"]
        lum["rom"] = float(le.max() - le.min())
        lum["max_ext_change"] = float((le - base).max()) if pd.notna(base) else 0.0
        lum["max_flex_change"] = (float(max(0.0, seated - rising.min()))
                                  if len(rising) and pd.notna(seated) else 0.0)
        lum["seated_change"] = float(seated - base) if pd.notna(seated) and pd.notna(base) else np.nan
        ph_rom = {ph: _phase_value(phase_summary_df, "lumbar_extension", ph, "ROM") for ph in phase_order}
        ph_rom = {k: v for k, v in ph_rom.items() if pd.notna(v)}
        lum["main_phase"] = max(ph_rom, key=ph_rom.get) if ph_rom else None
    d["lumbar"] = lum

    # ---------------- 反復全体（高さ・テンポ） ----------------
    if reps:
        heights = np.array([p["height_m"] for p in reps])
        asc_t = np.array([p["ascent_s"] for p in reps])
        desc_t = np.array([p["descent_s"] for p in reps])
        d["height_cv_pct"] = float(heights.std() / heights.mean() * 100) if heights.mean() else 0.0
        d["height_mean_cm"] = float(heights.mean() * 100)
        d["ascent_mean_s"] = float(asc_t.mean())
        d["descent_mean_s"] = float(desc_t.mean())
        d["tempo_ratio"] = float(desc_t.mean() / asc_t.mean()) if asc_t.mean() else np.nan
    return d


# ===============================================================
# 2. 臨床向けコメント（PDF Report）
# ===============================================================
def generate_single_sit_stand_auto_comment(lang_code, overall_score, details, feature_values=None):
    """専門職向けの詳細コメントを、部位ごとの見出し付きで生成する。"""
    th = THRESHOLDS
    N = NAMES[lang_code]
    J = details["joints"]
    ja = lang_code == "ja"
    side = N[details["side"]]
    out = []

    def head(ja_t, en_t):
        out.append("")
        out.append(f"【{ja_t}】" if ja else f"[{en_t}]")

    # ---------------- 総合 ----------------
    out.append("【総合】" if ja else "[Overall]")
    if ja:
        lvl = "動作全体は良好で、大きな問題は見られません" if overall_score >= 80 else (
            "一部に改善の余地があります" if overall_score >= 60 else "複数の注意すべき所見があります")
        out.append(f"総合スコアは{overall_score:.0f}/100で、{lvl}。"
                   f"解析側（支持脚）は{side}脚、解析した片脚立ち座りの回数は{details['n_reps']}回です。")
    else:
        lvl = "indicating generally good performance" if overall_score >= 80 else (
            "with some room for improvement" if overall_score >= 60 else "with several findings that warrant attention")
        out.append(f"Overall score: {overall_score:.0f}/100, {lvl}. Stance (analysed) leg: {side}; "
                   f"{details['n_reps']} single-leg sit-to-stand repetitions were analysed.")

    lean, pat = details.get("forward_lean_deg"), details.get("lean_pattern")
    if pat and pd.notna(lean):
        if ja:
            txt = {"low": "前傾が小さく、体幹前傾による重心移動を十分に使えていない（反動や下肢伸展筋への依存の可能性）",
                   "high": "前傾が大きく、体幹を大きく倒して立ち上がる（支持脚の筋力を体幹前傾で補っている可能性）",
                   "normal": "体幹前傾と支持脚の伸展がバランスよく使われた"}[pat]
            out.append(f"立ち上がり時の体幹前傾（座位からの股関節屈曲の追加量）は{lean:.1f}°で、{txt}立ち上がり方です。")
        else:
            txt = {"low": "limited forward lean, with little use of trunk momentum",
                   "high": "excessive forward lean (trunk flexion possibly compensating for stance-leg strength)",
                   "normal": "a balanced use of forward lean and stance-leg extension"}[pat]
            out.append(f"Forward trunk lean during rising (additional hip flexion from sitting) is {lean:.1f}°, suggesting {txt}.")

    # ---------------- 各関節（支持脚） ----------------
    for key, (ja_t, en_t) in {"hip": ("股関節", "Hip"), "knee": ("膝関節", "Knee"),
                              "ankle": ("足関節", "Ankle")}.items():
        if key not in J:
            continue
        j = J[key]
        head(f"{ja_t}（{side}・支持脚）", f"{en_t} ({side}, stance leg)")
        verb_ja = "背屈" if key == "ankle" else "屈曲"
        verb_en = "dorsiflexion" if key == "ankle" else "flexion"
        if ja:
            line = f"最大{verb_ja}は{j['peak']:.1f}°、可動域は{j['rom']:.1f}°。"
            if j.get("below"):
                line += f"正常範囲（{j['healthy_min']:.0f}〜{j['healthy_max']:.0f}°）を下回っており、可動域制限または支持脚の筋力不足で深く座れていない可能性があります。"
            elif j.get("above"):
                line += f"正常範囲（{j['healthy_min']:.0f}〜{j['healthy_max']:.0f}°）を上回っており、過可動または代償的な動きの可能性があります。"
            elif "healthy_min" in j:
                line += "正常範囲内です。"
        else:
            line = f"Peak {verb_en} {j['peak']:.1f}°, ROM {j['rom']:.1f}°."
            if j.get("below"):
                line += (f" Below the normal range ({j['healthy_min']:.0f}–{j['healthy_max']:.0f}°), suggesting restricted ROM "
                         "or insufficient stance-leg strength to sit deeply.")
            elif j.get("above"):
                line += (f" Above the normal range ({j['healthy_min']:.0f}–{j['healthy_max']:.0f}°), "
                         "suggesting hypermobility or compensation.")
            elif "healthy_min" in j:
                line += " Within the normal range."
        out.append(line)

        if "rep_cv_pct" in j:
            if j["rep_cv_pct"] > th["rep_cv_pct"]:
                out.append(f"反復ごとのピーク角度の変動係数は{j['rep_cv_pct']:.1f}%と大きく、動作の再現性にばらつきがあります。" if ja else
                           f"Rep-to-rep variability of peak angle is high (CV {j['rep_cv_pct']:.1f}%).")
            if abs(j.get("rep_trend_deg", 0)) > th["rep_trend_deg"]:
                trend = j["rep_trend_deg"]
                out.append((f"前半から後半にかけてピーク角度が{abs(trend):.1f}°{'増加' if trend > 0 else '減少'}しており、"
                            f"{'慣れ・座り方の変化' if trend > 0 else '疲労や座り込みの浅化'}の影響が考えられます。") if ja else
                           (f"Peak angle {'increased' if trend > 0 else 'decreased'} by {abs(trend):.1f}° from early to late reps, "
                            f"possibly reflecting {'adaptation' if trend > 0 else 'fatigue or shallower sitting'}."))

        if j["ecc_con_pct"] > th["ecc_con_pct"]:
            out.append(f"着座{j['desc_rom']:.1f}°／立ち上がり{j['asc_rom']:.1f}°で{j['ecc_con_pct']:.1f}%の差があり、"
                       "遠心性（着座）と求心性（立ち上がり）の制御に違いがある可能性があります。" if ja else
                       f"Sitting down {j['desc_rom']:.1f}° vs rising {j['asc_rom']:.1f}° ({j['ecc_con_pct']:.1f}% difference), "
                       "suggesting differing eccentric/concentric control.")

        if j["unstable_phases"]:
            ph = ("・" if ja else ", ").join(N[p] for p in j["unstable_phases"])
            out.append(f"{ph}局面で角度の標準偏差が大きく、保持中の動揺が見られます。" if ja else
                       f"Elevated SD during {ph}, indicating sway while holding the position.")

        if key == "ankle" and j["peak"] < th["ankle_df_limited"]:
            out.append(f"最大背屈が{j['peak']:.1f}°と小さく、足部を引き込めないことで体幹前傾の増加や"
                       "立ち上がりの困難さにつながっている可能性があります。" if ja else
                       f"Peak dorsiflexion is limited ({j['peak']:.1f}°), which may increase the trunk lean needed to rise.")

    # ---------------- Knee-in ----------------
    kin = details["knee_in"]
    if kin.get("available"):
        head("膝の内側への入り込み（Knee-in の目安）", "Knee Valgus (Dynamic Knee-in Indicator)")
        parts = []
        if "add_inc" in kin:
            parts.append(f"股関節内転 +{kin['add_inc']:.1f}°" if ja else f"hip adduction +{kin['add_inc']:.1f}°")
        if "rot_inc" in kin:
            parts.append(f"股関節内旋 +{kin['rot_inc']:.1f}°" if ja else f"hip internal rotation +{kin['rot_inc']:.1f}°")
        sep = "、" if ja else ", "
        out.append(("動作中の立位からの最大増加量：" if ja else "Maximum increase from standing during movement: ")
                   + sep.join(parts) + ("。" if ja else "."))
        if kin["flag"]:
            out.append("しきい値を超えており、支持脚の膝が内側に入る（動的外反）傾向が示唆されます。"
                       "中殿筋・大殿筋による股関節の外転／外旋コントロールの評価をおすすめします。" if ja else
                       "Above threshold, suggesting dynamic knee valgus of the stance leg. "
                       "Assessment of hip abductor/external rotator control (gluteus medius/maximus) is recommended.")
        else:
            out.append("しきい値内で、膝のアライメントは比較的保たれています。" if ja else
                       "Within threshold; knee alignment was relatively well maintained.")

    # ---------------- 骨盤 ----------------
    p = details["pelvis"]
    head("骨盤", "Pelvis")
    items = []
    if pd.notna(p["drop_deg"]):
        items.append((f"反対側への骨盤下制{p['drop_deg']:.1f}°" if ja else
                      f"contralateral pelvic drop {p['drop_deg']:.1f}°", p["drop_flag"]))
    if pd.notna(p["tilt_rom"]):
        items.append((f"前後傾の可動範囲{p['tilt_rom']:.1f}°" if ja else f"tilt range {p['tilt_rom']:.1f}°",
                      p["tilt_rom"] > th["pelvis_rom_deg"]))
    if pd.notna(p["rot_rom"]):
        items.append((f"回旋{p['rot_rom']:.1f}°" if ja else f"rotation range {p['rot_rom']:.1f}°",
                      p["rot_rom"] > th["pelvis_rom_deg"]))
    if pd.notna(p["sway_mm"]):
        items.append((f"動作中の左右の揺れ幅{p['sway_mm']:.0f}mm" if ja else f"medio-lateral sway {p['sway_mm']:.0f} mm",
                      p["sway_flag"]))
    if items:
        sep = "、" if ja else ", "
        out.append(("計測値：" if ja else "Measured: ") + sep.join(t for t, _ in items) + ("。" if ja else "."))
        flagged = [t for t, f in items if f]
        if flagged:
            out.append(("しきい値を超えた項目：" + "、".join(flagged) +
                        "。片脚支持での骨盤の制御が不十分で、代償的に骨盤が動いている可能性があります。") if ja else
                       ("Above threshold: " + ", ".join(flagged) +
                        ". Pelvic control in single-leg support may be insufficient."))
        else:
            out.append("いずれもしきい値内で、片脚支持でも骨盤は比較的安定しています。" if ja else
                       "All within thresholds; the pelvis remained relatively stable in single-leg support.")
        if p["drop_flag"]:
            out.append("反対側の骨盤が下がる所見は、支持脚側の股関節外転筋（中殿筋）の筋力・制御低下（トレンデレンブルグ様）を示唆します。" if ja else
                       "Contralateral pelvic drop suggests reduced stance-side hip abductor (gluteus medius) strength/control "
                       "(Trendelenburg-like pattern).")
    if pd.notna(p.get("forward_mm")):
        out.append(f"参考：立ち上がりで骨盤は前方へ{p['forward_mm']:.0f}mm移動しています。" if ja else
                   f"For reference, the pelvis travelled {p['forward_mm']:.0f} mm forward when rising.")
    if p["unstable_phases"]:
        ph = ("・" if ja else ", ").join(N[x] for x in p["unstable_phases"])
        out.append(f"{ph}局面で骨盤角度の標準偏差が大きく、保持中の骨盤動揺が見られます。" if ja else
                   f"Elevated pelvic SD during {ph}, indicating pelvic sway while holding.")

    # ---------------- 体幹・腰部 ----------------
    lum = details["lumbar"]
    tsr = details.get("trunk_side_rom")
    if lum or pd.notna(tsr):
        head("体幹・腰部", "Trunk & Lumbar")
    if pd.notna(tsr):
        if tsr > th["trunk_side_deg"]:
            out.append(f"動作中の体幹側屈の可動範囲は{tsr:.1f}°と大きく、体幹を横に倒してバランスをとっている可能性があります。" if ja else
                       f"Trunk lateral bending range during movement is {tsr:.1f}°, suggesting lateral trunk lean to maintain balance.")
        else:
            out.append(f"動作中の体幹側屈の可動範囲は{tsr:.1f}°で、しきい値内です。" if ja else
                       f"Trunk lateral bending range during movement is {tsr:.1f}°, within threshold.")
    if lum:
        nd = th["lumbar_note_deg"]
        parts = []
        if lum["max_ext_change"] >= nd:
            parts.append(f"立位より最大{lum['max_ext_change']:.1f}°反り" if ja else
                         f"up to {lum['max_ext_change']:.1f}° more extension than standing")
        if lum["max_flex_change"] >= nd:
            parts.append(f"立ち上がり中に座位よりさらに最大{lum['max_flex_change']:.1f}°丸まり" if ja else
                         f"up to {lum['max_flex_change']:.1f}° more flexion than seated while rising")
        if ja:
            chg = ("、".join(parts) + "。") if parts else "立ち上がり中の腰の反り・丸まりは小さい状態です。"
            out.append(f"腰椎伸展の可動範囲は{lum['rom']:.1f}°。{chg}")
        else:
            chg = ("; ".join(parts) + ".") if parts else "Little extra arching or rounding while rising."
            out.append(f"Lumbar extension range {lum['rom']:.1f}°. {chg}")
        if lum["max_ext_change"] > th["lumbar_change_deg"] and lum["max_ext_change"] >= lum["max_flex_change"]:
            out.append("主に腰を反らせる代償（腰椎過伸展）が見られ、立ち上がりの最後に腰で伸び上がっている可能性があります。" if ja else
                       "Predominantly extension-type compensation (lumbar hyperextension) at the end of rising.")
        elif lum["max_flex_change"] > th["lumbar_change_deg"]:
            out.append("立ち上がり中に腰が丸まる代償（腰椎屈曲）が見られ、股関節ではなく腰から前傾している可能性があります。" if ja else
                       "Flexion-type compensation while rising, suggesting forward lean from the lower back rather than the hips.")
        else:
            out.append("立位からの変化はしきい値内で、腰部の代償は少ない状態です。" if ja else
                       "Changes from standing are within threshold; minimal lumbar compensation.")
        if lum.get("main_phase"):
            out.append(f"腰部の動きが最も大きいのは{N.get(lum['main_phase'], lum['main_phase'])}局面です。" if ja else
                       f"Lumbar motion is greatest during the {N.get(lum['main_phase'], lum['main_phase'])} phase.")

    # ---------------- 反復・テンポ ----------------
    if details["n_reps"]:
        head("反復の一貫性・テンポ", "Consistency & Tempo")
        if ja:
            out.append(f"骨盤の平均上昇量は{details['height_mean_cm']:.1f}cm（変動係数{details['height_cv_pct']:.1f}%）、"
                       f"立ち上がり{details['ascent_mean_s']:.2f}秒／着座{details['descent_mean_s']:.2f}秒です。")
        else:
            out.append(f"Mean pelvic rise {details['height_mean_cm']:.1f} cm (CV {details['height_cv_pct']:.1f}%); "
                       f"rising {details['ascent_mean_s']:.2f} s / sitting down {details['descent_mean_s']:.2f} s.")
        tr = details.get("tempo_ratio")
        if pd.notna(tr) and tr < 1 / th["tempo_ratio"]:
            out.append("着座が立ち上がりに比べて速く、片脚での遠心性コントロールが不十分な（ドスンと座る）可能性があります。" if ja else
                       "Sitting down is notably faster than rising, suggesting limited single-leg eccentric control.")
        elif pd.notna(tr) and tr > th["tempo_ratio"]:
            out.append("立ち上がりが着座に比べて速く、反動を利用して立ち上がっている可能性があります。" if ja else
                       "Rising is notably faster than sitting down, possibly using momentum to stand.")
        if details["height_cv_pct"] > th["rep_cv_pct"]:
            out.append("反復ごとの立ち上がりの高さにばらつきがあり、動作の再現性に課題があります。" if ja else
                       "Rise height varied between reps, indicating limited consistency.")

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
                                 mobility_score, stability_score, compensation_score):
    """専門用語を減らし、「良い点」と「気をつけたい点」を平易な言葉で伝える。"""
    th = THRESHOLDS
    J = details["joints"]
    ja = lang_code == "ja"
    NM = CLIENT_NAMES_JA if ja else CLIENT_NAMES_EN
    leg = ("右" if details["side"] == "Right" else "左") if ja else details["side"].lower()
    good, care = [], []

    if ja:
        head = (f"今回の総合スコアは{overall_score:.0f}/100で、" +
                ("とても良い状態です。" if overall_score >= 80 else
                 "全体的には悪くありませんが、いくつか気をつけたい点があります。" if overall_score >= 60 else
                 "いくつか改善していきたいポイントが見つかりました。") +
                f"（{leg}脚での片脚立ち座り）")
    else:
        head = (f"Your overall score was {overall_score:.0f}/100 — " +
                ("a great result." if overall_score >= 80 else
                 "reasonably good, with a few points to keep an eye on." if overall_score >= 60 else
                 "we found a few areas worth working on.") +
                f" (single-leg sit-to-stand on your {leg} leg)")

    pat = details.get("lean_pattern")
    if pat == "low":
        care.append("立ち上がるときに、上半身をあまり前に倒さずに立っています。おじぎをするように少し前に倒すと、楽に立ち上がれます。"
                    if ja else "You stand up without leaning forward much — leaning forward a little, like a small bow, makes it easier.")
    elif pat == "high":
        care.append("立ち上がるときに、上半身を大きく前に倒して立っています。脚の力を上半身で補っている可能性があります。"
                    if ja else "You lean your upper body far forward to stand, which may be making up for leg strength.")
    elif pat == "normal":
        good.append("上半身の前傾と脚の力を、バランスよく使って立ち上がれています。"
                    if ja else "You combine a forward lean and leg strength in good balance when standing up.")

    below = [k for k, j in J.items() if j.get("below")]
    if below:
        where = "・".join(NM[k] for k in below) if ja else " and ".join(NM[k] for k in below)
        care.append(f"{leg}脚の{where}の動く範囲が、目安より少し小さめです。" if ja else
                    f"The range of motion in your {leg} {where} is a little smaller than typical.")
    elif J:
        good.append(f"{leg}脚の股関節・ひざ・足首は、しっかり動かせています。" if ja else
                    f"Your {leg} hip, knee and ankle all moved through a good range.")
    for key, j in J.items():
        if j.get("rep_trend_deg", 0) < -th["rep_trend_deg"]:
            care.append(f"回数を重ねるにつれて、{NM[key]}の曲がりが小さくなっていました（疲れのサインかもしれません）。" if ja else
                        f"Your {NM[key]} bent less as the reps went on — possibly a sign of fatigue.")
            break
    if "ankle" in J and J["ankle"]["peak"] < th["ankle_df_limited"]:
        care.append("足首がかたく、足を手前に引いて立ち上がりにくい状態かもしれません。" if ja else
                    "A stiff ankle may make it harder to draw your foot back before standing up.")

    kin = details["knee_in"]
    if kin.get("available"):
        if kin["flag"]:
            care.append("立ち座りの途中で、ひざが内側に入りやすい傾向があります。ひざとつま先を同じ方向に向ける意識を持ちましょう。"
                        if ja else "Your knee tends to cave inward during the movement — try to keep your knee pointing in the same direction as your toes.")
        else:
            good.append("ひざとつま先の向きがそろっていて、ひざが内側に入らずに動けています。"
                        if ja else "Your knee stayed nicely in line with your toes.")

    p = details["pelvis"]
    pel_flags = [
        pd.notna(p["tilt_rom"]) and p["tilt_rom"] > th["pelvis_rom_deg"],
        pd.notna(p["rot_rom"]) and p["rot_rom"] > th["pelvis_rom_deg"],
    ]
    if p["drop_flag"]:
        care.append("片脚で支えているときに、浮かせている脚の側の骨盤が下がりやすい傾向があります。お尻の横の筋肉（中殿筋）を鍛えると安定しやすくなります。"
                    if ja else "When standing on one leg, the pelvis tends to drop on the side of the lifted leg — strengthening the side hip muscles can help.")
    if p["sway_flag"]:
        care.append("動作中に、体が左右にグラつく場面がありました。"
                    if ja else "Your body swayed side to side during the movement.")
    if any(pel_flags):
        care.append("立ち座りの途中で、骨盤が傾いたりねじれたりしやすい傾向があります。"
                    if ja else "Your pelvis tends to tilt or twist as you sit down and stand up.")
    if not p["drop_flag"] and not p["sway_flag"] and not any(pel_flags):
        good.append("片脚で支えていても、骨盤（腰の土台）は安定していました。" if ja else
                    "Your pelvis stayed stable even on one leg.")

    tsr = details.get("trunk_side_rom")
    if pd.notna(tsr) and tsr > th["trunk_side_deg"]:
        care.append("バランスをとるために、上半身を横に倒しやすい傾向があります。"
                    if ja else "You tend to lean your upper body sideways to keep your balance.")

    lum = details["lumbar"]
    if lum:
        if lum["max_ext_change"] > th["lumbar_change_deg"] and lum["max_ext_change"] >= lum["max_flex_change"]:
            care.append("立ち上がりの最後に腰を反らせて伸び上がる傾向があり、腰まわりに負担がかかりやすい状態です。"
                        if ja else "You tend to arch your lower back at the end of standing up, which can strain the area.")
        elif lum["max_flex_change"] > th["lumbar_change_deg"]:
            care.append("立ち上がるときに腰が丸まりやすい傾向があります。お尻を後ろに引いて、股関節から体を倒す意識を持ちましょう。"
                        if ja else "Your lower back tends to round as you stand up — try hinging from your hips instead.")
        else:
            good.append("腰の反りや丸まりは少なく、よい姿勢を保てています。"
                        if ja else "Your lower back stayed in a good position.")

    tr = details.get("tempo_ratio")
    if pd.notna(tr) and tr < 1 / th["tempo_ratio"]:
        care.append("座るスピードが速めです（ドスンと座りやすい状態）。ゆっくり座ると、脚の筋力づくりにもなります。"
                    if ja else "You sit down quite fast — lowering yourself slowly is safer and also builds leg strength.")

    lines = [head]
    sep = "" if ja else " "
    if good:
        lines.append(("◎ よくできている点：" if ja else "Strengths: ") + sep.join(good[:3]))
    if care:
        lines.append(("△ 気をつけたい点：" if ja else "Points to watch: ") + sep.join(care[:4]))
    lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。" if ja else
                 "Try the recommended actions below at a comfortable pace before your next check.")
    return "\n".join(lines)
