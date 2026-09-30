"""
sit_stand_auto_comment.py
---------------------------------------------------------------
Sit-to-Stand Analysis の自動コメント（PDF Report / Client Report）用モジュール。
squat_auto_comment.py と同じ構成・同じ呼び出し方にそろえています。
 
実測データ（df_phase）から、股関節・膝関節・足関節・骨盤・腰部について
以下を読み取り、臨床向け／クライアント向けのコメント下書きを生成します。
 
  - 最大角度・可動域（左右別）と正常範囲との比較
  - 左右差（試技全体／フェーズ別で最大となる局面）
  - 反復ごとの一貫性（ピーク角度のばらつき・前半→後半の変化）
  - 左右のタイミング差（立ち上がりで最も速く伸びる時刻のずれ）
  - 着座と立ち上がりの可動域差（遠心性／求心性の制御差）
  - 体幹の前傾（立ち上がりで股関節が座位から追加で曲がる量）
  - 骨盤：前後傾・側方傾斜・回旋・左右方向の移動
  - 腰部：立位からの変化（反り／立ち上がり中の丸まり）とそのタイミング
  - 反復全体：立ち上がりの高さ・テンポ（立ち上がり／着座時間）
 
■ Squat 版との違い（Sit-to-Stand の動作に合わせた変更）
  - 反復検出：Squat は「最下点」を数えますが、こちらは「立位（骨盤の最高点）」を数えます。
  - 左右の骨盤移動：OpenSim/OpenCap の座標系では左右方向は pelvis_tz です
    （pelvis_tx は前後方向）。こちらは pelvis_tz を使っています。
    ※ squat_auto_comment.py は pelvis_tx を左右移動として使っているので、
      同じ修正を検討してください。
  - 股関節／膝関節の比率の代わりに、立ち上がり時の体幹前傾量を評価します。
  - 腰の丸まりは、座っている姿勢そのものではなく「立ち上がり中」の丸まりだけを見ます
    （座位で腰が立位より丸まるのは自然なため）。
  - テンポ：着座が立ち上がりより極端に速い場合を「ドスンと座る」傾向として扱います。
 
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
    "ecc_con_pct": 15.0,        # 着座／立ち上がりのROM差（%）
    "rep_cv_pct": 8.0,          # 反復ごとのピーク角度の変動係数（%）
    "rep_trend_deg": 8.0,       # 前半→後半のピーク角度の変化（°）
    "timing_lag_ms": 100.0,     # 左右の伸展タイミングのずれ（ms）
    "static_std_deg": 2.0,      # 静止局面（座位・立位）の標準偏差（°）
    "lean_low_deg": 10.0,       # 立ち上がり時の体幹前傾がこれ未満 → 前傾不足
    "lean_high_deg": 40.0,      # これ超 → 前傾が大きすぎる
    "ankle_df_limited": 10.0,   # 足関節背屈の最大値がこれ未満 → 制限の可能性
    "lumbar_change_deg": 10.0,  # 腰部の立位からの変化（°）
    "pelvis_rom_deg": 10.0,     # 骨盤（前後傾・回旋）の可動範囲（°）
    "pelvis_list_deg": 5.0,     # 骨盤の側方傾斜の可動範囲（°）
    "pelvis_shift_mm": 20.0,    # 骨盤の左右移動（mm）
    "tempo_ratio": 1.5,         # 着座時間／立ち上がり時間の比（逆数未満で「速く座る」）
    "phase_rom_min_deg": 10.0,  # フェーズ別左右差を評価する最小ROM
    "lumbar_note_deg": 2.0,     # 腰部の反り／丸まりを文中で言及する最小量
}
 
JOINTS = {
    "hip": ("hip_flexion_r", "hip_flexion_l"),
    "knee": ("knee_angle_r", "knee_angle_l"),
    "ankle": ("ankle_angle_r", "ankle_angle_l"),
}
 
NAMES = {
    "ja": {"hip": "股関節", "knee": "膝関節", "ankle": "足関節",
           "R": "右", "L": "左",
           "Bottom": "座位", "Ascending": "立ち上がり", "Standing": "立位", "Descending": "着座"},
    "en": {"hip": "Hip", "knee": "Knee", "ankle": "Ankle",
           "R": "right", "L": "left",
           "Bottom": "Seated", "Ascending": "Rising", "Standing": "Standing", "Descending": "Sitting down"},
}
CLIENT_NAMES_JA = {"hip": "股関節", "knee": "ひざ", "ankle": "足首"}
CLIENT_NAMES_EN = {"hip": "hips", "knee": "knees", "ankle": "ankles"}
 
 
# ===============================================================
# 1. データの読み取り
# ===============================================================
def _pct_diff(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m * 100
 
 
def _detect_reps(pelvis_ty, fs=FS):
    """骨盤の高さ（pelvis_ty）の極大点（立位）から反復を検出し、各反復の区間を返す。"""
    s = pd.Series(pelvis_ty).rolling(5, center=True).mean().bfill().ffill().to_numpy()
    lo, hi = s.min(), s.max()
    rng = hi - lo
    if rng <= 0:
        return [], s
 
    stand_line = lo + rng * 0.6
    min_gap = int(1.0 * fs)
 
    # 局所極大（立位）の候補
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
 
    # 反復の区切り＝隣り合う立位の間の最低点（座位）
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
        # 立ち上がり開始（座位ラインを最後に離れた時点）→ 立位到達
        pre_seat = np.where(s[start:m + 1] <= seat_line)[0]
        rise_start = start + pre_seat[-1] if len(pre_seat) else start
        reach = np.where(s[rise_start:m + 1] >= top_line)[0]
        rise_end = rise_start + reach[0] if len(reach) else m
        # 立位を離れた時点 → 座位到達
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
 
 
def analyze_sit_stand_details(df_phase, phase_summary_df, comparison_df,
                              phase_order=("Bottom", "Ascending", "Standing", "Descending"),
                              fs=FS):
    """実測データから、関節・骨盤・腰部ごとの詳細な特徴量をまとめて返す。"""
    th = THRESHOLDS
    reps, _ = _detect_reps(df_phase["pelvis_ty"].to_numpy(), fs)
    d = {"reps": reps, "n_reps": len(reps), "joints": {}}
 
    # ---------------- 下肢関節 ----------------
    for key, (rv, lv) in JOINTS.items():
        if rv not in df_phase.columns or lv not in df_phase.columns:
            continue
        r, l = df_phase[rv].to_numpy(), df_phase[lv].to_numpy()
        j = {
            "peak_R": float(np.nanmax(r)), "peak_L": float(np.nanmax(l)),
            "rom_R": float(np.nanmax(r) - np.nanmin(r)),
            "rom_L": float(np.nanmax(l) - np.nanmin(l)),
        }
        j["asym_pct"] = _pct_diff(j["rom_R"], j["rom_L"])
        j["larger_side"] = "R" if j["rom_R"] >= j["rom_L"] else "L"
 
        # 正常範囲との比較（左右別）
        for side, var in (("R", rv), ("L", lv)):
            row = comparison_df[comparison_df["Variable"] == var]
            if len(row):
                j[f"out_{side}"] = bool(row["Out_of_Range"].iloc[0])
                rom = row["Subject_ROM"].iloc[0]
                j[f"below_{side}"] = rom < row["Healthy_Min"].iloc[0]
                j[f"above_{side}"] = rom > row["Healthy_Max"].iloc[0]
 
        # フェーズ別の左右差（最大となる局面）
        phase_asym = {}
        for ph in phase_order:
            a = _phase_value(phase_summary_df, rv, ph, "ROM")
            b = _phase_value(phase_summary_df, lv, ph, "ROM")
            if pd.notna(a) and pd.notna(b) and max(a, b) >= th["phase_rom_min_deg"]:
                phase_asym[ph] = _pct_diff(a, b)
        if phase_asym:
            j["worst_phase"] = max(phase_asym, key=phase_asym.get)
            j["worst_phase_asym"] = phase_asym[j["worst_phase"]]
 
        # 着座（Descending）と立ち上がり（Ascending）のROM差
        desc = np.nanmean([_phase_value(phase_summary_df, v, "Descending", "ROM") for v in (rv, lv)])
        asc = np.nanmean([_phase_value(phase_summary_df, v, "Ascending", "ROM") for v in (rv, lv)])
        j["desc_rom"], j["asc_rom"] = desc, asc
        j["ecc_con_pct"] = _pct_diff(desc, asc) if pd.notna(desc) and pd.notna(asc) else 0.0
 
        # 静止局面（座位・立位）の安定性
        j["unstable_phases"] = []
        for ph in ("Bottom", "Standing"):
            stds = [_phase_value(phase_summary_df, v, ph, "Std") for v in (rv, lv)]
            if any(pd.notna(x) and x > th["static_std_deg"] for x in stds):
                j["unstable_phases"].append(ph)
 
        # 反復ごとのピーク角度・左右タイミング
        if reps:
            pk_r = [np.nanmax(r[p["start"]:p["end"] + 1]) for p in reps]
            pk_l = [np.nanmax(l[p["start"]:p["end"] + 1]) for p in reps]
            pk_mean = np.mean([pk_r, pk_l], axis=0)
            j["rep_peaks"] = pk_mean.tolist()
            j["rep_cv_pct"] = float(np.std(pk_mean) / np.mean(pk_mean) * 100) if np.mean(pk_mean) else 0.0
            half = max(1, len(pk_mean) // 2)
            j["rep_trend_deg"] = float(np.mean(pk_mean[-half:]) - np.mean(pk_mean[:half])) if len(pk_mean) >= 2 else 0.0
            # 座位ではピーク角度が長く続き時刻が定まらないため、
            # 立ち上がり区間で「伸展が半分まで進んだ時刻」の左右差をタイミング差とする
            def _half_time(x):
                mid = (np.nanmax(x) + np.nanmin(x)) / 2
                idx = np.where(x <= mid)[0]
                return int(idx[0]) if len(idx) else None
 
            lags = []
            for p in reps:
                a, b = p["rise_start"], p["rise_end"]
                if b - a < 3:
                    continue
                tr_, tl_ = _half_time(r[a:b + 1]), _half_time(l[a:b + 1])
                if tr_ is not None and tl_ is not None:
                    lags.append(abs(tr_ - tl_) / fs * 1000)
            j["timing_lag_ms"] = float(np.mean(lags)) if lags else 0.0
        d["joints"][key] = j
 
    # ---------------- 立ち上がり時の体幹前傾 ----------------
    # 股関節屈曲が座位から立ち上がり中にさらに増えた量（＝体幹を前に倒した量の目安）
    hip_cols = [c for c in JOINTS["hip"] if c in df_phase.columns]
    if hip_cols:
        hip = df_phase[hip_cols].mean(axis=1)
        seated = hip[df_phase["Phase"] == "Bottom"].mean()
        rising = hip[df_phase["Phase"] == "Ascending"]
        lean = float(rising.max() - seated) if len(rising) and pd.notna(seated) else np.nan
        d["forward_lean_deg"] = lean
        if pd.isna(lean):
            d["lean_pattern"] = None
        elif lean < th["lean_low_deg"]:
            d["lean_pattern"] = "low"
        elif lean > th["lean_high_deg"]:
            d["lean_pattern"] = "high"
        else:
            d["lean_pattern"] = "normal"
 
    # ---------------- 骨盤 ----------------
    pel = {}
 
    def rng(col):
        return float(df_phase[col].max() - df_phase[col].min()) if col in df_phase.columns else np.nan
 
    def standing_minus_bottom(col):
        if col not in df_phase.columns:
            return np.nan
        s = df_phase.loc[df_phase["Phase"] == "Standing", col].mean()
        b = df_phase.loc[df_phase["Phase"] == "Bottom", col].mean()
        return float(s - b) if pd.notna(b) and pd.notna(s) else np.nan
 
    pel["tilt_rom"] = rng("pelvis_tilt")
    pel["tilt_change"] = standing_minus_bottom("pelvis_tilt")
    pel["list_rom"] = rng("pelvis_list")
    pel["rot_rom"] = rng("pelvis_rotation")
    # 左右方向は pelvis_tz（+ = 右）。立位での位置 − 座位での位置
    shift = standing_minus_bottom("pelvis_tz")
    pel["shift_mm"] = shift * 1000 if pd.notna(shift) else np.nan
    pel["shift_side"] = None if pd.isna(shift) else ("R" if shift > 0 else "L")
    pel["ml_rom_mm"] = rng("pelvis_tz") * 1000 if "pelvis_tz" in df_phase.columns else np.nan
    # 前後方向（pelvis_tx）：立ち上がりで骨盤が前へ移動した量（参考値）
    fwd = standing_minus_bottom("pelvis_tx")
    pel["forward_mm"] = fwd * 1000 if pd.notna(fwd) else np.nan
    pel["unstable_phases"] = [
        ph for ph in ("Bottom", "Standing")
        if any(pd.notna(_phase_value(phase_summary_df, v, ph, "Std"))
               and _phase_value(phase_summary_df, v, ph, "Std") > th["static_std_deg"]
               for v in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"))
    ]
    d["pelvis"] = pel
 
    # ---------------- 腰部 ----------------
    lum = {}
    if "lumbar_extension" in df_phase.columns:
        le = df_phase["lumbar_extension"]
        base = df_phase.loc[df_phase["Phase"] == "Standing", "lumbar_extension"].mean()
        dev = le - base
        seated = df_phase.loc[df_phase["Phase"] == "Bottom", "lumbar_extension"].mean()
        rising = le[df_phase["Phase"] == "Ascending"]
        lum["rom"] = float(le.max() - le.min())
        lum["max_ext_change"] = float(dev.max())  # 立位より反った最大量（+）
        # 座位で腰が丸まるのは自然なので、「座位よりさらに丸まった量（立ち上がり中）」だけを見る
        lum["max_flex_change"] = (float(max(0.0, seated - rising.min()))
                                  if len(rising) and pd.notna(seated) else 0.0)
        # 座位の腰の位置（立位基準、−＝丸まり／＋＝反り）
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
def generate_sit_stand_auto_comment(lang_code, overall_score, details, feature_values=None):
    """専門職向けの詳細コメントを、部位ごとの見出し付きで生成する。"""
    th = THRESHOLDS
    N = NAMES[lang_code]
    J = details["joints"]
    ja = lang_code == "ja"
    out = []
 
    def head(ja_t, en_t):
        out.append("")
        out.append(f"【{ja_t}】" if ja else f"[{en_t}]")
 
    # ---------------- 総合 ----------------
    out.append("【総合】" if ja else "[Overall]")
    if ja:
        lvl = "動作全体は良好で、大きな問題は見られません" if overall_score >= 80 else (
            "一部に改善の余地があります" if overall_score >= 60 else "複数の注意すべき所見があります")
        out.append(f"総合スコアは{overall_score}/100で、{lvl}。"
                   f"解析した立ち座りの回数は{details['n_reps']}回です。")
    else:
        lvl = "indicating generally good performance" if overall_score >= 80 else (
            "with some room for improvement" if overall_score >= 60 else "with several findings that warrant attention")
        out.append(f"Overall score: {overall_score}/100, {lvl}. {details['n_reps']} sit-to-stand repetitions were analysed.")
 
    lean, pat = details.get("forward_lean_deg"), details.get("lean_pattern")
    if pat and pd.notna(lean):
        if ja:
            txt = {"low": "前傾が小さく、体幹前傾による重心移動を十分に使えていない（下肢伸展筋への依存・反動の可能性）",
                   "high": "前傾が大きく、体幹を大きく倒して立ち上がる（下肢筋力を体幹前傾で補っている可能性）",
                   "normal": "体幹前傾と下肢の伸展がバランスよく使われた"}[pat]
            out.append(f"立ち上がり時の体幹前傾（座位からの股関節屈曲の追加量）は{lean:.1f}°で、{txt}立ち上がり方です。")
        else:
            txt = {"low": "limited forward lean, with little use of trunk momentum (possible reliance on leg extensors or backward momentum)",
                   "high": "excessive forward lean (trunk flexion possibly compensating for lower-limb strength)",
                   "normal": "a balanced use of forward lean and leg extension"}[pat]
            out.append(f"Forward trunk lean during rising (additional hip flexion from sitting) is {lean:.1f}°, suggesting {txt}.")
 
    # ---------------- 各関節 ----------------
    for key, (ja_t, en_t) in {"hip": ("股関節", "Hip"), "knee": ("膝関節", "Knee"),
                              "ankle": ("足関節", "Ankle")}.items():
        if key not in J:
            continue
        j = J[key]
        head(ja_t, en_t)
        verb_ja = "背屈" if key == "ankle" else "屈曲"
        verb_en = "dorsiflexion" if key == "ankle" else "flexion"
 
        if ja:
            line = (f"最大{verb_ja}は右{j['peak_R']:.1f}°／左{j['peak_L']:.1f}°、"
                    f"可動域は右{j['rom_R']:.1f}°／左{j['rom_L']:.1f}°。")
        else:
            line = (f"Peak {verb_en} R {j['peak_R']:.1f}° / L {j['peak_L']:.1f}°; "
                    f"ROM R {j['rom_R']:.1f}° / L {j['rom_L']:.1f}°.")
        below = [N[s] for s in ("R", "L") if j.get(f"below_{s}")]
        above = [N[s] for s in ("R", "L") if j.get(f"above_{s}")]
        if ja:
            if below:
                line += f"{'・'.join(below)}で正常範囲を下回っており、可動域制限が疑われます。"
            elif above:
                line += f"{'・'.join(above)}で正常範囲を上回っており、過可動または代償的な動きの可能性があります。"
            else:
                line += "左右とも正常範囲内です。"
        else:
            if below:
                line += f" Below the normal range on the {', '.join(below)} side, suggesting restricted ROM."
            elif above:
                line += f" Above the normal range on the {', '.join(above)} side, suggesting hypermobility or compensation."
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
            out.append(f"立ち上がりで最も速く伸展する時刻に、左右で平均{j['timing_lag_ms']:.0f}msのずれがあり、"
                       "左右の協調性（タイミング）に差がある可能性があります。" if ja else
                       f"Mean left-right lag in the moment of fastest extension during rising is {j['timing_lag_ms']:.0f} ms, "
                       "suggesting a possible bilateral coordination difference.")
 
        if "rep_cv_pct" in j:
            if j["rep_cv_pct"] > th["rep_cv_pct"]:
                out.append(f"反復ごとのピーク角度の変動係数は{j['rep_cv_pct']:.1f}%と大きく、"
                           "動作の再現性にばらつきがあります。" if ja else
                           f"Rep-to-rep variability of peak angle is high (CV {j['rep_cv_pct']:.1f}%).")
            if abs(j.get("rep_trend_deg", 0)) > th["rep_trend_deg"]:
                trend = j["rep_trend_deg"]
                out.append((f"前半から後半にかけてピーク角度が{abs(trend):.1f}°"
                            f"{'増加' if trend > 0 else '減少'}しており、"
                            f"{'慣れ・座り方の変化' if trend > 0 else '疲労や座り込みの浅化'}の影響が考えられます。") if ja else
                           (f"Peak angle {'increased' if trend > 0 else 'decreased'} by {abs(trend):.1f}° "
                            f"from early to late reps, possibly reflecting "
                            f"{'adaptation / a change in sitting position' if trend > 0 else 'fatigue or shallower sitting'}."))
 
        if j["ecc_con_pct"] > th["ecc_con_pct"]:
            out.append(f"着座{j['desc_rom']:.1f}°／立ち上がり{j['asc_rom']:.1f}°で{j['ecc_con_pct']:.1f}%の差があり、"
                       "遠心性（着座）と求心性（立ち上がり）の制御に違いがある可能性があります。" if ja else
                       f"Sitting down {j['desc_rom']:.1f}° vs Rising {j['asc_rom']:.1f}° "
                       f"({j['ecc_con_pct']:.1f}% difference), suggesting differing eccentric/concentric control.")
 
        if j["unstable_phases"]:
            ph = "・".join(N[p] for p in j["unstable_phases"]) if ja else ", ".join(N[p] for p in j["unstable_phases"])
            out.append(f"{ph}局面で角度の標準偏差が大きく、保持中の動揺が見られます。" if ja else
                       f"Elevated SD during {ph}, indicating sway while holding the position.")
 
        if key == "ankle":
            pk = min(j["peak_R"], j["peak_L"])
            if pk < th["ankle_df_limited"]:
                out.append(f"最大背屈が{pk:.1f}°と小さく、足部を引き込めないことで体幹前傾の増加や"
                           "立ち上がりの困難さにつながっている可能性があります。" if ja else
                           f"Peak dorsiflexion is limited ({pk:.1f}°), which may prevent the feet from being drawn back "
                           "and increase the trunk lean needed to rise.")
 
    # ---------------- 骨盤 ----------------
    p = details["pelvis"]
    head("骨盤", "Pelvis")
    items = []
    if pd.notna(p["tilt_rom"]):
        flag = p["tilt_rom"] > th["pelvis_rom_deg"]
        items.append((f"前後傾の可動範囲{p['tilt_rom']:.1f}°（立位で座位から{p['tilt_change']:+.1f}°）"
                      if ja else f"tilt range {p['tilt_rom']:.1f}° ({p['tilt_change']:+.1f}° standing vs seated)", flag))
    if pd.notna(p["list_rom"]):
        flag = p["list_rom"] > th["pelvis_list_deg"]
        items.append((f"側方傾斜{p['list_rom']:.1f}°" if ja else f"obliquity range {p['list_rom']:.1f}°", flag))
    if pd.notna(p["rot_rom"]):
        flag = p["rot_rom"] > th["pelvis_rom_deg"]
        items.append((f"回旋{p['rot_rom']:.1f}°" if ja else f"rotation range {p['rot_rom']:.1f}°", flag))
    if pd.notna(p["shift_mm"]):
        flag = abs(p["shift_mm"]) > th["pelvis_shift_mm"]
        side_txt = (N[p["shift_side"]] + "へ") if ja else f"to the {N[p['shift_side']]}"
        items.append((f"立ち上がり後の左右移動{abs(p['shift_mm']):.0f}mm（{side_txt}）" if ja else
                      f"lateral shift after rising {abs(p['shift_mm']):.0f} mm ({side_txt})", flag))
    if items:
        sep = "、" if ja else ", "
        out.append(("計測値：" if ja else "Measured: ") + sep.join(t for t, _ in items) + ("。" if ja else "."))
        flagged = [t for t, f in items if f]
        if flagged:
            out.append(("しきい値を超えた項目：" + "、".join(t.split("（")[0] for t in flagged) +
                        "。骨盤の制御が不十分で、代償的に骨盤が動いている可能性があります。") if ja else
                       ("Above threshold: " + ", ".join(t.split(" (")[0] for t in flagged) +
                        ". Pelvic control may be insufficient, with compensatory pelvic motion."))
        else:
            out.append("いずれもしきい値内で、骨盤は比較的安定しています。" if ja else
                       "All within thresholds; the pelvis remained relatively stable.")
        asym_j = [k for k, j in J.items() if j["asym_pct"] > th["asym_pct"]]
        if asym_j and pd.notna(p["shift_mm"]) and abs(p["shift_mm"]) > th["pelvis_shift_mm"]:
            out.append("下肢の左右差と骨盤の側方移動が同時に見られ、片側への荷重偏りが示唆されます。" if ja else
                       "Lower-limb asymmetry together with lateral pelvic shift suggests uneven weight bearing.")
    if pd.notna(p.get("forward_mm")):
        out.append(f"参考：立ち上がりで骨盤は前方へ{p['forward_mm']:.0f}mm移動しています。" if ja else
                   f"For reference, the pelvis travelled {p['forward_mm']:.0f} mm forward when rising.")
    if p["unstable_phases"]:
        ph = "・".join(N[x] for x in p["unstable_phases"]) if ja else ", ".join(N[x] for x in p["unstable_phases"])
        out.append(f"{ph}局面で骨盤角度の標準偏差が大きく、保持中の骨盤動揺が見られます。" if ja else
                   f"Elevated pelvic SD during {ph}, indicating pelvic sway while holding.")
 
    # ---------------- 腰部 ----------------
    lum = details["lumbar"]
    if lum:
        head("腰部", "Lumbar")
        nd = th["lumbar_note_deg"]
        parts = []
        if lum["max_ext_change"] >= nd:
            parts.append(f"立位より最大{lum['max_ext_change']:.1f}°反り" if ja else f"up to {lum['max_ext_change']:.1f}° more extension than standing")
        if lum["max_flex_change"] >= nd:
            parts.append(f"立ち上がり中に座位よりさらに最大{lum['max_flex_change']:.1f}°丸まり" if ja else
                         f"up to {lum['max_flex_change']:.1f}° more flexion than seated while rising")
        sc = lum.get("seated_change")
        if ja:
            chg = ("、".join(parts) + "。") if parts else "立ち上がり中の腰の反り・丸まりは小さい状態です。"
            seat_txt = (f"座位では立位と比べて{abs(sc):.1f}°{'丸まった' if sc < 0 else '反った'}姿勢です。"
                        if pd.notna(sc) else "")
            out.append(f"腰椎伸展の可動範囲は{lum['rom']:.1f}°。{seat_txt}{chg}")
        else:
            chg = ("; ".join(parts) + ".") if parts else "Little extra arching or rounding while rising."
            seat_txt = (f"Seated, the lower back is {abs(sc):.1f}° more {'flexed' if sc < 0 else 'extended'} than standing. "
                        if pd.notna(sc) else "")
            out.append(f"Lumbar extension range {lum['rom']:.1f}°. {seat_txt}{chg}")
        if lum["max_ext_change"] > th["lumbar_change_deg"] and lum["max_ext_change"] >= lum["max_flex_change"]:
            out.append("主に腰を反らせる代償（腰椎過伸展）が見られ、立ち上がりの最後に腰で伸び上がっている可能性があります。" if ja else
                       "Predominantly extension-type compensation (lumbar hyperextension), possibly finishing the rise with the lower back.")
        elif lum["max_flex_change"] > th["lumbar_change_deg"]:
            out.append("立ち上がり中に腰が丸まる代償（腰椎屈曲）が見られ、股関節ではなく腰から前傾している可能性があります。" if ja else
                       "Flexion-type compensation while rising (lumbar flexion), suggesting forward lean from the lower back rather than the hips.")
        else:
            out.append("立位からの変化はしきい値内で、腰部の代償は少ない状態です。" if ja else
                       "Changes from standing are within threshold; minimal lumbar compensation.")
        if lum.get("main_phase"):
            out.append(f"腰部の動きが最も大きいのは{N[lum['main_phase']]}局面です。" if ja else
                       f"Lumbar motion is greatest during the {N[lum['main_phase']]} phase.")
 
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
            out.append("着座が立ち上がりに比べて速く、座る局面の遠心性コントロールが不十分な（ドスンと座る）可能性があります。" if ja else
                       "Sitting down is notably faster than rising, suggesting limited eccentric control (dropping into the seat).")
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
                                 mobility_score, stability_score, symmetry_score, compensation_score):
    """専門用語を減らし、「良い点」と「気をつけたい点」を平易な言葉で伝える。"""
    th = THRESHOLDS
    J = details["joints"]
    ja = lang_code == "ja"
    NM = CLIENT_NAMES_JA if ja else CLIENT_NAMES_EN
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
 
    # 立ち上がり方（体幹の前傾）
    pat = details.get("lean_pattern")
    if pat == "low":
        care.append("立ち上がるときに、上半身をあまり前に倒さずに立っています。おじぎをするように少し前に倒すと、楽に立ち上がれます。"
                    if ja else "You stand up without leaning forward much — leaning forward a little, like a small bow, makes standing easier.")
    elif pat == "high":
        care.append("立ち上がるときに、上半身を大きく前に倒して立っています。脚の力を上半身で補っている可能性があります。"
                    if ja else "You lean your upper body far forward to stand, which may be making up for leg strength.")
    elif pat == "normal":
        good.append("上半身の前傾と脚の力を、バランスよく使って立ち上がれています。"
                    if ja else "You combine a forward lean and leg strength in good balance when standing up.")
 
    for key, j in J.items():
        below = [s for s in ("R", "L") if j.get(f"below_{s}")]
        if below:
            where = "・".join(side[s] for s in below) if ja else " and ".join(side[s] for s in below)
            care.append(f"{where}の{NM[key]}の動く範囲が、目安より少し小さめです。" if ja else
                        f"The range of motion in your {where} {key} is a little smaller than typical.")
        if j["asym_pct"] > th["asym_pct"]:
            big = side[j["larger_side"]]
            care.append(f"{NM[key]}は、{big}側の方が大きく動いていて、左右で使い方に差があります。" if ja else
                        f"Your {NM[key]} move more on the {big} side — the two sides are being used differently.")
        if j.get("rep_trend_deg", 0) < -th["rep_trend_deg"]:
            care.append(f"回数を重ねるにつれて、{NM[key]}の曲がりが小さくなっていました（疲れのサインかもしれません）。" if ja else
                        f"Your {NM[key]} bent less as the reps went on — possibly a sign of fatigue.")
        if key == "ankle" and min(j["peak_R"], j["peak_L"]) < th["ankle_df_limited"]:
            care.append("足首がかたく、足を手前に引いて立ち上がりにくい状態かもしれません。" if ja else
                        "Stiff ankles may make it harder to draw your feet back before standing up.")
 
    if not any(j.get("below_R") or j.get("below_L") for j in J.values()):
        good.append("股関節・ひざ・足首は、しっかり動かせています。" if ja else
                    "Your hips, knees and ankles all moved through a good range.")
    if J and all(j["asym_pct"] <= th["asym_pct"] for j in J.values()):
        good.append("左右の脚をバランスよく使えています。" if ja else "You used both legs evenly.")
 
    p = details["pelvis"]
    pel_flags = [
        pd.notna(p["tilt_rom"]) and p["tilt_rom"] > th["pelvis_rom_deg"],
        pd.notna(p["list_rom"]) and p["list_rom"] > th["pelvis_list_deg"],
        pd.notna(p["rot_rom"]) and p["rot_rom"] > th["pelvis_rom_deg"],
    ]
    shift = pd.notna(p["shift_mm"]) and abs(p["shift_mm"]) > th["pelvis_shift_mm"]
    if shift:
        care.append(f"立ち上がったときに、骨盤（腰の土台）が{side[p['shift_side']]}に少しずれています。片側に体重がかかりやすい状態です。"
                    if ja else f"Your pelvis shifts slightly to the {side[p['shift_side']]} as you stand up, so more weight goes onto one leg.")
    if any(pel_flags):
        care.append("立ち座りの途中で、骨盤が傾いたりねじれたりしやすい傾向があります。"
                    if ja else "Your pelvis tends to tilt or twist as you sit down and stand up.")
    if not shift and not any(pel_flags):
        good.append("骨盤（腰の土台）は安定していました。" if ja else "Your pelvis stayed stable.")
 
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
