"""
squat_auto_comment.py
---------------------------------------------------------------
Squat Analysis の自動コメント（PDF Report / Client Report）用モジュール。

実測データ（df_phase）から、股関節・膝関節・足関節・骨盤・腰部について
以下を読み取り、臨床向け／クライアント向けのコメント下書きを生成します。

  - 最大角度・可動域（左右別）と健常範囲との比較
  - 左右差（試技全体／フェーズ別で最大となる局面）
  - 反復ごとの一貫性（ピーク角度のばらつき・前半→後半の変化）
  - 左右のタイミング差（ピーク到達時刻のずれ）
  - 下降と上昇の可動域差（遠心性／求心性の制御差）
  - 股関節／膝関節の比率（股関節優位・膝関節優位のしゃがみ方）
  - 骨盤：前後傾・側方傾斜・回旋・左右方向の移動
  - 腰部：立位からの変化（反り／丸まり）とそのタイミング
  - 反復全体：深さ・テンポ（下降／上昇時間）

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
    "asym_pct": 15.0,          # 左右差（%）
    "ecc_con_pct": 15.0,       # 下降／上昇のROM差（%）
    "rep_cv_pct": 8.0,         # 反復ごとのピーク角度の変動係数（%）
    "rep_trend_deg": 8.0,      # 前半→後半のピーク角度の変化（°）
    "timing_lag_ms": 100.0,    # 左右のピーク到達時刻のずれ（ms）
    "static_std_deg": 2.0,     # 静止局面の標準偏差（°）
    "hip_knee_ratio_low": 0.70,   # これ未満 → 膝関節優位
    "hip_knee_ratio_high": 1.00,  # これ超 → 股関節優位
    "ankle_df_limited": 15.0,  # 足関節背屈の最大値がこれ未満 → 制限の可能性
    "lumbar_change_deg": 10.0, # 腰部の立位からの変化（°）
    "pelvis_rom_deg": 10.0,    # 骨盤（前後傾・回旋）の可動範囲（°）
    "pelvis_list_deg": 5.0,    # 骨盤の側方傾斜の可動範囲（°）
    "pelvis_shift_mm": 20.0,   # 骨盤の左右移動（mm）
    "tempo_ratio": 1.5,        # 下降時間／上昇時間の比（これ超 or 逆数未満で偏り）
    "phase_rom_min_deg": 10.0, # フェーズ別左右差を評価する最小ROM（小さい局面の%は誤差が大きいため除外）
    "lumbar_note_deg": 2.0,    # 腰部の反り／丸まりを文中で言及する最小量
}

JOINTS = {
    "hip": ("hip_flexion_r", "hip_flexion_l"),
    "knee": ("knee_angle_r", "knee_angle_l"),
    "ankle": ("ankle_angle_r", "ankle_angle_l"),
}

NAMES = {
    "ja": {"hip": "股関節", "knee": "膝関節", "ankle": "足関節",
           "R": "右", "L": "左",
           "Standing": "立位", "Descending": "下降", "Bottom": "最下点", "Ascending": "上昇"},
    "en": {"hip": "Hip", "knee": "Knee", "ankle": "Ankle",
           "R": "right", "L": "left",
           "Standing": "Standing", "Descending": "Descending", "Bottom": "Bottom", "Ascending": "Ascending"},
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
    """骨盤の高さ（pelvis_ty）の極小点から反復を検出し、各反復の区間を返す。"""
    s = pd.Series(pelvis_ty).rolling(5, center=True).mean().bfill().ffill().to_numpy()
    lo, hi = s.min(), s.max()
    rng = hi - lo
    if rng <= 0:
        return [], s

    depth_line = lo + rng * 0.4
    min_gap = int(0.8 * fs)

    # 局所極小の候補
    cand = [i for i in range(1, len(s) - 1)
            if s[i] <= s[i - 1] and s[i] <= s[i + 1] and s[i] < depth_line]
    # 深い順に採用し、近すぎる候補を除外
    cand.sort(key=lambda i: s[i])
    minima = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in minima):
            minima.append(i)
    minima.sort()
    if not minima:
        return [], s

    # 反復の区切り＝隣り合う最下点の間の最高点
    bounds = [0]
    for a, b in zip(minima[:-1], minima[1:]):
        bounds.append(a + int(np.argmax(s[a:b + 1])))
    bounds.append(len(s) - 1)

    top_line = hi - rng * 0.10
    reps = []
    for k, m in enumerate(minima):
        start, end = bounds[k], bounds[k + 1]
        # 下降開始・上昇完了（立位ラインを下回る／上回る時点）
        pre = np.where(s[start:m + 1] >= top_line)[0]
        post = np.where(s[m:end + 1] >= top_line)[0]
        d_start = start + pre[-1] if len(pre) else start
        a_end = m + post[0] if len(post) else end
        reps.append({
            "start": start, "end": end, "bottom": m,
            "descent_s": (m - d_start) / fs,
            "ascent_s": (a_end - m) / fs,
            "depth_m": s[start:end + 1].max() - s[m],
        })
    return reps, s


def _phase_value(phase_summary_df, var, phase, stat):
    row = phase_summary_df[phase_summary_df["Variable"] == var]
    if len(row) == 0:
        return np.nan
    col = f"{phase}_{stat}"
    return row[col].iloc[0] if col in row.columns else np.nan


def analyze_squat_details(df_phase, phase_summary_df, comparison_df,
                          phase_order=("Standing", "Descending", "Bottom", "Ascending"),
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

        # 健常範囲との比較（comparison_df の結果を左右別に取得）
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

        # 下降と上昇のROM差
        desc = np.nanmean([_phase_value(phase_summary_df, v, "Descending", "ROM") for v in (rv, lv)])
        asc = np.nanmean([_phase_value(phase_summary_df, v, "Ascending", "ROM") for v in (rv, lv)])
        j["desc_rom"], j["asc_rom"] = desc, asc
        j["ecc_con_pct"] = _pct_diff(desc, asc) if pd.notna(desc) and pd.notna(asc) else 0.0

        # 静止局面（立位・最下点）の安定性
        j["unstable_phases"] = []
        for ph in ("Standing", "Bottom"):
            stds = [_phase_value(phase_summary_df, v, ph, "Std") for v in (rv, lv)]
            if any(pd.notna(x) and x > th["static_std_deg"] for x in stds):
                j["unstable_phases"].append(ph)

        # 反復ごとのピーク角度・左右タイミング
        if reps:
            pk_r = [np.nanmax(r[p["start"]:p["end"] + 1]) for p in reps]
            pk_l = [np.nanmax(l[p["start"]:p["end"] + 1]) for p in reps]
            lags = [abs(int(np.nanargmax(r[p["start"]:p["end"] + 1]))
                        - int(np.nanargmax(l[p["start"]:p["end"] + 1]))) / fs * 1000
                    for p in reps]
            pk_mean = np.mean([pk_r, pk_l], axis=0)
            j["rep_peaks"] = pk_mean.tolist()
            j["rep_cv_pct"] = float(np.std(pk_mean) / np.mean(pk_mean) * 100) if np.mean(pk_mean) else 0.0
            half = max(1, len(pk_mean) // 2)
            j["rep_trend_deg"] = float(np.mean(pk_mean[-half:]) - np.mean(pk_mean[:half])) if len(pk_mean) >= 2 else 0.0
            j["timing_lag_ms"] = float(np.mean(lags))
        d["joints"][key] = j

    # ---------------- 股関節／膝関節の比率 ----------------
    if "hip" in d["joints"] and "knee" in d["joints"]:
        hip_pk = np.mean([d["joints"]["hip"]["peak_R"], d["joints"]["hip"]["peak_L"]])
        knee_pk = np.mean([d["joints"]["knee"]["peak_R"], d["joints"]["knee"]["peak_L"]])
        ratio = hip_pk / knee_pk if knee_pk else np.nan
        d["hip_knee_ratio"] = ratio
        if pd.isna(ratio):
            d["strategy"] = None
        elif ratio < th["hip_knee_ratio_low"]:
            d["strategy"] = "knee"
        elif ratio > th["hip_knee_ratio_high"]:
            d["strategy"] = "hip"
        else:
            d["strategy"] = "balanced"

    # ---------------- 骨盤 ----------------
    pel = {}

    def rng(col):
        return float(df_phase[col].max() - df_phase[col].min()) if col in df_phase.columns else np.nan

    def bottom_minus_standing(col):
        if col not in df_phase.columns:
            return np.nan
        b = df_phase.loc[df_phase["Phase"] == "Bottom", col].mean()
        s = df_phase.loc[df_phase["Phase"] == "Standing", col].mean()
        return float(b - s) if pd.notna(b) and pd.notna(s) else np.nan

    pel["tilt_rom"] = rng("pelvis_tilt")
    pel["tilt_change"] = bottom_minus_standing("pelvis_tilt")
    pel["list_rom"] = rng("pelvis_list")
    pel["list_change"] = bottom_minus_standing("pelvis_list")
    pel["rot_rom"] = rng("pelvis_rotation")
    pel["rot_change"] = bottom_minus_standing("pelvis_rotation")
    shift = bottom_minus_standing("pelvis_tx")
    pel["shift_mm"] = shift * 1000 if pd.notna(shift) else np.nan
    pel["ml_rom_mm"] = rng("pelvis_tx") * 1000 if "pelvis_tx" in df_phase.columns else np.nan
    pel["unstable_phases"] = [
        ph for ph in ("Standing", "Bottom")
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
        lum["rom"] = float(le.max() - le.min())
        lum["max_ext_change"] = float(dev.max())    # 立位より反った最大量（+）
        lum["max_flex_change"] = float(-dev.min())  # 立位より丸まった最大量（+）
        lum["bottom_change"] = bottom_minus_standing("lumbar_extension")
        # 最も大きく動いた局面
        ph_rom = {ph: _phase_value(phase_summary_df, "lumbar_extension", ph, "ROM") for ph in phase_order}
        ph_rom = {k: v for k, v in ph_rom.items() if pd.notna(v)}
        lum["main_phase"] = max(ph_rom, key=ph_rom.get) if ph_rom else None
    d["lumbar"] = lum

    # ---------------- 反復全体（深さ・テンポ） ----------------
    if reps:
        depths = np.array([p["depth_m"] for p in reps])
        desc_t = np.array([p["descent_s"] for p in reps])
        asc_t = np.array([p["ascent_s"] for p in reps])
        d["depth_cv_pct"] = float(depths.std() / depths.mean() * 100) if depths.mean() else 0.0
        d["depth_mean_cm"] = float(depths.mean() * 100)
        d["descent_mean_s"] = float(desc_t.mean())
        d["ascent_mean_s"] = float(asc_t.mean())
        d["tempo_ratio"] = float(desc_t.mean() / asc_t.mean()) if asc_t.mean() else np.nan
    return d


# ===============================================================
# 2. 臨床向けコメント（PDF Report）
# ===============================================================
def generate_squat_auto_comment(lang_code, overall_score, details, feature_values=None):
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
                   f"解析した反復回数は{details['n_reps']}回です。")
    else:
        lvl = "indicating generally good performance" if overall_score >= 80 else (
            "with some room for improvement" if overall_score >= 60 else "with several findings that warrant attention")
        out.append(f"Overall score: {overall_score}/100, {lvl}. {details['n_reps']} repetitions were analysed.")

    strat = details.get("strategy")
    ratio = details.get("hip_knee_ratio")
    if strat and pd.notna(ratio):
        if ja:
            txt = {"knee": "膝関節優位（股関節の屈曲が相対的に少ない）",
                   "hip": "股関節優位（体幹前傾を伴いやすい）",
                   "balanced": "股関節と膝関節がバランスよく動員された"}[strat]
            out.append(f"最大屈曲角の股関節／膝関節比は{ratio:.2f}で、{txt}しゃがみ方です。")
        else:
            txt = {"knee": "a knee-dominant pattern (relatively less hip flexion)",
                   "hip": "a hip-dominant pattern (often with more trunk lean)",
                   "balanced": "a balanced hip-knee strategy"}[strat]
            out.append(f"Hip/knee peak flexion ratio is {ratio:.2f}, suggesting {txt}.")

    # ---------------- 各関節 ----------------
    for key, (ja_t, en_t) in {"hip": ("股関節", "Hip"), "knee": ("膝関節", "Knee"),
                              "ankle": ("足関節", "Ankle")}.items():
        if key not in J:
            continue
        j = J[key]
        head(ja_t, en_t)
        verb_ja = "背屈" if key == "ankle" else "屈曲"
        verb_en = "dorsiflexion" if key == "ankle" else "flexion"

        # 最大角度・ROM・健常範囲
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
                line += f"{'・'.join(below)}で健常範囲を下回っており、可動域制限が疑われます。"
            elif above:
                line += f"{'・'.join(above)}で健常範囲を上回っており、過可動または代償的な動きの可能性があります。"
            else:
                line += "左右とも健常範囲内です。"
        else:
            if below:
                line += f" Below the reference range on the {', '.join(below)} side, suggesting restricted ROM."
            elif above:
                line += f" Above the reference range on the {', '.join(above)} side, suggesting hypermobility or compensation."
            else:
                line += " Both sides within the reference range."
        out.append(line)

        # 左右差
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

        # タイミング差
        if j.get("timing_lag_ms", 0) > th["timing_lag_ms"]:
            out.append(f"左右のピーク到達時刻に平均{j['timing_lag_ms']:.0f}msのずれがあり、"
                       "左右の協調性（タイミング）に差がある可能性があります。" if ja else
                       f"Mean left-right lag in peak timing is {j['timing_lag_ms']:.0f} ms, "
                       "suggesting a possible bilateral coordination difference.")

        # 反復の一貫性
        if "rep_cv_pct" in j:
            if j["rep_cv_pct"] > th["rep_cv_pct"]:
                out.append(f"反復ごとのピーク角度の変動係数は{j['rep_cv_pct']:.1f}%と大きく、"
                           "動作の再現性にばらつきがあります。" if ja else
                           f"Rep-to-rep variability of peak angle is high (CV {j['rep_cv_pct']:.1f}%).")
            if abs(j.get("rep_trend_deg", 0)) > th["rep_trend_deg"]:
                trend = j["rep_trend_deg"]
                out.append((f"前半から後半にかけてピーク角度が{abs(trend):.1f}°"
                            f"{'増加' if trend > 0 else '減少'}しており、"
                            f"{'慣れ・可動域の拡大' if trend > 0 else '疲労や動作の浅化'}の影響が考えられます。") if ja else
                           (f"Peak angle {'increased' if trend > 0 else 'decreased'} by {abs(trend):.1f}° "
                            f"from early to late reps, possibly reflecting "
                            f"{'warm-up / increasing ROM' if trend > 0 else 'fatigue or shallower reps'}."))

        # 下降／上昇
        if j["ecc_con_pct"] > th["ecc_con_pct"]:
            out.append(f"下降{j['desc_rom']:.1f}°／上昇{j['asc_rom']:.1f}°で{j['ecc_con_pct']:.1f}%の差があり、"
                       "遠心性と求心性の制御に違いがある可能性があります。" if ja else
                       f"Descending {j['desc_rom']:.1f}° vs Ascending {j['asc_rom']:.1f}° "
                       f"({j['ecc_con_pct']:.1f}% difference), suggesting differing eccentric/concentric control.")

        # 静止局面の安定性
        if j["unstable_phases"]:
            ph = "・".join(N[p] for p in j["unstable_phases"]) if ja else ", ".join(N[p] for p in j["unstable_phases"])
            out.append(f"{ph}局面で角度の標準偏差が大きく、保持中の動揺が見られます。" if ja else
                       f"Elevated SD during {ph}, indicating sway while holding the position.")

        # 足関節固有：背屈制限
        if key == "ankle":
            pk = min(j["peak_R"], j["peak_L"])
            if pk < th["ankle_df_limited"]:
                out.append(f"最大背屈が{pk:.1f}°と小さく、背屈制限がしゃがみの深さや体幹前傾の増加に"
                           "影響している可能性があります。" if ja else
                           f"Peak dorsiflexion is limited ({pk:.1f}°), which may restrict depth and increase trunk lean.")

    # ---------------- 骨盤 ----------------
    p = details["pelvis"]
    head("骨盤", "Pelvis")
    items = []
    if pd.notna(p["tilt_rom"]):
        flag = p["tilt_rom"] > th["pelvis_rom_deg"]
        items.append((f"前後傾の可動範囲{p['tilt_rom']:.1f}°（最下点で立位から{p['tilt_change']:+.1f}°）"
                      if ja else f"tilt range {p['tilt_rom']:.1f}° ({p['tilt_change']:+.1f}° at bottom vs standing)", flag))
    if pd.notna(p["list_rom"]):
        flag = p["list_rom"] > th["pelvis_list_deg"]
        items.append((f"側方傾斜{p['list_rom']:.1f}°" if ja else f"obliquity range {p['list_rom']:.1f}°", flag))
    if pd.notna(p["rot_rom"]):
        flag = p["rot_rom"] > th["pelvis_rom_deg"]
        items.append((f"回旋{p['rot_rom']:.1f}°" if ja else f"rotation range {p['rot_rom']:.1f}°", flag))
    if pd.notna(p["shift_mm"]):
        flag = abs(p["shift_mm"]) > th["pelvis_shift_mm"]
        items.append((f"最下点での左右移動{abs(p['shift_mm']):.0f}mm" if ja else
                      f"lateral shift at bottom {abs(p['shift_mm']):.0f} mm", flag))
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
        # 左右差のある関節と骨盤の側方移動の関連
        asym_j = [k for k, j in J.items() if j["asym_pct"] > th["asym_pct"]]
        if asym_j and pd.notna(p["shift_mm"]) and abs(p["shift_mm"]) > th["pelvis_shift_mm"]:
            out.append("下肢の左右差と骨盤の側方移動が同時に見られ、片側への荷重偏りが示唆されます。" if ja else
                       "Lower-limb asymmetry together with lateral pelvic shift suggests uneven weight bearing.")
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
            parts.append(f"最大{lum['max_ext_change']:.1f}°反り" if ja else f"up to {lum['max_ext_change']:.1f}° more extension")
        if lum["max_flex_change"] >= nd:
            parts.append(f"最大{lum['max_flex_change']:.1f}°丸まり" if ja else f"up to {lum['max_flex_change']:.1f}° more flexion")
        if ja:
            chg = ("立位と比べて" + "、".join(parts) + "、") if parts else "立位からの変化は小さく、"
            out.append(f"腰椎伸展の可動範囲は{lum['rom']:.1f}°。{chg}最下点では立位から{lum['bottom_change']:+.1f}°の変化です。")
        else:
            chg = ("Relative to standing: " + " and ".join(parts) + "; ") if parts else "Little change from standing; "
            out.append(f"Lumbar extension range {lum['rom']:.1f}°. {chg}{lum['bottom_change']:+.1f}° at bottom vs standing.")
        if lum["max_ext_change"] > th["lumbar_change_deg"] and lum["max_ext_change"] >= lum["max_flex_change"]:
            out.append("主に腰を反らせる代償（腰椎過伸展）が見られ、腰部への負担増加に注意が必要です。" if ja else
                       "Predominantly extension-type compensation (lumbar hyperextension); monitor lumbar loading.")
        elif lum["max_flex_change"] > th["lumbar_change_deg"]:
            out.append("主に腰が丸まる代償（腰椎屈曲）が見られ、最下点付近での骨盤後傾を伴っている可能性があります。" if ja else
                       "Predominantly flexion-type compensation (lumbar flexion), possibly with posterior pelvic tilt near the bottom.")
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
            out.append(f"平均の沈み込みは{details['depth_mean_cm']:.1f}cm（変動係数{details['depth_cv_pct']:.1f}%）、"
                       f"下降{details['descent_mean_s']:.2f}秒／上昇{details['ascent_mean_s']:.2f}秒です。")
        else:
            out.append(f"Mean pelvic descent {details['depth_mean_cm']:.1f} cm (CV {details['depth_cv_pct']:.1f}%); "
                       f"descent {details['descent_mean_s']:.2f} s / ascent {details['ascent_mean_s']:.2f} s.")
        tr = details.get("tempo_ratio")
        if pd.notna(tr) and tr < 1 / th["tempo_ratio"]:
            out.append("下降が上昇に比べて速く、しゃがむ局面の遠心性コントロールが不十分な可能性があります。" if ja else
                       "Descent is notably faster than ascent, suggesting limited eccentric control.")
        elif pd.notna(tr) and tr > th["tempo_ratio"]:
            out.append("上昇が下降に比べて速く、反動を利用して立ち上がっている可能性があります。" if ja else
                       "Ascent is notably faster than descent, possibly using momentum to rise.")
        if details["depth_cv_pct"] > th["rep_cv_pct"]:
            out.append("反復ごとの深さにばらつきがあり、動作の再現性に課題があります。" if ja else
                       "Squat depth varied between reps, indicating limited consistency.")

    # ---------------- 締め ----------------
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
    """専門用語を減らし、部位ごとに「良い点」と「気をつけたい点」を平易な言葉で伝える。"""
    th = THRESHOLDS
    J = details["joints"]
    ja = lang_code == "ja"
    NM = CLIENT_NAMES_JA if ja else CLIENT_NAMES_EN
    side = {"R": "右", "L": "左"} if ja else {"R": "right", "L": "left"}
    good, care = [], []

    # 総合
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

    # しゃがみ方
    strat = details.get("strategy")
    if strat == "knee":
        care.append("しゃがむときに、ひざが主に動いていて、股関節（お尻を後ろに引く動き）があまり使えていません。"
                    if ja else "When squatting, your knees do most of the work and your hips (sitting back) are used less.")
    elif strat == "hip":
        care.append("しゃがむときに、上半身が前に倒れやすい傾向があります。"
                    if ja else "Your upper body tends to lean forward as you squat.")
    elif strat == "balanced":
        good.append("股関節とひざをバランスよく使ってしゃがめています。"
                    if ja else "You use your hips and knees in good balance.")

    # 各関節
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
        if abs(j.get("rep_trend_deg", 0)) > th["rep_trend_deg"] and j.get("rep_trend_deg", 0) < 0:
            care.append(f"回数を重ねるにつれて、{NM[key]}の曲がりが浅くなっていました（疲れのサインかもしれません）。" if ja else
                        f"Your {NM[key]} bent less as the reps went on — possibly a sign of fatigue.")
        if key == "ankle" and min(j["peak_R"], j["peak_L"]) < th["ankle_df_limited"]:
            care.append("足首がかたく、しゃがむ深さや姿勢に影響している可能性があります。" if ja else
                        "Stiff ankles may be limiting your depth and posture.")

    if not any(j.get("below_R") or j.get("below_L") for j in J.values()):
        good.append("股関節・ひざ・足首は、しっかり動かせています。" if ja else
                    "Your hips, knees and ankles all moved through a good range.")
    if all(j["asym_pct"] <= th["asym_pct"] for j in J.values()):
        good.append("左右の脚をバランスよく使えています。" if ja else "You used both legs evenly.")

    # 骨盤
    p = details["pelvis"]
    pel_flags = [
        pd.notna(p["tilt_rom"]) and p["tilt_rom"] > th["pelvis_rom_deg"],
        pd.notna(p["list_rom"]) and p["list_rom"] > th["pelvis_list_deg"],
        pd.notna(p["rot_rom"]) and p["rot_rom"] > th["pelvis_rom_deg"],
    ]
    shift = pd.notna(p["shift_mm"]) and abs(p["shift_mm"]) > th["pelvis_shift_mm"]
    if shift:
        care.append("しゃがんだときに、骨盤（腰の土台）が左右どちらかに少しずれています。片側に体重がかかりやすい状態です。"
                    if ja else "Your pelvis shifts slightly to one side at the bottom, so more weight goes onto one leg.")
    if any(pel_flags):
        care.append("動作の途中で、骨盤が傾いたりねじれたりしやすい傾向があります。"
                    if ja else "Your pelvis tends to tilt or twist during the movement.")
    if not shift and not any(pel_flags):
        good.append("骨盤（腰の土台）は安定していました。" if ja else "Your pelvis stayed stable.")

    # 腰部
    lum = details["lumbar"]
    if lum:
        if lum["max_ext_change"] > th["lumbar_change_deg"] and lum["max_ext_change"] >= lum["max_flex_change"]:
            care.append("しゃがむ・立ち上がるときに腰が反りやすく、腰まわりに負担がかかりやすい状態です。"
                        if ja else "Your lower back tends to arch while squatting and standing up, which can strain the area.")
        elif lum["max_flex_change"] > th["lumbar_change_deg"]:
            care.append("深くしゃがんだところで腰が丸まりやすい傾向があります。"
                        if ja else "Your lower back tends to round at the bottom of the squat.")
        else:
            good.append("腰の反りや丸まりは少なく、よい姿勢を保てています。"
                        if ja else "Your lower back stayed in a good position.")

    # テンポ
    tr = details.get("tempo_ratio")
    if pd.notna(tr) and tr < 1 / th["tempo_ratio"]:
        care.append("しゃがむスピードが速めです。ゆっくりしゃがむと、より安全で効果的です。"
                    if ja else "You squat down quite fast — lowering more slowly is safer and more effective.")

    # 組み立て
    lines = [head]
    sep = "" if ja else " "
    if good:
        lines.append(("◎ よくできている点：" if ja else "Strengths: ") + sep.join(good[:3]))
    if care:
        lines.append(("△ 気をつけたい点：" if ja else "Points to watch: ") + sep.join(care[:4]))
    lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。" if ja else
                 "Try the recommended actions below at a comfortable pace before your next check.")
    return "\n".join(lines)
