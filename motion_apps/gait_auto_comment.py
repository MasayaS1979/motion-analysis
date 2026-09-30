"""
gait_auto_comment.py
---------------------------------------------------------------
Gait Analysis の自動コメント（PDF Report / Client Report）用モジュール。
squat / sit_stand / arm_flexion の auto_comment と同じ構成・同じ呼び出し方にそろえています。

実測データ（df_phase）から、股関節・膝関節・足関節・骨盤・腰部・歩行全体について
以下を読み取り、臨床向け／クライアント向けのコメント下書きを生成します。

  - 各関節の可動域（左右別）と正常範囲（Healthy_ROM）との比較
  - 歩行で重要なピーク角度：股関節伸展（蹴り出し）、遊脚期の膝屈曲（足の振り出し）、
    足関節の背屈・底屈（蹴り出しの強さ）
  - 左右差（可動域・ピーク角度・ストライド時間）
  - 歩行の一貫性（ストライド時間の変動係数）
  - 骨盤：前後傾・側方傾斜・回旋・左右の揺れ（pelvis_tz）
  - 腰部：可動範囲
  - 歩行速度・歩幅の目安（pelvis_tx / pelvis_tz の水平移動量から推定）

■ ストライドの検出
  歩行周期は、遊脚期に最も大きく曲がる「膝屈曲のピーク」を左右それぞれ検出して区切ります
  （足関節角度より波形がはっきりしており、検出が安定するため）。

■ 他動作との違い
  - Healthy_ROM は「角度の範囲（最小〜最大）」で定義されているため、
    正常範囲との比較は「可動域の大きさ（最大−最小）」と「ピーク角度」の両方で見ます。
  - 骨盤・腰の指標は「動きが少ないほど良い」代償指標として扱い、小さすぎても問題にしません。
  - 左右の骨盤移動は pelvis_tz（OpenSim/OpenCap の左右方向）、進行方向は pelvis_tx です。

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
    "asym_pct": 15.0,             # 可動域の左右差（%）
    "rom_diff_pct": 20.0,         # 正常範囲の可動域との差（%）— アプリの Status 判定と同じ
    "stride_asym_pct": 10.0,      # ストライド時間の左右差（%）
    "stride_cv_pct": 5.0,         # ストライド時間の変動係数（%）
    "hip_ext_min_deg": 5.0,       # 股関節伸展（最小屈曲角の符号反転）がこれ未満 → 伸展不足
    "knee_swing_min_deg": 50.0,   # 遊脚期の膝屈曲ピークがこれ未満 → 振り出し時の膝の曲がり不足
    "ankle_pf_min_deg": 5.0,      # 蹴り出しの底屈がこれ未満 → 蹴り出しが弱い
    "ankle_df_min_deg": 5.0,      # 背屈がこれ未満 → 背屈制限
    "pelvis_rom_deg": 10.0,       # 骨盤前後傾・回旋の可動範囲（°）
    "pelvis_list_deg": 10.0,      # 骨盤側方傾斜の可動範囲（°）
    "lumbar_rom_deg": 10.0,       # 腰椎伸展の可動範囲（°）
    "lateral_sway_mm": 60.0,      # 骨盤の左右の揺れ幅（トレンド除去後、mm）
    "speed_slow_mps": 1.0,        # 歩行速度がこれ未満 → ゆっくり
    "cadence_low": 90.0,          # ケイデンス下限（歩/分）
    "cadence_high": 130.0,        # ケイデンス上限（歩/分）
    "phase_rom_min_deg": 10.0,    # フェーズ別左右差を評価する最小ROM
}

COMPENSATION_VARS = {"lumbar_extension", "pelvis_tilt", "pelvis_rotation", "pelvis_list"}

JOINTS = {
    "hip": ("hip_flexion_r", "hip_flexion_l"),
    "knee": ("knee_angle_r", "knee_angle_l"),
    "ankle": ("ankle_angle_r", "ankle_angle_l"),
}

NAMES = {
    "ja": {"hip": "股関節", "knee": "膝関節", "ankle": "足関節", "R": "右", "L": "左",
           "Heel Strike": "踵接地", "Mid Stance": "立脚中期", "Toe Off": "蹴り出し", "Swing": "遊脚期"},
    "en": {"hip": "Hip", "knee": "Knee", "ankle": "Ankle", "R": "right", "L": "left",
           "Heel Strike": "Heel Strike", "Mid Stance": "Mid Stance", "Toe Off": "Toe Off", "Swing": "Swing"},
}
CLIENT_NAMES_JA = {"hip": "股関節", "knee": "ひざ", "ankle": "足首"}
CLIENT_NAMES_EN = {"hip": "hips", "knee": "knees", "ankle": "ankles"}


# ===============================================================
# 1. データの読み取り
# ===============================================================
def _pct_diff(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m * 100


def _nanmean(values):
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _phase_value(phase_summary_df, var, phase, stat):
    row = phase_summary_df[phase_summary_df["Variable"] == var]
    if len(row) == 0:
        return np.nan
    col = f"{phase}_{stat}"
    return row[col].iloc[0] if col in row.columns else np.nan


def _detect_strides(knee_angle, fs=FS):
    """遊脚期の膝屈曲ピークを検出し、ピーク位置（フレーム）のリストを返す。"""
    s = pd.Series(knee_angle).rolling(5, center=True).mean().bfill().ffill().to_numpy()
    if len(s) < 3:
        return []
    lo, hi = np.nanmin(s), np.nanmax(s)
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 5:
        return []
    line = lo + rng * 0.6
    min_gap = int(0.6 * fs)
    cand = [i for i in range(1, len(s) - 1)
            if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > line]
    cand.sort(key=lambda i: -s[i])
    peaks = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in peaks):
            peaks.append(i)
    return sorted(peaks)


def analyze_gait_details(df_phase, phase_summary_df, comparison_df,
                         phase_order=("Heel Strike", "Mid Stance", "Toe Off", "Swing"),
                         fs=FS):
    """実測データから、歩行の詳細な特徴量をまとめて返す。"""
    th = THRESHOLDS
    n = len(df_phase)
    d = {"joints": {}, "duration_s": n / fs if n else 0.0}

    # ---------------- ストライド（膝屈曲ピーク） ----------------
    strides = {}
    for side, col in (("R", "knee_angle_r"), ("L", "knee_angle_l")):
        strides[side] = _detect_strides(df_phase[col].to_numpy(), fs) if col in df_phase.columns else []
    d["strides"] = strides
    stride_t = {s: np.diff(p) / fs for s, p in strides.items() if len(p) >= 2}
    d["n_strides"] = int(sum(max(0, len(p) - 1) for p in strides.values()))
    if stride_t:
        all_t = np.concatenate(list(stride_t.values()))
        d["stride_time_s"] = float(np.mean(all_t))
        d["stride_cv_pct"] = float(np.std(all_t) / np.mean(all_t) * 100) if np.mean(all_t) else 0.0
        d["cadence_est"] = float(120.0 / np.mean(all_t)) if np.mean(all_t) else np.nan  # 1ストライド＝2歩
        if "R" in stride_t and "L" in stride_t:
            d["stride_asym_pct"] = _pct_diff(float(np.mean(stride_t["R"])), float(np.mean(stride_t["L"])))

    # ---------------- 歩行速度・歩幅（参考値） ----------------
    if all(c in df_phase.columns for c in ("pelvis_tx", "pelvis_tz")) and n > fs:
        dx = float(df_phase["pelvis_tx"].iloc[-1] - df_phase["pelvis_tx"].iloc[0])
        dz = float(df_phase["pelvis_tz"].iloc[-1] - df_phase["pelvis_tz"].iloc[0])
        dist = float(np.hypot(dx, dz))
        speed = dist / d["duration_s"] if d["duration_s"] else np.nan
        # その場歩き・トレッドミルでは前進しないため、0.2 m/s 未満は算出しない
        d["speed_mps"] = speed if pd.notna(speed) and speed >= 0.2 else np.nan
        if pd.notna(d["speed_mps"]) and pd.notna(d.get("cadence_est", np.nan)):
            d["step_length_m"] = d["speed_mps"] * 60.0 / d["cadence_est"]

    # ---------------- 下肢関節 ----------------
    for key, (rv, lv) in JOINTS.items():
        if rv not in df_phase.columns or lv not in df_phase.columns:
            continue
        r, l = df_phase[rv].to_numpy(), df_phase[lv].to_numpy()
        j = {
            "max_R": float(np.nanmax(r)), "max_L": float(np.nanmax(l)),
            "min_R": float(np.nanmin(r)), "min_L": float(np.nanmin(l)),
        }
        j["rom_R"], j["rom_L"] = j["max_R"] - j["min_R"], j["max_L"] - j["min_L"]
        j["asym_pct"] = _pct_diff(j["rom_R"], j["rom_L"])
        j["larger_side"] = "R" if j["rom_R"] >= j["rom_L"] else "L"

        # 正常範囲（角度範囲 → 可動域）との比較
        for side, var in (("R", rv), ("L", lv)):
            row = comparison_df[comparison_df["Variable"] == var]
            if len(row) and "Healthy_Min" in row.columns and "Healthy_Max" in row.columns:
                ref = float(row["Healthy_Max"].iloc[0] - row["Healthy_Min"].iloc[0])
                rom = j[f"rom_{side}"]
                pct = (rom - ref) / ref * 100 if ref else 0.0
                j[f"ref_rom"] = ref
                j[f"below_{side}"] = pct < -th["rom_diff_pct"]
                j[f"above_{side}"] = pct > th["rom_diff_pct"]

        # フェーズ別左右差
        phase_asym = {}
        for ph in phase_order:
            a = _phase_value(phase_summary_df, rv, ph, "ROM")
            b = _phase_value(phase_summary_df, lv, ph, "ROM")
            if pd.notna(a) and pd.notna(b) and max(a, b) >= th["phase_rom_min_deg"]:
                phase_asym[ph] = _pct_diff(a, b)
        if phase_asym:
            j["worst_phase"] = max(phase_asym, key=phase_asym.get)
            j["worst_phase_asym"] = phase_asym[j["worst_phase"]]

        # 関節ごとの歩行特有の所見
        if key == "hip":
            j["ext_R"], j["ext_L"] = -j["min_R"], -j["min_L"]   # 股関節伸展量（+ が伸展）
            j["ext_limited"] = [s for s in ("R", "L") if j[f"ext_{s}"] < th["hip_ext_min_deg"]]
        elif key == "knee":
            # 遊脚期の膝屈曲ピーク（ストライドごとの平均）
            for side, arr in (("R", r), ("L", l)):
                pk = strides.get(side, [])
                j[f"swing_peak_{side}"] = float(np.mean(arr[pk])) if len(pk) else j[f"max_{side}"]
            j["swing_low"] = [s for s in ("R", "L") if j[f"swing_peak_{s}"] < th["knee_swing_min_deg"]]
        elif key == "ankle":
            j["pf_R"], j["pf_L"] = -j["min_R"], -j["min_L"]      # 底屈量（+ が底屈）
            j["pf_weak"] = [s for s in ("R", "L") if j[f"pf_{s}"] < th["ankle_pf_min_deg"]]
            j["df_limited"] = [s for s in ("R", "L") if j[f"max_{s}"] < th["ankle_df_min_deg"]]
        d["joints"][key] = j

    # ---------------- 骨盤 ----------------
    def rng(col):
        return float(df_phase[col].max() - df_phase[col].min()) if col in df_phase.columns else np.nan

    pel = {"tilt_rom": rng("pelvis_tilt"), "list_rom": rng("pelvis_list"), "rot_rom": rng("pelvis_rotation")}
    if "pelvis_tz" in df_phase.columns and n > 3:
        z = df_phase["pelvis_tz"].to_numpy(dtype=float)
        t = np.arange(n)
        ok = np.isfinite(z)
        if ok.sum() > 3:
            trend = np.polyval(np.polyfit(t[ok], z[ok], 1), t)   # 歩く向きのずれ（ドリフト）を除く
            resid = z - trend
            pel["sway_mm"] = float((np.nanmax(resid) - np.nanmin(resid)) * 1000)
    d["pelvis"] = pel

    # ---------------- 腰部 ----------------
    d["lumbar"] = {"rom": rng("lumbar_extension")} if "lumbar_extension" in df_phase.columns else {}
    return d


def _flag(value, key):
    return pd.notna(value) and value > THRESHOLDS[key]


def _sides(sides, lang):
    N = NAMES[lang]
    if len(sides) == 2:
        return "両側" if lang == "ja" else "both sides"
    return (N[sides[0]] + "側") if lang == "ja" else f"the {N[sides[0]]} side"


# ===============================================================
# 2. 臨床向けコメント（PDF Report）
# ===============================================================
def generate_gait_auto_comment(lang_code, overall_score, details, cadence=None, feature_values=None):
    """専門職向けの詳細コメントを、部位ごとの見出し付きで生成する。"""
    th = THRESHOLDS
    N = NAMES[lang_code]
    J = details["joints"]
    ja = lang_code == "ja"
    out = []

    def head(ja_t, en_t):
        out.append("")
        out.append(f"【{ja_t}】" if ja else f"[{en_t}]")

    # ---------------- 総合・時間距離因子 ----------------
    out.append("【総合】" if ja else "[Overall]")
    if ja:
        lvl = "良好な歩行パターンを示しています" if overall_score >= 80 else (
            "軽度〜中等度の逸脱が見られます" if overall_score >= 60 else "歩行パターンに明らかな逸脱が見られます")
        out.append(f"総合スコアは{overall_score}/100で、{lvl}。解析したストライド数は{details['n_strides']}回です。")
    else:
        lvl = "indicating a favorable gait pattern" if overall_score >= 80 else (
            "with mild to moderate deviation" if overall_score >= 60 else "with clear deviation from a typical pattern")
        out.append(f"Overall score: {overall_score}/100, {lvl}. {details['n_strides']} strides were analysed.")

    cad = cadence if cadence is not None else details.get("cadence_est", np.nan)
    parts = []
    if pd.notna(cad):
        parts.append(f"ケイデンス{cad:.1f}歩/分" if ja else f"cadence {cad:.1f} steps/min")
    if pd.notna(details.get("stride_time_s", np.nan)):
        parts.append(f"ストライド時間{details['stride_time_s']:.2f}秒" if ja else f"stride time {details['stride_time_s']:.2f} s")
    if pd.notna(details.get("speed_mps", np.nan)):
        parts.append(f"推定歩行速度{details['speed_mps']:.2f}m/秒" if ja else f"estimated speed {details['speed_mps']:.2f} m/s")
    if pd.notna(details.get("step_length_m", np.nan)):
        parts.append(f"推定歩幅{details['step_length_m'] * 100:.0f}cm" if ja else f"estimated step length {details['step_length_m'] * 100:.0f} cm")
    if parts:
        out.append(("時間距離因子：" + "、".join(parts) + "。") if ja else ("Spatiotemporal: " + ", ".join(parts) + "."))
    if pd.notna(cad) and cad < th["cadence_low"]:
        out.append("ケイデンスが目安（90〜130歩/分）を下回っています。" if ja else "Cadence is below the typical 90–130 steps/min.")
    elif pd.notna(cad) and cad > th["cadence_high"]:
        out.append("ケイデンスが目安（90〜130歩/分）を上回っています。" if ja else "Cadence is above the typical 90–130 steps/min.")
    if pd.notna(details.get("speed_mps", np.nan)) and details["speed_mps"] < th["speed_slow_mps"]:
        out.append("推定歩行速度が1.0m/秒を下回っており、歩行能力の低下に注意が必要です。" if ja else
                   "Estimated gait speed is below 1.0 m/s, which warrants attention to walking capacity.")
    if details.get("stride_asym_pct", 0) > th["stride_asym_pct"]:
        out.append(f"ストライド時間に左右で{details['stride_asym_pct']:.1f}%の差があり、時間的な左右非対称性が見られます。" if ja else
                   f"Stride time differs by {details['stride_asym_pct']:.1f}% between sides (temporal asymmetry).")
    if details.get("stride_cv_pct", 0) > th["stride_cv_pct"]:
        out.append(f"ストライド時間の変動係数が{details['stride_cv_pct']:.1f}%と大きく、歩行リズムのばらつき（転倒リスクとの関連に注意）が見られます。" if ja else
                   f"Stride time variability is high (CV {details['stride_cv_pct']:.1f}%), which may be associated with fall risk.")

    # ---------------- 各関節 ----------------
    for key, (ja_t, en_t) in {"hip": ("股関節", "Hip"), "knee": ("膝関節", "Knee"), "ankle": ("足関節", "Ankle")}.items():
        if key not in J:
            continue
        j = J[key]
        head(ja_t, en_t)
        if ja:
            line = f"可動域は右{j['rom_R']:.1f}°／左{j['rom_L']:.1f}°"
            line += f"（正常範囲の可動域{j['ref_rom']:.0f}°）。" if "ref_rom" in j else "。"
        else:
            line = f"ROM R {j['rom_R']:.1f}° / L {j['rom_L']:.1f}°"
            line += f" (normal-range ROM {j['ref_rom']:.0f}°)." if "ref_rom" in j else "."
        below = [s for s in ("R", "L") if j.get(f"below_{s}")]
        above = [s for s in ("R", "L") if j.get(f"above_{s}")]
        if below:
            line += (f"{_sides(below, 'ja')}で正常範囲より{th['rom_diff_pct']:.0f}%以上小さく、可動域の低下が疑われます。" if ja else
                     f" More than {th['rom_diff_pct']:.0f}% below the normal range on {_sides(below, 'en')}, suggesting reduced ROM.")
        elif above:
            line += (f"{_sides(above, 'ja')}で正常範囲より{th['rom_diff_pct']:.0f}%以上大きくなっています。" if ja else
                     f" More than {th['rom_diff_pct']:.0f}% above the normal range on {_sides(above, 'en')}.")
        out.append(line)

        if key == "hip" and j["ext_limited"]:
            out.append(f"{_sides(j['ext_limited'], 'ja')}で股関節伸展が不足しており（右{j['ext_R']:.1f}°／左{j['ext_L']:.1f}°）、"
                       "立脚後期の蹴り出し・歩幅の減少や、腰椎伸展による代償につながる可能性があります。" if ja else
                       f"Limited hip extension on {_sides(j['ext_limited'], 'en')} (R {j['ext_R']:.1f}° / L {j['ext_L']:.1f}°), "
                       "which may shorten step length or be compensated by lumbar extension.")
        if key == "knee" and j["swing_low"]:
            out.append(f"{_sides(j['swing_low'], 'ja')}で遊脚期の膝屈曲が小さく（右{j['swing_peak_R']:.1f}°／左{j['swing_peak_L']:.1f}°）、"
                       "つま先の引っかかり（トウクリアランス低下）に注意が必要です。" if ja else
                       f"Reduced swing-phase knee flexion on {_sides(j['swing_low'], 'en')} "
                       f"(R {j['swing_peak_R']:.1f}° / L {j['swing_peak_L']:.1f}°); monitor toe clearance.")
        if key == "ankle":
            if j["pf_weak"]:
                out.append(f"{_sides(j['pf_weak'], 'ja')}で蹴り出し時の底屈が小さく（右{j['pf_R']:.1f}°／左{j['pf_L']:.1f}°）、"
                           "推進力の低下が疑われます。" if ja else
                           f"Limited push-off plantarflexion on {_sides(j['pf_weak'], 'en')} "
                           f"(R {j['pf_R']:.1f}° / L {j['pf_L']:.1f}°), suggesting reduced propulsion.")
            if j["df_limited"]:
                out.append(f"{_sides(j['df_limited'], 'ja')}で背屈が小さく、立脚中期の下腿前傾が制限されている可能性があります。" if ja else
                           f"Limited dorsiflexion on {_sides(j['df_limited'], 'en')}, possibly restricting forward tibial progression.")

        if j["asym_pct"] > th["asym_pct"]:
            out.append(f"可動域の左右差は{j['asym_pct']:.1f}%でしきい値を超え、{N[j['larger_side']]}側が大きくなっています。" if ja else
                       f"ROM asymmetry is {j['asym_pct']:.1f}% (above threshold), larger on the {N[j['larger_side']]} side.")
        elif j.get("worst_phase_asym", 0) > th["asym_pct"]:
            out.append(f"試技全体の左右差は{j['asym_pct']:.1f}%と小さいものの、{N[j['worst_phase']]}では{j['worst_phase_asym']:.1f}%の差があります。" if ja else
                       f"Overall asymmetry is small ({j['asym_pct']:.1f}%), but reaches {j['worst_phase_asym']:.1f}% during {N[j['worst_phase']]}.")
        else:
            out.append(f"可動域の左右差は{j['asym_pct']:.1f}%で、しきい値内です。" if ja else
                       f"ROM asymmetry is {j['asym_pct']:.1f}%, within threshold.")

    # ---------------- 骨盤・腰部 ----------------
    p, lum = details["pelvis"], details["lumbar"]
    head("骨盤・腰部", "Pelvis & Lumbar")
    items = []
    if pd.notna(p.get("tilt_rom", np.nan)):
        items.append((f"前後傾{p['tilt_rom']:.1f}°" if ja else f"tilt {p['tilt_rom']:.1f}°", _flag(p["tilt_rom"], "pelvis_rom_deg")))
    if pd.notna(p.get("list_rom", np.nan)):
        items.append((f"側方傾斜{p['list_rom']:.1f}°" if ja else f"obliquity {p['list_rom']:.1f}°", _flag(p["list_rom"], "pelvis_list_deg")))
    if pd.notna(p.get("rot_rom", np.nan)):
        items.append((f"回旋{p['rot_rom']:.1f}°" if ja else f"rotation {p['rot_rom']:.1f}°", _flag(p["rot_rom"], "pelvis_rom_deg")))
    if pd.notna(p.get("sway_mm", np.nan)):
        items.append((f"左右の揺れ幅{p['sway_mm']:.0f}mm" if ja else f"lateral sway {p['sway_mm']:.0f} mm", _flag(p["sway_mm"], "lateral_sway_mm")))
    if lum and pd.notna(lum.get("rom", np.nan)):
        items.append((f"腰椎伸展{lum['rom']:.1f}°" if ja else f"lumbar extension {lum['rom']:.1f}°", _flag(lum["rom"], "lumbar_rom_deg")))
    if items:
        sep = "、" if ja else ", "
        out.append(("計測値（可動範囲）：" if ja else "Measured ranges: ") + sep.join(t for t, _ in items) + ("。" if ja else "."))
        flagged = [t for t, f in items if f]
        if flagged:
            out.append(("しきい値を超えた項目：" + "、".join(flagged) + "。体幹・骨盤で歩行を代償している可能性があります。") if ja else
                       ("Above threshold: " + ", ".join(flagged) + ". Trunk/pelvic compensation may be present."))
            if _flag(p.get("list_rom", np.nan), "pelvis_list_deg"):
                out.append("骨盤の側方傾斜が大きく、立脚側の中殿筋の機能低下（トレンデレンブルグ様の動き）に注意が必要です。" if ja else
                           "Large pelvic obliquity; consider stance-side gluteus medius weakness (Trendelenburg-like pattern).")
        else:
            out.append("いずれもしきい値内で、骨盤・腰部は比較的安定しています。" if ja else
                       "All within thresholds; the pelvis and lumbar spine were relatively stable.")
        hip = J.get("hip", {})
        if hip.get("ext_limited") and lum and _flag(lum.get("rom", np.nan), "lumbar_rom_deg"):
            out.append("股関節伸展の不足と腰椎の大きな動きが同時に見られ、股関節伸展を腰の反りで補っている可能性があります。" if ja else
                       "Limited hip extension with large lumbar motion suggests lumbar extension compensating for the hip.")

    out.append("")
    out.append("※ 実測値からの自動生成による下書きです。しきい値は仮の基準のため、観察・触診所見と合わせて"
               "内容を確認し、必要に応じて修正してください。" if ja else
               "* Auto-generated draft from measured values. Thresholds are provisional; please review alongside "
               "observational and clinical findings and edit as needed.")
    return "\n".join(out)


# ===============================================================
# 3. クライアント向けコメント（Client Report）
# ===============================================================
def generate_client_auto_comment(lang_code, overall_score, details,
                                 mobility_score, symmetry_score, cadence_score,
                                 pelvic_ml_score, lumbar_extension_score, cadence=None):
    """専門用語を減らし、「良い点」と「気をつけたい点」を平易な言葉で伝える。"""
    th = THRESHOLDS
    J = details["joints"]
    ja = lang_code == "ja"
    NM = CLIENT_NAMES_JA if ja else CLIENT_NAMES_EN
    side = {"R": "右", "L": "左"} if ja else {"R": "right", "L": "left"}
    good, care = [], []

    def where(sides):
        if len(sides) == 2:
            return "両脚" if ja else "both legs"
        return (side[sides[0]] + "脚") if ja else f"{side[sides[0]]} leg"

    if ja:
        head = (f"今回の総合スコアは{overall_score:.0f}/100で、" +
                ("とても良い歩き方です。" if overall_score >= 80 else
                 "全体的には悪くありませんが、いくつか気をつけたい点があります。" if overall_score >= 60 else
                 "いくつか改善していきたいポイントが見つかりました。"))
    else:
        head = (f"Your overall score was {overall_score:.0f}/100 — " +
                ("a great walking pattern." if overall_score >= 80 else
                 "reasonably good, with a few points to keep an eye on." if overall_score >= 60 else
                 "we found a few areas worth working on."))

    # 歩くペース・速さ
    cad = cadence if cadence is not None else details.get("cadence_est", np.nan)
    speed = details.get("speed_mps", np.nan)
    if pd.notna(speed) and speed < th["speed_slow_mps"]:
        care.append(f"歩く速さ（約{speed:.1f}m/秒）が、目安の1.0m/秒より少しゆっくりでした。" if ja else
                    f"Your walking speed (about {speed:.1f} m/s) was a little below the 1.0 m/s guideline.")
    elif pd.notna(cad) and cad < th["cadence_low"]:
        care.append(f"歩くテンポ（1分間に約{cad:.0f}歩）がゆっくりめでした。" if ja else
                    f"Your walking rhythm (about {cad:.0f} steps/min) was on the slow side.")
    elif pd.notna(cad) and cad > th["cadence_high"]:
        care.append(f"歩くテンポ（1分間に約{cad:.0f}歩）が速めで、歩幅が小さくなっている可能性があります。" if ja else
                    f"Your walking rhythm (about {cad:.0f} steps/min) was fast — your steps may be short.")
    elif pd.notna(cad):
        good.append("歩くテンポは、ちょうどよい範囲でした。" if ja else "Your walking rhythm was in a good range.")

    # 関節（歩行で重要な3つの動きを1文にまとめる）
    hip, knee, ankle = J.get("hip", {}), J.get("knee", {}), J.get("ankle", {})
    issues = []
    if hip.get("ext_limited"):
        issues.append((hip["ext_limited"], "後ろに蹴り出す動き（股関節を伸ばす動き）", "swinging the leg back behind you"))
    if knee.get("swing_low"):
        issues.append((knee["swing_low"], "前に振り出すときのひざの曲がり", "knee bend as the leg swings forward"))
    if ankle.get("pf_weak"):
        issues.append((ankle["pf_weak"], "地面をけり出す力", "push-off from the ground"))
    if issues:
        same_side = all(sorted(sd) == sorted(issues[0][0]) for sd, _, _ in issues)
        if ja:
            if same_side:
                txt = f"{where(issues[0][0])}の、" + "・".join(ph for _, ph, _ in issues)
            else:
                txt = "、".join(f"{where(sd)}の{ph}" for sd, ph, _ in issues)
            msg = f"{txt}が小さめです。"
            if knee.get("swing_low"):
                msg += "ひざの曲がりが小さいと、つまずきやすくなるので注意しましょう。"
        else:
            if same_side:
                txt = ", ".join(en for _, _, en in issues) + f" ({where(issues[0][0])})"
            else:
                txt = "; ".join(f"{en} ({where(sd)})" for sd, _, en in issues)
            msg = f"Reduced: {txt}."
            if knee.get("swing_low"):
                msg += " Less knee bend can make tripping more likely."
        care.append(msg)
    elif J:
        good.append("股関節・ひざ・足首を、歩くためにしっかり使えています。" if ja else
                    "You use your hips, knees and ankles well while walking.")

    asym = [k for k, j in J.items() if j["asym_pct"] > th["asym_pct"]]
    if asym:
        names = [NM[k] for k in asym]
        txt = "・".join(names) if ja else (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])
        care.append(f"{txt}の動きに左右差があり、片側に負担がかかりやすい歩き方です。" if ja else
                    f"Your {txt} move differently on each side, so one side may carry more load.")
    elif details.get("stride_asym_pct", 0) > th["stride_asym_pct"]:
        care.append("左右の足で、1歩にかかる時間に差がありました。" if ja else
                    "The timing of your steps differs between the left and right legs.")
    elif J:
        good.append("左右の脚をバランスよく使えています。" if ja else "You use both legs evenly.")

    if details.get("stride_cv_pct", 0) > th["stride_cv_pct"]:
        care.append("歩くリズムが一定になりにくく、ふらつきやすい状態かもしれません。" if ja else
                    "Your walking rhythm varies from step to step, which can make you less steady.")

    # 骨盤・腰
    p, lum = details["pelvis"], details["lumbar"]
    pel_flag = any([_flag(p.get("tilt_rom", np.nan), "pelvis_rom_deg"),
                    _flag(p.get("rot_rom", np.nan), "pelvis_rom_deg")])
    list_flag = _flag(p.get("list_rom", np.nan), "pelvis_list_deg")
    sway_flag = _flag(p.get("sway_mm", np.nan), "lateral_sway_mm")
    lum_flag = bool(lum) and _flag(lum.get("rom", np.nan), "lumbar_rom_deg")
    if list_flag or sway_flag:
        care.append("歩いているときに、骨盤（腰の土台）が左右に揺れやすい傾向があります。お尻の横の筋肉を鍛えると安定しやすくなります。" if ja else
                    "Your pelvis tends to sway side to side as you walk — strengthening the muscles at the side of your hips helps.")
    if lum_flag:
        care.append("歩くときに腰が反りやすく、腰まわりに負担がかかりやすい状態です。" if ja else
                    "Your lower back tends to arch as you walk, which can strain the area.")
    elif pel_flag:
        care.append("歩くときに、骨盤が前後に傾いたりねじれたりしやすい傾向があります。" if ja else
                    "Your pelvis tends to tilt or twist as you walk.")
    has_trunk = any(pd.notna(v) for v in list(p.values()) + list(lum.values()))
    if has_trunk and not (list_flag or sway_flag or lum_flag or pel_flag):
        good.append("骨盤や腰は安定していて、よい姿勢で歩けています。" if ja else
                    "Your pelvis and lower back stayed steady — good walking posture.")

    lines = [head]
    sep = "" if ja else " "
    if good:
        lines.append(("◎ よくできている点：" if ja else "Strengths: ") + sep.join(good[:3]))
    if care:
        lines.append(("△ 気をつけたい点：" if ja else "Points to watch: ") + sep.join(care[:4]))
    lines.append("次回までに、下のおすすめアクションを無理のない範囲で続けてみましょう。" if ja else
                 "Try the recommended actions below at a comfortable pace before your next check.")
    return "\n".join(lines)
