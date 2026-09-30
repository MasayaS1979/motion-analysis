# =========================
# Client Report (一般ユーザー向けPDF)
# =========================
with tab9:

    client_lang_choice = st.radio(
        "レポート言語 / Report Language",
        ["日本語", "English"],
        horizontal=True,
        key="arm_client_lang_radio"
    )
    client_lang_code = "ja" if client_lang_choice == "日本語" else "en"

    CLIENT_UI = {
        "ja": {
            "header": "クライアント向けレポート",
            "caption": "専門用語を減らし、スコアと図解を中心にした、対象者本人にそのまま渡せるレポートです。",
            "subject_name": "対象者名",
            "exam_date": "測定日",
            "examiner": "検者",
            "comment_heading": "総合コメント",
            "comment_label": "対象者向けの総合コメントを記入してください（PDFに反映されます）",
            "auto_generate_button": "🪄 コメントを自動生成（下書き）",
            "auto_generate_caption": "実測値をもとに、専門用語を減らした下書きコメントを自動生成します。内容を確認・編集してからPDFを生成してください。",
            "generate_button": "📄 クライアント向けレポートを生成",
            "download_label": "📥 クライアント向けレポートをダウンロード",
            "success_message": "クライアント向けレポートを生成しました。上のボタンからダウンロードしてください。",
            "healthy_heading": "正常範囲との比較（目立つ差）",
            "healthy_include": "正常範囲との比較コメントをPDFに含める",
            "healthy_label": "正常範囲との比較コメント（編集できます）",
            "healthy_auto_button": "🪄 正常範囲との差からコメントを自動生成",
        },
        "en": {
            "header": "Client Report",
            "caption": "A plain-language report with scores and visuals, ready to hand directly to the client.",
            "subject_name": "Subject Name",
            "exam_date": "Exam Date",
            "examiner": "Examiner",
            "comment_heading": "Comment",
            "comment_label": "Enter an overall comment for the client (included in the PDF)",
            "auto_generate_button": "🪄 Auto-generate Comment (Draft)",
            "auto_generate_caption": "Generates a plain-language draft comment from the measured values. Please review and edit before generating the PDF.",
            "generate_button": "📄 Generate Client Report",
            "download_label": "📥 Download Client Report",
            "success_message": "Client report generated. Use the button above to download it.",
            "healthy_heading": "Compared with Normal Range (Notable Differences)",
            "healthy_include": "Include normal-range comparison in the PDF",
            "healthy_label": "Normal-range comment (editable)",
            "healthy_auto_button": "🪄 Auto-generate from normal-range differences",
        },
    }
    CUI = CLIENT_UI[client_lang_code]

    st.subheader(CUI["header"])
    st.caption(CUI["caption"])

    client_col1, client_col2, client_col3 = st.columns(3)
    with client_col1:
        client_subject_name = st.text_input(
            CUI["subject_name"], value="", key="arm_client_subject_name_input"
        )
    with client_col2:
        client_exam_date = st.text_input(
            CUI["exam_date"], value="", key="arm_client_exam_date_input"
        )
    with client_col3:
        client_examiner_name = st.text_input(
            CUI["examiner"], value="", key="arm_client_examiner_name_input"
        )

    JOINT_LABEL = {
        "ja": {
            "arm_flex_r": "肩（右）", "arm_flex_l": "肩（左）",
            "pelvis_tilt": "骨盤の前後の傾き", "pelvis_rotation": "骨盤の左右の回旋",
            "lumbar_extension": "腰の反り（腰椎伸展）",
        },
        "en": {
            "arm_flex_r": "Shoulder (Right)", "arm_flex_l": "Shoulder (Left)",
            "pelvis_tilt": "Pelvic Tilt", "pelvis_rotation": "Pelvic Rotation",
            "lumbar_extension": "Lumbar Extension",
        },
    }
    JOINT_SIMPLE_JA = {"Shoulder": "肩"}

    # 腰の反り・骨盤の傾き／回旋は「動きが少ないほど良い」代償の指標なので、
    # 正常範囲の下限を下回っても問題として扱わない。
    COMPENSATION_VARS_C = {"lumbar_extension", "pelvis_tilt", "pelvis_rotation"}

    def _is_concern_row(row):
        if not bool(row["Out_of_Range"]):
            return False
        if row["Variable"] in COMPENSATION_VARS_C and row["Subject_ROM"] < row["Healthy_Min"]:
            return False
        return True

    def build_healthy_diff_items(comparison_df, top_n=3, min_diff=3.0):
        """正常範囲から外れた項目を、差が大きい順に最大 top_n 件返す"""
        items = []
        for _, row in comparison_df.iterrows():
            val, lo, hi = row["Subject_ROM"], row["Healthy_Min"], row["Healthy_Max"]
            if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
                continue
            if val < lo:
                if row["Variable"] in COMPENSATION_VARS_C:
                    continue
                diff, direction = lo - val, "low"
            elif val > hi:
                diff, direction = val - hi, "high"
            else:
                continue
            if diff < min_diff:
                continue
            width = max(hi - lo, 1.0)
            items.append({"var": row["Variable"], "value": val, "lo": lo, "hi": hi,
                          "diff": diff, "direction": direction, "rel": diff / width})
        items.sort(key=lambda d: d["rel"], reverse=True)
        return items[:top_n]

    def generate_healthy_diff_comment(items, lang_code):
        JL = JOINT_LABEL[lang_code]
        if not items:
            return ("正常範囲と比べて、大きな差は見られませんでした。"
                    if lang_code == "ja" else
                    "No notable differences from the normal range were found.")
        if lang_code == "ja":
            lines = ["正常範囲と比べて、特に差が大きかったのは次の点です。"]
            for it in items:
                name = JL.get(it["var"], it["var"])
                base = f"・{name}：{it['value']:.1f}°（正常範囲 {it['lo']:.0f}〜{it['hi']:.0f}°）"
                if it["direction"] == "low":
                    lines.append(base + f"で、正常範囲より約{it['diff']:.0f}°小さめでした。動きが硬くなっている可能性があります。")
                else:
                    lines.append(base + f"で、正常範囲より約{it['diff']:.0f}°大きめでした。動きすぎ、または他の部位をかばっている可能性があります。")
            lines.append("これらのポイントを中心に、次回までのエクササイズを進めていきましょう。")
        else:
            lines = ["Compared with the normal range, the biggest differences were:"]
            for it in items:
                name = JL.get(it["var"], it["var"])
                base = f"- {name}: {it['value']:.1f}° (normal range {it['lo']:.0f}–{it['hi']:.0f}°)"
                if it["direction"] == "low":
                    lines.append(base + f", about {it['diff']:.0f}° below the normal range. This may indicate some stiffness.")
                else:
                    lines.append(base + f", about {it['diff']:.0f}° above the normal range. This may indicate excess motion or compensation.")
            lines.append("We'll focus the exercises on these points before the next check.")
        return "\n".join(lines)

    def client_tier(score, lang_code):
        if score >= 80:
            hex_c, bg_hex, ja, en = "#2E7D32", "#E8F5E9", "良好", "Good"
        elif score >= 60:
            hex_c, bg_hex, ja, en = "#B8860B", "#FFF8E1", "この調子で", "Keep it up"
        else:
            hex_c, bg_hex, ja, en = "#C62828", "#FFEBEE", "サポートが必要", "Needs support"
        label = ja if lang_code == "ja" else en
        return hex_c, bg_hex, label

    def make_client_gauge(score, color_hex, width_cm=8.6):
        gfig, gax = plt.subplots(figsize=(4.6, 2.6), subplot_kw={"aspect": "equal"})
        theta_bg = np.linspace(180, 0, 200)
        r_outer, r_inner = 1.0, 0.72
        x_out = r_outer * np.cos(np.radians(theta_bg))
        y_out = r_outer * np.sin(np.radians(theta_bg))
        x_in = r_inner * np.cos(np.radians(theta_bg[::-1]))
        y_in = r_inner * np.sin(np.radians(theta_bg[::-1]))
        gax.fill(np.concatenate([x_out, x_in]), np.concatenate([y_out, y_in]), color="#E0E0E0")
        theta_end = 180 - (max(0, min(100, score)) / 100) * 180
        theta_score = np.linspace(180, theta_end, 200)
        xo = r_outer * np.cos(np.radians(theta_score))
        yo = r_outer * np.sin(np.radians(theta_score))
        xi = r_inner * np.cos(np.radians(theta_score[::-1]))
        yi = r_inner * np.sin(np.radians(theta_score[::-1]))
        gax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=color_hex)
        gax.text(0, -0.05, f"{score:.0f}", ha="center", va="center", fontsize=38, fontweight="bold", color=color_hex)
        gax.text(0, -0.38, "/ 100", ha="center", va="center", fontsize=12, color="#78909C")
        gax.set_xlim(-1.15, 1.15)
        gax.set_ylim(-0.5, 1.15)
        gax.axis("off")
        return fig_to_rl_image(gfig, width_cm=width_cm)

    def make_client_score_bars(score_items):
        bfig, bax = plt.subplots(figsize=(8.6, 0.62 * len(score_items) + 0.6))
        y_pos = np.arange(len(score_items))[::-1]
        labels = [lbl for lbl, _ in score_items]
        values = [val for _, val in score_items]
        bax.barh(y_pos, [100] * len(values), height=0.5, color="#ECEFF1", zorder=1)
        for y, v in zip(y_pos, values):
            c_hex, _, _ = client_tier(v, "en")
            bax.barh(y, max(0, min(100, v)), height=0.5, color=c_hex, zorder=2)
            bax.text(min(100, max(0, v)) + 2, y, f"{v:.0f}", va="center", ha="left", fontsize=11, fontweight="bold", color="#263238")
        for x in (60, 80):
            bax.axvline(x, color="#B0BEC5", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        bax.set_yticks(y_pos)
        bax.set_yticklabels(labels, fontsize=11, color="#263238")
        bax.set_xlim(0, 112)
        bax.set_xticks([])
        for spine in bax.spines.values():
            spine.set_visible(False)
        bax.tick_params(left=False)
        bfig.tight_layout()
        return fig_to_rl_image(bfig, width_cm=15.5)

    def make_client_phase_timeline(phase_boxes):
        # phase_boxes: [(english_title, color_hex), ...] — English-only inside the
        # raster image on purpose. matplotlib's default font has no CJK glyphs and we
        # don't want a dependency on a Japanese font being installed on the server,
        # so Japanese labels/notes are placed as native PDF text just below this
        # image (see the small table right after) instead of inside the chart.
        n = len(phase_boxes)
        tfig, tax = plt.subplots(figsize=(15.5 / 2.2, 1.25))
        box_w, gap = 3.0, 0.6
        total_w = n * box_w + (n - 1) * gap
        x0 = -total_w / 2
        for i, (title, color_hex) in enumerate(phase_boxes):
            x = x0 + i * (box_w + gap)
            tax.add_patch(plt.Rectangle((x, 0), box_w, 1.0, facecolor=color_hex, alpha=0.15, edgecolor=color_hex, linewidth=1.6))
            tax.text(x + box_w / 2, 0.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#263238")
            if i < n - 1:
                tax.annotate("", xy=(x + box_w + gap - 0.05, 0.5), xytext=(x + box_w + 0.05, 0.5),
                             arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=1.4))
        tax.set_xlim(x0 - 0.3, x0 + total_w + 0.3)
        tax.set_ylim(-0.15, 1.15)
        tax.axis("off")
        tfig.tight_layout()
        return fig_to_rl_image(tfig, width_cm=16)

    st.markdown(f"#### {CUI['comment_heading']}")
    if "arm_client_report_comment" not in st.session_state:
        st.session_state["arm_client_report_comment"] = ""
    if st.button(CUI["auto_generate_button"], key="arm_client_report_auto_comment_btn"):
        st.session_state["arm_client_report_comment"] = gen_client_comment(
            client_lang_code, overall_score, arm_flexion_details,
            mobility_score, symmetry_score, lumbar_score, pelvis_score
        )
    st.caption(CUI["auto_generate_caption"])
    client_comment = st.text_area(
        CUI["comment_label"],
        key="arm_client_report_comment",
        height=150
    )

    st.markdown(f"#### {CUI['healthy_heading']}")
    include_healthy_diff = st.checkbox(CUI["healthy_include"], value=True, key="arm_client_include_healthy_diff")
    if "arm_client_healthy_diff_comment" not in st.session_state:
        st.session_state["arm_client_healthy_diff_comment"] = ""
    if st.button(CUI["healthy_auto_button"], key="arm_client_healthy_diff_auto_btn", disabled=not include_healthy_diff):
        st.session_state["arm_client_healthy_diff_comment"] = generate_healthy_diff_comment(
            build_healthy_diff_items(comparison_df), client_lang_code
        )
    healthy_diff_comment = st.text_area(
        CUI["healthy_label"], key="arm_client_healthy_diff_comment",
        height=130, disabled=not include_healthy_diff
    )

    if st.button(CUI["generate_button"], key="arm_client_report_generate_btn"):

        from reportlab.platypus import PageBreak, HRFlowable

        # ---- 動的な所見の収集（実測値ベース） ----
        # 代償の指標（腰・骨盤）が正常範囲の下限を下回っている場合は問題にしない
        out_of_range_rows = comparison_df[comparison_df.apply(_is_concern_row, axis=1)] if len(comparison_df) else comparison_df
        asym_flags = [(joint, value) for joint, value in asymmetry_results.items() if value > 15]

        # 「グラつき」はこのアプリのMovement Scoreには含まれない（stability_scoreに相当する指標が
        # 存在しない）ため、Start/Topフェーズの標準偏差から直接判定する。
        STATIC_PHASES_C = ["Start", "Top"]
        STD_THRESHOLD_C = 2.0
        phase_std_flag = {p: False for p in STATIC_PHASES_C}
        for variable in ["pelvis_tilt", "pelvis_rotation", "lumbar_extension", "pelvis_list"]:
            var_row = phase_summary_df[phase_summary_df["Variable"] == variable]
            if len(var_row) == 0:
                continue
            for phase in STATIC_PHASES_C:
                std_v = var_row[f"{phase}_Std"].iloc[0]
                if pd.notna(std_v) and std_v > STD_THRESHOLD_C:
                    phase_std_flag[phase] = True

        concern_flags = {
            "asymmetry": len(asym_flags) > 0,
            "range": len(out_of_range_rows) > 0,
            "compensation": (lumbar_compensation > 10) or (pelvis_compensation > 10) or (pelvis_rotation_compensation > 10),
            "stability": any(phase_std_flag.values()),
        }

        JL = JOINT_LABEL[client_lang_code]

        concern_items = []
        if concern_flags["compensation"]:
            if client_lang_code == "ja":
                concern_items.append((
                    "腰・骨盤が反りやすい" if lumbar_compensation > 10 else "骨盤が傾き／回旋しやすい",
                    f"腕を挙げ下げする動きの中で、腰や骨盤が大きく動く場面が見られました"
                    f"(腰の反り {lumbar_compensation:.1f}°、骨盤の前後の傾き {pelvis_compensation:.1f}°、"
                    f"骨盤の回旋 {pelvis_rotation_compensation:.1f}°)。この状態が続くと、腰まわりへの負担が"
                    "蓄積しやすくなります。"
                ))
            else:
                concern_items.append((
                    "Lower back / pelvis compensation",
                    f"Noticeable lumbar and pelvic movement was observed while raising and lowering the arm "
                    f"(lumbar extension {lumbar_compensation:.1f}°, pelvic tilt {pelvis_compensation:.1f}°, "
                    f"pelvic rotation {pelvis_rotation_compensation:.1f}°). Over time this can add strain around the lower back."
                ))
        if concern_flags["asymmetry"]:
            if client_lang_code == "ja":
                joint_text = "・".join(JOINT_SIMPLE_JA.get(joint, joint) for joint, _ in asym_flags)
                concern_items.append((
                    "左右差がある",
                    f"{joint_text}で、左右の動きの差が基準(15%)を超えていました。"
                    "片側に負担が偏っている可能性があります。"
                ))
            else:
                joint_text = ", ".join(joint for joint, _ in asym_flags)
                concern_items.append((
                    "Left-right difference",
                    f"The {joint_text} showed a left-right difference beyond the 15% guideline, "
                    "which may indicate uneven loading between sides."
                ))
        if concern_flags["range"]:
            if client_lang_code == "ja":
                range_text = "・".join(JL.get(v, v) for v in out_of_range_rows["Variable"].tolist())
                concern_items.append((
                    "可動域が基準の範囲外",
                    f"{range_text}が、一般的な正常範囲の外にありました。可動域の制限、"
                    "またはやや動きすぎている可能性があります。"
                ))
            else:
                range_text = ", ".join(JOINT_LABEL["en"].get(v, v) for v in out_of_range_rows["Variable"].tolist())
                concern_items.append((
                    "Range of motion outside reference",
                    f"{range_text} fell outside the typical normal range, suggesting possible "
                    "restricted or excessive range of motion."
                ))
        if concern_flags["stability"]:
            if client_lang_code == "ja":
                concern_items.append((
                    "動作中のグラつき",
                    "腕を挙げ下げする動作の中で、体幹や骨盤が揺れる場面がありました。"
                    "体幹まわりの筋力を使って、じっと支える意識をすると安定しやすくなります。"
                ))
            else:
                concern_items.append((
                    "Movement wobble",
                    "Some trunk/pelvic movement was observed while raising and lowering the arm. "
                    "Engaging the core muscles to hold steady can help improve stability."
                ))
        if len(concern_items) == 0:
            if client_lang_code == "ja":
                concern_items.append(("良い状態です", "今回のチェックでは、特に大きな気になるポイントはありませんでした。この調子を維持しましょう。"))
            else:
                concern_items.append(("Looking good", "No major concerns were found in this check. Keep up the good work."))

        # ---- アクション（気になるポイントに応じて選択、最大3件） ----
        action_pool_ja = [
            ("compensation", "毎日 10回", "お腹に軽く力を入れたまま、ゆっくり腕を挙げ下げ",
             "腰を反らさず、体幹で支える感覚をつかむ練習になります。"),
            ("asymmetry", "毎日 左右10回ずつ", "左右片方ずつ、ゆっくり腕を挙げる練習",
             "左右差を意識しながら、均等に力を伝える練習になります。"),
            ("stability", "毎日 左右5秒×5回", "腕を真上に挙げて5秒キープ",
             "静止姿勢を保つ練習になり、挙上時の揺れを減らします。"),
            ("range", "毎日 5回", "痛みのない範囲で、ゆっくり腕を高く挙げる練習",
             "無理のない範囲で可動域を広げていく練習になります。"),
            ("default", "毎日 5秒×3回", "椅子に深く座り、背すじを伸ばしてキープ",
             "姿勢を保つための筋力を養います。"),
        ]
        action_pool_en = [
            ("compensation", "Daily x10", "Slow arm raises with gentle core engagement",
             "Builds the habit of supporting the movement with your core instead of arching your back."),
            ("asymmetry", "Daily x10/side", "Slow single-arm raises, one side at a time",
             "Helps even out effort between the left and right sides."),
            ("stability", "Daily 5s x5/side", "Raise the arm overhead and hold for 5 seconds",
             "Builds the ability to hold a steady position and reduces wobble during the raise."),
            ("range", "Daily x5", "Slow, pain-free practice raising the arm as high as comfortable",
             "Gradually improves range of motion within a comfortable limit."),
            ("default", "Daily 5s x3", "Sit tall in a chair and hold your posture",
             "Builds the postural strength needed to hold a good position."),
        ]
        action_pool = action_pool_ja if client_lang_code == "ja" else action_pool_en
        selected_actions = [a for a in action_pool if a[0] != "default" and concern_flags.get(a[0], False)]
        if len(selected_actions) == 0:
            selected_actions = [a for a in action_pool if a[0] == "default"]
        selected_actions = (selected_actions + [a for a in action_pool if a[0] == "default"])[:3]

        # 図(画像)側は英語のみ。日本語ラベル・所見はネイティブPDFテキスト(下の表)で表示する。
        phase_titles_en = ["① Start", "② Raising", "③ Top", "④ Lowering"]
        phase_colors = [
            "#2E7D32" if not phase_std_flag["Start"] else "#B8860B",
            "#C62828" if lumbar_compensation > 10 else "#2E7D32",
            "#B8860B" if phase_std_flag["Top"] else "#2E7D32",
            "#B8860B" if len(asym_flags) > 0 else "#2E7D32",
        ]
        phase_boxes = list(zip(phase_titles_en, phase_colors))

        if client_lang_code == "ja":
            phase_titles_local = ["① 開始位置", "② 挙上中", "③ 最大挙上", "④ 下降中"]
            phase_notes_local = [
                "開始姿勢は安定しています" if not phase_std_flag["Start"] else "開始姿勢でやや不安定な様子がありました",
                "腰が反りやすい傾向があります" if lumbar_compensation > 10 else "スムーズに挙上できています",
                "最大挙上位置は安定しています" if not phase_std_flag["Top"] else "最大挙上位置でやや不安定な様子がありました",
                "左右差が見られました" if len(asym_flags) > 0 else "安定して下降できています",
            ]
        else:
            phase_titles_local = phase_titles_en
            phase_notes_local = [
                "Stable" if not phase_std_flag["Start"] else "Slightly unstable",
                "Back tends to arch" if lumbar_compensation > 10 else "Smooth raise",
                "Stable" if not phase_std_flag["Top"] else "Slightly unstable",
                "Left-right difference observed" if len(asym_flags) > 0 else "Stable, even lowering",
            ]

        # ---- スタイル ----
        c_styles = getSampleStyleSheet()
        c_title_style = ParagraphStyle("CTitle", parent=c_styles["Title"], fontName="HeiseiKakuGo-W5", fontSize=21, alignment=TA_CENTER, textColor=colors.white, spaceAfter=2)
        c_subtitle_style = ParagraphStyle("CSubtitle", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, alignment=TA_CENTER, textColor=colors.white)
        c_section_style = ParagraphStyle("CSection", parent=c_styles["Heading2"], fontName="HeiseiKakuGo-W5", fontSize=13.5, textColor=colors.HexColor("#263238"), spaceBefore=9, spaceAfter=5)
        c_lead_style = ParagraphStyle("CLead", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, leading=16, textColor=colors.HexColor("#263238"), alignment=TA_CENTER, spaceAfter=4)
        c_body_style = ParagraphStyle("CBody", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.6, leading=14.5, textColor=colors.HexColor("#263238"))
        c_small_muted_style = ParagraphStyle("CSmallMuted", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8.3, leading=12.5, textColor=colors.HexColor("#607D8B"))
        c_card_title_style = ParagraphStyle("CCardTitle", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=10.5, textColor=colors.HexColor("#263238"), alignment=TA_CENTER)
        c_card_status_style = ParagraphStyle("CCardStatus", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=12, alignment=TA_CENTER)
        c_score_headline_style = ParagraphStyle("CScoreHeadline", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=12.5, alignment=TA_CENTER, textColor=colors.HexColor("#263238"), spaceBefore=8)
        c_footer_style = ParagraphStyle("CFooter", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=7.8, leading=11.5, textColor=colors.HexColor("#607D8B"), alignment=TA_CENTER)
        c_page_label_style = ParagraphStyle("CPageLabel", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8, textColor=colors.HexColor("#607D8B"))
        c_action_head_style = ParagraphStyle("CActionHead", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.3, textColor=colors.white)
        c_action_body_style = ParagraphStyle("CActionBody", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=9.2, leading=13.5, textColor=colors.HexColor("#263238"))
        c_action_note_style = ParagraphStyle("CActionNote", parent=c_styles["Normal"], fontName="HeiseiKakuGo-W5", fontSize=8.2, leading=12, textColor=colors.HexColor("#607D8B"))

        LINE_HEX_C = "#CFD8DC"
        BAND_C = colors.HexColor("#0D47A1")
        BLUE_C = colors.HexColor("#1565C0")
        BLUE_BG_C = "#E3F2FD"

        def c_tier_pack(score):
            return client_tier(score, client_lang_code)

        def make_client_card(label, score, desc):
            color_hex, bg_hex, tier_text = c_tier_pack(score)
            inner = Table(
                [
                    [Paragraph(label, c_card_title_style)],
                    [Spacer(1, 0.12 * cm)],
                    [Paragraph(f'<font color="{color_hex}"><b>{tier_text}</b></font>', c_card_status_style)],
                    [Spacer(1, 0.16 * cm)],
                    [Paragraph(desc, c_small_muted_style)],
                ],
                colWidths=[4.3 * cm]
            )
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(color_hex)),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            return inner

        if client_lang_code == "ja":
            card_data = [
                ("可動域", mobility_score, "腕を挙げる高さは十分に出ています。" if mobility_score >= 80 else "腕を挙げる高さに、もう少し伸びしろがあります。"),
                ("左右差", symmetry_score, "左右の動きはよく揃っています。" if symmetry_score >= 80 else "左右で動きの差がやや見られます。"),
                ("腰の代償", lumbar_score, "腰の反りは少なめです。" if lumbar_score >= 80 else "腕を挙げ下げする中で、腰が反りやすい傾向があります。"),
                ("骨盤の代償", pelvis_score, "骨盤の傾きは少なめです。" if pelvis_score >= 80 else "動作の中で、骨盤が傾きやすい傾向があります。"),
            ]
        else:
            card_data = [
                ("Mobility", mobility_score, "Good arm raise height." if mobility_score >= 80 else "There is room to raise the arm higher."),
                ("Symmetry", symmetry_score, "Left and right sides move very evenly." if symmetry_score >= 80 else "Some left-right difference was observed."),
                ("Lumbar", lumbar_score, "Minimal lower-back compensation." if lumbar_score >= 80 else "The lower back tends to arch during the movement."),
                ("Pelvis", pelvis_score, "Minimal pelvic compensation." if pelvis_score >= 80 else "The pelvis tends to tilt during the movement."),
            ]

        elements_c = []

        band = Table(
            [[Paragraph(
                "あなたの腕の挙上チェック結果" if client_lang_code == "ja" else "Your Arm Flexion Check Results",
                c_title_style
            )],
                [Paragraph(
                    (f"対象者：{client_subject_name or '-'}　｜　測定日：{client_exam_date or '-'}　｜　検者：{client_examiner_name or '-'}"
                     if client_lang_code == "ja" else
                     f"Subject: {client_subject_name or '-'}  |  Exam Date: {client_exam_date or '-'}  |  Examiner: {client_examiner_name or '-'}"),
                    c_subtitle_style
                )]],
            colWidths=[19 * cm]
        )
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BAND_C),
            ("TOPPADDING", (0, 0), (0, 0), 14),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
            ("TOPPADDING", (0, 1), (0, 1), 2),
            ("BOTTOMPADDING", (0, 1), (0, 1), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements_c.append(band)
        elements_c.append(Spacer(1, 0.5 * cm))

        gauge_color_hex, _, gauge_tier_label = c_tier_pack(overall_score)
        gauge_img_c = make_client_gauge(overall_score, gauge_color_hex)
        gauge_table_c = Table([[gauge_img_c]], colWidths=[19 * cm])
        gauge_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(gauge_table_c)

        if client_lang_code == "ja":
            elements_c.append(Paragraph(f"総合評価：<b>{gauge_tier_label}</b>", c_score_headline_style))
            elements_c.append(Paragraph(
                "腕の挙上動作を、可動域・左右差・腰の代償・骨盤の代償の4つの視点でチェックしました。",
                c_lead_style
            ))
        else:
            elements_c.append(Paragraph(f"Overall: <b>{gauge_tier_label}</b>", c_score_headline_style))
            elements_c.append(Paragraph(
                "Your arm flexion was checked across four areas: mobility, symmetry, lumbar compensation, and pelvis compensation.",
                c_lead_style
            ))

        elements_c.append(Spacer(1, 0.45 * cm))
        elements_c.append(Paragraph("4つのポイント" if client_lang_code == "ja" else "Four Key Areas", c_section_style))

        cards_row_c = Table([[make_client_card(*c) for c in card_data]], colWidths=[4.75 * cm] * 4)
        cards_row_c.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements_c.append(cards_row_c)
        elements_c.append(Spacer(1, 0.4 * cm))

        elements_c.append(Paragraph(
            "4つのスコアを見比べる" if client_lang_code == "ja" else "Comparing the Four Scores",
            c_section_style
        ))
        elements_c.append(Paragraph(
            ("点線は「この調子で(60点)」「良好(80点)」の目安ラインです。"
             if client_lang_code == "ja" else
             "The dotted lines mark the 60 (“keep it up”) and 80 (“good”) reference points."),
            c_body_style
        ))
        elements_c.append(Spacer(1, 0.12 * cm))
        score_bar_labels = ["Mobility", "Symmetry", "Lumbar", "Pelvis"]
        bars_img_c = make_client_score_bars(list(zip(score_bar_labels, [mobility_score, symmetry_score, lumbar_score, pelvis_score])))
        bars_table_c = Table([[bars_img_c]], colWidths=[19 * cm])
        bars_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(bars_table_c)

        elements_c.append(Spacer(1, 0.25 * cm))
        elements_c.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(LINE_HEX_C)))
        elements_c.append(Spacer(1, 0.12 * cm))
        elements_c.append(Paragraph(
            "続きは次のページで、動きの流れとおすすめアクションをご紹介します。" if client_lang_code == "ja"
            else "Continued on the next page: movement flow and recommended actions.",
            c_page_label_style
        ))
        elements_c.append(PageBreak())

        elements_c.append(Paragraph("動きの流れをチェック" if client_lang_code == "ja" else "Movement Flow", c_section_style))
        elements_c.append(Paragraph(
            ("腕の挙上動作を4つの場面に分けてみると、どこで体に負担がかかりやすいかが見えてきます。"
             if client_lang_code == "ja" else
             "Breaking the movement into four phases makes it easier to see where load tends to build up."),
            c_body_style
        ))
        elements_c.append(Spacer(1, 0.15 * cm))
        timeline_img_c = make_client_phase_timeline(phase_boxes)
        timeline_table_c = Table([[timeline_img_c]], colWidths=[19 * cm])
        timeline_table_c.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements_c.append(timeline_table_c)
        elements_c.append(Spacer(1, 0.15 * cm))

        phase_note_row_titles = [
            Paragraph(f"<font color='{phase_colors[i]}'><b>{phase_titles_local[i]}</b></font>", c_small_muted_style)
            for i in range(4)
        ]
        phase_note_row_notes = [
            Paragraph(phase_notes_local[i], c_small_muted_style)
            for i in range(4)
        ]
        phase_note_table = Table([phase_note_row_titles, phase_note_row_notes], colWidths=[4.75 * cm] * 4)
        phase_note_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ]))
        elements_c.append(phase_note_table)
        elements_c.append(Spacer(1, 0.22 * cm))

        elements_c.append(Paragraph("気になるポイント" if client_lang_code == "ja" else "Points to Note", c_section_style))
        for title, desc in concern_items:
            row_c = Table(
                [[Paragraph(f"<font color='#C62828'><b>●</b></font>  <b>{title}</b>", c_action_body_style),
                  Paragraph(desc, c_action_body_style)]],
                colWidths=[4.4 * cm, 14.6 * cm]
            )
            row_c.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements_c.append(row_c)
        elements_c.append(Spacer(1, 0.16 * cm))

        elements_c.append(Paragraph(CUI["comment_heading"], c_section_style))
        comment_display = client_comment.strip() if client_comment and client_comment.strip() else (
            "(記入なし)" if client_lang_code == "ja" else "(No comment entered)"
        )
        comment_html = escape(comment_display).replace("\n", "<br/>")
        elements_c.append(Paragraph(comment_html, c_body_style))
        elements_c.append(Spacer(1, 0.16 * cm))

        if include_healthy_diff:
            healthy_items = build_healthy_diff_items(comparison_df)
            elements_c.append(Paragraph(CUI["healthy_heading"], c_section_style))

            if healthy_items:
                hdr = (["項目", "あなた", "正常範囲", "差"] if client_lang_code == "ja"
                       else ["Item", "You", "Normal Range", "Diff"])
                h_rows = [hdr]
                range_sep = "〜" if client_lang_code == "ja" else "–"
                for it in healthy_items:
                    sign = "-" if it["direction"] == "low" else "+"
                    h_rows.append([
                        JOINT_LABEL[client_lang_code].get(it["var"], it["var"]),
                        f"{it['value']:.1f}°",
                        f"{it['lo']:.0f}{range_sep}{it['hi']:.0f}°",
                        f"{sign}{it['diff']:.1f}°",
                    ])
                h_table = Table(h_rows, colWidths=[6 * cm, 3.5 * cm, 5 * cm, 4.5 * cm])
                h_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#78909C")),
                    ("TEXTCOLOR", (0, 1), (-2, -1), colors.HexColor("#263238")),
                    ("TEXTCOLOR", (-1, 1), (-1, -1), colors.HexColor("#C62828")),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE_HEX_C)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]))
                elements_c.append(h_table)
                elements_c.append(Spacer(1, 0.12 * cm))

            h_text = (healthy_diff_comment.strip() if healthy_diff_comment and healthy_diff_comment.strip()
                      else generate_healthy_diff_comment(healthy_items, client_lang_code))
            elements_c.append(Paragraph(escape(h_text).replace("\n", "<br/>"), c_body_style))
            elements_c.append(Spacer(1, 0.16 * cm))

        elements_c.append(Paragraph(
            "おすすめのアクション（今週から）" if client_lang_code == "ja" else "Recommended Actions (Starting This Week)",
            c_section_style
        ))
        action_rows_c = [[
            Paragraph("回数の目安" if client_lang_code == "ja" else "Frequency", c_action_head_style),
            Paragraph("やること" if client_lang_code == "ja" else "What to Do", c_action_head_style),
            Paragraph("ねらい" if client_lang_code == "ja" else "Why", c_action_head_style),
        ]]
        for _, freq, what, why in selected_actions:
            action_rows_c.append([freq, Paragraph(what, c_action_body_style), Paragraph(why, c_action_note_style)])
        action_table_c = Table(action_rows_c, colWidths=[3.2 * cm, 8.4 * cm, 7.4 * cm])
        action_table_c.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, 0), BLUE_C),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor(BLUE_BG_C)),
            ("TEXTCOLOR", (0, 1), (0, -1), BLUE_C),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor(LINE_HEX_C)),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(LINE_HEX_C)),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements_c.append(action_table_c)
        elements_c.append(Spacer(1, 0.16 * cm))

        elements_c.append(Paragraph(
            "参考：今回の測定値（くわしく知りたい方向け）" if client_lang_code == "ja" else "Reference: Measured Values (For Those Who Want Detail)",
            c_section_style
        ))
        if client_lang_code == "ja":
            detail_rows_c = [["項目", "測定値", "備考"]]
        else:
            detail_rows_c = [["Item", "Value", "Note"]]

        def in_range_note(var):
            row = comparison_df[comparison_df["Variable"] == var]
            if len(row) == 0:
                return "-"
            out = bool(row["Out_of_Range"].iloc[0])
            if client_lang_code == "ja":
                return "正常範囲外" if out else "正常範囲内"
            return "Outside normal range" if out else "Within normal range"

        for var in ["arm_flex_r", "arm_flex_l"]:
            row = comparison_df[comparison_df["Variable"] == var]
            if len(row) == 0:
                continue
            val = row["Subject_ROM"].iloc[0]
            detail_rows_c.append([JOINT_LABEL[client_lang_code].get(var, var), f"{val:.1f}°", in_range_note(var)])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["lumbar_extension"], f"{lumbar_compensation:.1f}°",
            ("やや大きめ" if lumbar_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if lumbar_compensation > 10 else "Small")
        ])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["pelvis_tilt"], f"{pelvis_compensation:.1f}°",
            ("やや大きめ" if pelvis_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if pelvis_compensation > 10 else "Small")
        ])
        detail_rows_c.append([
            JOINT_LABEL[client_lang_code]["pelvis_rotation"], f"{pelvis_rotation_compensation:.1f}°",
            ("やや大きめ" if pelvis_rotation_compensation > 10 else "少なめ") if client_lang_code == "ja" else ("Somewhat large" if pelvis_rotation_compensation > 10 else "Small")
        ])

        detail_table_c = Table(detail_rows_c, colWidths=[6 * cm, 4 * cm, 9 * cm])
        detail_table_c.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#78909C")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#607D8B")),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(LINE_HEX_C)),
            ("TOPPADDING", (0, 0), (-1, -1), 2.8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements_c.append(detail_table_c)
        elements_c.append(Spacer(1, 0.16 * cm))

        elements_c.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(LINE_HEX_C)))
        elements_c.append(Spacer(1, 0.12 * cm))
        elements_c.append(Paragraph(
            ("このレポートはスマートフォンの動画をもとにした簡易チェックの結果であり、医学的な診断ではありません。"
             "痛みや強い違和感がある場合は、無理をせず医療・専門家にご相談ください。")
            if client_lang_code == "ja" else
            ("This report is based on a simplified check from smartphone video and is not a medical diagnosis. "
             "If you experience pain or significant discomfort, please consult a healthcare professional."),
            c_footer_style
        ))

        def draw_client_bg(canvas, doc_):
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#FAFAFA"))
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.restoreState()

        # ページ数は「気になるポイント」の件数やコメントの長さによって変わるため、
        # 固定の「1/2」を書く代わりに、実際の総ページ数を描画時に計算してfooterに描く。
        from reportlab.pdfgen import canvas as canvas_module

        class _ClientReportNumberedCanvas(canvas_module.Canvas):
            def __init__(self, *args, **kwargs):
                canvas_module.Canvas.__init__(self, *args, **kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                total_pages = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self._draw_totalized_page_number(total_pages)
                    canvas_module.Canvas.showPage(self)
                canvas_module.Canvas.save(self)

            def _draw_totalized_page_number(self, total_pages):
                self.setFont("HeiseiKakuGo-W5", 8)
                self.setFillColor(colors.HexColor("#607D8B"))
                self.drawString(1.6 * cm, 1.0 * cm, f"{self._pageNumber} / {total_pages}")

        client_report_buffer = BytesIO()
        client_doc = SimpleDocTemplate(
            client_report_buffer,
            pagesize=A4,
            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
            leftMargin=1.6 * cm, rightMargin=1.6 * cm
        )
        client_doc.build(
            elements_c,
            onFirstPage=draw_client_bg,
            onLaterPages=draw_client_bg,
            canvasmaker=_ClientReportNumberedCanvas
        )

        st.download_button(
            CUI["download_label"],
            data=client_report_buffer.getvalue(),
            file_name="Arm_Flexion_Client_Report.pdf",
            mime="application/pdf",
            key="arm_client_report_download_btn"
        )
        st.success(CUI["success_message"])
