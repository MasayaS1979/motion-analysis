"""
snowflake_store.py
---------------------------------------------------------------
Streamlit アプリから Snowflake に「生データ（時系列）」と「要約（スコア）」を
保存するための共通モジュール。motion_apps/ 直下（app.py と同じ場所）に置く。

使い方（各ページの最後、tab9 の後など）:
    from snowflake_store import render_save_section
    render_save_section(df_phase, movement="Single Sit-to-Stand",
                        summary={"OVERALL_SCORE": overall_score, ...})
"""
import datetime
import hashlib
import os
import tempfile
import uuid

import pandas as pd
import streamlit as st

RAW_TABLE = "RAW_FRAMES"
SUMMARY_TABLE = "ASSESSMENTS"


# ---------------------------------------------------------------
# 接続（キーペア認証。秘密鍵は Secrets から一時ファイルに書き出して使う）
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    pem = st.secrets["snowflake_private_key"]["pem"]
    key_path = os.path.join(tempfile.gettempdir(), "sf_key.p8")
    with open(key_path, "w") as f:
        f.write(pem)
    return st.connection("snowflake", private_key_file=key_path)


def file_hash_of(uploaded):
    """アップロードファイル（またはファイルパス）の SHA-256。同じファイルなら同じ値。"""
    if isinstance(uploaded, (str, os.PathLike)):
        with open(uploaded, "rb") as f:
            data = f.read()
    else:
        data = uploaded.getvalue()
    return hashlib.sha256(data).hexdigest()


def _already_saved(conn, table, fhash, movement):
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT COUNT(*) FROM {table} WHERE FILE_HASH = %(h)s AND MOVEMENT = %(m)s",
            {"h": fhash, "m": movement},
        )
        return cur.fetchone()[0] > 0
    except Exception:
        # テーブルがまだ無い（初回）場合
        return False
    finally:
        cur.close()


def _prepare_raw(df_phase, fhash, movement, fs=60):
    raw = df_phase.copy()
    if "time" in raw.columns:
        raw = raw.rename(columns={"time": "time_s"})
    else:
        raw.insert(0, "time_s", [i / fs for i in range(len(raw))])
    raw.insert(0, "frame", range(len(raw)))
    raw.insert(0, "movement", movement)
    raw.insert(0, "file_hash", fhash)
    # 列名を Snowflake 向けに整える（大文字・英数字とアンダースコアのみ）
    raw.columns = [
        "".join(ch if ch.isalnum() else "_" for ch in str(c)).upper() for c in raw.columns
    ]
    return raw


def save_raw(df_phase, uploaded, movement):
    conn = get_conn()
    fhash = file_hash_of(uploaded)
    if _already_saved(conn, RAW_TABLE, fhash, movement):
        return "exists", 0
    raw = _prepare_raw(df_phase, fhash, movement)
    conn.write_pandas(raw, RAW_TABLE, auto_create_table=True, quote_identifiers=False)
    return "saved", len(raw)


def save_summary(summary, uploaded, movement, profile=None):
    conn = get_conn()
    fhash = file_hash_of(uploaded)
    if _already_saved(conn, SUMMARY_TABLE, fhash, movement):
        return "exists"
    row = {"RECORD_ID": str(uuid.uuid4()),
           "CREATED_AT": datetime.datetime.utcnow(),
           "FILE_HASH": fhash, "MOVEMENT": movement}
    row.update({k.upper(): (float(v) if isinstance(v, (int, float)) else v)
                for k, v in (summary or {}).items()})
    row.update(profile or {})
    conn.write_pandas(pd.DataFrame([row]), SUMMARY_TABLE,
                      auto_create_table=True, quote_identifiers=False)
    return "saved"


# ---------------------------------------------------------------
# ページに貼る UI（同意チェック＋保存ボタン）
# ---------------------------------------------------------------
def render_save_section(df_phase, movement, summary=None, key_prefix=None):
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        return
    kp = key_prefix or movement.replace(" ", "_").replace("-", "_").lower()

    st.markdown("---")
    st.markdown("#### データ保存（研究・サービス改善用）")
    consent = st.checkbox(
        "本人の同意を得たうえで、個人を特定しない測定データ（関節の動きの数値・スコア）を保存します",
        key=f"{kp}_sf_consent",
    )
    # --- 対象者の基本情報（任意。未回答でも保存できる） ---
    c1, c2 = st.columns(2)
    with c1:
        age_group = st.selectbox(
            "年代（任意）",
            ["未回答", "10代以下", "20代", "30代", "40代", "50代", "60代", "70代", "80代以上"],
            key=f"{kp}_sf_age",
        )
    with c2:
        sex = st.selectbox("性別（任意）", ["未回答", "男性", "女性", "その他"], key=f"{kp}_sf_sex")
    profile = {
        "AGE_GROUP": None if age_group == "未回答" else age_group,
        "SEX": None if sex == "未回答" else sex,
    }

    done_key = f"{kp}_sf_saved_{file_hash_of(uploaded)}"

    if st.button("💾 Snowflake に保存", key=f"{kp}_sf_save_btn", disabled=not consent):
        if st.session_state.get(done_key):
            st.info("このデータはこの画面ですでに保存済みです。")
            return
        try:
            with st.spinner("保存中..."):
                status, n = save_raw(df_phase, uploaded, movement)
                s_status = save_summary(summary, uploaded, movement, profile)
            if status == "exists":
                st.info("同じデータがすでに登録されているため、保存しませんでした。")
            else:
                st.success(f"生データ {n} フレームを保存しました。"
                           + (" 要約も保存しました。" if s_status == "saved" else ""))
            st.session_state[done_key] = True
        except Exception as e:
            st.error(f"保存に失敗しました：{e}")
