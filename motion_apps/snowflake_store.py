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
import numbers
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


def data_hash_of(df):
    """測定データの中身から作る ID。同じデータなら、ファイルを開き直しても同じ値になる。"""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def _already_saved(conn, table, fhash, movement):
    """保存済みなら True。テーブルが無い（初回）ときだけ False。
    それ以外のエラーは握りつぶさずに止める（二重保存を防ぐため）。"""
    cur = conn.cursor()
    try:
        # 接続の設定によって %(h)s 形式が使えないため、値を安全に埋め込む
        h = "".join(c for c in str(fhash) if c in "0123456789abcdef")
        m = str(movement).replace("'", "''")
        cur.execute(
            f"SELECT COUNT(*) FROM {table} WHERE FILE_HASH = '{h}' AND MOVEMENT = '{m}'"
        )
        return cur.fetchone()[0] > 0
    except Exception as e:
        if "does not exist" in str(e):
            return False
        raise
    finally:
        cur.close()


def _clean_name(c):
    return "".join(ch if ch.isalnum() else "_" for ch in str(c)).upper()


def _ensure_columns(conn, table, df):
    """動作ごとに列が違っても保存できるよう、足りない列をテーブルに追加する。"""
    cur = conn.cursor()
    try:
        try:
            cur.execute(f"SHOW COLUMNS IN TABLE {table}")
        except Exception as e:
            if "does not exist" in str(e):
                return  # 初回：write_pandas がテーブルを作る
            raise
        existing = {row[2].upper() for row in cur.fetchall()}
        for col in df.columns:
            if col.upper() in existing:
                continue
            s = df[col].dropna()
            if pd.api.types.is_datetime64_any_dtype(df[col]) or (
                len(s) and isinstance(s.iloc[0], datetime.datetime)
            ):
                typ = "TIMESTAMP_NTZ"
            elif pd.api.types.is_numeric_dtype(df[col]):
                typ = "FLOAT"
            else:
                typ = "VARCHAR"
            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
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
    raw.columns = [_clean_name(c) for c in raw.columns]
    return raw


def save_raw(df_phase, fhash, movement):
    conn = get_conn()
    if _already_saved(conn, RAW_TABLE, fhash, movement):
        return "exists", 0
    raw = _prepare_raw(df_phase, fhash, movement)
    _ensure_columns(conn, RAW_TABLE, raw)
    conn.write_pandas(raw, RAW_TABLE, auto_create_table=True, quote_identifiers=False)
    return "saved", len(raw)


def save_summary(summary, fhash, movement, profile=None):
    conn = get_conn()
    if _already_saved(conn, SUMMARY_TABLE, fhash, movement):
        return "exists"
    row = {"RECORD_ID": str(uuid.uuid4()),
           "CREATED_AT": datetime.datetime.utcnow(),
           "FILE_HASH": fhash, "MOVEMENT": movement}
    for k, v in (summary or {}).items():
        if v is None:
            continue  # 値が無い項目は保存しない（NULL のまま）
        if isinstance(v, numbers.Number) and not isinstance(v, bool):
            v = float(v)
        row[_clean_name(k)] = v
    row.update({k: v for k, v in (profile or {}).items() if v is not None})
    df_row = pd.DataFrame([row])
    _ensure_columns(conn, SUMMARY_TABLE, df_row)
    conn.write_pandas(df_row, SUMMARY_TABLE,
                      auto_create_table=True, quote_identifiers=False)
    return "saved"


# ---------------------------------------------------------------
# アップロード画面（app.py）に置く：同意・年代・性別の入力
# ---------------------------------------------------------------
AGE_OPTIONS = ["未回答", "10代以下", "20代", "30代", "40代", "50代", "60代", "70代", "80代以上"]
SEX_OPTIONS = ["未回答", "男性", "女性", "その他"]
CONSENT_STATE = "sf_upload_consent"


def _file_id(uploaded):
    """アップロードされたファイルの目印（名前＋サイズ）。別ファイルなら同意をやり直す。"""
    if uploaded is None:
        return None
    if isinstance(uploaded, (str, os.PathLike)):
        return str(uploaded)
    return f"{getattr(uploaded, 'name', '')}|{getattr(uploaded, 'size', '')}"


def render_upload_consent(uploaded=None):
    """app.py のファイルアップロードの直後に呼ぶ。
    ここで同意したファイルは、各動作ページを開いたときに自動で Snowflake に保存される。"""
    uploaded = uploaded if uploaded is not None else st.session_state.get("uploaded_file")
    if uploaded is None:
        return
    fid = _file_id(uploaded)
    saved = st.session_state.get(CONSENT_STATE) or {}
    if saved.get("file_id") != fid:  # 新しいファイル → 未同意から
        saved = {"file_id": fid, "consent": False, "age": "未回答", "sex": "未回答"}

    kp = "sf_up_" + hashlib.md5(fid.encode()).hexdigest()[:8]
    with st.container(border=True):
        st.markdown("#### データ保存の同意（研究・サービス改善用）")
        consent = st.checkbox(
            "本人の同意を得たうえで、個人を特定しない測定データ（関節の動きの数値・スコア）を保存します",
            value=saved["consent"], key=f"{kp}_consent",
        )
        age = st.radio("年代（任意）", AGE_OPTIONS, horizontal=True,
                       index=AGE_OPTIONS.index(saved["age"]), key=f"{kp}_age")
        sex = st.radio("性別（任意）", SEX_OPTIONS, horizontal=True,
                       index=SEX_OPTIONS.index(saved["sex"]), key=f"{kp}_sex")
        st.session_state[CONSENT_STATE] = {"file_id": fid, "consent": consent, "age": age, "sex": sex}
        if consent:
            st.caption("✅ 解析ページを開くと、測定データとスコアが自動で保存されます。")
        else:
            st.caption("同意がない場合、データは保存されません（解析はそのまま使えます）。")


def _current_consent():
    """今アップロードされているファイルに対する同意内容。無ければ None。"""
    info = st.session_state.get(CONSENT_STATE)
    uploaded = st.session_state.get("uploaded_file")
    if not info or uploaded is None or info.get("file_id") != _file_id(uploaded):
        return None
    return info


# ---------------------------------------------------------------
# 各ページの最後で呼ぶ：同意済みなら自動保存（ボタン不要）
# ---------------------------------------------------------------
def render_save_section(df_phase, movement, summary=None, key_prefix=None):
    if st.session_state.get("uploaded_file") is None:
        return
    info = _current_consent()
    if not info or not info.get("consent"):
        st.caption("💾 データ保存：同意なし（保存していません）。アップロード画面で変更できます。")
        return

    profile = {
        "AGE_GROUP": None if info["age"] == "未回答" else info["age"],
        "SEX": None if info["sex"] == "未回答" else info["sex"],
    }
    kp = key_prefix or movement.replace(" ", "_").replace("-", "_").lower()
    fhash = data_hash_of(df_phase)
    done_key = f"{kp}_sf_saved_{fhash}"

    if st.session_state.get(done_key):
        st.caption(f"💾 データ保存：{st.session_state[done_key]}")
        return
    try:
        with st.spinner("Snowflake に保存中..."):
            status, n = save_raw(df_phase, fhash, movement)
            s_status = save_summary(summary, fhash, movement, profile)
        if status == "exists":
            msg = "このデータは保存済みです（重複して保存はしません）。"
        else:
            msg = f"生データ {n} フレームと要約を保存しました。" if s_status == "saved" \
                else f"生データ {n} フレームを保存しました。"
        st.session_state[done_key] = msg
        st.caption(f"💾 データ保存：{msg}")
    except Exception as e:
        st.error(f"Snowflake への保存に失敗しました：{e}")


# ---------------------------------------------------------------
# どのページでも使える簡単版：ページの変数を名前で探して保存する
# ---------------------------------------------------------------
DF_CANDIDATES = ("df_phase", "df", "df_raw", "df_all", "data")


def render_save_from_page(page_globals, movement, summary_vars=(), df_names=DF_CANDIDATES):
    """page_globals には各ページで globals() を渡す。
    summary_vars の変数がページに無い場合は、その項目だけ保存しない（エラーにしない）。"""
    df = None
    for name in df_names:
        obj = page_globals.get(name)
        if isinstance(obj, pd.DataFrame) and len(obj):
            df = obj
            break
    if df is None:
        if st.session_state.get("uploaded_file") is not None:
            st.warning("保存用の測定データ（DataFrame）が見つかりませんでした。")
        return
    summary = {v: page_globals.get(v) for v in summary_vars}
    render_save_section(df, movement=movement, summary=summary)
