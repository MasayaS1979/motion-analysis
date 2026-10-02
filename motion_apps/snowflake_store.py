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


def _already_saved(conn, table, fhash, movement=None):
    """同じ測定データがすでにあれば、その動作名を返す（無ければ None）。
    動作に関係なく、同じデータは1回しか保存しない。
    テーブルが無い（初回）ときだけ None。それ以外のエラーは止める（二重保存を防ぐため）。"""
    cur = conn.cursor()
    try:
        # 接続の設定によって %(h)s 形式が使えないため、値を安全に埋め込む
        h = "".join(c for c in str(fhash) if c in "0123456789abcdef")
        cur.execute(f"SELECT MOVEMENT FROM {table} WHERE FILE_HASH = '{h}' LIMIT 1")
        row = cur.fetchone()
        return (row[0] or "不明") if row else None
    except Exception as e:
        if "does not exist" in str(e):
            return None
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
    prev = _already_saved(conn, RAW_TABLE, fhash)
    if prev:
        return "exists", prev
    raw = _prepare_raw(df_phase, fhash, movement)
    _ensure_columns(conn, RAW_TABLE, raw)
    conn.write_pandas(raw, RAW_TABLE, auto_create_table=True, quote_identifiers=False)
    return "saved", len(raw)


def save_summary(summary, fhash, movement, profile=None):
    conn = get_conn()
    if _already_saved(conn, SUMMARY_TABLE, fhash):
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


CONSENT_TEXT = (
    "- **保存するもの**：関節の角度などの動きの数値、解析スコア、年代・性別（任意）\n"
    "- **保存しないもの**：氏名・連絡先・顔の映像など、個人を特定できる情報\n"
    "- **利用目的**：動作評価の精度向上、正常値の研究、サービス改善\n"
    "- **削除**：ご本人の申し出があれば、保存したデータを削除します"
)


def consent_and_upload(label="📁 測定ファイルをアップロード", **uploader_kwargs):
    """app.py の st.file_uploader(...) の代わりに使う。
    同意にチェックを入れるまでアップロードできない。別のファイル（次の人）を
    アップロードしようとすると、同意をやり直す。"""
    state = st.session_state.get(CONSENT_STATE) or {
        "file_id": None, "consent": False, "age": "未回答", "sex": "未回答", "nonce": 0}
    state.setdefault("nonce", 0)
    kp = f"sf_gate_{state['nonce']}"

    if st.session_state.pop("sf_gate_msg", None):
        st.warning("新しい測定ファイルです。ご本人の同意を確認してから、もう一度アップロードしてください。")

    with st.container(border=True):
        st.markdown("#### ① データ保存の同意（研究・サービス改善用）")
        with st.expander("📄 保存する内容と利用目的（必ずご本人に説明してください）"):
            st.markdown(CONSENT_TEXT)
        consent = st.checkbox(
            "本人の同意を得たうえで、個人を特定しない測定データ（関節の動きの数値・スコア）を保存します",
            value=state["consent"], key=f"{kp}_consent")
        age = st.radio("年代（任意）", AGE_OPTIONS, horizontal=True,
                       index=AGE_OPTIONS.index(state["age"]), key=f"{kp}_age")
        sex = st.radio("性別（任意）", SEX_OPTIONS, horizontal=True,
                       index=SEX_OPTIONS.index(state["sex"]), key=f"{kp}_sex")
    state.update(consent=consent, age=age, sex=sex)

    st.markdown("#### ② 測定ファイルのアップロード")
    f = st.file_uploader(label, disabled=not consent, key=f"{kp}_uploader", **uploader_kwargs)

    if not consent:
        st.info("☝️ 同意にチェックを入れると、アップロードできるようになります。")
        st.session_state["uploaded_file"] = None
        state["file_id"] = None
        st.session_state[CONSENT_STATE] = state
        return None

    if f is not None:
        fid = _file_id(f)
        if state["file_id"] is None:
            state["file_id"] = fid                      # この同意をこのファイルに結びつける
        elif state["file_id"] != fid:                   # 別のファイル＝次の人 → 同意をやり直す
            st.session_state[CONSENT_STATE] = {
                "file_id": None, "consent": False, "age": "未回答", "sex": "未回答",
                "nonce": state["nonce"] + 1}
            st.session_state["uploaded_file"] = None
            st.session_state["sf_gate_msg"] = True
            st.rerun()
        st.session_state["uploaded_file"] = f
    st.session_state[CONSENT_STATE] = state

    current = st.session_state.get("uploaded_file")
    if current is not None:
        st.success(f"アップロード済み：{getattr(current, 'name', current)}　→ 下の解析ページから動作を選んでください。")
        if st.button("🔄 次の人の測定を始める（同意からやり直す）", key=f"{kp}_reset"):
            st.session_state[CONSENT_STATE] = {
                "file_id": None, "consent": False, "age": "未回答", "sex": "未回答",
                "nonce": state["nonce"] + 1}
            st.session_state["uploaded_file"] = None
            st.rerun()
    return current


def render_upload_consent(uploaded=None):
    """（旧方式）アップロード後に同意欄を出す。consent_and_upload を使う場合は不要。"""
    uploaded = uploaded if uploaded is not None else st.session_state.get("uploaded_file")
    if uploaded is None:
        return
    fid = _file_id(uploaded)
    saved = st.session_state.get(CONSENT_STATE) or {}
    if saved.get("file_id") != fid:
        saved = {"file_id": fid, "consent": False, "age": "未回答", "sex": "未回答"}
    kp = "sf_up_" + hashlib.md5(fid.encode()).hexdigest()[:8]
    with st.container(border=True):
        st.markdown("#### データ保存の同意（研究・サービス改善用）")
        with st.expander("📄 保存する内容と利用目的（必ずご本人に説明してください）"):
            st.markdown(CONSENT_TEXT)
        consent = st.checkbox(
            "本人の同意を得たうえで、個人を特定しない測定データ（関節の動きの数値・スコア）を保存します",
            value=saved["consent"], key=f"{kp}_consent")
        age = st.radio("年代（任意）", AGE_OPTIONS, horizontal=True,
                       index=AGE_OPTIONS.index(saved["age"]), key=f"{kp}_age")
        sex = st.radio("性別（任意）", SEX_OPTIONS, horizontal=True,
                       index=SEX_OPTIONS.index(saved["sex"]), key=f"{kp}_sex")
        st.session_state[CONSENT_STATE] = {"file_id": fid, "consent": consent, "age": age, "sex": sex}


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
            s_status = None if status == "exists" else save_summary(summary, fhash, movement, profile)
        if status == "exists":
            if n == movement:
                msg = "このデータは保存済みです（重複して保存はしません）。"
            else:
                msg = (f"このデータはすでに「{n}」として保存済みのため、保存しませんでした。"
                       "（同じデータは1回だけ保存します）")
                st.warning(f"⚠️ {msg}")
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
