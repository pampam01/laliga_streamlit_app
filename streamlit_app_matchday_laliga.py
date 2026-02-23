
import os
import re
import glob
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import sys

# Ensure local modules can be found
if "." not in sys.path:
    sys.path.append(".")

# Import custom module to ensure unpickling works
try:
    import laliga_forecaster
except ImportError:
    pass

# ============================================================
# Streamlit: Forecast Rank per Matchday (Joblib-backed)
# Input hanya: Nama Club + Matchday
# Semua logika forecasting harus ada di artifact joblib.
# ============================================================

CANDIDATE_MODEL_PATHS = [
    "data/models/laliga_forecast_model.joblib",
    "laliga_forecast_model.joblib",
]

TOTAL_MATCHES_DEFAULT = 38


def _norm_team(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower().strip())


@st.cache_resource
def load_joblib_model() -> Any:
    """Load model artifact dari file joblib lokal."""
    for p in CANDIDATE_MODEL_PATHS:
        if os.path.exists(p):
            return joblib.load(p)

    any_joblibs = sorted(glob.glob("*.joblib"))
    if any_joblibs:
        return joblib.load(any_joblibs[0])

    raise FileNotFoundError(
        "Tidak menemukan file model *.joblib. "
        "Taruh file model di folder yang sama dengan script ini, "
        "misalnya bernama 'laliga_forecast_model.joblib'."
    )


def _extract_team_list(model_obj: Any) -> Optional[List[str]]:
    """Ambil daftar tim jika artifact menyediakannya."""
    for attr in ["teams", "team_list", "clubs", "club_list", "classes_"]:
        if hasattr(model_obj, attr):
            v = getattr(model_obj, attr)
            if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
                return [str(x) for x in list(v)]
    if isinstance(model_obj, dict):
        for k in ["teams", "team_list", "clubs", "club_list"]:
            if k in model_obj and isinstance(model_obj[k], (list, tuple, np.ndarray, pd.Index)):
                return [str(x) for x in list(model_obj[k])]
    return None


def _rank_from_table(df: Any, club: str, matchday: int) -> Tuple[int, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predict_table(matchday) harus mengembalikan pandas.DataFrame")

    # Kolom Team
    team_col = None
    for c in df.columns:
        if str(c).lower() in ["team", "club", "tim", "nama_tim", "nama_club"]:
            team_col = c
            break
    if team_col is None:
        team_col = df.columns[0]

    # Kolom Rank
    rank_col = None
    for c in df.columns:
        if str(c).lower() in ["rank", "peringkat", "posisi"]:
            rank_col = c
            break
    if rank_col is None:
        raise ValueError("Tabel dari predict_table() wajib punya kolom 'Rank' (atau 'Peringkat/Posisi').")

    df2 = df.copy()
    df2["_team_norm"] = df2[team_col].map(_norm_team)
    m = df2.loc[df2["_team_norm"] == _norm_team(club)]
    if m.empty:
        raise ValueError(f"Tim '{club}' tidak ditemukan di tabel prediksi matchday {matchday}.")

    row = m.iloc[0].to_dict()
    rank = int(row[rank_col])

    # Meta: Pts/GD jika ada
    meta: Dict[str, Any] = {}
    for key in df.columns:
        lk = str(key).lower()
        if lk in ["pts", "points", "point", "poin"]:
            meta["Pts"] = float(row[key])
        if lk in ["gd", "goal_difference", "selisih_goal", "selisihgol"]:
            meta["GD"] = float(row[key])
    return rank, meta


def _call_predict_rank(model_obj: Any, club: str, matchday: int) -> Tuple[int, Dict[str, Any]]:
    """
    Kontrak minimal:
      - Return rank (int). Meta boleh kosong.

    Interface yang didukung:
      1) model_obj.predict_rank(club, matchday) -> rank atau (rank, meta)
      2) model_obj["predict_rank"](club, matchday) -> rank atau (rank, meta)
      3) model_obj.predict_table(matchday) -> DataFrame kolom Team & Rank (+ optional Pts/GD)
      4) model_obj["predict_table"](matchday) -> DataFrame
    """
    if hasattr(model_obj, "predict_rank") and callable(getattr(model_obj, "predict_rank")):
        out = model_obj.predict_rank(club, matchday)
        if isinstance(out, tuple) and len(out) == 2:
            rank, meta = out
            return int(rank), dict(meta) if isinstance(meta, dict) else {"meta": meta}
        return int(out), {}

    if isinstance(model_obj, dict) and "predict_rank" in model_obj and callable(model_obj["predict_rank"]):
        out = model_obj["predict_rank"](club, matchday)
        if isinstance(out, tuple) and len(out) == 2:
            rank, meta = out
            return int(rank), dict(meta) if isinstance(meta, dict) else {"meta": meta}
        return int(out), {}

    if hasattr(model_obj, "predict_table") and callable(getattr(model_obj, "predict_table")):
        df = model_obj.predict_table(matchday)
        return _rank_from_table(df, club, matchday)

    if isinstance(model_obj, dict) and "predict_table" in model_obj and callable(model_obj["predict_table"]):
        df = model_obj["predict_table"](matchday)
        return _rank_from_table(df, club, matchday)

    raise TypeError(
        "Artifact joblib tidak punya interface yang dikenali. "
        "Minimal sediakan salah satu:\n"
        "- method: predict_rank(club, matchday)\n"
        "- atau dict: {'predict_rank': callable}\n"
        "- atau method/function: predict_table(matchday) -> DataFrame kolom Team & Rank."
    )


def _build_timeline(model_obj: Any, club: str, total_matches: int) -> pd.DataFrame:
    recs = []
    for md in range(1, total_matches + 1):
        try:
            rank, meta = _call_predict_rank(model_obj, club, md)
            recs.append({
                "Matchday": md,
                "Rank": int(rank),
                "Pts": float(meta.get("Pts")) if "Pts" in meta else np.nan,
                "GD": float(meta.get("GD")) if "GD" in meta else np.nan,
            })
        except Exception:
            continue
    return pd.DataFrame(recs)


# =========================
# UI
# =========================
st.set_page_config(page_title="Forecast Rank LaLiga (Joblib)", layout="wide")
st.title("Forecast Peringkat Tim LaLiga per Matchday (Joblib)")
st.caption("Input hanya: nama club + matchday. Prediksi diambil dari model/joblib.")

# Load model

try:
    model_obj = load_joblib_model()
except Exception as e:
    st.error(str(e))
    st.stop()

team_list = _extract_team_list(model_obj)

col1, col2 = st.columns([2, 1])
with col1:
    if team_list:
        default_team = "Real Madrid" if "Real Madrid" in team_list else team_list[0]
        club = st.selectbox("Nama Club", team_list, index=team_list.index(default_team))
    else:
        club = st.text_input("Nama Club", value="Real Madrid")

with col2:
    matchday = st.number_input(
        "Matchday / Pekan",
        min_value=1,
        max_value=TOTAL_MATCHES_DEFAULT,
        value=15,
        step=1,
    )

run = st.button("Forecast")

if run:
    try:
        rank, meta = _call_predict_rank(model_obj, club, int(matchday))

        st.subheader("Hasil Forecast")
        st.metric("Rank", int(rank))

        if "Pts" in meta or "GD" in meta:
            pts_txt = f"{meta.get('Pts'):.1f}" if meta.get("Pts") is not None else "-"
            gd_txt = f"{meta.get('GD'):.1f}" if meta.get("GD") is not None else "-"
            st.write(f"**Club:** {club}  |  **Poin:** {pts_txt}  |  **GD:** {gd_txt}")
        else:
            st.write(f"**Club:** {club}")

        st.divider()
        st.subheader("Visualisasi Tren Rank (Matchday 1–38)")
        timeline = _build_timeline(model_obj, club, TOTAL_MATCHES_DEFAULT)

        if timeline.empty:
            st.info("Timeline tidak tersedia dari model (prediksi tidak berhasil untuk matchday mana pun).")
        else:
            fig = plt.figure(figsize=(12, 4))
            plt.plot(timeline["Matchday"], timeline["Rank"], marker="o")
            plt.gca().invert_yaxis()
            plt.axvline(int(matchday))
            plt.xlabel("Matchday")
            plt.ylabel("Rank (1 = teratas)")
            plt.title(f"Tren Rank {club} (highlight MD={int(matchday)})")
            plt.grid(True)
            st.pyplot(fig, clear_figure=True)

            if timeline["Pts"].notna().sum() >= 3:
                fig2 = plt.figure(figsize=(12, 4))
                plt.plot(timeline["Matchday"], timeline["Pts"], marker="o")
                plt.axvline(int(matchday))
                plt.xlabel("Matchday")
                plt.ylabel("Poin (estimasi)")
                plt.title(f"Tren Poin {club} (highlight MD={int(matchday)})")
                plt.grid(True)
                st.pyplot(fig2, clear_figure=True)

    except Exception as e:
        st.error(f"Gagal forecasting: {e}")

    # --- ADDED: Tampilkan Klasemen (Predict Table) jika support ---
    if hasattr(model_obj, "predict_table") or (isinstance(model_obj, dict) and "predict_table" in model_obj):
        st.divider()
        st.subheader(f"Klasemen Prediksi Pekan {int(matchday)}")
        try:
            if hasattr(model_obj, "predict_table"):
                df_table = model_obj.predict_table(int(matchday))
            else:
                df_table = model_obj["predict_table"](int(matchday))
            
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Gagal menampilkan tabel detail: {e}")

st.caption(
    "Catatan untuk artefak joblib: sediakan salah satu interface berikut:\n"
    "1) method predict_rank(club, matchday)\n"
    "2) dict {'predict_rank': callable}\n"
    "3) method/function predict_table(matchday) -> DataFrame kolom Team & Rank"
)
