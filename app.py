# app.py
import re
import difflib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# =============================================================================
# BACKEND PATH CONFIG (ubah sesuai struktur folder kamu)
# =============================================================================
DATA_DIR = Path("data")

# Matches dataset (WAJIB ADA)
MATCHES_PATH = Path("matches_laliga_2025_2026 (1).csv")

# Pilih salah satu sumber prediksi akhir (untuk FORECAST matchday > played):
# Opsi A (recommended): predicted_df sudah jadi dari notebook
PREDICTED_DF_PATH = DATA_DIR / "predicted_df.csv"       # kolom: Team, Total_Points, Total_GD

# Opsi B: generate predicted_df via joblib + team_features
MODEL_JOBLIB_PATH = DATA_DIR / "model.joblib"
TEAM_FEATURES_PATH = DATA_DIR / "team_features.csv"     # kolom: Team + fitur

# =============================================================================
# UTIL: Normalisasi & fuzzy team
# =============================================================================
def _norm_team(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def resolve_team_name(team_query: str, available_teams: list[str], cutoff: float = 0.55) -> dict:
    q = str(team_query).strip()
    qn = _norm_team(q)

    norm_map = {_norm_team(t): t for t in available_teams}
    if qn in norm_map:
        return {"found": True, "team": norm_map[qn], "suggestions": []}

    close_norm = difflib.get_close_matches(qn, list(norm_map.keys()), n=5, cutoff=cutoff)
    if close_norm:
        best = norm_map[close_norm[0]]
        sugg = [norm_map[k] for k in close_norm]
        return {"found": True, "team": best, "suggestions": sugg}

    lower_to_orig = {t.lower(): t for t in available_teams}
    close_raw = difflib.get_close_matches(q.lower(), list(lower_to_orig.keys()), n=5, cutoff=cutoff)
    if close_raw:
        best = lower_to_orig[close_raw[0]]
        sugg = [lower_to_orig[x] for x in close_raw]
        return {"found": True, "team": best, "suggestions": sugg}

    return {"found": False, "team": None, "suggestions": []}

# =============================================================================
# IO: Read CSV/XLSX
# =============================================================================
def read_table_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # fallback
    return pd.read_csv(path)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _pick_exact(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

# =============================================================================
# FIX: mapping schema matches
# =============================================================================
def map_matches_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output schema:
      Matchday (int), Home (str), Away (str), HG (float), AG (float), Date (optional)
    FIX utama:
      - Kolom Gol_Tuan_Rumah / Gol_Tim_Tamu tidak boleh dianggap Home/Away.
    """
    df = standardize_columns(df)
    cols = list(df.columns)

    col_md   = _pick_exact(cols, ["Matchday", "Pekan", "Round", "Week"])
    col_home = _pick_exact(cols, ["Home", "Tuan_Rumah", "Home_Team", "Tim_Rumah"])
    col_away = _pick_exact(cols, ["Away", "Tim_Tamu", "Away_Team", "Tim_Tandang"])
    col_hg   = _pick_exact(cols, ["HG", "Gol_Tuan_Rumah", "Home_Goals", "FT_HG"])
    col_ag   = _pick_exact(cols, ["AG", "Gol_Tim_Tamu", "Away_Goals", "FT_AG"])
    col_date = _pick_exact(cols, ["Date", "Tanggal"])

    out = pd.DataFrame(index=df.index)

    if col_md is not None:
        out["Matchday"] = pd.to_numeric(df[col_md], errors="coerce")
    if col_home is not None:
        out["Home"] = df[col_home].astype(str).str.strip()
    if col_away is not None:
        out["Away"] = df[col_away].astype(str).str.strip()
    if col_hg is not None:
        out["HG"] = pd.to_numeric(df[col_hg], errors="coerce")
    if col_ag is not None:
        out["AG"] = pd.to_numeric(df[col_ag], errors="coerce")
    if col_date is not None:
        out["Date"] = pd.to_datetime(df[col_date], errors="coerce")

    def is_goal_col(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in ["gol", "goal", "hg", "ag", "score", "hasil"])

    # fallback Matchday
    if "Matchday" not in out.columns:
        for col in cols:
            c = col.lower()
            if any(k in c for k in ["matchday", "pekan", "round", "week", "md"]):
                out["Matchday"] = pd.to_numeric(df[col], errors="coerce")
                break

    # fallback Home/Away (SKIP goal columns!)
    if "Home" not in out.columns:
        for col in cols:
            if is_goal_col(col):
                continue
            c = col.lower()
            if c in ["home", "tuan_rumah", "tim_rumah", "home_team"] or c.endswith("_home") or c.endswith("home"):
                out["Home"] = df[col].astype(str).str.strip()
                break

    if "Away" not in out.columns:
        for col in cols:
            if is_goal_col(col):
                continue
            c = col.lower()
            if c in ["away", "tim_tamu", "tim_tandang", "away_team"] or c.endswith("_away") or c.endswith("away"):
                out["Away"] = df[col].astype(str).str.strip()
                break

    # fallback HG/AG
    if "HG" not in out.columns:
        for col in cols:
            c = col.lower()
            if c in ["hg", "gol_tuan_rumah", "home_goals", "ft_hg"] or "gol_tuan" in c:
                out["HG"] = pd.to_numeric(df[col], errors="coerce")
                break

    if "AG" not in out.columns:
        for col in cols:
            c = col.lower()
            if c in ["ag", "gol_tim_tamu", "away_goals", "ft_ag"] or "gol_tim_tamu" in c:
                out["AG"] = pd.to_numeric(df[col], errors="coerce")
                break

    # fallback skor "2-1"
    if ("HG" not in out.columns) or ("AG" not in out.columns):
        for col in cols:
            c = col.lower()
            if any(k in c for k in ["score", "ft", "hasil", "skor"]) and col not in ["Home", "Away"]:
                s = df[col].astype(str)
                m = s.str.extract(r"(\d+)\D+(\d+)")
                if "HG" not in out.columns:
                    out["HG"] = pd.to_numeric(m[0], errors="coerce")
                if "AG" not in out.columns:
                    out["AG"] = pd.to_numeric(m[1], errors="coerce")
                break

    out = out.dropna(subset=["Home", "Away"]).copy()
    out["Matchday"] = pd.to_numeric(out["Matchday"], errors="coerce").fillna(1).astype(int)
    out["HG"] = pd.to_numeric(out["HG"], errors="coerce") if "HG" in out.columns else np.nan
    out["AG"] = pd.to_numeric(out["AG"], errors="coerce") if "AG" in out.columns else np.nan
    return out

# =============================================================================
# Logic: infer played matchday & standings
# =============================================================================
def infer_matches_played(matches_std: pd.DataFrame) -> int:
    played = matches_std.dropna(subset=["HG", "AG"])
    if played.empty:
        return 0
    return int(played["Matchday"].max())

def standings_from_matches(matches_std: pd.DataFrame, matchday: int) -> pd.DataFrame:
    m = matches_std[matches_std["Matchday"] <= matchday].copy()
    m = m.dropna(subset=["HG", "AG"])  # hanya yang sudah ada skor

    teams = pd.unique(pd.concat([m["Home"], m["Away"]], ignore_index=True))
    tab = pd.DataFrame({"Team": teams}).set_index("Team")
    tab["Pts"] = 0
    tab["GF"] = 0
    tab["GA"] = 0

    for _, r in m.iterrows():
        h, a = r["Home"], r["Away"]
        hg, ag = int(r["HG"]), int(r["AG"])

        tab.loc[h, "GF"] += hg
        tab.loc[h, "GA"] += ag
        tab.loc[a, "GF"] += ag
        tab.loc[a, "GA"] += hg

        if hg > ag:
            tab.loc[h, "Pts"] += 3
        elif hg < ag:
            tab.loc[a, "Pts"] += 3
        else:
            tab.loc[h, "Pts"] += 1
            tab.loc[a, "Pts"] += 1

    tab["GD"] = tab["GF"] - tab["GA"]
    tab = tab.reset_index()
    tab = tab.sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    tab["Rank"] = np.arange(1, len(tab) + 1)
    return tab

def build_forecast_table_for_matchday(
    matchday: int,
    current_md: int,
    current_table: pd.DataFrame,   # Team, Pts, GD at current_md
    final_pred: pd.DataFrame,      # Team, Total_Points, Total_GD
    total_matches: int
) -> pd.DataFrame:
    fp = final_pred[["Team", "Total_Points", "Total_GD"]].copy()
    fp["Team_norm"] = fp["Team"].map(_norm_team)

    cur = current_table[["Team", "Pts", "GD"]].copy()
    cur["Team_norm"] = cur["Team"].map(_norm_team)

    merged = cur.merge(fp[["Team_norm", "Total_Points", "Total_GD"]], on="Team_norm", how="left")
    merged["Total_Points"] = merged["Total_Points"].fillna(merged["Pts"])
    merged["Total_GD"] = merged["Total_GD"].fillna(merged["GD"])

    if matchday <= current_md:
        out = merged[["Team", "Pts", "GD"]].copy()
    else:
        denom = max(1, (total_matches - current_md))
        frac = (matchday - current_md) / denom
        out = pd.DataFrame({
            "Team": merged["Team"],
            "Pts": merged["Pts"] + (merged["Total_Points"] - merged["Pts"]) * frac,
            "GD": merged["GD"] + (merged["Total_GD"] - merged["GD"]) * frac,
        })

    out["Pts"] = out["Pts"].astype(float)
    out["GD"] = out["GD"].astype(float)
    out = out.sort_values(["Pts", "GD"], ascending=[False, False]).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out

# =============================================================================
# JOBLIB integration: generate predicted_df (opsional)
# =============================================================================
def predict_final_table_from_model(model, team_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    team_features_df wajib punya 'Team' + fitur numerik.
    Jika model output 2 kolom -> (points, gd)
    Jika model output 1 kolom -> points saja, GD=0
    """
    df = team_features_df.copy()
    if "Team" not in df.columns:
        raise ValueError("team_features.csv harus punya kolom 'Team'.")

    X = df.drop(columns=["Team"])

    if hasattr(model, "feature_names_in_"):
        needed = list(model.feature_names_in_)
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise ValueError(f"Fitur kurang untuk model: {missing}")
        X = X[needed]

    y = np.asarray(model.predict(X))
    if y.ndim == 2 and y.shape[1] >= 2:
        pts = y[:, 0]
        gd = y[:, 1]
    else:
        pts = y.reshape(-1)
        gd = np.zeros_like(pts)

    return pd.DataFrame({
        "Team": df["Team"].astype(str),
        "Total_Points": pts.astype(float),
        "Total_GD": gd.astype(float),
    })

def load_predicted_df_backend() -> tuple[pd.DataFrame | None, str]:
    """
    Return (predicted_df, source_note)
    Prioritas:
      1) predicted_df.csv jika ada
      2) model.joblib + team_features.csv jika ada
      else None
    """
    if PREDICTED_DF_PATH.exists():
        df = read_table_path(PREDICTED_DF_PATH)
        df = standardize_columns(df)

        # mapping kolom minimal (robust)
        def pick_col(cands):
            for c in df.columns:
                if c.lower() in [x.lower() for x in cands]:
                    return c
            return None

        c_team = pick_col(["Team", "Tim"])
        c_pts  = pick_col(["Total_Points", "Pred_Points", "Points", "Poin_Akhir", "TotalPoint"])
        c_gd   = pick_col(["Total_GD", "Pred_GD", "GD", "Selisih_Goal_Akhir", "TotalGD"])

        if c_team and c_pts and c_gd:
            out = pd.DataFrame({
                "Team": df[c_team].astype(str),
                "Total_Points": pd.to_numeric(df[c_pts], errors="coerce"),
                "Total_GD": pd.to_numeric(df[c_gd], errors="coerce"),
            }).dropna(subset=["Team"]).copy()
            return out, f"predicted_df dari file: {PREDICTED_DF_PATH.name}"

        return None, f"File {PREDICTED_DF_PATH.name} ada, tapi kolomnya tidak sesuai (butuh Team, Total_Points, Total_GD)."

    if MODEL_JOBLIB_PATH.exists() and TEAM_FEATURES_PATH.exists():
        model = joblib.load(MODEL_JOBLIB_PATH)
        feat = read_table_path(TEAM_FEATURES_PATH)
        feat = standardize_columns(feat)
        out = predict_final_table_from_model(model, feat)
        return out, f"predicted_df digenerate dari model: {MODEL_JOBLIB_PATH.name}"

    return None, "predicted_df tidak tersedia (tidak ada predicted_df.csv atau model.joblib + team_features.csv)."

# =============================================================================
# STREAMLIT UI
# =============================================================================
# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="LaLiga Rank per Matchday", layout="wide")

# =============================================================================
# Load matches from backend (MOVED UP)
# =============================================================================
try:
    matches_raw = read_table_path(MATCHES_PATH)
except Exception as e:
    st.error(f"Gagal load matches dari backend: {e}")
    st.stop()

matches_std = map_matches_schema(matches_raw)
teams_detected = sorted(pd.unique(pd.concat([matches_std["Home"], matches_std["Away"]], ignore_index=True)).tolist())
MATCHES_PLAYED_AUTO = infer_matches_played(matches_std)

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.title("📊 Forecasting Posisi Tim per Matchday")

# Sidebar controls
with st.sidebar:
    st.header("Parameter")
    with st.form("run_form"):
        TOTAL_MATCHES = st.number_input("TOTAL_MATCHES (LaLiga = 38)", min_value=1, max_value=60, value=38, step=1)
        TEAM_QUERY = st.text_input("Nama tim", value="Real Madrid")
        MATCHDAY_QUERY = st.number_input("Matchday yang ingin dicek", min_value=1, max_value=int(TOTAL_MATCHES), value=15, step=1)
        submitted = st.form_submit_button("Running Model")


    
    # LIST TEAM DIAWAL (For Copy Paste)
    st.subheader("📋 Daftar Tim (Copy-Paste)")
    st.text_area("List Tim:", value="\n".join(teams_detected), height=300)

if not submitted:
    st.warning("Silakan tekan **Running Model** di sidebar untuk memproses.")
    st.stop()




# =============================================================================
# Load predicted_df backend (for forecast only)
# =============================================================================
predicted_df, pred_note = load_predicted_df_backend()
# st.caption(f"Sumber prediksi akhir: {pred_note}")

# =============================================================================
# Build table for query matchday
# =============================================================================
# current table at matches_played
current_table_md = standings_from_matches(matches_std, int(MATCHES_PLAYED_AUTO))
cur_tbl = current_table_md[["Team", "Pts", "GD"]].copy()

if int(MATCHDAY_QUERY) <= int(MATCHES_PLAYED_AUTO):
    table_q = standings_from_matches(matches_std, int(MATCHDAY_QUERY))[["Team", "Pts", "GD", "Rank"]]
    mode_note = "Akurat"
else:
    if predicted_df is None:
        st.error("Matchday > MATCHES_PLAYED, tapi predicted_df tidak tersedia. Tambahkan predicted_df.csv atau model.joblib + team_features.csv di folder data/.")
        st.stop()
    table_q = build_forecast_table_for_matchday(
        matchday=int(MATCHDAY_QUERY),
        current_md=int(MATCHES_PLAYED_AUTO),
        current_table=cur_tbl,
        final_pred=predicted_df,
        total_matches=int(TOTAL_MATCHES),
    )
    mode_note = "FORECAST (interpolasi linear dari kondisi saat ini ke prediksi akhir)"

# =============================================================================
# Output utama: rank tim pada matchday tertentu
# =============================================================================
st.subheader(f"📌 {mode_note} — Posisi Tim pada Matchday {int(MATCHDAY_QUERY)}")

teams_available = table_q["Team"].dropna().astype(str).unique().tolist()
res = resolve_team_name(TEAM_QUERY, teams_available, cutoff=0.55)

if not res["found"]:
    st.error(f"Tim '{TEAM_QUERY}' tidak ditemukan di data.")
    # kasih saran
    sugg = res.get("suggestions") or []
    if sugg:
        st.write("Mungkin maksud kamu:")
        st.write(sugg[:5])
else:
    team_resolved = res["team"]
    row_team = table_q.loc[table_q["Team"].map(_norm_team) == _norm_team(team_resolved)]

    if row_team.empty:
        st.error(f"Tim '{team_resolved}' tidak ditemukan di klasemen matchday {int(MATCHDAY_QUERY)} (cek data).")
    else:
        r = row_team.iloc[0]
        if TEAM_QUERY.strip().lower() != team_resolved.strip().lower():
            st.info(f"Input '{TEAM_QUERY}' dicocokkan menjadi **{team_resolved}**")

        # Ini jawaban yang kamu minta:
        st.success(f"✅ **{r['Team']}** pada matchday **{int(MATCHDAY_QUERY)}** berada di peringkat **#{int(r['Rank'])}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Points", f"{float(r['Pts']):.1f}")
        m2.metric("Goal Difference", f"{float(r['GD']):.1f}")
        m3.metric("Matchday Mode", mode_note)

# =============================================================================
# Tampilkan Top 10 + download
# =============================================================================
st.markdown("### Top 10 Klasemen")
st.dataframe(table_q.head(10), use_container_width=True)

csv_out = table_q.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Tabel Matchda",
    data=csv_out,
    file_name=f"table_matchday_{int(MATCHDAY_QUERY)}.csv",
    mime="text/csv",
)

# =============================================================================
# Timeline rank tim
# =============================================================================
st.markdown("### 📈 Timeline Rank (1..TOTAL_MATCHES)")
if res["found"]:
    team_norm = _norm_team(res["team"])
    timeline = []
    for md in range(1, int(TOTAL_MATCHES) + 1):
        if md <= int(MATCHES_PLAYED_AUTO):
            tmd = standings_from_matches(matches_std, md)[["Team", "Pts", "GD", "Rank"]]
        else:
            if predicted_df is None:
                break
            tmd = build_forecast_table_for_matchday(
                matchday=md,
                current_md=int(MATCHES_PLAYED_AUTO),
                current_table=cur_tbl,
                final_pred=predicted_df,
                total_matches=int(TOTAL_MATCHES),
            )[["Team", "Pts", "GD", "Rank"]]

        rt = tmd.loc[tmd["Team"].map(_norm_team) == team_norm]
        if rt.empty:
            continue
        timeline.append({
            "Matchday": md,
            "Team": rt.iloc[0]["Team"],
            "Rank": int(rt.iloc[0]["Rank"]),
            "Pts": float(rt.iloc[0]["Pts"]),
            "GD": float(rt.iloc[0]["GD"]),
        })

    timeline_df = pd.DataFrame(timeline)
    if timeline_df.empty:
        st.warning("Timeline kosong (tim tidak ditemukan pada matchday tertentu).")
    else:
        st.dataframe(timeline_df, use_container_width=True)

        fig = plt.figure(figsize=(12, 4))
        plt.plot(timeline_df["Matchday"], timeline_df["Rank"], marker="o")
        plt.gca().invert_yaxis()
        plt.title(f"Perubahan Peringkat: {timeline_df.iloc[0]['Team']}")
        plt.xlabel("Matchday")
        plt.ylabel("Rank (1 = terbaik)")
        plt.grid(True)
        st.pyplot(fig)

        csv_tl = timeline_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Timeline (CSV)",
            data=csv_tl,
            file_name=f"timeline_rank_{team_norm}.csv",
            mime="text/csv",
        )
else:
    st.info("Masukkan nama tim yang valid untuk melihat timeline.")
