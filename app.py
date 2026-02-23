# app.py
import re
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib


# =============================================================================
# 0) KONFIG BACKEND (ubah sesuai struktur folder kamu)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

MATCHES_PATH = DATA_DIR / "matches_laliga_2025_2026 (1).csv"

MODEL_POINTS_PATH = MODEL_DIR / "model_points_laliga.joblib"
MODEL_GD_PATH = MODEL_DIR / "model_goal_diff_laliga.joblib"
SCALER_PATH = MODEL_DIR / "scaler_laliga.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder_team.joblib"

TOTAL_MATCHES_DEFAULT = 38


# =============================================================================
# 1) HELPER: NORMALIZE + FUZZY MATCH TEAM
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
# 2) LOAD & CLEAN MATCHES (schema standar)
# =============================================================================
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

def map_matches_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output schema: Matchday, Home, Away, HG, AG, Date(optional)
    FIX: jangan sampai kolom gol kebaca Home/Away.
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

    if col_md is not None:   out["Matchday"] = pd.to_numeric(df[col_md], errors="coerce")
    if col_home is not None: out["Home"] = df[col_home].astype(str).str.strip()
    if col_away is not None: out["Away"] = df[col_away].astype(str).str.strip()
    if col_hg is not None:   out["HG"] = pd.to_numeric(df[col_hg], errors="coerce")
    if col_ag is not None:   out["AG"] = pd.to_numeric(df[col_ag], errors="coerce")
    if col_date is not None: out["Date"] = pd.to_datetime(df[col_date], errors="coerce")

    def is_goal_col(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in ["gol", "goal", "hg", "ag", "score", "hasil"])

    # fallback matchday
    if "Matchday" not in out.columns:
        for col in cols:
            c = col.lower()
            if any(k in c for k in ["matchday", "pekan", "round", "week", "md"]):
                out["Matchday"] = pd.to_numeric(df[col], errors="coerce")
                break

    # fallback home/away (skip goal cols)
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
    if "HG" in out.columns: out["HG"] = pd.to_numeric(out["HG"], errors="coerce")
    if "AG" in out.columns: out["AG"] = pd.to_numeric(out["AG"], errors="coerce")
    return out

def infer_matches_played(matches_std: pd.DataFrame) -> int:
    played = matches_std.dropna(subset=["HG", "AG"])
    if played.empty:
        return 0
    return int(played["Matchday"].max())


# =============================================================================
# 3) STANDINGS (AKURAT)
# =============================================================================
def standings_from_matches(matches_std: pd.DataFrame, matchday: int) -> pd.DataFrame:
    m = matches_std[matches_std["Matchday"] <= matchday].copy()
    m = m.dropna(subset=["HG", "AG"])

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


# =============================================================================
# 4) FEATURE ENGINEERING dari MATCHES (untuk inferensi model)
# =============================================================================
def build_team_features_from_matches(matches_std: pd.DataFrame, matchday: int) -> pd.DataFrame:
    m = matches_std[matches_std["Matchday"] <= matchday].copy()
    m = m.dropna(subset=["HG", "AG"])

    teams = pd.unique(pd.concat([m["Home"], m["Away"]], ignore_index=True))
    tab = pd.DataFrame({"Team": teams}).set_index("Team")

    tab["Menang"] = 0
    tab["Seri"] = 0
    tab["Kalah"] = 0
    tab["Goal_Masuk"] = 0
    tab["Goal_Kebobolan"] = 0
    tab["Point"] = 0

    for _, r in m.iterrows():
        h, a = r["Home"], r["Away"]
        hg, ag = int(r["HG"]), int(r["AG"])

        tab.loc[h, "Goal_Masuk"] += hg
        tab.loc[h, "Goal_Kebobolan"] += ag
        tab.loc[a, "Goal_Masuk"] += ag
        tab.loc[a, "Goal_Kebobolan"] += hg

        if hg > ag:
            tab.loc[h, "Menang"] += 1
            tab.loc[a, "Kalah"] += 1
            tab.loc[h, "Point"] += 3
        elif hg < ag:
            tab.loc[a, "Menang"] += 1
            tab.loc[h, "Kalah"] += 1
            tab.loc[a, "Point"] += 3
        else:
            tab.loc[h, "Seri"] += 1
            tab.loc[a, "Seri"] += 1
            tab.loc[h, "Point"] += 1
            tab.loc[a, "Point"] += 1

    df = tab.reset_index().rename(columns={"index": "Team"})
    df["Selisih_Goal"] = df["Goal_Masuk"] - df["Goal_Kebobolan"]

    mp = max(1, int(matchday))
    df["Win_Rate"] = df["Menang"] / mp
    df["Draw_Rate"] = df["Seri"] / mp
    df["Loss_Rate"] = df["Kalah"] / mp
    df["Points_Per_Match"] = df["Point"] / mp
    df["Goals_Scored_Per_Match"] = df["Goal_Masuk"] / mp
    df["Goals_Conceded_Per_Match"] = df["Goal_Kebobolan"] / mp
    df["Goal_Diff_Per_Match"] = df["Selisih_Goal"] / mp
    return df


# =============================================================================
# 5) MODEL INFERENCE: prediksi target akhir musim (Total Points & Total GD)
# =============================================================================
def predict_final_table_from_current_features(current_features_df: pd.DataFrame,
                                             model_points, model_gd,
                                             scaler, le_team) -> pd.DataFrame:
    df = current_features_df.copy()
    df["Team"] = df["Team"].astype(str).str.strip()

    def enc(x):
        try:
            return int(le_team.transform([x])[0])
        except Exception:
            return -1

    df["Team_Encoded"] = df["Team"].apply(enc)

    feature_cols = [
        "Win_Rate", "Draw_Rate", "Loss_Rate",
        "Points_Per_Match",
        "Goals_Scored_Per_Match", "Goals_Conceded_Per_Match",
        "Goal_Diff_Per_Match",
        "Team_Encoded"
    ]
    X = df[feature_cols].fillna(0)
    Xs = scaler.transform(X)

    pred_points_season = model_points.predict(Xs)
    pred_gd_season = model_gd.predict(Xs)

    return pd.DataFrame({
        "Team": df["Team"],
        "Total_Points": pred_points_season.astype(float),
        "Total_GD": pred_gd_season.astype(float),
    })


# =============================================================================
# 6) FORECAST TABLE pada matchday N (proyeksi current -> target akhir)
# =============================================================================
def build_forecast_table_for_matchday(matchday: int,
                                      current_md: int,
                                      current_table: pd.DataFrame,
                                      final_pred: pd.DataFrame,
                                      total_matches: int) -> pd.DataFrame:
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
# 7) STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="LaLiga Rank per Matchday", layout="wide")
st.title("📊 LaLiga 2025/26 — Posisi Tim per Matchday (AKURAT & FORECAST)")

with st.sidebar:
    st.header("Input")
    with st.form("input_form"):
        total_matches = st.number_input("Total match (default LaLiga 38)", min_value=1, max_value=60, value=TOTAL_MATCHES_DEFAULT, step=1)
        
        # New: Manual override for "Current Matchday" to force Forecast mode if dataset is full
        sim_current_md = st.number_input("Simulasi Pekan Saat Ini (Realita)", min_value=1, max_value=60, value=19, step=1, help="Anggap liga baru berjalan sampai pekan ini. Pekan setelahnya akan dianggap belum main (Forecast).")
        
        team_query = st.text_input("Nama tim", value="Real Madrid")
        
        # Ensure initial value doesn't exceed max_value to avoid StreamlitValueAboveMaxError
        # Default to total_matches to prioritize "FORECAST" mode
        safe_md_val = int(total_matches)
        matchday_query = st.number_input("Matchday yang ingin dicek", min_value=1, max_value=int(total_matches), value=safe_md_val, step=1)
        
        submitted = st.form_submit_button("Running")

    st.divider()
    
    st.info("""
    **Cara Membaca Forecast:**
    
    1. **Matchday 1 s/d Terakhir Main**: Menampilkan klasemen ASLI.
    2. **Matchday > Terakhir Main**: Menampilkan PREDIKSI (Forecast) tim akan finis di peringkat berapa.
    3. **Total Match**: Target akhir musim (biasanya 38).
    
    **Simulasi Pekan Saat Ini**:
    - Jika dataset penuh (38 match), atur ini ke misal **19**. 
    - Maka Matchday 20-38 akan dianggap "masa depan" (Forecast).
    
    Gunakan tombol **Running** setiap ganti input.
    """)

    st.caption("Backend files")
    st.code(
        "\n".join([
            str(MATCHES_PATH),
            str(MODEL_POINTS_PATH),
            str(MODEL_GD_PATH),
            str(SCALER_PATH),
            str(ENCODER_PATH),
        ]),
        language="text",
    )

# ---- Load matches ----
if not MATCHES_PATH.exists():
    st.error(f"File matches tidak ditemukan: {MATCHES_PATH}")
    st.stop()

matches_raw = pd.read_csv(MATCHES_PATH)
matches_std = map_matches_schema(matches_raw)
matches_played_real = infer_matches_played(matches_std)
matches_played = min(matches_played_real, int(sim_current_md))

teams_detected = sorted(pd.unique(pd.concat([matches_std["Home"], matches_std["Away"]], ignore_index=True)).tolist())

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("🔎 Validasi Dataset")
    st.write(f"- Tim terdeteksi: **{len(teams_detected)}** (ideal 20)")
    st.write(f"- MATCHES_PLAYED (auto dari skor): **{matches_played}**")
    st.write("Contoh tim:", teams_detected[:10])
with c2:
    st.write("Contoh data (setelah mapping):")
    st.dataframe(matches_std.head(6), use_container_width=True)

# ---- Build table matchday ----
matchday_query = int(matchday_query)

if matchday_query <= matches_played:
    # AKURAT
    table_q = standings_from_matches(matches_std, matchday_query)[["Team", "Pts", "GD", "Rank"]]
    mode_note = "AKURAT (berdasarkan hasil pertandingan)"
else:
    # FORECAST
    needed = [MODEL_POINTS_PATH, MODEL_GD_PATH, SCALER_PATH, ENCODER_PATH]
    missing = [p for p in needed if not p.exists()]
    if missing:
        st.error("File model/scaler/encoder belum lengkap di backend:\n" + "\n".join([str(p) for p in missing]))
        st.stop()

    model_points = joblib.load(MODEL_POINTS_PATH)
    model_gd = joblib.load(MODEL_GD_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_team = joblib.load(ENCODER_PATH)

    # current standings pada matches_played
    current_table_md = standings_from_matches(matches_std, matches_played)
    cur_tbl = current_table_md[["Team", "Pts", "GD"]].copy()

    # current features (dari match results)
    current_feat = build_team_features_from_matches(matches_std, matches_played)

    # prediksi target akhir musim (Total Points + Total GD)
    predicted_df = predict_final_table_from_current_features(current_feat, model_points, model_gd, scaler, le_team)

    # proyeksi matchday query
    table_q = build_forecast_table_for_matchday(
        matchday=matchday_query,
        current_md=matches_played,
        current_table=cur_tbl,
        final_pred=predicted_df,
        total_matches=int(total_matches),
    )[["Team", "Pts", "GD", "Rank"]]

    mode_note = "FORECAST (model joblib + proyeksi menuju prediksi akhir)"

# ---- Jawaban utama (sesuai dosen) ----
st.subheader(f"📌 Posisi Tim pada Matchday {matchday_query} — {mode_note}")

teams_available = table_q["Team"].dropna().astype(str).unique().tolist()
res = resolve_team_name(team_query, teams_available)

if not res["found"]:
    st.error(f"Tim '{team_query}' tidak ditemukan.")
else:
    team_resolved = res["team"]
    row = table_q.loc[table_q["Team"].map(_norm_team) == _norm_team(team_resolved)]

    if row.empty:
        st.error(f"Tim '{team_resolved}' tidak ada di tabel matchday {matchday_query}.")
    else:
        r = row.iloc[0]
        if team_query.strip().lower() != team_resolved.strip().lower():
            st.info(f"Input '{team_query}' dicocokkan menjadi **{team_resolved}**")
        st.success(f"✅ **{r['Team']}** di matchday **{matchday_query}** berada di peringkat **#{int(r['Rank'])}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Points", f"{float(r['Pts']):.1f}")
        m2.metric("Goal Difference", f"{float(r['GD']):.1f}")
        m3.metric("Mode", mode_note)

# ---- Top 10 ----
st.markdown("### Top 10 Klasemen")
st.dataframe(table_q.head(10), use_container_width=True)

# ---- Timeline Rank ----
st.markdown("### 📈 Timeline Rank (1..TOTAL_MATCHES)")
if res["found"]:
    team_norm = _norm_team(res["team"])
    timeline = []

    # Precompute forecast objects only if needed
    can_forecast = all(p.exists() for p in [MODEL_POINTS_PATH, MODEL_GD_PATH, SCALER_PATH, ENCODER_PATH])

    if can_forecast:
        model_points = joblib.load(MODEL_POINTS_PATH)
        model_gd = joblib.load(MODEL_GD_PATH)
        scaler = joblib.load(SCALER_PATH)
        le_team = joblib.load(ENCODER_PATH)

        current_table_md = standings_from_matches(matches_std, matches_played)
        cur_tbl = current_table_md[["Team", "Pts", "GD"]].copy()

        current_feat = build_team_features_from_matches(matches_std, matches_played)
        predicted_df = predict_final_table_from_current_features(current_feat, model_points, model_gd, scaler, le_team)
    else:
        predicted_df = None
        cur_tbl = None

    for md in range(1, int(total_matches) + 1):
        if md <= matches_played:
            tmd = standings_from_matches(matches_std, md)[["Team", "Pts", "GD", "Rank"]]
        else:
            if not can_forecast:
                break
            tmd = build_forecast_table_for_matchday(
                matchday=md,
                current_md=matches_played,
                current_table=cur_tbl,
                final_pred=predicted_df,
                total_matches=int(total_matches),
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
        st.warning("Timeline kosong (tim tidak ditemukan).")
    else:
        st.dataframe(timeline_df, use_container_width=True)

        fig = plt.figure(figsize=(12, 4))
        plt.plot(timeline_df["Matchday"], timeline_df["Rank"], marker="o")
        plt.gca().invert_yaxis()
        plt.title(f"Timeline Rank: {timeline_df.iloc[0]['Team']}")
        plt.xlabel("Matchday")
        plt.ylabel("Rank (1 = terbaik)")
        plt.grid(True)
        st.pyplot(fig)
