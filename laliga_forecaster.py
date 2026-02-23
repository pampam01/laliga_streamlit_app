
import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Optional, List

# ==============================================================================
# Helper Functions
# ==============================================================================

def _norm_team(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

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
    df = standardize_columns(df)
    cols = list(df.columns)
    col_md   = _pick_exact(cols, ["Matchday", "Pekan", "Round", "Week"])
    col_home = _pick_exact(cols, ["Home", "Tuan_Rumah", "Home_Team", "Tim_Rumah"])
    col_away = _pick_exact(cols, ["Away", "Tim_Tamu", "Away_Team", "Tim_Tandang"])
    col_hg   = _pick_exact(cols, ["HG", "Gol_Tuan_Rumah", "Home_Goals", "FT_HG"])
    col_ag   = _pick_exact(cols, ["AG", "Gol_Tim_Tamu", "Away_Goals", "FT_AG"])
    
    out = pd.DataFrame(index=df.index)
    if col_md: out["Matchday"] = pd.to_numeric(df[col_md], errors="coerce")
    if col_home: out["Home"] = df[col_home].astype(str).str.strip()
    if col_away: out["Away"] = df[col_away].astype(str).str.strip()
    if col_hg: out["HG"] = pd.to_numeric(df[col_hg], errors="coerce")
    if col_ag: out["AG"] = pd.to_numeric(df[col_ag], errors="coerce")
    
    out = out.dropna(subset=["Home", "Away"])
    out["Matchday"] = out["Matchday"].fillna(1).astype(int)
    return out

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

def get_final_season_targets(matches_std: pd.DataFrame) -> pd.DataFrame:
    final_md = matches_std["Matchday"].max()
    final_ft = build_team_features_from_matches(matches_std, final_md)
    return final_ft[["Team", "Point", "Selisih_Goal"]].rename(columns={
        "Point": "Target_Points_Final",
        "Selisih_Goal": "Target_GD_Final"
    })

# ==============================================================================
# LaLigaForecaster Class
# ==============================================================================

class LaLigaForecaster:
    def __init__(self, model_points, model_gd, scaler, le_team, matches_df):
        self.model_points = model_points
        self.model_gd = model_gd
        self.scaler = scaler
        self.le_team = le_team
        self.matches_df = matches_df
        
        # Expose team list for the app
        self.teams = sorted(list(self.matches_df["Home"].unique()))

    def predict_table(self, matchday: int) -> pd.DataFrame:
        """
        Prediksi klasemen akhir berdasarkan data hingga matchday tertentu.
        Return DataFrame dengan kolom: Team, Rank, Pts, GD
        """
        # 1. Build features at current matchday
        feats = build_team_features_from_matches(self.matches_df, matchday)
        
        # 2. Encode Teams
        # Use classes_ to ensure safe lookup
        # We create a mapping first
        team_map = {t: i for i, t in enumerate(self.le_team.classes_)}
        
        def get_team_code(t):
            ts = str(t).strip()
            return team_map.get(ts, -1)
            
        feats["Team_Encoded"] = feats["Team"].apply(get_team_code)
        
        # 3. Prepare X
        feature_cols = [
            "Win_Rate", "Draw_Rate", "Loss_Rate",
            "Points_Per_Match",
            "Goals_Scored_Per_Match", "Goals_Conceded_Per_Match",
            "Goal_Diff_Per_Match",
            "Team_Encoded"
        ]
        
        X = feats[feature_cols].values
        
        # 4. Predict
        X_scaled = self.scaler.transform(X)
        pred_points = self.model_points.predict(X_scaled)
        pred_gd = self.model_gd.predict(X_scaled)
        
        feats["Pts"] = pred_points
        feats["GD"] = pred_gd
        
        # 5. Ranking
        # Sort by Pts (desc), then GD (desc)
        feats = feats.sort_values(by=["Pts", "GD"], ascending=[False, False])
        feats = feats.reset_index(drop=True)
        feats.index += 1
        feats["Rank"] = feats.index
        
        return feats[["Team", "Rank", "Pts", "GD"]]

    def predict_rank(self, club: str, matchday: int):
        """
        Return (rank, meta_dict)
        """
        df = self.predict_table(matchday)
        
        # Normalize for search
        df["_team_norm"] = df["Team"].map(_norm_team)
        target = _norm_team(club)
        
        row = df[df["_team_norm"] == target]
        if row.empty:
            return 0, {}
        
        rec = row.iloc[0]
        rank = int(rec["Rank"])
        meta = {
            "Pts": float(rec["Pts"]),
            "GD": float(rec["GD"])
        }
        return rank, meta
