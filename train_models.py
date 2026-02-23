
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from laliga_forecaster import (
    map_matches_schema, 
    _norm_team, 
    get_final_season_targets, 
    build_team_features_from_matches, 
    LaLigaForecaster
)

# ==============================================================================
# Main Execution (Training & Saving)
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = DATA_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    CSV_PATH = DATA_DIR / "matches_laliga_2025_2026 (1).csv"
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df_raw = pd.read_csv(CSV_PATH)
    df_std = map_matches_schema(df_raw)

    # 1. Prepare Targets (Final Season Results)
    final_targets = get_final_season_targets(df_std)
    final_targets["Team_Norm"] = final_targets["Team"].map(_norm_team)

    # 2. Prepare Training Data 
    X_list = []
    y_points_list = []
    y_gd_list = []

    all_teams = final_targets["Team"].unique()
    le_team = LabelEncoder()
    le_team.fit([t.strip() for t in all_teams])

    sample_matchdays = range(10, 35, 2)
    for md in sample_matchdays:
        feats = build_team_features_from_matches(df_std, md)
        
        # Encode Teams
        feats["Team_Encoded"] = -1
        for idx, row in feats.iterrows():
            t_str = str(row["Team"]).strip()
            if t_str in le_team.classes_:
                feats.loc[idx, "Team_Encoded"] = le_team.transform([t_str])[0]
                
        # Merge with targets
        feats["Team_Norm"] = feats["Team"].map(_norm_team)
        merged = feats.merge(final_targets[["Team_Norm", "Target_Points_Final", "Target_GD_Final"]], on="Team_Norm", how="left")
        
        feature_cols = [
            "Win_Rate", "Draw_Rate", "Loss_Rate",
            "Points_Per_Match",
            "Goals_Scored_Per_Match", "Goals_Conceded_Per_Match",
            "Goal_Diff_Per_Match",
            "Team_Encoded"
        ]
        
        # Filter out rows with missing targets (if any)
        merged = merged.dropna(subset=["Target_Points_Final", "Target_GD_Final"])
        
        if merged.empty:
            continue

        X_part = merged[feature_cols].values
        y_p_part = merged["Target_Points_Final"].values
        y_g_part = merged["Target_GD_Final"].values
        
        X_list.append(X_part)
        y_points_list.append(y_p_part)
        y_gd_list.append(y_g_part)

    X_train = np.vstack(X_list)
    y_train_points = np.concatenate(y_points_list)
    y_train_gd = np.concatenate(y_gd_list)

    # 3. Train Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model_points = LinearRegression()
    model_points.fit(X_train_scaled, y_train_points)

    model_gd = LinearRegression()
    model_gd.fit(X_train_scaled, y_train_gd)

    # 4. Save Artifact (The Forecaster Object)
    forecaster = LaLigaForecaster(
        model_points=model_points,
        model_gd=model_gd,
        scaler=scaler,
        le_team=le_team,
        matches_df=df_std
    )
    
    # Save the main artifact
    output_path = MODELS_DIR / "laliga_forecast_model.joblib"
    joblib.dump(forecaster, output_path)

    # Optional: Save individual parts if needed for debugging
    joblib.dump(model_points, MODELS_DIR / "model_points_laliga.joblib")
    joblib.dump(model_gd, MODELS_DIR / "model_goal_diff_laliga.joblib")
    
    print(f"Success! Artifact saved to: {output_path}")
    print("Interface check:")
    print(f"- Has predict_rank: {hasattr(forecaster, 'predict_rank')}")
    print(f"- Has predict_table: {hasattr(forecaster, 'predict_table')}")
