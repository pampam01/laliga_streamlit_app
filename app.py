from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIG
# ==========================================================
APP_TITLE = "Testing Model Prediksi LaLiga"
APP_SUBTITLE = (
    "Streamlit untuk uji model joblib: model_points_laliga.joblib dan model_goal_diff_laliga.joblib.\n"
    "Sudah dilengkapi tombol demo supaya dosen penguji tinggal klik."
)

DEFAULT_TOTAL_MATCHES = 38
DEFAULT_MATCHES_PLAYED = 15

# Fitur yang dipakai pada notebook Anda
DEFAULT_FEATURES: List[str] = [
    "Win_Rate",
    "Draw_Rate",
    "Loss_Rate",
    "Points_Per_Match",
    "Goals_Scored_Per_Match",
    "Goals_Conceded_Per_Match",
    "Goal_Diff_Per_Match",
    "Team_Encoded",
]

MODEL_POINTS_FILENAME = "model_points_laliga.joblib"
MODEL_GOAL_DIFF_FILENAME = "model_goal_diff_laliga.joblib"
PREPROCESS_BUNDLE_FILENAME = "preprocessing_laliga.joblib"  # optional (recommended)


# ==========================================================
# HELPERS: LOAD FILES
# ==========================================================

def _possible_model_dirs(base_dir: Path) -> List[Path]:
    return [
        base_dir / "models",
        base_dir,
    ]


def resolve_existing_path(base_dir: Path, filename: str) -> Optional[Path]:
    for d in _possible_model_dirs(base_dir):
        p = d / filename
        if p.exists():
            return p
    return None


def read_table_from_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV/Excel from a Streamlit UploadedFile."""
    if uploaded_file is None:
        raise ValueError("File belum dipilih")

    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")

    # CSV
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8")
    except Exception:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


# ==========================================================
# PREPROCESSING (mengikuti notebook)
# ==========================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "", regex=False)
    )
    return df


def map_to_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Petakan kolom dataset ke schema standar yang dipakai model."""
    df = standardize_columns(df)

    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        c = str(col).lower()
        if any(x in c for x in ["team", "tim", "club"]):
            out["Team"] = df[col].astype(str).str.strip().values
        elif any(x in c for x in ["menang", "win"]):
            out["Menang"] = df[col].values
        elif any(x in c for x in ["seri", "draw"]):
            out["Seri"] = df[col].values
        elif any(x in c for x in ["kalah", "loss"]):
            out["Kalah"] = df[col].values
        elif any(x in c for x in ["goal_masuk", "goals_for", "gf", "gol_masuk"]):
            out["Goal_Masuk"] = df[col].values
        elif any(
            x in c
            for x in [
                "goal_kebobolan",
                "goals_against",
                "ga",
                "gol_kebobolan",
            ]
        ):
            out["Goal_Kebobolan"] = df[col].values
        elif any(x in c for x in ["point", "points", "pts"]):
            out["Point"] = df[col].values

    # Pastikan kolom minimal ada
    for k in ["Menang", "Seri", "Kalah"]:
        if k not in out.columns:
            out[k] = 0

    if "Point" not in out.columns:
        out["Point"] = out["Menang"] * 3 + out["Seri"]

    if "Goal_Masuk" not in out.columns:
        out["Goal_Masuk"] = out["Menang"] * 2 + out["Seri"] * 1 + out["Kalah"] * 0.5

    if "Goal_Kebobolan" not in out.columns:
        out["Goal_Kebobolan"] = out["Menang"] * 0.8 + out["Seri"] * 1.2 + out["Kalah"] * 2

    if "Team" not in out.columns:
        out["Team"] = [f"Team_{i}" for i in range(len(out))]

    # Force numeric types where possible
    for k in ["Menang", "Seri", "Kalah", "Goal_Masuk", "Goal_Kebobolan", "Point"]:
        out[k] = pd.to_numeric(out[k], errors="coerce").fillna(0)

    return out


def add_features(df: pd.DataFrame, matches_played: int, total_matches: int) -> pd.DataFrame:
    df = df.copy()
    
    # Calculate matches played based on W+D+L if available (more accurate for mixed tables)
    # If W+D+L are 0 (e.g. data only has Points), fallback to the scalar argument `matches_played`
    calculated_mp = df["Menang"] + df["Seri"] + df["Kalah"]
    df["Match_Played"] = np.where(calculated_mp > 0, calculated_mp, matches_played)
    
    df["Total_Match"] = total_matches
    df["Selisih_Goal"] = df["Goal_Masuk"] - df["Goal_Kebobolan"]

    # Use the per-row Match_Played for rate calculations
    mp = df["Match_Played"].replace(0, 1).astype(float)
    
    df["Win_Rate"] = df["Menang"] / mp
    df["Draw_Rate"] = df["Seri"] / mp
    df["Loss_Rate"] = df["Kalah"] / mp
    df["Points_Per_Match"] = df["Point"] / mp
    df["Goals_Scored_Per_Match"] = df["Goal_Masuk"] / mp
    df["Goals_Conceded_Per_Match"] = df["Goal_Kebobolan"] / mp
    df["Goal_Diff_Per_Match"] = df["Selisih_Goal"] / mp

    return df


def encode_teams_with_existing_encoder(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    """Encode Team memakai LabelEncoder yang sudah ada.

    Jika ada team baru yang tidak ada di encoder, diberi kode baru berurutan
    (meniru logic notebook Anda).
    """
    df = df.copy()
    df["Team_Clean"] = df["Team"].astype(str).str.strip()

    class_map = {cls: i for i, cls in enumerate(le.classes_)}
    next_code = len(le.classes_)

    codes = []
    for t in df["Team_Clean"].tolist():
        if t in class_map:
            codes.append(int(class_map[t]))
        else:
            codes.append(int(next_code))
            next_code += 1

    df["Team_Encoded"] = codes
    return df


def build_preprocessing_from_historical(
    historical_raw: pd.DataFrame,
    *,
    total_matches: int,
    features: List[str],
) -> Tuple[StandardScaler, LabelEncoder, List[str], pd.DataFrame]:
    """Bangun scaler + label encoder dari dataset historis.

    Catatan:
    - Ini paling mendekati proses training (notebook) jika historis yang dipakai sama.
    """
    hist = map_to_standard_schema(historical_raw)
    hist = add_features(hist, matches_played=total_matches, total_matches=total_matches)

    le = LabelEncoder()
    hist["Team_Clean"] = hist["Team"].astype(str).str.strip()
    hist["Team_Encoded"] = le.fit_transform(hist["Team_Clean"])

    # Pastikan fitur ada
    for f in features:
        if f not in hist.columns:
            hist[f] = 0

    X = hist[features].fillna(0)
    scaler = StandardScaler().fit(X)

    return scaler, le, features, hist


# ==========================================================
# DEMO DATA (agar penguji tinggal klik)
# ==========================================================

def demo_current_table() -> pd.DataFrame:
    """Contoh dataset 'musim ini' (15 match) hanya untuk demo UI.

    Anda boleh ganti angka-angkanya agar sesuai dataset asli.
    """
    rows = [
        {"Team": "Real Madrid", "Menang": 12, "Seri": 2, "Kalah": 1, "Goal_Masuk": 32, "Goal_Kebobolan": 12},
        {"Team": "Barcelona", "Menang": 11, "Seri": 2, "Kalah": 2, "Goal_Masuk": 34, "Goal_Kebobolan": 16},
        {"Team": "Atletico Madrid", "Menang": 10, "Seri": 3, "Kalah": 2, "Goal_Masuk": 28, "Goal_Kebobolan": 13},
        {"Team": "Girona", "Menang": 9, "Seri": 4, "Kalah": 2, "Goal_Masuk": 29, "Goal_Kebobolan": 18},
        {"Team": "Real Sociedad", "Menang": 8, "Seri": 4, "Kalah": 3, "Goal_Masuk": 22, "Goal_Kebobolan": 15},
        {"Team": "Athletic Club", "Menang": 8, "Seri": 3, "Kalah": 4, "Goal_Masuk": 24, "Goal_Kebobolan": 19},
        {"Team": "Real Betis", "Menang": 7, "Seri": 5, "Kalah": 3, "Goal_Masuk": 20, "Goal_Kebobolan": 16},
        {"Team": "Villarreal", "Menang": 7, "Seri": 4, "Kalah": 4, "Goal_Masuk": 25, "Goal_Kebobolan": 22},
        {"Team": "Valencia", "Menang": 6, "Seri": 5, "Kalah": 4, "Goal_Masuk": 18, "Goal_Kebobolan": 16},
        {"Team": "Sevilla", "Menang": 5, "Seri": 6, "Kalah": 4, "Goal_Masuk": 16, "Goal_Kebobolan": 18},
        {"Team": "Getafe", "Menang": 5, "Seri": 5, "Kalah": 5, "Goal_Masuk": 14, "Goal_Kebobolan": 15},
        {"Team": "Osasuna", "Menang": 5, "Seri": 4, "Kalah": 6, "Goal_Masuk": 15, "Goal_Kebobolan": 19},
        {"Team": "Celta Vigo", "Menang": 4, "Seri": 6, "Kalah": 5, "Goal_Masuk": 16, "Goal_Kebobolan": 20},
        {"Team": "Mallorca", "Menang": 4, "Seri": 5, "Kalah": 6, "Goal_Masuk": 13, "Goal_Kebobolan": 18},
        {"Team": "Rayo Vallecano", "Menang": 4, "Seri": 4, "Kalah": 7, "Goal_Masuk": 12, "Goal_Kebobolan": 19},
        {"Team": "Alaves", "Menang": 4, "Seri": 3, "Kalah": 8, "Goal_Masuk": 12, "Goal_Kebobolan": 22},
        {"Team": "Las Palmas", "Menang": 3, "Seri": 5, "Kalah": 7, "Goal_Masuk": 11, "Goal_Kebobolan": 21},
        {"Team": "Granada", "Menang": 3, "Seri": 4, "Kalah": 8, "Goal_Masuk": 14, "Goal_Kebobolan": 26},
        {"Team": "Cadiz", "Menang": 2, "Seri": 5, "Kalah": 8, "Goal_Masuk": 10, "Goal_Kebobolan": 22},
        {"Team": "Almeria", "Menang": 1, "Seri": 4, "Kalah": 10, "Goal_Masuk": 11, "Goal_Kebobolan": 28},
    ]
    df = pd.DataFrame(rows)
    df["Point"] = df["Menang"] * 3 + df["Seri"]
    return df


def synthetic_historical_table(team_names: List[str], seasons: int = 3, seed: int = 42) -> pd.DataFrame:
    """Dataset historis sintetik (full season 38 match) untuk fallback demo.

    Kalau Anda punya dataset historis asli, lebih baik upload dan gunakan itu.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for season in range(1, seasons + 1):
        for t in team_names:
            # Sample wins and draws, then compute losses to make total 38
            w = int(rng.integers(5, 28))
            d = int(rng.integers(0, 16))
            if w + d > 38:
                d = 38 - w
            l = 38 - w - d

            gf = int(rng.integers(25, 90))
            ga = int(rng.integers(20, 85))
            pts = 3 * w + d

            rows.append(
                {
                    "Season": f"S{season}",
                    "Team": t,
                    "Menang": w,
                    "Seri": d,
                    "Kalah": l,
                    "Goal_Masuk": gf,
                    "Goal_Kebobolan": ga,
                    "Point": pts,
                }
            )

    return pd.DataFrame(rows)


# ==========================================================
# PREDICTION
# ==========================================================

@dataclass
class PredictArtifacts:
    model_points: Any
    model_goal_diff: Any
    scaler: Optional[StandardScaler]
    label_encoder: Optional[LabelEncoder]
    features: List[str]
    mode: str  # just for display


def _ensure_feature_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = df.copy()
    for f in features:
        if f not in df.columns:
            df[f] = 0
    return df


def predict_standings(
    current_raw: pd.DataFrame,
    artifacts: PredictArtifacts,
    *,
    matches_played: int,
    total_matches: int,
) -> pd.DataFrame:
    current = map_to_standard_schema(current_raw)
    current = add_features(current, matches_played=matches_played, total_matches=total_matches)

    # Encode teams
    if artifacts.label_encoder is not None:
        current = encode_teams_with_existing_encoder(current, artifacts.label_encoder)
    else:
        # fallback: fit encoder on current teams only
        le = LabelEncoder()
        current["Team_Clean"] = current["Team"].astype(str).str.strip()
        current["Team_Encoded"] = le.fit_transform(current["Team_Clean"])

    current = _ensure_feature_columns(current, artifacts.features)

    X_current = current[artifacts.features].fillna(0)

    # Scale
    X_current_scaled = X_current
    if artifacts.scaler is not None:
        try:
            X_current_scaled = artifacts.scaler.transform(X_current)
        except Exception:
            # last resort: no scaling
            X_current_scaled = X_current.values
    else:
        X_current_scaled = X_current.values

    # Predict totals for full season
    pred_total_points = np.asarray(artifacts.model_points.predict(X_current_scaled), dtype=float)
    pred_total_gd = np.asarray(artifacts.model_goal_diff.predict(X_current_scaled), dtype=float)

    # Use per-row matches played for accurate remaining matches
    current_mp = current["Match_Played"].values
    remaining_matches = np.maximum(total_matches - current_mp, 0)

    pred_ppm = pred_total_points / float(total_matches)
    pred_gdpm = pred_total_gd / float(total_matches)

    rows = []
    for i in range(len(current)):
        team = str(current.iloc[i]["Team"])
        current_points = float(current.iloc[i]["Point"])
        current_gd = float(current.iloc[i]["Selisih_Goal"])

        rem = float(remaining_matches[i])
        projected_points = float(pred_ppm[i]) * rem
        projected_gd = float(pred_gdpm[i]) * rem

        total_points = current_points + projected_points
        total_gd = current_gd + projected_gd

        current_wins = float(current.iloc[i]["Menang"])
        # Estimate wins from points (approx 2.3 pts/win usually, but simplified here)
        projected_wins = (float(pred_ppm[i]) / 2.3) * rem

        rows.append(
            {
                "Team": team,
                "Current_Points": round(current_points, 1),
                "Current_GD": round(current_gd, 1),
                "Projected_Rest_Points": round(projected_points, 1),
                "Projected_Rest_GD": round(projected_gd, 1),
                "Total_Points": round(total_points, 1),
                "Total_GD": round(total_gd, 1),
                "Projected_Wins": round(current_wins + projected_wins, 1),
                "PPM_Current": round(current_points / float(max(current_mp[i], 1)), 2),
                "PPM_Projected": round(float(pred_ppm[i]), 2),
            }
        )

    predicted_df = pd.DataFrame(rows)
    predicted_df = predicted_df.sort_values(["Total_Points", "Total_GD"], ascending=[False, False]).reset_index(drop=True)
    predicted_df.insert(0, "Rank", np.arange(1, len(predicted_df) + 1))

    # Probabilitas juara (softmax sederhana dari Total Points)
    pts = predicted_df["Total_Points"].to_numpy(dtype=float)
    exp_pts = np.exp(pts - np.max(pts))
    predicted_df["Champion_%"] = np.round(exp_pts / exp_pts.sum() * 100, 1)

    return predicted_df


# ==========================================================
# LOAD MODELS + ARTIFACTS (cache)
# ==========================================================

@st.cache_resource
def load_models_and_artifacts(app_dir: str, use_preprocess_bundle: bool = True) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    base_dir = Path(app_dir)

    # Load models
    p_points = resolve_existing_path(base_dir, MODEL_POINTS_FILENAME)
    p_gd = resolve_existing_path(base_dir, MODEL_GOAL_DIFF_FILENAME)

    if p_points is None or p_gd is None:
        raise FileNotFoundError(
            "Model file tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py atau di ./models/.\n"
            f"- {MODEL_POINTS_FILENAME}\n"
            f"- {MODEL_GOAL_DIFF_FILENAME}"
        )

    model_points = joblib.load(p_points)
    model_goal_diff = joblib.load(p_gd)

    preprocess_bundle: Optional[Dict[str, Any]] = None
    if use_preprocess_bundle:
        p_bundle = resolve_existing_path(base_dir, PREPROCESS_BUNDLE_FILENAME)
        if p_bundle is not None:
            preprocess_bundle = joblib.load(p_bundle)

    return model_points, model_goal_diff, preprocess_bundle


def make_artifacts_from_bundle(model_points: Any, model_goal_diff: Any, bundle: Dict[str, Any]) -> PredictArtifacts:
    scaler = bundle.get("scaler")
    le = bundle.get("label_encoder_team") or bundle.get("label_encoder")
    features = bundle.get("features") or DEFAULT_FEATURES

    # Basic validation
    if not isinstance(features, list) or len(features) == 0:
        features = DEFAULT_FEATURES

    return PredictArtifacts(
        model_points=model_points,
        model_goal_diff=model_goal_diff,
        scaler=scaler,
        label_encoder=le,
        features=features,
        mode="bundle (preprocessing_laliga.joblib)",
    )


def make_artifacts_fallback(model_points: Any, model_goal_diff: Any, *, mode: str, scaler: Optional[StandardScaler], le: Optional[LabelEncoder]) -> PredictArtifacts:
    return PredictArtifacts(
        model_points=model_points,
        model_goal_diff=model_goal_diff,
        scaler=scaler,
        label_encoder=le,
        features=DEFAULT_FEATURES,
        mode=mode,
    )


# ==========================================================
# STREAMLIT UI
# ==========================================================


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    app_dir = str(Path(__file__).parent.resolve())

    with st.sidebar:
        st.header("Pengaturan")

        total_matches = st.number_input(
            "Total match 1 musim",
            min_value=1,
            max_value=100,
            value=DEFAULT_TOTAL_MATCHES,
            step=1,
        )
        matches_played = st.number_input(
            "Match sudah dimainkan (current)",
            min_value=0,
            max_value=int(total_matches),
            value=min(DEFAULT_MATCHES_PLAYED, int(total_matches)),
            step=1,
        )

        use_bundle = st.toggle(
            "Pakai preprocessing bundle (recommended)",
            value=True,
            help=(
                "Jika Anda menyimpan preprocessing_laliga.joblib (isi scaler + label encoder + features), "
                "aktifkan ini supaya hasil sesuai training."
            ),
        )

        st.divider()
        st.subheader("Lokasi file")
        st.write("Folder app:")
        st.code(app_dir)
        st.write("Model dicari di:")
        st.code(str(Path(app_dir) / "models"))

    # Load models
    try:
        model_points, model_goal_diff, bundle = load_models_and_artifacts(app_dir, use_preprocess_bundle=use_bundle)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build artifacts
    artifacts: Optional[PredictArtifacts] = None
    if bundle is not None:
        artifacts = make_artifacts_from_bundle(model_points, model_goal_diff, bundle)

    tab_demo, tab_upload, tab_manual, tab_export = st.tabs(
        ["✅ Demo (1-klik)", "📁 Upload Dataset", "🧮 Input Manual 1 Tim", "🧾 Export Preprocessing Bundle"]
    )

    with tab_demo:
        st.subheader("Demo cepat")
        st.write(
            "Klik tombol di bawah untuk menjalankan prediksi memakai data contoh. "
            "Ini dibuat supaya dosen penguji bisa langsung melihat output tanpa input apa-apa."
        )

        demo_df = demo_current_table()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Data contoh (bisa diedit kalau mau):**")
            try:
                edited_df = st.data_editor(demo_df, use_container_width=True, num_rows="dynamic")
            except Exception:
                st.dataframe(demo_df, use_container_width=True)
                edited_df = demo_df

        with col2:
            st.write("**Aksi**")

            build_demo_preprocess = st.toggle(
                "Jika bundle tidak ada, buat preprocessing dari data historis sintetik",
                value=True,
                help=(
                    "Kalau preprocessing_laliga.joblib tidak tersedia, aplikasi butuh scaler/encoder. "
                    "Mode ini membuatnya otomatis dari data historis sintetik (demo)."
                ),
            )

            if artifacts is None:
                if build_demo_preprocess:
                    teams = map_to_standard_schema(edited_df)["Team"].astype(str).str.strip().tolist()
                    hist_syn = synthetic_historical_table(teams, seasons=3, seed=42)
                    scaler, le, feats, _hist_processed = build_preprocessing_from_historical(
                        hist_syn,
                        total_matches=int(total_matches),
                        features=DEFAULT_FEATURES,
                    )
                    artifacts = make_artifacts_fallback(
                        model_points,
                        model_goal_diff,
                        mode="demo-preprocess (synthetic historical)",
                        scaler=scaler,
                        le=le,
                    )
                else:
                    artifacts = make_artifacts_fallback(
                        model_points,
                        model_goal_diff,
                        mode="fallback (tanpa scaler/encoder)",
                        scaler=None,
                        le=None,
                    )

            st.info(f"Mode preprocessing: **{artifacts.mode}**")

            run_demo = st.button("▶️ Jalankan Prediksi Demo", type="primary", use_container_width=True)

        if run_demo:
            try:
                out = predict_standings(
                    edited_df,
                    artifacts,
                    matches_played=int(matches_played),
                    total_matches=int(total_matches),
                )
                st.success("Prediksi selesai ✅")

                st.write("### Hasil Prediksi Klasemen")
                st.dataframe(out, use_container_width=True)

                # Champion probability chart
                st.write("### Probabilitas Juara (berdasarkan softmax Total_Points)")
                chart_df = out[["Team", "Champion_%"]].set_index("Team")
                st.bar_chart(chart_df)

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download hasil (CSV)",
                    data=csv,
                    file_name="prediksi_kalsemen_laliga.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")

        st.divider()
        st.write("### Download template input")
        st.write("Kalau mau upload dataset sendiri, gunakan format kolom seperti di bawah.")

        template = pd.DataFrame(
            [
                {
                    "Team": "Contoh FC",
                    "Menang": 0,
                    "Seri": 0,
                    "Kalah": 0,
                    "Goal_Masuk": 0,
                    "Goal_Kebobolan": 0,
                    "Point": 0,
                }
            ]
        )
        st.download_button(
            "⬇️ Download template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="template_current_laliga.csv",
            mime="text/csv",
        )

    with tab_upload:
        st.subheader("Upload dataset (current + opsional historical)")
        st.write(
            "- **Current**: klasemen sementara (kolom minimal: Team, Menang, Seri, Kalah, Goal_Masuk, Goal_Kebobolan, Point)\n"
            "- **Historical** (opsional tapi recommended): untuk membangun scaler + label encoder jika preprocessing bundle tidak ada."
        )

        colu1, colu2 = st.columns(2)
        with colu1:
            current_file = st.file_uploader("Upload CURRENT (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="cur")
        with colu2:
            hist_file = st.file_uploader("Upload HISTORICAL (CSV/XLSX) - opsional", type=["csv", "xlsx", "xls"], key="hist")

        if current_file is not None:
            try:
                cur_df_raw = read_table_from_upload(current_file)
                st.write("**Preview current:**")
                st.dataframe(cur_df_raw.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Gagal baca current: {e}")
                st.stop()

            # Decide preprocessing
            if artifacts is None:
                if hist_file is not None:
                    try:
                        hist_df_raw = read_table_from_upload(hist_file)
                        scaler, le, feats, _hist_processed = build_preprocessing_from_historical(
                            hist_df_raw,
                            total_matches=int(total_matches),
                            features=DEFAULT_FEATURES,
                        )
                        artifacts = make_artifacts_fallback(
                            model_points,
                            model_goal_diff,
                            mode="built-from-uploaded-historical",
                            scaler=scaler,
                            le=le,
                        )
                        st.success("Preprocessing berhasil dibuat dari HISTORICAL yang Anda upload ✅")
                    except Exception as e:
                        st.warning(f"Gagal membuat preprocessing dari historical: {e}")
                        artifacts = make_artifacts_fallback(
                            model_points,
                            model_goal_diff,
                            mode="fallback (tanpa scaler/encoder)",
                            scaler=None,
                            le=None,
                        )
                else:
                    st.warning(
                        "Preprocessing bundle tidak ditemukan dan Anda tidak upload historical. "
                        "Aplikasi akan jalan dengan fallback (hasil bisa berbeda dari skripsi)."
                    )
                    artifacts = make_artifacts_fallback(
                        model_points,
                        model_goal_diff,
                        mode="fallback (tanpa scaler/encoder)",
                        scaler=None,
                        le=None,
                    )

            st.info(f"Mode preprocessing: **{artifacts.mode}**")

            if st.button("▶️ Jalankan Prediksi (Upload)", type="primary"):
                try:
                    out = predict_standings(
                        cur_df_raw,
                        artifacts,
                        matches_played=int(matches_played),
                        total_matches=int(total_matches),
                    )
                    st.success("Prediksi selesai ✅")
                    st.dataframe(out, use_container_width=True)

                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download hasil (CSV)",
                        data=csv,
                        file_name="prediksi_kalsemen_laliga_upload.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")

    with tab_manual:
        st.subheader("Input manual 1 tim")
        st.write("Cocok untuk demo cepat: isi angka, lalu klik prediksi.")

        example_teams = [
            "Real Madrid",
            "Barcelona",
            "Atletico Madrid",
            "Sevilla",
            "Valencia",
            "Villarreal",
            "Real Sociedad",
            "Athletic Club",
        ]

        with st.form("form_manual"):
            team = st.selectbox("Team", options=example_teams, index=0)

            c1, c2, c3 = st.columns(3)
            with c1:
                menang = st.number_input("Menang", min_value=0, max_value=38, value=10, step=1)
                goals_for = st.number_input("Goal Masuk", min_value=0, max_value=200, value=25, step=1)
            with c2:
                seri = st.number_input("Seri", min_value=0, max_value=38, value=3, step=1)
                goals_against = st.number_input("Goal Kebobolan", min_value=0, max_value=200, value=15, step=1)
            with c3:
                kalah = st.number_input("Kalah", min_value=0, max_value=38, value=2, step=1)
                points = st.number_input(
                    "Point (opsional, otomatis jika 0)",
                    min_value=0,
                    max_value=200,
                    value=0,
                    step=1,
                )

            submit = st.form_submit_button("▶️ Prediksi 1 Tim", type="primary")

        if submit:
            mp = int(menang + seri + kalah)
            if mp == 0:
                mp = int(matches_played)

            pts = int(points) if int(points) > 0 else int(menang) * 3 + int(seri)

            one = pd.DataFrame(
                [
                    {
                        "Team": team,
                        "Menang": int(menang),
                        "Seri": int(seri),
                        "Kalah": int(kalah),
                        "Goal_Masuk": int(goals_for),
                        "Goal_Kebobolan": int(goals_against),
                        "Point": int(pts),
                    }
                ]
            )

            # If artifacts still None, create demo fallback preprocessing from synthetic historical
            if artifacts is None:
                hist_syn = synthetic_historical_table([team], seasons=3, seed=42)
                scaler, le, feats, _ = build_preprocessing_from_historical(
                    hist_syn,
                    total_matches=int(total_matches),
                    features=DEFAULT_FEATURES,
                )
                artifacts = make_artifacts_fallback(
                    model_points,
                    model_goal_diff,
                    mode="demo-preprocess (synthetic historical)",
                    scaler=scaler,
                    le=le,
                )

            # Contextualize with Demo Data (other teams)
            # Supaya 'Champion %' masuk akal (tidak 100%), kita gabungkan dengan tim lain
            demo_ctx = demo_current_table()
            # Hapus tim yang sedang diinput dari demo context (bila ada)
            demo_ctx = demo_ctx[demo_ctx["Team"] != team]
            
            # Gabungkan input user + sisa liga
            combined = pd.concat([one, demo_ctx], ignore_index=True)

            st.info(f"Memprediksi **{team}** melawan {len(demo_ctx)} tim lain (data demo) agar persentase juara valid.")

            out = predict_standings(
                combined,
                artifacts,
                matches_played=mp, # will be overridden by per-row W/D/L in add_features, but strictly used if W/D/L=0
                total_matches=int(total_matches),
            )

            # Tampilkan hasil spesifik tim sendiri
            my_team_res = out[out["Team"] == team]
            
            st.write("### Hasil Prediksi Tim Anda")
            if not my_team_res.empty:
                st.dataframe(my_team_res, use_container_width=True)
            else:
                st.warning("Tim Anda tidak ditemukan di hasil prediksi.")

            st.write("### Klasemen Lengkap (Simulasi)")
            st.dataframe(out, use_container_width=True)

    with tab_export:
        st.subheader("Export preprocessing bundle (opsional tapi recommended)")
        st.write(
            "Agar hasil **konsisten** dengan training, sebaiknya Anda menyimpan scaler + label encoder.\n\n"
            "Caranya: upload dataset historis, lalu klik tombol **Generate & Save preprocessing_laliga.joblib**.\n"
            "Setelah itu, taruh file tersebut di folder yang sama dengan app.py atau di ./models/."
        )

        hist_file2 = st.file_uploader(
            "Upload HISTORICAL untuk membuat preprocessing bundle",
            type=["csv", "xlsx", "xls"],
            key="hist2",
        )

        if hist_file2 is not None:
            try:
                hist_df_raw2 = read_table_from_upload(hist_file2)
                st.write("**Preview historical:**")
                st.dataframe(hist_df_raw2.head(20), use_container_width=True)

                if st.button("🧩 Generate & Save preprocessing_laliga.joblib", type="primary"):
                    scaler, le, feats, hist_processed = build_preprocessing_from_historical(
                        hist_df_raw2,
                        total_matches=int(total_matches),
                        features=DEFAULT_FEATURES,
                    )

                    bundle_out = {
                        "scaler": scaler,
                        "label_encoder_team": le,
                        "features": feats,
                        "meta": {
                            "total_matches": int(total_matches),
                            "note": "Bundle dibuat dari Streamlit export tab.",
                        },
                    }

                    # Save to disk next to app
                    out_path = Path(app_dir) / PREPROCESS_BUNDLE_FILENAME
                    joblib.dump(bundle_out, out_path)

                    st.success(f"Bundle tersimpan: {out_path}")
                    st.info(
                        "Sekarang aktifkan toggle 'Pakai preprocessing bundle' di sidebar dan refresh app."
                    )

                    # Provide download
                    buf = io.BytesIO()
                    joblib.dump(bundle_out, buf)
                    st.download_button(
                        "⬇️ Download preprocessing_laliga.joblib",
                        data=buf.getvalue(),
                        file_name=PREPROCESS_BUNDLE_FILENAME,
                        mime="application/octet-stream",
                    )

            except Exception as e:
                st.error(f"Gagal membuat bundle: {e}")


if __name__ == "__main__":
    main()
