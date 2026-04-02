import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# =====================================================
# CONFIG / CONSTANTS
# =====================================================

U_COLS = [
    "encroachment.length.u0",
    "encroachment.length.u1",
    "encroachment.length.u2",
    "encroachment.length.u3",
    "encroachment.length.u4",
]

CLEARANCE_MAP = [2.0, 5.0, 7.0, 9.0, 11.0]
RISK_LEGEND_PATCHES = [
    mpatches.Patch(color="red", label="< 4 ft"),
    mpatches.Patch(color="orange", label="4–6 ft"),
    mpatches.Patch(color="yellow", label="4–6 ft"),
    mpatches.Patch(color="green", label="Healthy"),
    mpatches.Patch(color="lightgray", label="Safe"),
]


# =====================================================
# BASIC HELPERS
# =====================================================


def parse_year_from_filename(filepath: str) -> int:
    name = os.path.basename(filepath)
    match = re.search(r"(20\d{2})", name)
    if not match:
        raise ValueError(f"Could not parse year from filename: {name}")
    return int(match.group(1))



def compute_clearance(row: pd.Series) -> float:
    u0 = row[U_COLS[0]]
    if pd.isna(u0) or str(u0).strip().upper() == "NA":
        return np.nan

    for i, col in enumerate(U_COLS):
        if float(row[col]) > 0:
            return CLEARANCE_MAP[i]

    return 0.0



def pick_substation_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.strip().lower() == "substation"]
    if not candidates:
        raise ValueError("CSV must include a 'substation' column.")
    return candidates[0]



def to_projected_gdf(df: pd.DataFrame, lon_col: str, lat_col: str, epsg: int = 3395) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=epsg)


# =====================================================
# DATA LOADING / STANDARDIZATION
# =====================================================


def load_year_csv(
    filepath: str,
    year: Optional[int] = None,
    force_compute_clearance: bool = False,
) -> Tuple[int, pd.DataFrame]:
    if year is None:
        year = parse_year_from_filename(filepath)

    df = pd.read_csv(filepath)

    required_basic = ["Latitude", "Longitude"]
    for col in required_basic:
        if col not in df.columns:
            raise ValueError(f"{filepath} is missing required column: {col}")

    substation_col = pick_substation_column(df)
    has_u_cols = all(c in df.columns for c in U_COLS)

    if force_compute_clearance:
        if not has_u_cols:
            raise ValueError(
                f"{filepath} was marked to compute Clearance, but is missing one or more encroachment columns."
            )
        df["Clearance"] = df.apply(compute_clearance, axis=1)
    elif "Clearance" in df.columns:
        pass
    elif has_u_cols:
        df["Clearance"] = df.apply(compute_clearance, axis=1)
    else:
        raise ValueError(
            f"{filepath} has neither a usable 'Clearance' column nor all encroachment columns."
        )

    out = df[["Latitude", "Longitude", substation_col, "Clearance"]].copy()
    out = out.rename(
        columns={
            "Latitude": f"lat_{year}",
            "Longitude": f"lon_{year}",
            substation_col: f"substation_{year}",
            "Clearance": f"clearance_{year}",
        }
    )

    out[f"lat_{year}"] = pd.to_numeric(out[f"lat_{year}"], errors="coerce")
    out[f"lon_{year}"] = pd.to_numeric(out[f"lon_{year}"], errors="coerce")
    out[f"clearance_{year}"] = pd.to_numeric(out[f"clearance_{year}"], errors="coerce")
    out = out.dropna(subset=[f"lat_{year}", f"lon_{year}"])

    print(
        f"{year}: usable clearance rows = "
        f"{out[f'clearance_{year}'].notna().sum()} / {len(out)}"
    )

    return year, out



def load_all_years(
    filepaths: Iterable[str],
    compute_clearance_for: Optional[Iterable[str]] = None,
) -> Dict[int, pd.DataFrame]:
    compute_clearance_for = set(compute_clearance_for or [])
    year_buckets: Dict[int, List[pd.DataFrame]] = {}

    for fp in filepaths:
        force_compute = (fp in compute_clearance_for) or (os.path.basename(fp) in compute_clearance_for)
        year, df = load_year_csv(fp, force_compute_clearance=force_compute)

        if year not in year_buckets:
            year_buckets[year] = []
        year_buckets[year].append(df)

    if len(year_buckets) < 2:
        raise ValueError("At least two yearly CSV files are required.")

    year_data: Dict[int, pd.DataFrame] = {}
    for year, dfs in year_buckets.items():
        combined = pd.concat(dfs, ignore_index=True)
        print(f"{year}: combined {len(dfs)} file(s), total rows = {len(combined)}")
        year_data[year] = combined

    return dict(sorted(year_data.items()))


# =====================================================
# MATCHING ACROSS YEARS
# =====================================================


def match_consecutive_years(year_data: Dict[int, pd.DataFrame], max_distance: float = 15.0) -> pd.DataFrame:
    years = list(year_data.keys())
    projected = {
        y: to_projected_gdf(year_data[y], f"lon_{y}", f"lat_{y}", epsg=3395)
        for y in years
    }

    matched_frames = []
    first_year = years[0]
    first_sub_col = f"substation_{first_year}"
    substations = projected[first_year][first_sub_col].dropna().unique()

    for substation in substations:
        sub_matched = projected[first_year][projected[first_year][first_sub_col] == substation].copy()
        if sub_matched.empty:
            continue

        success = True
        for i in range(len(years) - 1):
            y1 = years[i]
            y2 = years[i + 1]
            right = projected[y2].copy()
            right_sub_col = f"substation_{y2}"
            right = right[right[right_sub_col] == substation].copy()

            if right.empty or sub_matched.empty:
                success = False
                break

            sub_matched = gpd.sjoin_nearest(
                sub_matched,
                right,
                how="inner",
                max_distance=max_distance,
                distance_col=f"dist_{y1}_{y2}",
            )

            if "index_right" in sub_matched.columns:
                sub_matched = sub_matched.drop(columns=["index_right"])

        if success and not sub_matched.empty:
            matched_frames.append(pd.DataFrame(sub_matched.drop(columns="geometry")))

    if not matched_frames:
        raise ValueError("No matched rows were found across consecutive years within substations.")

    matched = pd.concat(matched_frames, ignore_index=True)
    print(f"Matched rows across all substations: {len(matched)}")
    return matched


# =====================================================
# SUBSTATION GROUPING
# =====================================================


def assign_plot_substation(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    df = df.copy()
    sub_cols = [f"substation_{y}" for y in years if f"substation_{y}" in df.columns]

    def choose_substation(row: pd.Series) -> str:
        for col in reversed(sub_cols):
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                return str(val)
        return "Unknown"

    df["substation_plot"] = df.apply(choose_substation, axis=1)
    return df


# =====================================================
# GROWTH / R-VALUE / FILTERING
# =====================================================


def compute_growth_and_r(
    df: pd.DataFrame,
    years: List[int],
    rainfall_by_interval: Dict[Tuple[int, int], float],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    growth_cols: List[str] = []
    r_cols: List[str] = []

    for i in range(len(years) - 1):
        y1 = years[i]
        y2 = years[i + 1]
        interval = (y1, y2)
        if interval not in rainfall_by_interval:
            raise ValueError(f"Missing rainfall value for interval {interval}")

        c1 = f"clearance_{y1}"
        c2 = f"clearance_{y2}"
        gcol = f"growth_{y1}_{y2}"
        rcol = f"r_{y1}_{y2}"

        df[gcol] = df[c1] - df[c2]
        df[rcol] = df[gcol] / rainfall_by_interval[interval]
        growth_cols.append(gcol)
        r_cols.append(rcol)

    df["avg_growth"] = df[growth_cols].mean(axis=1)
    df["r_point"] = df[r_cols].mean(axis=1)
    q_low, q_high = df["r_point"].quantile([0.01, 0.99])
    df["r_point_clipped"] = df["r_point"].clip(q_low, q_high)
    return df, growth_cols, r_cols



def filter_growth_rows(
    df: pd.DataFrame,
    growth_cols: List[str],
    min_growth: float = -4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before = df.copy()
    keep_mask = np.ones(len(df), dtype=bool)
    for gcol in growth_cols:
        keep_mask &= df[gcol] >= min_growth

    kept = before.loc[keep_mask].copy()
    filtered_out = before.loc[~keep_mask].copy()
    return before, kept, filtered_out


# =====================================================
# NEAREST-NEIGHBOR SMOOTHING FOR NEGATIVE R
# =====================================================


def smooth_negative_r(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        print("Warning: smooth_negative_r received an empty dataframe. Skipping smoothing.")
        df["r_point_nn5"] = pd.Series(dtype=float)
        df["r_point_nn5_clipped"] = pd.Series(dtype=float)
        return df

    gdf = to_projected_gdf(df, lon_col, lat_col, epsg=3395)
    df["x_m"] = gdf.geometry.x
    df["y_m"] = gdf.geometry.y
    coords = df[["x_m", "y_m"]].values

    if len(coords) == 0:
        print("Warning: no coordinates available for smoothing. Skipping smoothing.")
        df["r_point_nn5"] = df["r_point"]
        df["r_point_nn5_clipped"] = df["r_point_clipped"]
        return df

    n_neighbors = min(6, len(df))
    if n_neighbors < 2:
        print("Warning: fewer than 2 rows available. Skipping neighbor smoothing.")
        df["r_point_nn5"] = df["r_point"]
        df["r_point_nn5_clipped"] = df["r_point_clipped"]
        return df

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    nn.fit(coords)
    _, idxs = nn.kneighbors(coords)

    r = df["r_point"].astype(float).values
    r_new = r.copy()

    for i in range(len(df)):
        if np.isfinite(r[i]) and r[i] < 0:
            neigh_idx = idxs[i, 1:]
            neigh_r = r[neigh_idx]
            neigh_r = neigh_r[np.isfinite(neigh_r) & (neigh_r >= 0)]
            if len(neigh_r) > 0:
                r_new[i] = neigh_r.mean()

    df["r_point_nn5"] = r_new
    q_low, q_high = pd.Series(df["r_point_nn5"]).quantile([0.01, 0.99])
    df["r_point_nn5_clipped"] = df["r_point_nn5"].clip(q_low, q_high)
    return df


# =====================================================
# PREDICTION
# =====================================================


def predict_next_year(
    df: pd.DataFrame,
    years: List[int],
    prediction_rainfall: float,
    danger_threshold: float = 4.0,
) -> Tuple[pd.DataFrame, str, str, int]:
    df = df.copy()
    latest_year = years[-1]
    next_year = latest_year + 1

    pred_growth_col = f"predicted_growth_{latest_year}_{next_year}"
    pred_clearance_col = f"predicted_clearance_{next_year}"

    df[pred_growth_col] = df["r_point_nn5_clipped"] * prediction_rainfall
    df[pred_clearance_col] = df[f"clearance_{latest_year}"] - df[pred_growth_col]

    conditions = [
        (df[pred_clearance_col] < danger_threshold),
        (df[pred_clearance_col] >= danger_threshold) & (df[pred_clearance_col] < danger_threshold + 2),
        (df[pred_clearance_col] >= danger_threshold + 2) & (df[pred_clearance_col] < danger_threshold + 4),
        (df[pred_clearance_col] >= danger_threshold + 4) & (df[pred_clearance_col] < 10),
        (df[pred_clearance_col] >= 10),
    ]
    colors = ["#ff0000", "#ffa500", "#ffff00", "#008000", "#ffffff"]
    buckets = ["Critical", "High Priority", "Watch List", "Healthy", "Safe"]

    df["risk_color"] = np.select(conditions, colors, default="#ffffff")
    df["risk_bucket"] = np.select(conditions, buckets, default="Safe")

    return df, pred_clearance_col, pred_growth_col, next_year


# =====================================================
# OUTPUT TABLES
# =====================================================


def build_risk_summary_by_substation(df: pd.DataFrame, pred_clearance_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["substation_plot", "risk_bucket"])
        .agg(
            total_points=(pred_clearance_col, "size"),
            mean_clearance_ft=(pred_clearance_col, "mean"),
            min_clearance_ft=(pred_clearance_col, "min"),
            max_clearance_ft=(pred_clearance_col, "max"),
        )
        .reset_index()
    )

    totals = df.groupby("substation_plot").size().rename("substation_total").reset_index()
    summary = summary.merge(totals, on="substation_plot", how="left")
    summary["percent_of_substation"] = 100.0 * summary["total_points"] / summary["substation_total"]
    return summary


# =====================================================
# MAIN PIPELINE FOR STREAMLIT
# =====================================================


def run_growth_pipeline(
    filepaths: Iterable[str],
    rainfall_by_interval: Dict[Tuple[int, int], float],
    prediction_rainfall: float,
    compute_clearance_for: Optional[Iterable[str]] = None,
    output_dir: str = ".",
    make_plots: bool = False,
    danger_threshold: float = 4.0,
) -> Dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    year_data = load_all_years(filepaths, compute_clearance_for=compute_clearance_for)
    years = list(year_data.keys())
    print(f"Loaded years: {years}")

    matched = match_consecutive_years(year_data, max_distance=15.0)
    print(f"Matched rows across years: {len(matched)}")
    if matched.empty:
        raise ValueError("No rows remained after spatial matching. Check coordinate overlap or max_distance.")

    matched = assign_plot_substation(matched, years)
    print("Substation counts:")
    print(matched["substation_plot"].value_counts(dropna=False))

    clearance_cols = [f"clearance_{y}" for y in years]
    print("\nClearance columns:", clearance_cols)
    for c in clearance_cols:
        print(f"{c}: non-null = {matched[c].notna().sum()} / {len(matched)}")

    matched = matched.dropna(subset=clearance_cols)
    print(f"Rows after dropping missing clearance: {len(matched)}")
    if matched.empty:
        raise ValueError("No rows remained after dropping missing clearance values.")

    matched, growth_cols, r_cols = compute_growth_and_r(matched, years, rainfall_by_interval)
    matched_before_filter, matched_kept, matched_filtered_out = filter_growth_rows(
        matched, growth_cols, min_growth=-4.0
    )

    print(f"Rows kept after growth filter: {len(matched_kept)}")
    print(f"Rows filtered out by growth filter: {len(matched_filtered_out)}")

    if matched_kept.empty:
        raise ValueError(
            "No rows remained after the growth filter. This likely means your growth threshold is too strict or the input years do not align well."
        )

    base_year = years[0]
    lon_col = f"lon_{base_year}"
    lat_col = f"lat_{base_year}"
    matched_kept = smooth_negative_r(matched_kept, lon_col, lat_col)

    matched_kept, pred_clearance_col, pred_growth_col, pred_year = predict_next_year(
        matched_kept,
        years,
        prediction_rainfall,
        danger_threshold=danger_threshold,
    )

    # Streamlit-friendly aliases
    matched_kept["lon"] = matched_kept[lon_col]
    matched_kept["lat"] = matched_kept[lat_col]
    matched_kept["substation"] = matched_kept["substation_plot"]

    matched_filtered_out.to_csv(output_path / "filtered_out_main_model_points.csv", index=False)
    matched_kept.to_csv(output_path / "per_point_growth_predictions.csv", index=False)
    risk_summary = build_risk_summary_by_substation(matched_kept, pred_clearance_col)
    risk_summary.to_csv(output_path / "table_risk_by_substation_and_bucket.csv", index=False)

    return {
        "years": years,
        "pred_year": pred_year,
        "matched_before_filter": matched_before_filter,
        "results_df": matched_kept,
        "filtered_out_growth_df": matched_filtered_out,
        "risk_summary_df": risk_summary,
        "pred_clearance_col": pred_clearance_col,
        "pred_growth_col": pred_growth_col,
        "base_lon_col": lon_col,
        "base_lat_col": lat_col,
    }
