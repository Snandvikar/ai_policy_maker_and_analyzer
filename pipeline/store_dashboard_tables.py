"""
pipeline/store_dashboard_tables.py
===================================
Builds and persists the 4 dashboard tables from mci_scores output.
Run AFTER mci_pipeline.py.

Usage:
    python -m pipeline.store_dashboard_tables
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    DB_PATH, BASELINE_YEAR,
    SCORE_TABLES, RAW_TABLES,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_col(series: pd.Series) -> pd.Series:
    p5, p95 = series.quantile(0.05), series.quantile(0.95)
    if p95 == p5:
        return pd.Series(50.0, index=series.index)
    return ((series.clip(p5, p95) - p5) / (p95 - p5) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — mci_scores  (already written by pipeline; add cluster columns)
# TABLE 1b — mci_timeseries_scores  (same, all years)
# We update both with MCI_class_sort, cluster_id, cluster_label
# ─────────────────────────────────────────────────────────────────────────────

CLASS_SORT = {
    "Severe desert": 1, "Moderate desert": 2, "Partial connectivity": 3,
    "Near-connected": 4, "Connected": 5,
}

INTERVENTION_MAP = {
    "IFS": ("Prioritise tower densification and fibre rollout. "
            "Apply for USOF/BharatNet grants. Push OFC fiberisation targets."),
    "DLS": ("Launch community digital literacy camps. "
            "Partner with CSCs for smartphone training. Focus on school digital access."),
    "SES": ("Expand Jan Dhan outreach, SHG formation, and PM scheme coverage. "
            "Address structural poverty barriers."),
    "WDI": ("Deploy women-only digital skilling under PMGDISHA. "
            "Subsidise women's smartphone ownership. Strengthen Mahila SHGs."),
}


def build_cluster_profiles(
    df: pd.DataFrame,
    factor_cols: list = ["IFS", "DLS", "SES", "WDI"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs K-Means on factor scores for the latest year.
    Returns (cluster_profiles_df, df_with_cluster_cols).
    """
    latest = df[df["year"] == df["year"].max()]
    cluster_input = latest[factor_cols].dropna()
    idx_cluster = cluster_input.index

    best_k, best_sil = 4, -1
    for k in range(3, min(8, len(cluster_input))):
        km_test = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbl = km_test.fit_predict(cluster_input.values)
        if len(set(lbl)) < 2:
            continue
        sil = silhouette_score(cluster_input.values, lbl)
        if sil > best_sil:
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    cluster_labels = km.fit_predict(cluster_input.values)

    # Map cluster IDs back onto full df (all years) by area+city key
    area_cluster_map = dict(zip(
        zip(latest.loc[idx_cluster, "city"], latest.loc[idx_cluster, "area"]),
        cluster_labels,
    ))
    df = df.copy()
    df["cluster_id"] = df.apply(
        lambda r: area_cluster_map.get((r["city"], r["area"]), np.nan), axis=1
    )

    LABEL_MAP = {
        "IFS": "Infrastructure gap", "DLS": "Digital literacy gap",
        "SES": "Socio-economic barrier", "WDI": "Women exclusion gap",
    }
    cluster_rows = []
    for cid in range(best_k):
        centroid   = km.cluster_centers_[cid]
        cent_dict  = dict(zip(factor_cols, centroid))
        weakest    = min(cent_dict, key=cent_dict.get)
        mean_score = float(np.mean(centroid))
        severity   = "severe" if mean_score < 35 else ("moderate" if mean_score < 55 else "mild")
        label = ("Comprehensive desert" if mean_score < 25
                 else LABEL_MAP.get(weakest, "Mixed gap"))
        label = f"{label} ({severity})"

        mask = df["cluster_id"] == cid
        areas_in = df[mask & (df["year"] == df["year"].max())]["area"].tolist()

        cluster_rows.append({
            "cluster_id":     cid,
            "label":          label,
            "weakest_factor": weakest,
            "mean_score":     round(mean_score, 1),
            "severity":       severity,
            "IFS_centroid":   round(float(cent_dict["IFS"]), 1),
            "DLS_centroid":   round(float(cent_dict["DLS"]), 1),
            "SES_centroid":   round(float(cent_dict["SES"]), 1),
            "WDI_centroid":   round(float(cent_dict["WDI"]), 1),
            "area_count":     len(areas_in),
            "area_list":      ", ".join(areas_in),
            "intervention":   INTERVENTION_MAP.get(weakest, "Multi-factor intervention required."),
            "silhouette_score": round(best_sil, 4),
        })

    cluster_df = pd.DataFrame(cluster_rows)

    # Add cluster label column to main df
    cluster_label_map = {r["cluster_id"]: r["label"] for _, r in cluster_df.iterrows()}
    df["cluster_label"] = df["cluster_id"].map(cluster_label_map)
    df["MCI_class_sort"] = df["MCI_class"].astype(str).map(CLASS_SORT).fillna(3)

    return cluster_df, df


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — factor_subcomponent_scores
# ─────────────────────────────────────────────────────────────────────────────

SUBCOMPONENTS = [
    # (display_name, factor, proxy_raw_col, weight_in_factor, invert)
    ("Coverage density",      "IFS", "towers_per_100k_pop",          0.20, False),
    ("Network quality",       "IFS", "cell_dl_mbps",                 0.30, False),
    ("Fiber backbone",        "IFS", "percent_ge_25_mbps",           0.25, False),
    ("Last-mile access",      "IFS", "pm_wani_hotspots",             0.15, False),
    ("Redundancy",            "IFS", "ul_median_mbps",               0.10, False),
    ("HH connectivity",       "DLS", "hh_internet_percent",          0.20, False),
    ("Skills",                "DLS", "digital_skill_percent",        0.20, False),
    ("Gender inclusion",      "DLS", "women_internet_percent",       0.30, False),
    ("Wealth equity",         "DLS", "poorest_q_internet_percent",   0.15, False),
    ("Structural literacy",   "DLS", "dl_literacy_rate",             0.15, False),
    ("Economic baseline",     "SES", "hdi_score",                    0.20, False),
    ("Education parity",      "SES", "gender_parity_index_gpi",      0.25, False),
    ("Women's participation", "SES", "female_lfpr_percent",          0.20, False),
    ("Financial inclusion",   "SES", "jan_dhan_account_percent",     0.15, False),
    ("Life infrastructure",   "SES", "electricity_hrs_per_day",      0.10, False),
    ("Scheme reach",          "SES", "scheme_coverage_percent",      0.10, False),
    ("Women internet",        "WDI", "women_internet_percent",       0.25, False),
    ("Women mobile",          "WDI", "women_mobile_percent",         0.25, False),
    ("Women skills",          "WDI", "women_skill_percent",          0.25, False),
    ("Gender gap",            "WDI", "gender_gap_pp",                0.25, True),
]


def build_subcomponent_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sub_name, factor, proxy_col, weight, invert in SUBCOMPONENTS:
        if proxy_col not in df.columns:
            continue
        raw_score = normalize_col(df[proxy_col])
        if invert:
            raw_score = 100 - raw_score

        for _, row in df.iterrows():
            rows.append({
                "city":           row["city"],
                "area":           row["area"],
                "district":       row.get("district", ""),
                "area_type":      row["area_type"],
                "year":           row["year"],
                "factor":         factor,
                "subcomponent":   sub_name,
                "sub_score":      round(float(raw_score.loc[row.name]), 2),
                "weight_in_factor": weight,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — rf_feature_importance
# ─────────────────────────────────────────────────────────────────────────────

PREDICTIVE_FEATURES = [
    "towers_per_100k_pop", "gen4_5g_pct", "cell_dl_mbps",
    "ofc_km_per_area_km2", "percent_ge_25_mbps",
    "hh_internet_percent", "women_internet_percent", "gender_gap_pp",
    "digital_skill_percent", "women_skill_percent",
    "hdi_score", "poverty_rate_percent", "gini_coeff",
    "female_lfpr_percent", "female_digital_skill_percent",
    "electricity_hrs_per_day", "sanitation_coverage_percent",
    "jan_dhan_account_percent",
]


def build_rf_importance(df: pd.DataFrame) -> pd.DataFrame:
    available = [f for f in PREDICTIVE_FEATURES if f in df.columns]
    # Use latest year for RF
    latest = df[df["year"] == df["year"].max()]
    feat_df = latest[available + ["MCI"]].dropna(subset=available)

    if len(feat_df) < 10:
        print(f"  [warn] only {len(feat_df)} rows — RF importance unreliable")
        return pd.DataFrame(columns=["feature", "importance", "rank", "pct"])

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    rf.fit(feat_df[available].values, feat_df["MCI"].values)

    imp_df = pd.DataFrame({
        "feature":    available,
        "importance": rf.feature_importances_,
        "rank":       pd.Series(rf.feature_importances_).rank(ascending=False).astype(int).values,
        "pct":        (rf.feature_importances_ * 100).round(2),
    }).sort_values("rank")
    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(con: duckdb.DuckDBPyConnection):
    # Load timeseries scores (written by mci_pipeline.py)
    df = con.execute("SELECT * FROM mci_timeseries_scores").df()
    print(f"[load] mci_timeseries_scores: {len(df)} rows, years={sorted(df['year'].unique())}")

    # ── TABLE 3: RF importance ───────────────────────────────────────────────
    print("[store] building rf_feature_importance...")
    imp_df = build_rf_importance(df)
    con.execute("DROP TABLE IF EXISTS rf_feature_importance")
    con.execute("CREATE TABLE rf_feature_importance AS SELECT * FROM imp_df")
    print(f"  ✓ {len(imp_df)} features")

    # ── TABLE 4: Cluster profiles ────────────────────────────────────────────
    print("[store] running K-Means clustering...")
    cluster_df, df = build_cluster_profiles(df)
    con.execute("DROP TABLE IF EXISTS cluster_profiles")
    con.execute("CREATE TABLE cluster_profiles AS SELECT * FROM cluster_df")
    print(f"  ✓ {len(cluster_df)} clusters")

    # ── Update mci_scores and mci_timeseries_scores with cluster columns ─────
    con.execute("DROP TABLE IF EXISTS mci_timeseries_scores")
    con.execute("CREATE TABLE mci_timeseries_scores AS SELECT * FROM df")

    latest_df = df[df["year"] == df["year"].max()].copy()
    con.execute("DROP TABLE IF EXISTS mci_scores")
    con.execute("CREATE TABLE mci_scores AS SELECT * FROM latest_df")
    print(f"  ✓ mci_scores ({len(latest_df)} rows) and mci_timeseries_scores ({len(df)} rows) updated")

    # ── TABLE 2: Subcomponent scores ─────────────────────────────────────────
    print("[store] building factor_subcomponent_scores...")
    sub_df = build_subcomponent_scores(df)
    con.execute("DROP TABLE IF EXISTS factor_subcomponent_scores")
    con.execute("CREATE TABLE factor_subcomponent_scores AS SELECT * FROM sub_df")
    print(f"  ✓ {len(sub_df)} rows")

    print("\n[done] All dashboard tables stored.")
    for tbl in con.execute("SHOW TABLES").fetchall():
        cnt = con.execute(f"SELECT COUNT(*) FROM \"{tbl[0]}\"").fetchone()[0]
        print(f"  {tbl[0]:<40} {cnt} rows")


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    run(con)
    con.close()