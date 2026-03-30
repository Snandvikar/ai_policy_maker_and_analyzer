"""
store_dashboard_tables.py
=========================
Run this AFTER mci_pipeline.py completes.
Saves the 4 tables the dashboard needs into DuckDB.

Usage:
    python store_dashboard_tables.py
"""

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_PATH = "database.duckdb"
con = duckdb.connect(DB_PATH)

# ── Load the scores table written by mci_pipeline.py ──────────────────────────
df = con.execute("SELECT * FROM mci_scores").df()
print(f"[load] mci_scores: {len(df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — mci_scores  (already exists, just ensure columns are complete)
#
# Add MCI_class_sort so the dashboard can order classifications correctly.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ORDER = {
    "Severe desert": 1,
    "Moderate desert": 2,
    "Partial connectivity": 3,
    "Near-connected": 4,
    "Connected": 5,
}
df["MCI_class_sort"] = df["MCI_class"].astype(str).map(CLASS_ORDER).fillna(3)

con.execute("DROP TABLE IF EXISTS mci_scores")
con.execute("CREATE TABLE mci_scores AS SELECT * FROM df")
print("[store] mci_scores ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — factor_subcomponent_scores
#
# One row per (area × sub-component).
# The pipeline computed factor scores as weighted sums of sub-components.
# We need to re-derive those sub-component values here so the drill-down
# radar can show WHERE within a factor an area is weak.
#
# If you stored sub-component scores in mci_pipeline.py, load them instead.
# This version re-derives them from mci_scores columns if available,
# otherwise approximates from the factor scores.
# ─────────────────────────────────────────────────────────────────────────────

# Sub-component definitions: (name, factor, proxy_column_in_mci_scores, weight)
# Adjust proxy columns to match whatever columns your mci_scores table has.
# If a proxy col is missing, we set the sub-score = factor_score (fallback).
SUBCOMPONENTS = [
    # Infrastructure sub-components
    ("Coverage density",    "IFS", "towers_per_100k_pop",      0.20),
    ("Network quality",     "IFS", "cell_dl_mbps",             0.30),
    ("Fiber backbone",      "IFS", "percent_ge_25_mbps",       0.25),
    ("Last-mile access",    "IFS", "pm_wani_hotspots",         0.15),
    ("Redundancy",          "IFS", "ul_median_mbps",           0.10),
    # Digital literacy sub-components
    ("HH connectivity",     "DLS", "hh_internet_percent",      0.20),
    ("Skills",              "DLS", "digital_skill_percent",    0.20),
    ("Gender inclusion",    "DLS", "women_internet_percent",   0.30),
    ("Wealth equity",       "DLS", "poorest_q_internet_percent", 0.15),
    ("Structural literacy", "DLS", "literacy_rate_percent",    0.15),
    # Socio-economic sub-components
    ("Economic baseline",   "SES", "hdi_score",                0.20),
    ("Education parity",    "SES", "gender_parity_index_gpi",  0.25),
    ("Women's participation","SES","female_lfpr_percent",      0.20),
    ("Financial inclusion", "SES", "jan_dhan_account_percent", 0.15),
    ("Life infrastructure", "SES", "electricity_hrs_per_day",  0.10),
    ("Scheme reach",        "SES", "scheme_coverage_percent",  0.10),
    # WDI sub-components
    ("Women internet",      "WDI", "women_internet_percent",   0.25),
    ("Women mobile",        "WDI", "women_mobile_percent",     0.25),
    ("Women skills",        "WDI", "women_skill_percent",      0.25),
    ("Gender gap",          "WDI", "gender_gap_pp",            0.25),
]

def normalize_col(series):
    p5, p95 = series.quantile(0.05), series.quantile(0.95)
    if p95 == p5:
        return pd.Series(50.0, index=series.index)
    return ((series.clip(p5, p95) - p5) / (p95 - p5) * 100)

rows = []
for sub_name, factor, proxy_col, weight in SUBCOMPONENTS:
    if proxy_col in df.columns:
        # Invert if known "bad when high" columns
        invert = proxy_col in {"gender_gap_pp", "poverty_rate_percent",
                               "gini_coeff", "latency_ms", "overall_unempl_percent",
                               "girls_school_dropout_percent"}
        raw_score = normalize_col(df[proxy_col])
        if invert:
            raw_score = 100 - raw_score
    else:
        # Fallback: approximate from parent factor score
        raw_score = df[factor]

    for _, area_row in df.iterrows():
        rows.append({
            "city":           area_row["city"],
            "area":           area_row["area"],
            "district":       area_row.get("district", ""),
            "area_type":      area_row["area_type"],
            "factor":         factor,
            "subcomponent":   sub_name,
            "sub_score":      round(float(raw_score.loc[area_row.name]), 2),
            "weight_in_factor": weight,
        })

sub_df = pd.DataFrame(rows)
con.execute("DROP TABLE IF EXISTS factor_subcomponent_scores")
con.execute("CREATE TABLE factor_subcomponent_scores AS SELECT * FROM sub_df")
print(f"[store] factor_subcomponent_scores ✓  ({len(sub_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — rf_feature_importance
#
# Fit RF on available features to get importance scores.
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

available_features = [f for f in PREDICTIVE_FEATURES if f in df.columns]
feat_df = df[available_features + ["MCI"]].dropna(subset=available_features)

if len(feat_df) >= 10:
    rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                               min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(feat_df[available_features].values, feat_df["MCI"].values)
    imp_df = pd.DataFrame({
        "feature":    available_features,
        "importance": rf.feature_importances_,
        "rank":       pd.Series(rf.feature_importances_).rank(ascending=False).astype(int).values,
        "pct":        (rf.feature_importances_ * 100).round(2),
    }).sort_values("rank")
else:
    print("[warn] not enough rows for RF — storing empty importance table")
    imp_df = pd.DataFrame(columns=["feature", "importance", "rank", "pct"])

con.execute("DROP TABLE IF EXISTS rf_feature_importance")
con.execute("CREATE TABLE rf_feature_importance AS SELECT * FROM imp_df")
print(f"[store] rf_feature_importance ✓  ({len(imp_df)} features)")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — cluster_profiles
#
# K-Means on factor scores. Finds optimal k, labels each cluster,
# and stores centroids + policy recommendations.
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_COLS = ["IFS", "DLS", "SES", "WDI"]
cluster_input = df[FACTOR_COLS].dropna()
idx_cluster   = cluster_input.index

best_k, best_sil = 4, -1
for k in range(3, min(8, len(cluster_input))):
    km_test = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels  = km_test.fit_predict(cluster_input.values)
    if len(set(labels)) < 2:
        continue
    sil = silhouette_score(cluster_input.values, labels)
    if sil > best_sil:
        best_sil, best_k = sil, k

km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
cluster_labels = km.fit_predict(cluster_input.values)
df.loc[idx_cluster, "cluster_id"] = cluster_labels
con.execute("DROP TABLE IF EXISTS mci_scores")
con.execute("CREATE TABLE mci_scores AS SELECT * FROM df")

INTERVENTION_MAP = {
    "IFS": "Prioritise tower densification and fibre rollout. Apply for USOF/BharatNet grants. Push OFC fiberisation targets.",
    "DLS": "Launch community digital literacy camps. Partner with CSCs for smartphone training. Focus on school digital access.",
    "SES": "Expand Jan Dhan outreach, SHG formation, and PM scheme coverage. Address structural poverty barriers.",
    "WDI": "Deploy women-only digital skilling under PMGDISHA. Subsidise women's smartphone ownership. Strengthen Mahila SHGs.",
}

cluster_rows = []
for cid in range(best_k):
    centroid  = km.cluster_centers_[cid]
    cent_dict = dict(zip(FACTOR_COLS, centroid))
    weakest   = min(cent_dict, key=cent_dict.get)
    mean_score = float(np.mean(centroid))
    severity  = "severe" if mean_score < 35 else ("moderate" if mean_score < 55 else "mild")

    LABEL_MAP = {
        "IFS": "Infrastructure gap",
        "DLS": "Digital literacy gap",
        "SES": "Socio-economic barrier",
        "WDI": "Women exclusion gap",
    }
    label = f"{LABEL_MAP[weakest]} ({severity})"
    if mean_score < 25:
        label = f"Comprehensive desert ({severity})"

    areas_in_cluster = df.loc[idx_cluster][df.loc[idx_cluster, "cluster_id"] == cid]["area"].tolist()

    cluster_rows.append({
        "cluster_id":        cid,
        "label":             label,
        "weakest_factor":    weakest,
        "mean_score":        round(mean_score, 1),
        "severity":          severity,
        "IFS_centroid":      round(float(cent_dict["IFS"]), 1),
        "DLS_centroid":      round(float(cent_dict["DLS"]), 1),
        "SES_centroid":      round(float(cent_dict["SES"]), 1),
        "WDI_centroid":      round(float(cent_dict["WDI"]), 1),
        "area_count":        len(areas_in_cluster),
        "area_list":         ", ".join(areas_in_cluster),
        "intervention":      INTERVENTION_MAP.get(weakest, "Multi-factor intervention required."),
        "silhouette_score":  round(best_sil, 4),
    })

cluster_df = pd.DataFrame(cluster_rows)
con.execute("DROP TABLE IF EXISTS cluster_profiles")
con.execute("CREATE TABLE cluster_profiles AS SELECT * FROM cluster_df")
print(f"[store] cluster_profiles ✓  ({best_k} clusters, silhouette={best_sil:.3f})")

print("\n[done] All 4 dashboard tables stored in mci.db")
print("\nTables available:")
for tbl in con.execute("SHOW TABLES").fetchall():
    cnt = con.execute(f"SELECT COUNT(*) FROM {tbl[0]}").fetchone()[0]
    print(f"  {tbl[0]:<35} {cnt} rows")

con.close()