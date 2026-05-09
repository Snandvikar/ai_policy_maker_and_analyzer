"""
pipeline/store_dashboard_tables.py
====================================
Builds and persists the 4 dashboard support tables after mci_pipeline.py runs.

Tables written:
  mci_scores              — updated in-place: adds cluster_id + cluster_label
  mci_timeseries_scores   — updated in-place: adds cluster_id + cluster_label
  factor_subcomponent_scores — one row per (canonical_state, districtname, year, subcomponent)
  rf_feature_importance   — RF trained on new schema columns
  cluster_profiles        — K-Means centroids + labels + intervention text

Schema alignment (new columns in mci_scores / mci_timeseries_scores):
  canonical_state, statename, statecode, districtname, districtcode, year
  + IFS, DLS, SES, WDI, MCI, WSI, WEI, MCI_class, Safety_risk, Employment_gap
  + rules vars: mobile_only_pct, no_phone_pct, illiteracy_percent,
                gender_vulnerability_index, crime_against_women_rate,
                msme_female_share_pct, share_of_slum_population,
                dependent_women_percent, wireless_rural_teledensity_pct,
                e_transactions_per_1000_population, maternal_mortality_rate,
                destitute_pct, non_agri_enterprise_pct, income_lt5k_pct,
                any_phone_pct

Run order:
  python -m pipeline.preprocessor
  python -m pipeline.mci_pipeline
  python -m pipeline.store_dashboard_tables    ← this file
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Identifier columns (never used as features) ──────────────────────────────
ID_COLS = {
    "canonical_state", "statename", "statecode",
    "districtname", "districtcode", "year",
    "MCI_class", "Safety_risk", "Employment_gap",
    "cluster_id", "cluster_label", "MCI_class_sort",
}

# ── Columns that are "bad when high" — inverted during subcomponent scoring ───
INVERT_COLS = {
    "no_phone_pct", "illiteracy_percent", "gender_vulnerability_index",
    "maternal_mortality_rate", "dependent_women_percent",
    "share_of_slum_population", "crime_against_women_rate",
    "destitute_pct", "income_lt5k_pct",
}

# ── Cluster intervention text per weakest factor ──────────────────────────────
INTERVENTION_MAP = {
    "IFS": (
        "Prioritise wireless coverage expansion and last-mile connectivity. "
        "Apply for USOF/BharatNet grants. Increase rural teledensity targets for TSPs."
    ),
    "DLS": (
        "Launch community digital literacy camps via CSCs. "
        "Subsidise entry-level smartphones. Focus on reducing no-phone household rate."
    ),
    "SES": (
        "Expand Jan Dhan outreach, SHG formation, and PM scheme coverage. "
        "Address structural poverty and illiteracy barriers."
    ),
    "WDI": (
        "Deploy women-only digital skilling under PMGDISHA. "
        "Reduce gender vulnerability index through targeted welfare convergence. "
        "Strengthen Mahila SHGs and women's MSME registration."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
    logger.info(f"Stored '{table}': {len(df)} rows")


def _normalise_col(series: pd.Series, invert: bool = False) -> pd.Series:
    p5, p95 = series.quantile(0.05), series.quantile(0.95)
    if p95 == p5:
        return pd.Series(50.0, index=series.index)
    scaled = (series.clip(p5, p95) - p5) / (p95 - p5) * 100
    return (100 - scaled) if invert else scaled


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — CLUSTER PROFILES + update mci_scores with cluster columns
#
# Runs K-Means on latest-year factor scores.
# Adds cluster_id and cluster_label to both mci_scores and mci_timeseries_scores.
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_profiles(
    con: duckdb.DuckDBPyConnection,
    df_latest: pd.DataFrame,
    df_ts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (cluster_profiles_df, df_latest_with_clusters, df_ts_with_clusters).
    """
    FACTOR_COLS = ["IFS", "DLS", "SES", "WDI"]
    cluster_input = df_latest[FACTOR_COLS].dropna()
    idx_cluster   = cluster_input.index

    # Find optimal k via silhouette score (k = 3..7)
    best_k, best_sil = 4, -1
    max_k = min(8, len(cluster_input))
    for k in range(3, max_k):
        km_test = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels  = km_test.fit_predict(cluster_input.values)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(cluster_input.values, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k

    logger.info(f"K-Means: optimal k={best_k}, silhouette={best_sil:.4f}")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    cluster_labels_arr = km.fit_predict(cluster_input.values)

    LABEL_MAP = {
        "IFS": "Infrastructure gap",
        "DLS": "Digital literacy gap",
        "SES": "Socio-economic barrier",
        "WDI": "Women exclusion gap",
    }

    cluster_rows = []
    cluster_id_to_label: dict[int, str] = {}

    for cid in range(best_k):
        centroid   = km.cluster_centers_[cid]
        cent_dict  = dict(zip(FACTOR_COLS, centroid))
        weakest    = min(cent_dict, key=cent_dict.get)
        mean_score = float(np.mean(centroid))
        severity   = (
            "severe"   if mean_score < 35 else
            "moderate" if mean_score < 55 else
            "mild"
        )
        label = (
            f"Comprehensive desert ({severity})"
            if mean_score < 25
            else f"{LABEL_MAP[weakest]} ({severity})"
        )
        cluster_id_to_label[cid] = label

        # Districts in this cluster (latest year only)
        mask = (cluster_labels_arr == cid)
        districts_in = df_latest.loc[idx_cluster[mask], "districtname"].tolist()

        cluster_rows.append({
            "cluster_id":       cid,
            "label":            label,
            "weakest_factor":   weakest,
            "mean_score":       round(mean_score, 1),
            "severity":         severity,
            "IFS_centroid":     round(float(cent_dict["IFS"]), 1),
            "DLS_centroid":     round(float(cent_dict["DLS"]), 1),
            "SES_centroid":     round(float(cent_dict["SES"]), 1),
            "WDI_centroid":     round(float(cent_dict["WDI"]), 1),
            "area_count":       len(districts_in),
            "area_list":        ", ".join(districts_in),
            "intervention":     INTERVENTION_MAP.get(weakest, "Multi-factor intervention required."),
            "silhouette_score": round(best_sil, 4),
        })

    cluster_df = pd.DataFrame(cluster_rows)

    # ── Map cluster_id + cluster_label back onto latest df ────────────────────
    df_latest = df_latest.copy()
    df_latest["cluster_id"]    = np.nan
    df_latest["cluster_label"] = ""

    df_latest.loc[idx_cluster, "cluster_id"]    = cluster_labels_arr.astype(float)
    df_latest.loc[idx_cluster, "cluster_label"] = [
        cluster_id_to_label[c] for c in cluster_labels_arr
    ]

    # ── Propagate cluster assignment to timeseries by (canonical_state, districtname) ──
    # Cluster is determined on latest year; broadcast to all years for same district.
    cluster_map = df_latest.set_index(
        ["canonical_state", "districtname"]
    )[["cluster_id", "cluster_label"]].to_dict("index")

    df_ts = df_ts.copy()
    df_ts["cluster_id"]    = np.nan
    df_ts["cluster_label"] = ""

    for idx, row in df_ts.iterrows():
        key = (row["canonical_state"], row["districtname"])
        if key in cluster_map:
            df_ts.at[idx, "cluster_id"]    = cluster_map[key]["cluster_id"]
            df_ts.at[idx, "cluster_label"] = cluster_map[key]["cluster_label"]

    # ── Add MCI_class_sort for ordered display ────────────────────────────────
    CLASS_SORT = {
        "Severe desert": 1, "Moderate desert": 2,
        "Partial connectivity": 3, "Near-connected": 4, "Connected": 5,
    }
    df_latest["MCI_class_sort"] = df_latest["MCI_class"].astype(str).map(CLASS_SORT).fillna(3)
    df_ts["MCI_class_sort"]     = df_ts["MCI_class"].astype(str).map(CLASS_SORT).fillna(3)

    return cluster_df, df_latest, df_ts


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — FACTOR SUBCOMPONENT SCORES
#
# One row per (canonical_state, districtname, year, factor, subcomponent).
# Uses proxy columns that ACTUALLY EXIST in mci_scores (new schema).
#
# Subcomponent definitions: (display_name, factor, proxy_col, weight, invert)
# proxy_col must be a column in mci_scores after mci_pipeline.py runs.
# If proxy_col is absent, falls back to the parent factor score.
# ─────────────────────────────────────────────────────────────────────────────

SUBCOMPONENTS = [
    # ── IFS: Infrastructure Factor Score ──────────────────────────────────────
    # proxy: wireless teledensity rates and phone access from new schema
    ("Rural wireless coverage",  "IFS", "wireless_rural_teledensity_pct",      0.35, False),
    ("Urban wireless coverage",  "IFS", "wireless_urban_teledensity_pct",      0.15, False),
    ("Phone household access",   "IFS", "any_phone_pct",                       0.20, False),
    ("Mobile-only access",       "IFS", "mobile_only_pct",                     0.15, False),
    ("Digital transaction use",  "IFS", "e_transactions_per_1000_population",  0.15, False),

    # ── DLS: Digital Literacy Factor Score ────────────────────────────────────
    ("Phone access (DLS proxy)", "DLS", "any_phone_pct",                       0.20, False),
    ("Illiteracy (inverted)",    "DLS", "illiteracy_percent",                  0.25, True),
    ("Gender vulnerability",     "DLS", "gender_vulnerability_index",          0.30, True),
    ("Women MSME share",         "DLS", "msme_female_share_pct",               0.25, False),

    # ── SES: Socio-Economic Factor Score ─────────────────────────────────────
    ("Income deprivation",       "SES", "income_lt5k_pct",                     0.25, True),
    ("Destitution rate",         "SES", "destitute_pct",                       0.25, True),
    ("Slum population share",    "SES", "share_of_slum_population",            0.25, True),
    ("Non-agri enterprise",      "SES", "non_agri_enterprise_pct",             0.25, False),

    # ── WDI: Women Digital Inclusion Factor Score ─────────────────────────────
    ("Gender vulnerability idx", "WDI", "gender_vulnerability_index",          0.25, True),
    ("Women MSME participation", "WDI", "msme_female_share_pct",               0.25, False),
    ("Crime against women",      "WDI", "crime_against_women_rate",            0.25, True),
    ("Women dependency rate",    "WDI", "dependent_women_percent",             0.25, True),
]


def build_subcomponent_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives sub-component scores for each district × year row.
    Returns a long-format DataFrame.
    """
    rows = []
    for sub_name, factor, proxy_col, weight, invert in SUBCOMPONENTS:
        if proxy_col in df.columns:
            raw_score = _normalise_col(df[proxy_col], invert=invert)
        else:
            logger.warning(f"Subcomponent proxy '{proxy_col}' not in mci_scores — "
                           f"falling back to {factor} score")
            raw_score = df[factor]

        for idx, row in df.iterrows():
            rows.append({
                "canonical_state":  row["canonical_state"],
                "statename":        row.get("statename", ""),
                "districtname":     row["districtname"],
                "districtcode":     row.get("districtcode", ""),
                "year":             row["year"],
                "factor":           factor,
                "subcomponent":     sub_name,
                "sub_score":        round(float(raw_score.loc[idx]), 2),
                "weight_in_factor": weight,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — RF FEATURE IMPORTANCE
#
# Trains a Random Forest on columns that ACTUALLY EXIST in mci_scores.
# ─────────────────────────────────────────────────────────────────────────────

# Features from the new schema (rules vars written by mci_pipeline.py)
RF_CANDIDATE_FEATURES = [
    "mobile_only_pct",
    "no_phone_pct",
    "any_phone_pct",
    "illiteracy_percent",
    "gender_vulnerability_index",
    "crime_against_women_rate",
    "msme_female_share_pct",
    "share_of_slum_population",
    "dependent_women_percent",
    "wireless_rural_teledensity_pct",
    "e_transactions_per_1000_population",
    "maternal_mortality_rate",
    "destitute_pct",
    "non_agri_enterprise_pct",
    "income_lt5k_pct",
]


def build_rf_importance(df: pd.DataFrame) -> pd.DataFrame:
    available = [f for f in RF_CANDIDATE_FEATURES if f in df.columns]
    if not available:
        logger.warning("No RF features available in mci_scores — skipping RF")
        return pd.DataFrame(columns=["feature", "importance", "rank", "pct"])

    feat_df = df[available + ["MCI"]].dropna(subset=available + ["MCI"])
    if len(feat_df) < 10:
        logger.warning(f"Only {len(feat_df)} rows available — RF unreliable, skipping")
        return pd.DataFrame(columns=["feature", "importance", "rank", "pct"])

    logger.info(f"Training RF on {len(feat_df)} rows, {len(available)} features...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8,
        min_samples_leaf=max(2, len(feat_df) // 20),
        random_state=42, n_jobs=-1,
    )
    rf.fit(feat_df[available].values, feat_df["MCI"].values)

    imp_df = pd.DataFrame({
        "feature":    available,
        "importance": rf.feature_importances_,
        "rank":       pd.Series(rf.feature_importances_)
                        .rank(ascending=False).astype(int).values,
        "pct":        (rf.feature_importances_ * 100).round(2),
    }).sort_values("rank").reset_index(drop=True)

    logger.info(f"RF top feature: {imp_df.iloc[0]['feature']} "
                f"({imp_df.iloc[0]['pct']:.1f}%)")
    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("=" * 60)
    logger.info("STORE DASHBOARD TABLES START")
    logger.info("=" * 60)

    # ── Load mci_scores (latest year) and timeseries ──────────────────────────
    df_latest = con.execute("SELECT * FROM mci_scores").df()
    df_ts     = con.execute("SELECT * FROM mci_timeseries_scores").df()
    logger.info(f"Loaded mci_scores: {len(df_latest)} rows")
    logger.info(f"Loaded mci_timeseries_scores: {len(df_ts)} rows")

    # Confirm key columns exist
    required = ["canonical_state", "districtname", "year", "IFS", "DLS", "SES", "WDI", "MCI"]
    missing  = [c for c in required if c not in df_latest.columns]
    if missing:
        raise ValueError(
            f"mci_scores is missing required columns: {missing}. "
            "Run mci_pipeline.py first."
        )

    # ── Step 1: Clustering + update mci_scores + mci_timeseries_scores ────────
    logger.info("Step 1/3  Running K-Means clustering...")
    cluster_df, df_latest, df_ts = build_cluster_profiles(con, df_latest, df_ts)

    _write(con, "mci_scores",           df_latest)
    _write(con, "mci_timeseries_scores", df_ts)
    _write(con, "cluster_profiles",      cluster_df)

    # ── Step 2: Subcomponent scores ───────────────────────────────────────────
    logger.info("Step 2/3  Building factor subcomponent scores...")
    sub_df = build_subcomponent_scores(df_latest)
    _write(con, "factor_subcomponent_scores", sub_df)

    # ── Step 3: RF feature importance ────────────────────────────────────────
    logger.info("Step 3/3  Training RF for feature importance...")
    imp_df = build_rf_importance(df_latest)
    _write(con, "rf_feature_importance", imp_df)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STORE DASHBOARD TABLES DONE")
    logger.info("Tables in DuckDB:")
    for (tbl,) in con.execute("SHOW TABLES").fetchall():
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            logger.info(f"  {tbl:<45} {n:>7} rows")
        except Exception:
            pass
    logger.info("=" * 60)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    run(con)
    con.close()