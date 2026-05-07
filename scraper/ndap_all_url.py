"""
NDAP Multi-Dataset Scraper
──────────────────────────
Datasets:
  7973  – Original dataset (kept as-is)
  9022  – Slum 2011 / Census slum data
  9429  – Socio-Economic / Vulnerable Women
  7150  – Digital Literacy / Internet Access
  7073  – Women who filed cases (NCRB crimes against women)
  3429  – Youth Unemployment by Sex, Age, Education

How it works (no Selenium needed):
  1. For each dataset, the scraper visits
         https://ndap.niti.gov.in/get-api/<dataset_id>
     and extracts the exact  ind=  dim=  API_Key=  parameters
     that NDAP already shows on that page.
  2. It then paginates through  https://loadqa.ndapapi.com/v1/openapi
     using those parameters to download all rows.
  3. Each dataset is saved as  csv/<dataset_id>_<slug>.csv

API_Key resolution order:
  1. NDAP_API_KEY  environment variable
  2. .api_key      file (written after first manual entry)
  3. Interactive   prompt  (run once, then cached)

If the live scrape of the get-api page fails (e.g. NDAP changes its SPA)
you can hard-code the parameters in  DATASET_REGISTRY['manual_api_url'].
"""

import sys
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from pre_processing_utils import preprocess_dataset

# ── sibling imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from ndap_get_api_key      import get_api_key
from ndap_page_scraper import scrape_dataset_params

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
CSV_DIR    = BASE_DIR / "csv"           # individual dataset CSVs
NDAP_DIR   = BASE_DIR / "ndap_data"    # legacy folder kept for compatibility
CSV_DIR.mkdir(exist_ok=True)
NDAP_DIR.mkdir(exist_ok=True)

LOG_FILE = NDAP_DIR / "scraper.log"

# Re-configure logging so this module can be run standalone
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ndap")

# ─────────────────────────────────────────────────────────────────────────────
# NDAP API CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

API_BASE  = "https://loadqa.ndapapi.com/v1/openapi"   # confirmed from ndap_simple.py
TIMEOUT   = 30
RETRY_MAX = 3
RETRY_WAIT = 2.0
PAGE_DELAY = 0.5   # seconds between paginated calls (be polite)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Each entry:
#   dataset_id   – numeric ID from the /dataset/<id> or /get-api/<id> URL
#   slug         – safe filename suffix
#   description  – human-readable label
#   manual_api_url – (optional) paste the full "Copy API" URL here if auto-
#                    scraping the get-api page fails. Leave as "" to use
#                    auto-discovery.  Example from task description:
#                    "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAA...
#                     &ind=I9022_3&dim=Country,StateName,StateCode,Year&pageno=1"
#
# IMPORTANT: ind= and dim= are NOT hardcoded here.
# They are scraped live from  https://ndap.niti.gov.in/get-api/<dataset_id>
# so that when NDAP updates a dataset's schema the scraper still works.
# Only override them in  manual_api_url  if auto-discovery fails.

DATASET_REGISTRY = [
    {
        "dataset_id"    : 7973,
        "slug"          : "gender_wise_total_msme_7973",
        "description"   : "Gender wise total Micro, Small, and Medium Enterprises (MSMEs)",
        "manual_api_url": "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAABp3gtIgV0nyAaP2S4iN37HUlLsGnWuQvKKTlfViTTTEK247x8xAWCXpKgxoTpf_oOe5GcuNWt9WXvQdrfdpuFsY-X70P3_e4KXIr5-OjAEH3yqjyAewRLeJ33E2kEZJAypwXLNnZcZLTZL4UCSviUqt9rhR58Qv3jxFMswUUZ3tc091MXpyADIxKjqNHsPpnRaRiov&ind=I7973_6,I7973_7,I7973_8,I7973_9,I7973_10&dim=Country,StateName,StateCode,DistrictName,DistrictCode,Year,CalendarDay&pageno=1",   # leave blank → use live scrape
    },
    {
        "dataset_id"    : 9022,
        "slug"          : "slum_census_2011_9022",
        "description"   : "Slum Population & Households — Census 2011",
        # Example URL from task description — used as fallback if scrape fails
        "manual_api_url": "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAABp3g6XbmrmPrpr-kVUQS858nH1Khd_25njBEWoY9U4eK_TZ2gPk_Q2As3fkqcTGuXw2ytc3ld3UFaEMNUuLK6jzKoAF4p6Af8Lx5rxhaLn9Mk39DOCqp4JZua-0IY_2Am6G5CiforGf0MIbgj8e11ZqBn7Fd4YrAa6yaT54Mhu5DIDNVp8mAzVKMQAVb3LKqBGtQ0X&ind=I9022_3&dim=Country,StateName,StateCode,Year&pageno=1",
    },
    {
        "dataset_id"    : 9429,
        "slug"          : "vulnerable_women_socio_economic_9429",
        "description"   : "Vulnerable Socio-Economic Conditions — Women",
        "manual_api_url": "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAABp3gr8A16ZRXBIzZ8CmiTtcuTkmQxCoOFhHy9vP7XtWlCRfpaQnxEKA0EGPsE7u524htEZ_deb05RchEOhD_1HfhPRPAEri9v2Xq5yqHenuuwbDz80BtbmLkePWnvH7uSWw2u_cj_5cR0Qfxf8md7czwVt5J4WMnDxWoaQiWXrVpIvXBq_7pCHTRhKrO4GP1oSM52U&ind=I9429_3,I9429_4,I9429_5,I9429_6,I9429_7,I9429_8,I9429_9,I9429_10&dim=Country,StateName,StateCode,Year&pageno=1",
    },
    {
        "dataset_id"    : 9131,
        "slug"          : "electronic_transaction_aggregation_per_1000_population_9131",
        "description"   : "Electronic Transaction Aggregation and Analysis Layer (ETAAL) -State-wise Transaction Per 1000 Population",
        "manual_api_url": "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAABp4UXYuRQilpyZhVKFQ5oJv80vdMun95f5nC5gRXU4XgS_GHqzA_ndnuGAvqwLfP7cggAftJHiMs9yoKTx6hH9DUSSxsTEOOE3LegxQ8iuTdOIhsuv-h-vIQcZsaYheFrXJaQPR18g-OKj32GH71AXnLawlwZLjBvKgTrhiu2GM69TIOUzgBh6_-W-fGlGGeKivqVW&ind=I9131_4,I9131_5,I9131_6&dim=Country,StateName,StateCode,Year,CalendarDay&pageno=1",
    },
    {
        "dataset_id"    : 7086,
        "slug"          : "socio_economic_census_7086",
        "description"   : "Socio Economic Census",
        "manual_api_url": "https://loadqa.ndapapi.com/v1/openapi?API_Key=gAAAAABp4UrZapuaeuRJX9nm8Fe1P_wyLFaW8u9Z7geww0711rVpN28oXl6FsSVX_0NzyQO0ybaRfLZf-9PB4NhNwo8SJTsL7DNbFA2NjxlRsLGOn2Ii28FTsMuEQ30QaYCkWIRxhtJBZDsrOSBmyBeF0DoH5lwyX6JU58-fVLcNwHVbYqdCO8T5Ehv1XEbO_KMyZX19igF4&ind=I7086_4,I7086_5,I7086_6,I7086_7,I7086_8,I7086_9,I7086_10,I7086_11,I7086_12,I7086_13,I7086_14,I7086_15,I7086_16,I7086_17,I7086_18,I7086_19,I7086_20,I7086_21,I7086_22,I7086_23,I7086_24,I7086_25,I7086_26,I7086_27,I7086_28,I7086_29,I7086_30,I7086_31,I7086_32,I7086_33&dim=Country,StateName,StateCode,DistrictName,DistrictCode,Year&pageno=1",
    },
    
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Resolve API parameters for one dataset
# ─────────────────────────────────────────────────────────────────────────────

def _parse_manual_url(url: str, fallback_key: str) :
    """
    Parse  ind=  dim=  API_Key=  from a manually pasted full API URL.
    Returns a params dict or None.
    """
    from urllib.parse import urlparse, parse_qs
    p  = urlparse(url)
    qs = parse_qs(p.query)
    ak = (qs.get("API_Key") or qs.get("api_key") or [""])[0].strip()
    ind = (qs.get("ind") or [""])[0].strip()
    dim = (qs.get("dim") or [""])[0].strip()
    if ind and dim:
        # Use the API key from the manual URL if available, otherwise fallback
        api_key = ak or fallback_key
        return {
            "api_key" : api_key,
            "ind"     : ind,
            "dim"     : dim,
            "api_base": f"{p.scheme}://{p.netloc}{p.path}",
        }
    return None


def resolve_params(
    session      : requests.Session,
    dataset      : dict,
    fallback_key : str,
) :
    """
    Return the API call parameters (api_key, ind, dim, api_base) for a dataset.

    Priority:
      1. manual_api_url in DATASET_REGISTRY  (if set and parseable)
      2. Live scrape of get-api page          (ndap_page_scraper)
      3. Fallback_key + scrape fails          → return None, skip dataset
    """
    did  = dataset["dataset_id"]
    slug = dataset["slug"]

    # ── 1. Manual URL override ────────────────────────────────────────────────
    manual = dataset.get("manual_api_url", "").strip()
    if manual:
        params = _parse_manual_url(manual, fallback_key)
        if params:
            log.info(f"[{slug}] Using manual URL parameters")
            log.info(f"[{slug}]   ind = {params['ind']}")
            log.info(f"[{slug}]   dim = {params['dim']}")
            return params
            return params

    # ── 2. Live scrape ────────────────────────────────────────────────────────
    scraped = scrape_dataset_params(session, did)
    if scraped:
        # Use the scraped API key, or fallback if none
        if not scraped.get("api_key") and fallback_key:
            scraped["api_key"] = fallback_key
        return scraped

    log.error(
        f"[{slug}] Cannot resolve ind/dim for dataset {did}.\n"
        f"  → Open https://ndap.niti.gov.in/get-api/{did} in a browser,\n"
        f"    click 'Copy API', paste the full URL into the 'manual_api_url'\n"
        f"    field for this dataset in DATASET_REGISTRY inside ndap_simple.py."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Paginated data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dataset(
    session : requests.Session,
    dataset : dict,
    params  : dict,
) :
    """
    Download all pages for one dataset using the NDAP OpenAPI endpoint.

    NDAP response structure (confirmed from ndap_simple.py):
      {
        "IsError" : false,
        "Message" : "...",
        "Data"    : [ { col: val, ... }, ... ]   ← capital D
      }
    Pagination: keep incrementing pageno until Data is an empty list.
    """
    slug   = dataset["slug"]
    did    = dataset["dataset_id"]
    ak     = params["api_key"]
    ind    = params["ind"]
    dim    = params["dim"]
    base   = params.get("api_base", API_BASE)

    log.info("")
    log.info("─" * 70)
    log.info(f"Fetching  [{did}]  {dataset['description']}")
    log.info(f"  ind  = {ind}")
    log.info(f"  dim  = {dim}")
    log.info(f"  base = {base}")
    log.info("─" * 70)

    all_rows = []
    page     = 1   # NDAP uses 1-indexed pages

    while True:
        call_params = {
            "API_Key": ak,
            "ind"    : ind,
            "dim"    : dim,
            "pageno" : page,
        }

        # ── retry loop ────────────────────────────────────────────────────────
        success = False
        for attempt in range(1, RETRY_MAX + 1):
            try:
                resp = session.get(base, params=call_params, timeout=TIMEOUT)

                if resp.status_code == 200:
                    data = resp.json()

                    if data.get("IsError"):
                        msg = data.get("Message", "Unknown API error")
                        log.error(f"[{slug}] API error on page {page}: {msg}")
                        # Common cause: expired API_Key → tell user
                        if "key" in msg.lower() or "auth" in msg.lower() or "token" in msg.lower():
                            log.error(
                                "  API_Key may have expired. Refresh it:\n"
                                "  1. Log in at https://ndap.niti.gov.in\n"
                                "  2. Open any /get-api/<id> page\n"
                                "  3. Click 'Copy API' and re-paste the key"
                            )
                        return pd.DataFrame(all_rows) if all_rows else None

                    rows = data.get("Data", [])

                    if not isinstance(rows, list):
                        log.error(f"[{slug}] 'Data' is not a list: {type(rows)}")
                        return pd.DataFrame(all_rows) if all_rows else None

                    if not rows:
                        log.info(f"[{slug}] Page {page}: empty -> all pages fetched")
                        success = True
                        break   # done with this dataset

                    all_rows.extend(rows)
                    log.info(
                        f"[{slug}] Page {page:>4}: {len(rows):>5} rows  "
                        f"(running total: {len(all_rows):,})"
                    )
                    page   += 1
                    success = True
                    break   # move to next page

                elif resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", RETRY_WAIT * attempt * 2))
                    log.warning(f"[{slug}] Rate-limited. Waiting {wait}s (attempt {attempt})…")
                    time.sleep(wait)

                else:
                    log.warning(
                        f"[{slug}] HTTP {resp.status_code} on page {page} "
                        f"(attempt {attempt}/{RETRY_MAX})"
                    )
                    if attempt < RETRY_MAX:
                        time.sleep(RETRY_WAIT * attempt)

            except Exception as e:
                log.warning(f"[{slug}] Exception on page {page} attempt {attempt}: {e}")
                if attempt < RETRY_MAX:
                    time.sleep(RETRY_WAIT * attempt)

        if not success:
            log.error(f"[{slug}] All {RETRY_MAX} retries failed on page {page}. Stopping.")
            break

        # If we got an empty page, the while-loop's break already exited
        if rows == [] if 'rows' in dir() else False:
            break

        time.sleep(PAGE_DELAY)

    if not all_rows:
        log.warning(f"[{slug}] No rows retrieved for dataset {did}.")
        return None

    df = pd.DataFrame(all_rows)
    log.info(f"[{slug}] OK  {len(df):,} rows x {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Save CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, dataset: dict) -> Path:
    """
    Save to  csv/<dataset_id>_<slug>.csv
    Also writes a copy to  ndap_data/<slug>.csv  for backwards compatibility.
    """
    slug = dataset["slug"]
    did  = dataset["dataset_id"]

    # Primary output: csv/ folder
    primary = CSV_DIR / f"{did}_{slug}.csv"
    df.to_csv(primary, index=False, encoding="utf-8-sig")
    kb = primary.stat().st_size // 1024
    log.info(f"  OK csv/{primary.name}  ({len(df):,} rows, {kb} KB)")

    # Legacy copy: ndap_data/ folder
    legacy = NDAP_DIR / f"{slug}.csv"
    df.to_csv(legacy, index=False, encoding="utf-8-sig")

    return primary

## SAVE TO DUCKDB 
import duckdb

DUCK_DB_PATH = "database.duckdb"
con = duckdb.connect(DUCK_DB_PATH)

def save_to_duckdb(df: pd.DataFrame, table_name: str):
    try:
        # Drop if exists
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        # Register dataframe temporarily
        con.register("temp_df", df)

        # Create table
        con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM temp_df')

        # Row count
        row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

        # written.append(table_name)

        # print(f"  ✅ {label}")
        print(f"      → table: {table_name} | {row_count} rows × {len(df.columns)} cols")
        print(f"         cols: {list(df.columns)}\n")

    except Exception as e:
        print(f"  ❌ Failed to write {table_name}: {e}")
        # skipped.append(table_name)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("  NDAP Multi-Dataset Scraper  (No Selenium)")
    log.info(f"  Datasets : {len(DATASET_REGISTRY)}")
    log.info(f"  CSV dir  : {CSV_DIR.resolve()}")
    log.info("=" * 70)

    # ── Get API_Key ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Resolving API_Key…")
    print("=" * 70)
    api_key = get_api_key(force_input=False)
    if not api_key:
        api_key = get_api_key(force_input=True)
    if not api_key:
        log.error("No API_Key available. Exiting.")
        sys.exit(1)
    print(f"✓ API_Key: {api_key[:40]}…\n")

    # ── Build shared session ───────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update({
        "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept"         : "application/json",
        "Referer"        : "https://ndap.niti.gov.in/",
    })

    # ── Process each dataset ───────────────────────────────────────────────────
    results = {}

    for ds in DATASET_REGISTRY:
        slug = ds["slug"]
        did  = ds["dataset_id"]

        log.info("")
        log.info("═" * 70)
        log.info(f"  [{did}]  {ds['description']}")
        log.info("═" * 70)

        # Step 1: resolve parameters (ind, dim) from get-api page
        params = resolve_params(session, ds, api_key)
        if not params:
            results[slug] = {"status": "✗ PARAM_FAIL", "rows": 0}
            continue

        # Step 2: download all pages
        df = fetch_dataset(session, ds, params)
        if df is None or df.empty:
            results[slug] = {"status": "✗ NO_DATA",   "rows": 0}
            continue
        
        df = preprocess_dataset(df, slug)

        # Step 3: save CSV
        save_csv(df, ds)
        save_to_duckdb(df, ds["slug"])
        results[slug] = {"status": "OK SUCCESS", "rows": len(df)}

        time.sleep(0.5)

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)

    success = 0
    for ds in DATASET_REGISTRY:
        slug = ds["slug"]
        r    = results.get(slug, {"status": "✗ SKIPPED", "rows": 0})
        rows = f"{r['rows']:>8,} rows" if r['rows'] > 0 else "       —"
        log.info(f"  [{ds['dataset_id']:>5}]  {r['status']:<12}  {rows}   {slug}")
        if r["status"].startswith("OK"):
            success += 1

    log.info("")
    log.info(f"  {success}/{len(DATASET_REGISTRY)} datasets fetched successfully")
    log.info(f"  CSVs in : {CSV_DIR.resolve()}")
    log.info(f"  Log     : {LOG_FILE.resolve()}")
    log.info("=" * 70)

    # Print the csv/ folder contents
    print("\nFiles written to csv/:")
    for f in sorted(CSV_DIR.glob("*.csv")):
        kb = f.stat().st_size // 1024
        print(f"  {f.name:<55}  {kb:>6} KB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — single-dataset mode
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one(dataset_id: int) :
    """
    Convenience function to fetch a single dataset by its numeric ID.
    Uses the matching entry in DATASET_REGISTRY if found, otherwise creates
    a minimal entry and uses live page scraping.

    Example:
        df = fetch_one(9022)
        df.to_csv("slum.csv", index=False)
    """
    # Find registry entry or create a stub
    ds = next(
        (d for d in DATASET_REGISTRY if d["dataset_id"] == dataset_id),
        {
            "dataset_id"    : dataset_id,
            "slug"          : f"dataset_{dataset_id}",
            "description"   : f"NDAP Dataset {dataset_id}",
            "manual_api_url": "",
        }
    )

    api_key = get_api_key(force_input=False) or get_api_key(force_input=True)
    if not api_key:
        log.error("No API_Key available.")
        return None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 Chrome/124.0.0.0",
        "Accept"    : "application/json",
    })

    params = resolve_params(session, ds, api_key)
    if not params:
        return None

    df = fetch_dataset(session, ds, params)
    
    if df is not None and not df.empty:
        save_csv(df, ds)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NDAP Multi-Dataset Scraper")
    parser.add_argument("--id",  type=int, help="Fetch a single dataset by ID")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Fetch all datasets in registry (default)")
    args = parser.parse_args()

    if args.id:
        df = fetch_one(args.id)
        if df is not None:
            print(f"\n✓  {len(df):,} rows × {len(df.columns)} columns")
            print(df.head(3).to_string())
    else:
        main()