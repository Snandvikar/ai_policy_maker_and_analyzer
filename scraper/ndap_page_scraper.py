"""
NDAP Get-API Page Scraper
─────────────────────────
Visits  https://ndap.niti.gov.in/get-api/<dataset_id>
and scrapes the pre-built API URL that NDAP shows on that page.
That URL contains the exact:
  • API_Key  (fresh, valid token)
  • ind      (comma-separated indicator codes, e.g. I9022_3)
  • dim      (comma-separated dimension names)

This is the correct way to discover indicators and dimensions
WITHOUT assumptions – every dataset has its own combination.

Returns a dict:
  {
    "dataset_id" : 9022,
    "api_key"    : "gAAAAABp...",
    "ind"        : "I9022_3",
    "dim"        : "Country,StateName,StateCode,Year",
    "api_url"    : "https://loadqa.ndapapi.com/v1/openapi?...",
  }

How NDAP renders the page
─────────────────────────
The /get-api/<id> page is a React SPA, so the main HTML is just
  <div id="root"></div>
However, NDAP also injects a <script> tag or makes an XHR call to
  https://ndap.niti.gov.in/api/v2/dataset/<id>/get-api
which returns JSON with the pre-built API URL.  We call that JSON
endpoint directly – no Selenium needed.

Fallback: if the JSON endpoint fails, we try the raw HTML and
regex-search for the loadqa.ndapapi.com URL that NDAP embeds.
"""

import re
import time
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional

import requests

log = logging.getLogger("ndap_page_scraper")

# ── NDAP internal REST endpoints that return the pre-built API URL ────────────
# Pattern A  – dataset metadata JSON (most reliable)
API_META_URL  = "https://ndap.niti.gov.in/api/v2/dataset/{dataset_id}/get-api"
API_META_URL2 = "https://ndap.niti.gov.in/api/dataset/{dataset_id}/getapi"
API_META_URL3 = "https://ndap.niti.gov.in/ndap-api/dataset/{dataset_id}/api-details"

# Pattern B  – the page HTML sometimes contains the full URL in a data attribute
PAGE_URL      = "https://ndap.niti.gov.in/get-api/{dataset_id}"

TIMEOUT  = 20
DELAY    = 0.4   # polite pause between requests


# ── regex to find the openapi URL anywhere in text ────────────────────────────
_OPENAPI_RE = re.compile(
    r'(https?://(?:loadqa\.ndapapi\.com|kdap\.ndapapi\.com|[^"\'>\s]*ndapapi\.com)'
    r'/v\d+/openapi[^"\'>\s&]*)',
    re.IGNORECASE,
)


def _parse_openapi_url(url: str) -> Optional[dict]:
    """
    Extract API_Key, ind, dim from a full loadqa.ndapapi.com URL.
    Returns None if any required parameter is missing.
    """
    parsed = urlparse(url)
    qs     = parse_qs(parsed.query)

    api_key = (qs.get("API_Key") or qs.get("api_key") or [""])[0].strip()
    ind     = (qs.get("ind")     or [""])[0].strip()
    dim     = (qs.get("dim")     or [""])[0].strip()

    if not api_key or not ind or not dim:
        return None

    # Reconstruct the clean base URL (without pageno so caller adds it)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    return {
        "api_key" : api_key,
        "ind"     : ind,
        "dim"     : dim,
        "api_base": base,
        "raw_url" : url,
    }


def scrape_dataset_params(
    session: requests.Session,
    dataset_id: int,
) -> Optional[dict]:
    """
    Discover indicators, dimensions, and a valid API_Key for a dataset
    by scraping its /get-api/<dataset_id> page (no Selenium).

    Strategy (tried in order until one succeeds):
      1. Call the internal JSON metadata endpoint → parse JSON for the URL
      2. Fetch the raw page HTML → regex-search for the openapi URL
      3. Try alternate internal endpoints

    Returns a dict with keys: dataset_id, api_key, ind, dim, api_base
    Returns None if all attempts fail.
    """
    log.info(f"[{dataset_id}] Scraping get-api page for indicators & dimensions…")

    headers = {
        "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept"         : "application/json, text/html, */*",
        "Referer"        : f"https://ndap.niti.gov.in/get-api/{dataset_id}",
        "Origin"         : "https://ndap.niti.gov.in",
    }

    # ── Strategy 1: JSON metadata endpoints ──────────────────────────────────
    meta_endpoints = [
        API_META_URL.format(dataset_id=dataset_id),
        API_META_URL2.format(dataset_id=dataset_id),
        API_META_URL3.format(dataset_id=dataset_id),
    ]

    for endpoint in meta_endpoints:
        try:
            r = session.get(endpoint, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    continue

                # Hunt for a URL in the JSON at any depth
                text = str(data)
                match = _OPENAPI_RE.search(text)
                if match:
                    parsed = _parse_openapi_url(match.group(1))
                    if parsed:
                        parsed["dataset_id"] = dataset_id
                        log.info(f"[{dataset_id}] ✓ Found via JSON endpoint: {endpoint}")
                        log.info(f"[{dataset_id}]   ind={parsed['ind']}")
                        log.info(f"[{dataset_id}]   dim={parsed['dim']}")
                        return parsed

        except Exception as e:
            log.debug(f"[{dataset_id}] Meta endpoint {endpoint} failed: {e}")
        time.sleep(DELAY)

    # ── Strategy 2: HTML page (React SPA – usually just a shell, but worth a try) ─
    page_url = PAGE_URL.format(dataset_id=dataset_id)
    try:
        r = session.get(page_url, headers={**headers, "Accept": "text/html"}, timeout=TIMEOUT)
        if r.status_code == 200:
            match = _OPENAPI_RE.search(r.text)
            if match:
                parsed = _parse_openapi_url(match.group(1))
                if parsed:
                    parsed["dataset_id"] = dataset_id
                    log.info(f"[{dataset_id}] ✓ Found openapi URL in page HTML")
                    return parsed
    except Exception as e:
        log.debug(f"[{dataset_id}] HTML page fetch failed: {e}")

    # ── Strategy 3: NDAP internal React bundle references ────────────────────
    # Some NDAP pages expose the API URL through a data- attribute or JS bundle
    alt_endpoints = [
        f"https://ndap.niti.gov.in/api/dataset/{dataset_id}",
        f"https://ndap.niti.gov.in/api/v1/dataset/{dataset_id}",
        f"https://ndap.niti.gov.in/api/openapi/{dataset_id}",
    ]
    for endpoint in alt_endpoints:
        try:
            r = session.get(endpoint, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                text = r.text
                match = _OPENAPI_RE.search(text)
                if match:
                    parsed = _parse_openapi_url(match.group(1))
                    if parsed:
                        parsed["dataset_id"] = dataset_id
                        log.info(f"[{dataset_id}] ✓ Found via alt endpoint {endpoint}")
                        return parsed
        except Exception as e:
            log.debug(f"[{dataset_id}] Alt endpoint {endpoint} failed: {e}")
        time.sleep(DELAY)

    log.warning(
        f"[{dataset_id}] ✗ Could not scrape parameters from get-api page.\n"
        f"           Manual fallback: open https://ndap.niti.gov.in/get-api/{dataset_id}\n"
        f"           in a browser, click 'Copy API', and paste the URL into DATASET_REGISTRY\n"
        f"           in ndap_scraper.py  as  'manual_api_url'."
    )
    return None


def scrape_multiple(
    session: requests.Session,
    dataset_ids: list[int],
) -> dict[int, dict]:
    """
    Scrape parameters for a list of dataset IDs.
    Returns {dataset_id: params_dict}  (missing IDs are skipped).
    """
    results = {}
    for did in dataset_ids:
        params = scrape_dataset_params(session, did)
        if params:
            results[did] = params
        time.sleep(DELAY)
    return results