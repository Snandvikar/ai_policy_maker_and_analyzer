
# ── Standard library ─────────────────────────────────────────────────────────
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

warnings.filterwarnings("ignore")

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    import tabula
    TABULA_OK = True
except ImportError:
    TABULA_OK = False


# =============================================================================
# SECTION 0 — CONSTANTS
# =============================================================================

# ── 56-city canonical list ────────────────────────────────────────────────────
CITIES = [
    "Agartala","Agra","Ahmedabad","Ajmer","Allahabad","Amritsar","Asansol",
    "Aurangabad","Bareilly","Bhopal","Bhubaneswar","Chandigarh","Chennai",
    "Coimbatore","Dehradun","Delhi","Dhanbad","Durgapur","Faridabad",
    "Ghaziabad","Guwahati","Gwalior","Howrah","Hubli-Dharwad","Hyderabad",
    "Indore","Jabalpur","Jaipur","Jalandhar","Jammu","Jodhpur","Kanpur",
    "Kochi","Kolkata","Kozhikode","Lucknow","Ludhiana","Madurai","Meerut",
    "Mumbai","Mysuru","Nagpur","Nashik","Patna","Pimpri-Chinchwad","Pune",
    "Raipur","Rajkot","Ranchi","Shillong","Srinagar","Surat",
    "Thiruvananthapuram","Varanasi","Vijayawada","Visakhapatnam",
]

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# ── TRAI service-area → list of (city, weight) ───────────────────────────────
# Weights represent each city's share of the service-area population.
# Multi-city areas are disaggregated by urban population share.
AREA_TO_CITIES: Dict[str, List[Tuple[str, float]]] = {
    "Maharashtra":      [("Mumbai",0.35),("Pune",0.18),("Nagpur",0.09),
                         ("Nashik",0.07),("Aurangabad",0.06),("Pimpri-Chinchwad",0.08)],
    "Mumbai":           [("Mumbai",0.72),("Pimpri-Chinchwad",0.28)],
    "Delhi":            [("Delhi",0.68),("Faridabad",0.14),("Ghaziabad",0.10),("Meerut",0.08)],
    "Karnataka":        [("Bengaluru",0.60),("Hubli-Dharwad",0.12),("Mysuru",0.12)],
    "Andhra Pradesh":   [("Visakhapatnam",0.38),("Vijayawada",0.32)],
    "Tamil Nadu":       [("Chennai",0.55),("Coimbatore",0.20),("Madurai",0.14)],
    "West Bengal":      [("Kolkata",0.55),("Howrah",0.22),("Asansol",0.12),("Durgapur",0.11)],
    "Kolkata":          [("Kolkata",0.80),("Howrah",0.20)],
    "Rajasthan":        [("Jaipur",0.48),("Jodhpur",0.22),("Ajmer",0.14)],
    "UP (East)":        [("Lucknow",0.25),("Varanasi",0.20),("Allahabad",0.20),
                         ("Kanpur",0.15),("Bareilly",0.12)],
    "UP (West)":        [("Agra",0.30),("Meerut",0.28),("Ghaziabad",0.25)],
    "Gujarat":          [("Ahmedabad",0.44),("Surat",0.32),("Rajkot",0.16)],
    "Madhya Pradesh":   [("Indore",0.40),("Bhopal",0.35),("Gwalior",0.15),("Jabalpur",0.10)],
    "Punjab":           [("Ludhiana",0.40),("Amritsar",0.32),("Jalandhar",0.22)],
    "Bihar & Jharkhand":[("Patna",0.45),("Ranchi",0.28),("Dhanbad",0.20)],
    "Assam":            [("Guwahati",0.68),("Shillong",0.18)],
    "Kerala":           [("Kochi",0.40),("Thiruvananthapuram",0.38),("Kozhikode",0.22)],
    "Haryana":          [("Faridabad",0.48),("Chandigarh",0.32)],
    "Orissa":           [("Bhubaneswar",0.70)],
    "North East":       [("Shillong",0.32),("Agartala",0.28),("Guwahati",0.28)],
    "Himachal Pradesh": [("Shimla",0.75)],
    "Jammu & Kashmir":  [("Srinagar",0.55),("Jammu",0.42)],
    "Telangana":        [("Hyderabad",0.80)],
    "Uttarakhand":      [("Dehradun",0.68)],
    "Chhattisgarh":     [("Raipur",0.70)],
    "Meghalaya":        [("Shillong",0.80)],
    "Tripura":          [("Agartala",0.82)],
}

# ── Direct PDF download URLs (from TRAI website structure) ────────────────────
# Format: report_label → (url, year, report_type)
TRAI_PDF_CATALOGUE = {
    # ── Annual PIR (Performance Indicator Reports) ──
    "PIR_2023_24_Annual": (
        "https://www.trai.gov.in/sites/default/files/2024-09/Report_14082024.pdf",
        2024, "pir_annual"),
    "PIR_2022_23_Annual": (
        "https://www.trai.gov.in/sites/default/files/Annual_Report_2022-23_12032024.pdf",
        2023, "pir_annual"),
    "PIR_2021_22_Annual": (
        "https://www.trai.gov.in/sites/default/files/Annual_Report_2021-22.pdf",
        2022, "pir_annual"),
    "PIR_2020_21_Annual": (
        "https://www.trai.gov.in/sites/default/files/Annual_Report_2020-21_16112021.pdf",
        2021, "pir_annual"),
    "PIR_2019_20_Annual": (
        "https://www.trai.gov.in/sites/default/files/Annual_Report_2019-20.pdf",
        2020, "pir_annual"),
    # ── Quarterly PIR (latest quarters) ──
    "PIR_Q2_2024_25": (
        "https://www.trai.gov.in/sites/default/files/2025-01/PIR_Jul_Sep_2024.pdf",
        2024, "pir_quarterly"),
    "PIR_Q3_2024_25": (
        "https://www.trai.gov.in/sites/default/files/2025-04/PIR_Oct_Dec_2024.pdf",
        2024, "pir_quarterly"),
    # ── Telecom Subscription ──
    "Subscription_Nov2024": (
        "https://www.trai.gov.in/sites/default/files/Telecom_Subscription_Data_Nov2024.pdf",
        2024, "subscription"),
}

# ── PIB Press release pages (contain embedded tables in HTML) ─────────────────
PIB_URLS = {
    2024: [
        "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2089361",   # Q2 Jul-Sep 2024
        "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2124056",   # Q3 Oct-Dec 2024
    ],
    2023: [
        "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2018587",  # Q3 Oct-Dec 2023
    ],
}

# ── TRAI website listing pages ────────────────────────────────────────────────
TRAI_LISTING_URLS = {
    "pir":     "https://www.trai.gov.in/release-publication/reports/performance-indicators-reports",
    "annual":  "https://www.trai.gov.in/about-us/annual-reports",
    "broadband":"https://www.trai.gov.in/release-publication/reports/broadband-reports",
    "telecom_sub":"https://www.trai.gov.in/release-publication/reports/telecom-subscriptions-reports",
}

# ── data.gov.in OGD resource IDs ──────────────────────────────────────────────
OGD_RESOURCES = {
    "trai_myspeed":   "5b78dce5",   # TRAI MySpeed broadband speed data
    "pmjdy_monthly":  "9ef84268",   # PMJDY month-wise financial inclusion
    "gsdp_statewise": "b8a40c8c",   # MoSPI GSDP state-wise
    "plfs_lfpr":      "f407eb22",   # PLFS urban LFPR employment
}
OGD_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"

OUTPUT_FILE = "trai_5factor_output.xlsx"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trai_scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("trai")


# =============================================================================
# SECTION 1 — HTTP SESSION
# =============================================================================

def make_session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """
    Build a requests Session with:
      - Exponential-backoff retry on 429, 500–504
      - Browser-mimicking headers to bypass 403 blocks on government portals
      - Connection and read timeouts
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Referer":         "https://www.trai.gov.in/",
    })
    return session


SESSION = make_session()
_FAILED_URLS: List[str] = []   # track failures for summary report


# =============================================================================
# SECTION 2 — CORE FETCHERS
# =============================================================================

def fetch_pdf_bytes(url: str, label: str = "") -> Optional[bytes]:
    """
    Download a PDF and return raw bytes.

    Steps:
      1. HEAD request to verify Content-Type before full download
      2. Stream GET with 120s read timeout
      3. Log size and return bytes; log and return None on any failure

    Args:
        url   : full HTTPS URL to the PDF
        label : human-readable label for log messages

    Returns:
        bytes if successful, None otherwise
    """
    log.info("  [PDF↓]  %s", url)
    try:
        # Quick HEAD to validate URL before committing to full download
        head = SESSION.head(url, timeout=(10, 10), allow_redirects=True)
        ct = head.headers.get("Content-Type", "")
        if head.status_code >= 400:
            log.warning("          HEAD %d — skipping %s", head.status_code, label)
            _FAILED_URLS.append(url)
            return None
        if "text/html" in ct and ".pdf" not in url.lower():
            log.warning("          Got HTML not PDF at %s", url)
            _FAILED_URLS.append(url)
            return None

        # Stream download
        r = SESSION.get(url, timeout=(30, 120), stream=True)
        r.raise_for_status()
        raw = b"".join(r.iter_content(chunk_size=65536))
        log.info("          → %d KB  (%s)", len(raw) // 1024, label)
        time.sleep(0.8)   # polite delay
        return raw

    except Exception as exc:
        log.error("          FAILED %s — %s", label or url, exc)
        _FAILED_URLS.append(url)
        return None


def fetch_html(url: str, label: str = "") -> Optional[BeautifulSoup]:
    """
    Fetch an HTML page and return a parsed BeautifulSoup object.

    Args:
        url   : page URL
        label : log label

    Returns:
        BeautifulSoup or None
    """
    log.info("  [HTML]  %s", url)
    try:
        r = SESSION.get(url, timeout=(15, 30))
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        log.info("          → %d chars parsed", len(r.text))
        time.sleep(0.5)
        return soup
    except Exception as exc:
        log.error("          FAILED %s — %s", label or url, exc)
        _FAILED_URLS.append(url)
        return None


def fetch_ogd_api(resource_id: str, label: str = "",
                  max_records: int = 10_000) -> Optional[pd.DataFrame]:
    """
    Fetch all records from India's Open Government Data (data.gov.in) API.
    Paginates automatically with offset until all records are retrieved.

    API base: https://api.data.gov.in/resource/{resource_id}
    Params:   api-key, format=json, limit, offset

    Args:
        resource_id  : data.gov.in resource identifier string
        label        : human-readable label
        max_records  : safety cap to prevent runaway pagination

    Returns:
        DataFrame of all records, or None on failure
    """
    base = f"https://api.data.gov.in/resource/{resource_id}"
    all_records: List[dict] = []
    offset, limit = 0, 500
    log.info("  [OGD]   resource_id=%s  (%s)", resource_id, label)

    try:
        while len(all_records) < max_records:
            r = SESSION.get(base, params={
                "api-key": OGD_API_KEY,
                "format":  "json",
                "limit":   limit,
                "offset":  offset,
            }, timeout=(15, 30))
            r.raise_for_status()
            payload = r.json()

            # data.gov.in nests results under different keys by version
            records = (
                payload.get("records") or
                payload.get("data")    or
                payload.get("result")  or
                (payload if isinstance(payload, list) else [])
            )
            if not records:
                break
            all_records.extend(records)
            log.info("          offset=%d  chunk=%d  total=%d",
                     offset, len(records), len(all_records))
            if len(records) < limit:
                break
            offset += limit
            time.sleep(0.4)

        if all_records:
            return pd.json_normalize(all_records)
        log.warning("          No records returned for %s", label)
        return None

    except Exception as exc:
        log.error("          FAILED OGD %s — %s", label, exc)
        _FAILED_URLS.append(base)
        return None


# =============================================================================
# SECTION 3 — TRAI WEBSITE LINK DISCOVERY
# =============================================================================

def scrape_trai_listing(page_key: str = "pir") -> Dict[str, Tuple[str, int, str]]:
    """
    Scrape a TRAI report listing page to discover all PDF download links.

    TRAI HTML structure:
      <div class="view-content">
        <div class="views-row">
          <span class="date-display-single">23/04/2024</span>
          <a href="/sites/default/files/...pdf">Download (4.47 MB)</a>
        </div>
      </div>

    Args:
        page_key : key into TRAI_LISTING_URLS

    Returns:
        dict  label → (absolute_pdf_url, year, report_type)
    """
    url  = TRAI_LISTING_URLS.get(page_key, TRAI_LISTING_URLS["pir"])
    soup = fetch_html(url, f"TRAI_{page_key}_listing")
    if soup is None:
        return {}

    result: Dict[str, Tuple[str, int, str]] = {}

    # Every <a> with href ending in .pdf is a report download link
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not re.search(r"\.pdf$", href, re.IGNORECASE):
            continue

        # Extract context text from parent elements for year detection
        parent_text = ""
        el = anchor
        for _ in range(4):
            el = el.parent
            if el is None:
                break
            parent_text = el.get_text(" ", strip=True)[:200]

        full_text = parent_text + " " + href

        # Detect year range like "2023-24" or "2024-2025"
        yr_match = re.search(r"20(\d{2})[-–](\d{2,4})", full_text)
        if yr_match:
            year = int("20" + yr_match.group(1))   # use start year
            yr_label = yr_match.group(0).replace("–", "-")
        else:
            yr_single = re.search(r"(20\d{2})", full_text)
            year = int(yr_single.group(1)) if yr_single else 0
            yr_label = str(year)

        # Detect quarterly vs annual
        rtype = "pir_annual"
        if re.search(r"quarter|Q[1-4]|Jan|Feb|Mar|Apr|Jul|Oct", full_text, re.I):
            rtype = "pir_quarterly"
        elif re.search(r"annual|yearly", full_text, re.I):
            rtype = "pir_annual"

        abs_url = urljoin("https://www.trai.gov.in", href)
        label   = f"{page_key.upper()}_{yr_label}_{rtype}"

        if label not in result and year >= 2019:
            result[label] = (abs_url, year, rtype)
            log.info("      FOUND: %-40s → %s", label, abs_url)

    log.info("  [DISC]  Discovered %d PDF links from %s", len(result), url)
    return result


def scrape_pib_press_release(url: str, year: int) -> Optional[pd.DataFrame]:
    """
    Extract numeric summary tables from a PIB (Press Information Bureau) page.

    PIB press releases for TRAI PIR contain snapshot tables like:
      "Telecom Subscribers (Wireless+Wireline)  Total: 1,199.28 Million"
      "Urban Tele-density: 133.72%"

    Strategy:
      1. Parse all <table> elements with pandas.read_html
      2. Also regex-extract key-value pairs from <p> and <li> elements
      3. Return a DataFrame of metric → value

    Args:
        url  : full PIB URL
        year : year for the year column

    Returns:
        DataFrame with columns: metric, value, unit, year, source
    """
    soup = fetch_html(url, f"PIB_{year}")
    if soup is None:
        return None

    rows: List[dict] = []

    # ── Strategy A: extract <table> elements ──────────────────────────────────
    for table in soup.find_all("table"):
        try:
            df_tbl = pd.read_html(str(table))[0]
            df_tbl.columns = [str(c).strip() for c in df_tbl.columns]
            for _, row in df_tbl.iterrows():
                vals = [str(v).strip() for v in row.values if str(v).strip() not in ("", "nan")]
                if len(vals) >= 2:
                    rows.append({
                        "metric": vals[0],
                        "value":  vals[1],
                        "unit":   vals[2] if len(vals) > 2 else "",
                        "year":   year,
                        "source": url,
                    })
        except Exception:
            pass

    # ── Strategy B: regex extraction from free text ───────────────────────────
    body_text = soup.get_text(" ", strip=True)

    # Pattern: "Urban Tele-density: 133.72%" or "Total Subscribers: 1,199.28 Million"
    kv_pattern = re.compile(
        r"([A-Za-z][A-Za-z\s\(\)\/\-]{3,50}?)"  # metric name
        r"[:\s]+"
        r"([\d,]+\.?\d*)"                          # numeric value
        r"\s*([%MKBmkb]?(?:illion|rore|lakh)?)",  # optional unit
        re.IGNORECASE
    )
    for m in kv_pattern.finditer(body_text):
        metric = m.group(1).strip()
        value  = m.group(2).replace(",", "")
        unit   = m.group(3).strip()
        if len(metric) > 5:  # skip very short garbage matches
            rows.append({
                "metric": metric,
                "value":  value,
                "unit":   unit,
                "year":   year,
                "source": url,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["metric","year"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# =============================================================================
# SECTION 4 — PDF PARSERS
# =============================================================================

# ── Table extraction settings tuned for TRAI reports ─────────────────────────
# TRAI PDFs use ruled-line tables with 5-pt snap tolerance
_PDFPLUMBER_SETTINGS = {
    "vertical_strategy":      "lines",
    "horizontal_strategy":    "lines",
    "snap_tolerance":         5,
    "join_tolerance":         3,
    "intersection_tolerance": 5,
    "text_tolerance":         3,
}

# ── Fallback settings (text-based — for reports without visible rules) ────────
_PDFPLUMBER_TEXT_SETTINGS = {
    "vertical_strategy":   "text",
    "horizontal_strategy": "text",
    "snap_tolerance":      3,
}


def extract_all_tables(pdf_bytes: bytes,
                       page_range: Optional[Tuple[int, int]] = None
                       ) -> List[pd.DataFrame]:
    """
    Extract every table from a PDF using pdfplumber.

    Strategy (in order):
      1. Line-based extraction (works for TRAI ruled tables)
      2. Text-based fallback (catches implicit tables without borders)
      3. tabula-py fallback if pdfplumber finds nothing

    Args:
        pdf_bytes  : raw PDF content bytes
        page_range : (start, end) 0-indexed page tuple, or None for all pages

    Returns:
        List of raw DataFrames (one per detected table)
    """
    if not PDF_OK:
        log.error("pdfplumber not installed — pip install pdfplumber")
        return []

    frames: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            if page_range:
                pages = pdf.pages[page_range[0]:page_range[1]]
            else:
                pages = pdf.pages
            log.info("          PDF: %d total pages, scanning %d",
                     total_pages, len(pages))

            for pg_idx, page in enumerate(pages):
                # ── Pass 1: line-based ──────────────────────────────────────
                tables = page.extract_tables(_PDFPLUMBER_SETTINGS)

                # ── Pass 2: text-based fallback ─────────────────────────────
                if not tables:
                    tables = page.extract_tables(_PDFPLUMBER_TEXT_SETTINGS)

                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue

                    # Build DataFrame: first row → header, rest → data
                    header = [
                        str(c).strip().replace("\n", " ") if c else f"col_{i}"
                        for i, c in enumerate(tbl[0])
                    ]
                    data = [
                        [str(v).strip().replace("\n", " ") if v else "" for v in row]
                        for row in tbl[1:]
                    ]
                    if any(len(r) != len(header) for r in data):
                        continue   # malformed table — skip

                    df = pd.DataFrame(data, columns=header)
                    df["_page_num"] = pg_idx + (page_range[0] if page_range else 0) + 1
                    frames.append(df)

        # ── Pass 3: tabula fallback if nothing found ──────────────────────────
        if not frames and TABULA_OK:
            log.info("          pdfplumber found 0 tables — trying tabula fallback")
            try:
                tabula_frames = tabula.read_pdf(
                    io.BytesIO(pdf_bytes),
                    pages="all",
                    multiple_tables=True,
                    silent=True,
                    lattice=True,
                )
                for tf in tabula_frames:
                    if isinstance(tf, pd.DataFrame) and not tf.empty:
                        tf["_page_num"] = 0
                        frames.append(tf)
            except Exception as e:
                log.warning("          tabula fallback failed: %s", e)

        log.info("          Extracted %d tables total", len(frames))
    except Exception as exc:
        log.error("          PDF parse error: %s", exc)

    return frames


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract full plain text from a PDF using pdfplumber.
    Used as fallback when table extraction finds nothing useful.
    """
    if not PDF_OK:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception:
        return ""


def _filter_tables_by_keyword(frames: List[pd.DataFrame],
                               keywords: List[str]) -> List[pd.DataFrame]:
    """
    Return only DataFrames whose column names OR first-row values
    contain at least one of the supplied keywords (case-insensitive).
    """
    result = []
    for df in frames:
        cols_text = " ".join(str(c) for c in df.columns).lower()
        first_row = " ".join(str(v) for v in df.iloc[0].values).lower() if len(df) else ""
        combined  = cols_text + " " + first_row
        if any(kw.lower() in combined for kw in keywords):
            result.append(df)
    return result


def parse_pir_pdf(pdf_bytes: bytes, year: int) -> Dict[str, pd.DataFrame]:
    """
    Parse one TRAI Performance Indicator Report PDF.

    Table locations in TRAI PIR PDFs (by page range):
      Pages  1-30  : Snapshot / Executive Summary
      Pages 31-60  : Chapter 1 — Subscriber base (wireless + wireline)
      Pages 61-90  : Chapter 2 — Internet & broadband
      Pages 91-120 : Chapter 3 — Quality of Service
      Pages 121-150: Chapter 4 — Financial data
      Pages 151+   : Annexures (detailed state/city tables)

    Extracts:
      'wireless'    → Annexure 1.1: state-wise wireless subscriber + tele-density
      'internet'    → Chapter 2: internet subscriber + broadband counts
      'towers'      → Annexure: BTS tower counts by technology
      'wireline'    → Annexure 1.2: wireline subscribers
      'financial'   → Chapter 4: revenue, AGR
      'summary'     → regex-extracted key metrics from full text

    Args:
        pdf_bytes : raw PDF bytes
        year      : the report year (for the year column)

    Returns:
        dict of {table_type: DataFrame}
    """
    parsed: Dict[str, pd.DataFrame] = {}
    log.info("  [PARSE] PIR PDF for year %d", year)

    # ── Extract ALL tables first (one pass) ───────────────────────────────────
    all_tables = extract_all_tables(pdf_bytes)
    log.info("          Total tables found in PDF: %d", len(all_tables))

    # ── Wireless subscriber tables ────────────────────────────────────────────
    wireless_tables = _filter_tables_by_keyword(
        all_tables,
        ["wireless", "subscriber", "tele-density", "teledensity", "urban sub"]
    )
    if wireless_tables:
        df = pd.concat(wireless_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "wireless_subscribers"
        parsed["wireless"] = df
        log.info("          wireless: %d rows", len(df))

    # ── Internet / broadband tables ───────────────────────────────────────────
    inet_tables = _filter_tables_by_keyword(
        all_tables,
        ["internet", "broadband", "data subscriber", "wireline bb",
         "wireless bb", "narrowband"]
    )
    if inet_tables:
        df = pd.concat(inet_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "internet_subscribers"
        parsed["internet"] = df
        log.info("          internet: %d rows", len(df))

    # ── Tower / BTS tables ────────────────────────────────────────────────────
    tower_tables = _filter_tables_by_keyword(
        all_tables,
        ["bts", "base station", "tower", "4g site", "5g site",
         "lte", "nr site", "nr bts"]
    )
    if tower_tables:
        df = pd.concat(tower_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "bts_towers"
        parsed["towers"] = df
        log.info("          towers: %d rows", len(df))

    # ── Wireline subscriber tables ────────────────────────────────────────────
    wire_tables = _filter_tables_by_keyword(
        all_tables,
        ["wireline", "fixed line", "landline", "pstn"]
    )
    if wire_tables:
        df = pd.concat(wire_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "wireline_subscribers"
        parsed["wireline"] = df
        log.info("          wireline: %d rows", len(df))

    # ── QoS / Speed tables ────────────────────────────────────────────────────
    speed_tables = _filter_tables_by_keyword(
        all_tables,
        ["download speed", "upload speed", "mbps", "latency", "qos", "myspeed"]
    )
    if speed_tables:
        df = pd.concat(speed_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "network_speed"
        parsed["speed"] = df
        log.info("          speed: %d rows", len(df))

    # ── OFC / Infrastructure tables ───────────────────────────────────────────
    ofc_tables = _filter_tables_by_keyword(
        all_tables,
        ["ofc", "fiber", "route km", "ftth", "fttx", "wireline bb"]
    )
    if ofc_tables:
        df = pd.concat(ofc_tables, ignore_index=True)
        df = _clean_table(df)
        df["year"] = year
        df["table_type"] = "ofc_fiber"
        parsed["ofc"] = df
        log.info("          ofc/fiber: %d rows", len(df))

    # ── Regex fallback on full text ───────────────────────────────────────────
    if len(parsed) < 2:
        log.info("          Few tables found — running regex text extraction")
        full_text = extract_text_from_pdf(pdf_bytes)
        df_summary = _parse_pir_text_regex(full_text, year)
        if df_summary is not None and not df_summary.empty:
            parsed["summary_text"] = df_summary
            log.info("          text regex: %d key-value pairs", len(df_summary))

    return parsed


def _parse_pir_text_regex(text: str, year: int) -> Optional[pd.DataFrame]:
    """
    Regex fallback parser for TRAI PIR text.
    Extracts snapshot metrics when table extraction fails.

    Handles patterns like:
      "Total Subscribers  1,199.28 Million"
      "Urban Tele-density  133.72%"
      "4G BTS  6,82,345"
    """
    if not text:
        return None

    rows: List[dict] = []

    # Key-value with optional unit suffix
    kv = re.compile(
        r"((?:[A-Z][a-z]*\s?){1,6})"   # metric: 1-6 title-case words
        r"[\s:\-–]{1,5}"
        r"([\d,]+\.?\d*)"               # number (with possible commas)
        r"\s*([%BMKmkb]?(?:illion|rore|lakh)?)",
        re.MULTILINE
    )

    for m in kv.finditer(text):
        metric = m.group(1).strip()
        value  = m.group(2).replace(",", "")
        unit   = m.group(3).strip()
        if len(metric) >= 5 and len(value) >= 1:
            rows.append({
                "service_area": "India",
                "metric":       metric,
                "value":        float(value) if value else None,
                "unit":         unit,
                "year":         year,
                "table_type":   "text_summary",
            })

    return pd.DataFrame(rows) if rows else None


# =============================================================================
# SECTION 5 — DATA CLEANING
# =============================================================================

def _to_snake(text: str) -> str:
    """Convert any string to clean snake_case."""
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)   # punctuation → space
    s = re.sub(r"\s+", "_", s)        # whitespace  → underscore
    s = re.sub(r"_{2,}", "_", s)      # collapse multiples
    return s.strip("_") or "col"


# def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for a raw PDF table DataFrame:

    Steps:
      1. Stringify and strip all cells
      2. Replace separator values (-, --, n.a.) with NaN
      3. Drop rows that are ≥70% NaN (separator / header bleed rows)
      4. Convert column names to snake_case
      5. Coerce numeric-looking columns to float
      6. Rename common TRAI column name variants to canonical names
      7. Reset index
    """
    # 1. Strip strings
    df = df.apply(lambda col: col.map(
        lambda x: x.strip().replace("\n", " ") if isinstance(x, str) else x
    ))

    # 2. Replace separators
    df = df.replace(["", "-", "--", "---", "n.a.", "N.A.", "NA", "nil", "Nil"], np.nan)

    # 3. Drop mostly-empty rows
    threshold = max(1, int(len(df.columns) * 0.30))
    df = df.dropna(thresh=threshold)

    # 4. Snake-case column names
    df.columns = [_to_snake(c) for c in df.columns]

    # 5. Coerce numerics — strip commas then convert
    for col in df.columns:
        if col.startswith("_"):
            continue
        try:
            cleaned = df[col].astype(str).str.replace(",", "").str.strip()
            numeric = pd.to_numeric(cleaned, errors="coerce")
            # Only replace if >30% of non-null values converted successfully
            if numeric.notna().sum() > 0.30 * df[col].notna().sum():
                df[col] = numeric
        except Exception:
            pass

    # 6. Canonical column name aliases
    aliases = {
        r"service.?area|circle|lsa|state":            "service_area",
        r"total.?subscr|total.?sub\b":                "total_subscribers",
        r"urban.?subscr|urban.?sub\b":                "urban_subscribers",
        r"rural.?subscr|rural.?sub\b":                "rural_subscribers",
        r"tele.?density|teledensity":                 "tele_density_pct",
        r"4g.?bts|lte.?bts|4g.?site":                "towers_4g",
        r"5g.?bts|nr.?bts|5g.?site":                 "towers_5g",
        r"2g.?bts|gsm.?bts":                          "towers_2g",
        r"3g.?bts|wcdma.?bts":                        "towers_3g",
        r"wireline.?bb|fixed.?bb|ftth|fttx":          "fixed_broadband",
        r"wireless.?bb|mobile.?bb|mobile.?internet":  "wireless_broadband",
        r"total.?internet|total.?broadband":           "total_internet",
        r"download.?speed|dl.?speed|avg.?dl":         "avg_download_mbps",
        r"upload.?speed|ul.?speed|avg.?ul":           "avg_upload_mbps",
        r"ofc.?route|route.?km|fibre.?km|fiber.?km":  "ofc_route_km",
    }
    renamed = {}
    for col in df.columns:
        for pattern, canonical in aliases.items():
            if re.search(pattern, col, re.IGNORECASE):
                renamed[col] = canonical
                break
    df = df.rename(columns=renamed)

    return df.reset_index(drop=True)

def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw tables extracted from TRAI PDFs into a safe, uniform format.

    Handles:
      - Duplicate columns from pd.concat of heterogeneous tables
      - Duplicate columns after snake_case renaming
      - Blank / merged headers
      - Whitespace noise
      - Numeric strings with commas, %
      - Duplicate index labels (root cause of InvalidIndexError)
    """

    # ------------------------------------------------------------------
    # STEP 1 — Remove duplicate columns BEFORE anything
    # ------------------------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ------------------------------------------------------------------
    # STEP 2 — Normalize column names to snake_case
    # ------------------------------------------------------------------
    df.columns = [_to_snake(c) for c in df.columns]

    # ------------------------------------------------------------------
    # STEP 3 — Remove duplicate columns AGAIN after renaming
    # ------------------------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ------------------------------------------------------------------
    # STEP 4 — Strip whitespace from all cells
    # ------------------------------------------------------------------
    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)

    # ------------------------------------------------------------------
    # STEP 5 — Replace common blanks with NaN and drop empty rows/cols
    # ------------------------------------------------------------------
    df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # ------------------------------------------------------------------
    # STEP 6 — Clean numeric-looking columns
    # ------------------------------------------------------------------
    for col in df.columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(cleaned, errors="ignore")

    # ------------------------------------------------------------------
    # STEP 7 — CRITICAL: Reset index to avoid InvalidIndexError later
    # ------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)
    df.index = pd.RangeIndex(len(df))

    return df




def map_service_area_to_cities(df: pd.DataFrame,
                                area_col: str = "service_area") -> pd.DataFrame:
    """
    Disaggregate TRAI service-area rows into city-level rows.

    TRAI reports data at telecom service-area level (22 circles).
    Each service area maps to 1-6 cities via population-share weights.

    The weight is applied to all numeric columns so city-level values
    are proportional estimates.

    Args:
        df       : DataFrame with a service_area column
        area_col : name of the column containing service-area strings

    Returns:
        Expanded DataFrame with one row per (city, year) combination
    """
    if area_col not in df.columns:
        # Try to detect area column by common names
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in ["service","area","state","circle","lsa"])]
        if candidates:
            df = df.rename(columns={candidates[0]: area_col})
        else:
            df["city"] = "Unknown"
            return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_weight = {"year", "_page_num", "year_"}
    weight_cols = [c for c in numeric_cols if c not in exclude_from_weight]

    output_rows: List[dict] = []
    for _, row in df.iterrows():
        area     = str(row.get(area_col, "")).strip()
        mappings = AREA_TO_CITIES.get(area)

        if not mappings:
            # Fuzzy fallback: partial match
            for key in AREA_TO_CITIES:
                if key.lower() in area.lower() or area.lower() in key.lower():
                    mappings = AREA_TO_CITIES[key]
                    break

        if not mappings:
            mappings = [(area, 1.0)]   # pass through as-is

        for city, weight in mappings:
            new_row = row.to_dict()
            new_row["city"] = city
            for col in weight_cols:
                val = new_row.get(col)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    new_row[col] = round(val * weight, 4)
            output_rows.append(new_row)

    result = pd.DataFrame(output_rows)
    result = result.drop(columns=[area_col], errors="ignore")
    return result.reset_index(drop=True)




def fill_all_city_years(df: pd.DataFrame,
                        city_col: str = "city") -> pd.DataFrame:
    """
    Ensure every city × year combination exists in the DataFrame.

    Missing rows are forward-filled then backward-filled within each city
    group to carry last-known values forward.

    Args:
        df       : DataFrame with city and year columns
        city_col : name of the city column

    Returns:
        Complete DataFrame with all city × year combinations
    """
    if city_col not in df.columns or "year" not in df.columns:
        return df

    idx = pd.MultiIndex.from_product(
        [CITIES, YEARS], names=[city_col, "year"]
    )
    df = (
        df.set_index([city_col, "year"])
          .reindex(idx)
          .reset_index()
          .sort_values([city_col, "year"])
    )
    df = df.groupby(city_col, group_keys=False).apply(
        lambda g: g.ffill().bfill()
    )
    return df.reset_index(drop=True)


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass: fill remaining NaN with 0 (numeric) or 'Unknown' (string),
    and ensure city/year are correct types.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip().str.title()
    return df


# =============================================================================
# SECTION 6 — FACTOR BUILDERS (live scrape path)
# =============================================================================

def _download_and_parse_all_pirs() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Download all TRAI PIR PDFs (from catalogue + discovered links)
    and parse each one.

    Returns:
        dict  year_label → {table_type: DataFrame}
    """
    # Merge catalogue with live-discovered links
    live_discovered = scrape_trai_listing("pir")
    live_annual     = scrape_trai_listing("annual")

    all_pdfs: Dict[str, Tuple[str, int, str]] = {}
    for label, (url, yr, rtype) in TRAI_PDF_CATALOGUE.items():
        all_pdfs[label] = (url, yr, rtype)
    for label, (url, yr, rtype) in {**live_discovered, **live_annual}.items():
        if label not in all_pdfs:
            all_pdfs[label] = (url, yr, rtype)

    parsed_by_year: Dict[str, Dict[str, pd.DataFrame]] = {}

    for label, (url, year, rtype) in all_pdfs.items():
        if year not in YEARS:
            continue

        raw = fetch_pdf_bytes(url, label)
        if raw is None:
            continue

        parsed = parse_pir_pdf(raw, year)
        if parsed:
            key = f"{year}_{rtype}"
            if key not in parsed_by_year:
                parsed_by_year[key] = {}
            for ttype, df in parsed.items():
                df["source_label"] = label
                parsed_by_year[key][ttype] = df

    # Also extract from PIB press releases (HTML tables)
    for year, urls in PIB_URLS.items():
        for url in urls:
            df_pib = scrape_pib_press_release(url, year)
            if df_pib is not None:
                key = f"{year}_pib"
                if key not in parsed_by_year:
                    parsed_by_year[key] = {}
                parsed_by_year[key]["pib_summary"] = df_pib

    return parsed_by_year


def build_factor_network_connectivity(parsed_data: Dict) -> pd.DataFrame:
    """
    Factor 1: Network Connectivity

    Source tables used:
      - PIR wireless subscriber table → tele_density_pct, urban_subscribers
      - PIR towers table              → towers_4g, towers_5g, towers_2g
      - PIR summary text              → total_subscribers snapshot

    Final columns:
      city, year, towers_2g, towers_3g, towers_4g, towers_5g, towers_total,
      tele_density_pct, urban_subscribers_mn, wireless_subscribers_mn,
      table_type, source_label
    """
    log.info("── Factor: Network Connectivity ──")
    frames: List[pd.DataFrame] = []

    for year_key, tables in parsed_data.items():
        for ttype in ["wireless", "towers", "summary_text", "pib_summary"]:
            df = tables.get(ttype)
            if df is None or df.empty:
                continue
            df = map_service_area_to_cities(df)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)

    # Compute towers_total if components exist
    tower_cols = [c for c in df_out.columns if re.match(r"towers_\d", c)]
    if tower_cols:
        df_out["towers_total"] = df_out[tower_cols].sum(axis=1, skipna=True)

    # Scale subscribers from millions if raw numbers are very large
    for col in ["total_subscribers","urban_subscribers","wireless_subscribers_mn"]:
        if col in df_out.columns:
            median_val = df_out[col].replace(0, np.nan).median()
            if median_val and median_val > 1e7:   # raw count → millions
                df_out[col] = df_out[col] / 1e6
                df_out = df_out.rename(columns={col: col.rstrip("_mn") + "_mn"})

    df_out = fill_all_city_years(df_out)
    df_out = final_clean(df_out)

    keep = ["city","year","towers_2g","towers_3g","towers_4g","towers_5g",
            "towers_total","tele_density_pct","urban_subscribers_mn",
            "wireless_subscribers_mn","table_type","source_label"]
    return df_out.reindex(columns=[c for c in keep if c in df_out.columns])


def build_factor_digital_literacy(parsed_data: Dict) -> pd.DataFrame:
    """
    Factor 2: Digital Literacy

    Source tables:
      - PIR internet table → total_internet, wireless_broadband, fixed_broadband
      - PIR speed table    → avg_download_mbps, avg_upload_mbps
      - OGD MySpeed API    → quarterly DL/UL speeds by state

    Final columns:
      city, year, internet_subscribers_mn, broadband_subscribers_mn,
      fixed_broadband_mn, wireless_broadband_mn,
      avg_download_mbps, avg_upload_mbps,
      hh_internet_pct, table_type, source_label
    """
    log.info("── Factor: Digital Literacy ──")
    frames: List[pd.DataFrame] = []

    for year_key, tables in parsed_data.items():
        for ttype in ["internet","speed"]:
            df = tables.get(ttype)
            if df is None or df.empty:
                continue
            df = map_service_area_to_cities(df)
            frames.append(df)

    # Supplement with OGD MySpeed API
    df_speed = fetch_ogd_api(OGD_RESOURCES["trai_myspeed"], "TRAI_MySpeed")
    if df_speed is not None:
        df_speed.columns = [_to_snake(c) for c in df_speed.columns]
        # Detect year from quarter column
        quarter_col = next((c for c in df_speed.columns
                           if "quarter" in c or "period" in c), None)
        if quarter_col:
            df_speed["year"] = (
                df_speed[quarter_col].astype(str)
                                     .str.extract(r"(20\d{2})")[0]
                                     .astype(float)
            )
        df_speed = map_service_area_to_cities(
            df_speed.rename(columns={c: "service_area"
                                     for c in df_speed.columns if "state" in c.lower()})
        )
        df_speed["source_label"] = "TRAI_MySpeed_OGD_API"
        df_speed["table_type"]   = "broadband_speed"
        frames.append(df_speed)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)

    # Convert raw subscriber counts to millions
    for col in ["total_internet","wireless_broadband","fixed_broadband"]:
        if col in df_out.columns:
            med = df_out[col].replace(0, np.nan).median()
            if med and med > 1e6:
                df_out[col] = df_out[col] / 1e6

    df_out = fill_all_city_years(df_out)
    df_out = final_clean(df_out)

    keep = ["city","year","internet_subscribers_mn","broadband_subscribers_mn",
            "fixed_broadband_mn","wireless_broadband_mn",
            "avg_download_mbps","avg_upload_mbps","hh_internet_pct",
            "table_type","source_label"]
    return df_out.reindex(columns=[c for c in keep if c in df_out.columns])


def build_factor_infrastructure(parsed_data: Dict) -> pd.DataFrame:
    """
    Factor 3: Infrastructure

    Source tables:
      - PIR towers  → towers_4g, towers_5g, towers_2g, towers_3g
      - PIR ofc     → ofc_route_km, fixed_broadband (FTTH)
      - PIR wireline→ wireline_subscribers

    Final columns:
      city, year, total_telecom_towers, towers_4g, towers_5g,
      ofc_route_km, ftth_connections_mn, wireline_subscribers_mn,
      pm_wani_hotspots, table_type, source_label
    """
    log.info("── Factor: Infrastructure ──")
    frames: List[pd.DataFrame] = []

    for year_key, tables in parsed_data.items():
        for ttype in ["towers","ofc","wireline"]:
            df = tables.get(ttype)
            if df is None or df.empty:
                continue
            df = map_service_area_to_cities(df)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)

    tower_cols = [c for c in df_out.columns
                  if re.match(r"towers_\d|tower_\d", c)]
    if tower_cols:
        df_out["total_telecom_towers"] = df_out[tower_cols].sum(axis=1, skipna=True)

    df_out = fill_all_city_years(df_out)
    df_out = final_clean(df_out)

    keep = ["city","year","total_telecom_towers","towers_4g","towers_5g",
            "ofc_route_km","ftth_connections_mn","wireline_subscribers_mn",
            "pm_wani_hotspots","table_type","source_label"]
    return df_out.reindex(columns=[c for c in keep if c in df_out.columns])


def build_factor_electricity(parsed_data: Dict) -> pd.DataFrame:
    """
    Factor 4: Electricity

    Source tables:
      - CEA Annual Report PDFs (downloaded separately)
        → power supply hours/day, SAIDI, installed capacity
      - PIB press releases with CEA summary data
      - TRAI PIR cross-reference (telecom towers need reliable power)

    Final columns:
      city, year, electricity_hours_per_day, saidi_hours_per_year,
      electrification_pct, installed_capacity_mw, avg_tariff_rs_unit,
      table_type, source_label
    """
    log.info("── Factor: Electricity ──")
    frames: List[pd.DataFrame] = []

    # Download CEA Annual Report PDFs
    for label, (url, yr, rtype) in {
        "CEA_2023_24": ("https://cea.nic.in/wp-content/uploads/annual_report/2024/annual_report-2023-24.pdf", 2024, "cea"),
        "CEA_2022_23": ("https://cea.nic.in/wp-content/uploads/annual_report/2023/Annual_Report_2022-23.pdf", 2023, "cea"),
        "CEA_2021_22": ("https://cea.nic.in/wp-content/uploads/annual_report/2022/annual_report-2021-22.pdf", 2022, "cea"),
        "CEA_2020_21": ("https://cea.nic.in/wp-content/uploads/annual_report/2021/annual_report-2020-21.pdf", 2021, "cea"),
        "CEA_2019_20": ("https://cea.nic.in/wp-content/uploads/annual_report/2020/annual_report-2019-20.pdf", 2020, "cea"),
    }.items():
        raw = fetch_pdf_bytes(url, label)
        if raw is None:
            continue
        all_tables = extract_all_tables(raw)
        elec_tables = _filter_tables_by_keyword(
            all_tables,
            ["hours","supply","saidi","tariff","energy","capacity","mw","gw","kwh"]
        )
        if elec_tables:
            df = pd.concat(elec_tables, ignore_index=True)
            df = _clean_table(df)
            df["year"] = yr
            df["source_label"] = label
            df["table_type"] = "electricity_supply"
            df = map_service_area_to_cities(df)
            frames.append(df)

    # Cross-reference from PIR data (power-related commentary)
    for year_key, tables in parsed_data.items():
        df_pib = tables.get("pib_summary")
        if df_pib is not None and not df_pib.empty:
            elec_rows = df_pib[df_pib["metric"].str.contains(
                "power|electr|energy|saidi|tariff|kwh", case=False, na=False
            )] if "metric" in df_pib.columns else pd.DataFrame()
            if not elec_rows.empty:
                frames.append(elec_rows)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)
    df_out = fill_all_city_years(df_out)
    df_out = final_clean(df_out)

    keep = ["city","year","electricity_hours_per_day","saidi_hours_per_year",
            "electrification_pct","installed_capacity_mw","avg_tariff_rs_unit",
            "table_type","source_label"]
    return df_out.reindex(columns=[c for c in keep if c in df_out.columns])


def build_factor_socio_economic() -> pd.DataFrame:
    """
    Factor 5: Socio-Economic

    Sources:
      - data.gov.in OGD API:
          PMJDY monthly accounts  → jan_dhan_accounts_lakh
          GSDP state-wise         → gdp_per_capita_rs
          PLFS LFPR               → female_lfpr_pct, literacy_rate_pct
      - MoSPI press release PDFs (PLFS annual reports)
      - PIB press releases (budget / economic survey snippets)

    Final columns:
      city, year, state, gdp_per_capita_rs, urban_poverty_rate_pct,
      female_lfpr_pct, literacy_rate_pct, jan_dhan_accounts_lakh,
      pmjay_beneficiaries_lakh, gini_coefficient, source_label
    """
    log.info("── Factor: Socio-Economic ──")
    frames: List[pd.DataFrame] = []

    # ── PMJDY Jan Dhan accounts ───────────────────────────────────────────────
    df_pmjdy = fetch_ogd_api(OGD_RESOURCES["pmjdy_monthly"], "PMJDY_Monthly")
    if df_pmjdy is not None:
        df_pmjdy.columns = [_to_snake(c) for c in df_pmjdy.columns]
        for col in df_pmjdy.columns:
            if "state" in col or "district" in col:
                df_pmjdy = df_pmjdy.rename(columns={col: "service_area"})
                break
        if "month" in df_pmjdy.columns or "date" in df_pmjdy.columns:
            date_col = "month" if "month" in df_pmjdy.columns else "date"
            df_pmjdy["year"] = pd.to_datetime(
                df_pmjdy[date_col], errors="coerce"
            ).dt.year
        df_pmjdy["source_label"] = "OGD_PMJDY"
        df_pmjdy["table_type"]   = "financial_inclusion"
        df_pmjdy = map_service_area_to_cities(df_pmjdy)
        frames.append(df_pmjdy)

    # ── GSDP state-wise ───────────────────────────────────────────────────────
    df_gsdp = fetch_ogd_api(OGD_RESOURCES["gsdp_statewise"], "GSDP_Statewise")
    if df_gsdp is not None:
        df_gsdp.columns = [_to_snake(c) for c in df_gsdp.columns]
        for col in df_gsdp.columns:
            if "state" in col:
                df_gsdp = df_gsdp.rename(columns={col: "service_area"})
                break
        df_gsdp["source_label"] = "OGD_GSDP"
        df_gsdp["table_type"]   = "gdp_income"
        df_gsdp = map_service_area_to_cities(df_gsdp)
        frames.append(df_gsdp)

    # ── PLFS LFPR ─────────────────────────────────────────────────────────────
    df_plfs = fetch_ogd_api(OGD_RESOURCES["plfs_lfpr"], "PLFS_LFPR")
    if df_plfs is not None:
        df_plfs.columns = [_to_snake(c) for c in df_plfs.columns]
        df_plfs["source_label"] = "OGD_PLFS"
        df_plfs["table_type"]   = "employment"
        df_plfs = map_service_area_to_cities(df_plfs)
        frames.append(df_plfs)

    # ── PLFS Annual Report press release PDFs ─────────────────────────────────
    plfs_pdfs = {
        2024: "https://mospi.gov.in/sites/default/files/press_release/Press_note_AR_PLFS_2023_24_22092024.pdf",
        2023: "https://www.mospi.gov.in/sites/default/files/press_release/Press_note_AR_PLFS_2022_23.pdf",
    }
    for yr, url in plfs_pdfs.items():
        raw = fetch_pdf_bytes(url, f"PLFS_{yr}")
        if raw is None:
            continue
        all_tables = extract_all_tables(raw)
        plfs_tbl = _filter_tables_by_keyword(
            all_tables, ["lfpr","labour","employment","female","urban"]
        )
        if plfs_tbl:
            df = pd.concat(plfs_tbl, ignore_index=True)
            df = _clean_table(df)
            df["year"] = yr
            df["source_label"] = f"PLFS_{yr}_PDF"
            df["table_type"] = "employment"
            df = map_service_area_to_cities(df)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)
    df_out = fill_all_city_years(df_out)
    df_out = final_clean(df_out)

    keep = ["city","year","state","gdp_per_capita_rs","urban_poverty_rate_pct",
            "female_lfpr_pct","literacy_rate_pct","jan_dhan_accounts_lakh",
            "pmjay_beneficiaries_lakh","gini_coefficient",
            "table_type","source_label"]
    return df_out.reindex(columns=[c for c in keep if c in df_out.columns])


# =============================================================================
# SECTION 7 — DEMO DATA (exact same schema, offline-safe)
# =============================================================================

def _make_demo_data() -> Dict[str, pd.DataFrame]:
    """
    Generate realistic demo DataFrames with the exact same schema
    as the live-scrape output.

    Data values are based on publicly-known TRAI statistics:
      - India had ~1.19 billion telecom subscribers end-2024
      - Urban tele-density ~133%
      - 4G towers: ~760K nationally (scaled to cities)
      - 5G towers: ~470K nationally (Oct 2022 onwards)
      - Avg download speed: ~95 Mbps (Q3 2024 TRAI MySpeed)
      - Broadband subscribers: ~940 million (Q2 2024)
    """
    np.random.seed(2024)

    # ── City weights (population share for scaling) ───────────────────────────
    city_weight = {
        "Mumbai":1.00,"Delhi":0.95,"Kolkata":0.75,"Chennai":0.70,
        "Bengaluru":0.85,"Hyderabad":0.80,"Ahmedabad":0.65,"Pune":0.60,
        "Surat":0.55,"Jaipur":0.50,"Lucknow":0.48,"Kanpur":0.46,
        "Nagpur":0.42,"Indore":0.40,"Bhopal":0.38,"Patna":0.36,
    }
    default_wt = 0.28

    def wt(city): return city_weight.get(city, default_wt)

    rows_nc, rows_dl, rows_inf, rows_el, rows_se = [], [], [], [], []

    STATE_MAP = {
        "Mumbai":"Maharashtra","Delhi":"Delhi","Bengaluru":"Karnataka",
        "Hyderabad":"Telangana","Chennai":"Tamil Nadu","Kolkata":"West Bengal",
        "Pune":"Maharashtra","Ahmedabad":"Gujarat","Jaipur":"Rajasthan",
        "Lucknow":"Uttar Pradesh","Surat":"Gujarat","Kanpur":"Uttar Pradesh",
        "Nagpur":"Maharashtra","Indore":"Madhya Pradesh","Bhopal":"Madhya Pradesh",
        "Patna":"Bihar","Ludhiana":"Punjab","Agra":"Uttar Pradesh",
        "Nashik":"Maharashtra","Faridabad":"Haryana","Meerut":"Uttar Pradesh",
        "Rajkot":"Gujarat","Varanasi":"Uttar Pradesh","Amritsar":"Punjab",
    }

    for city in CITIES:
        w = wt(city)
        # ── Baseline 2019 values ──────────────────────────────────────────────
        base_4g   = int(3800 * w * np.random.uniform(0.85, 1.15))
        base_5g   = 0
        base_dl   = 12 + 8 * w + np.random.uniform(-2, 3)
        base_inet = 35 + 25 * w + np.random.uniform(-5, 5)
        base_ofc  = int(800 * w * np.random.uniform(0.8, 1.2))
        base_ftth = int(15000 * w * np.random.uniform(0.7, 1.3))
        base_pov  = max(4, 28 - 15 * w + np.random.uniform(-3, 3))
        base_gdp  = int((65000 + 180000 * w) * np.random.uniform(0.9, 1.1))
        base_lfpr = 20 + 18 * w + np.random.uniform(-3, 3)
        base_elec = 16 + 6 * w + np.random.uniform(-1, 1)
        base_saidi= max(3, 40 - 30 * w + np.random.uniform(-3, 3))

        for yr in YEARS:
            t = yr - 2019   # years from baseline

            # ── Network Connectivity ─────────────────────────────────────────
            growth_4g = 1 + np.random.uniform(0.04, 0.10)
            base_4g   = int(base_4g * growth_4g)
            base_5g   = (0 if yr < 2022 else
                         int(base_4g * (0.18 if yr==2022 else 0.52 if yr==2023 else 0.85)))
            rows_nc.append({
                "city":city,"year":yr,
                "towers_2g":     int(500 * w * max(0.3, 1-0.12*t)),
                "towers_3g":     int(1200 * w * max(0.2, 1-0.08*t)),
                "towers_4g":     base_4g,
                "towers_5g":     base_5g,
                "towers_total":  int(500*w*max(0.3,1-0.12*t))+int(1200*w*max(0.2,1-0.08*t))+base_4g+base_5g,
                "tele_density_pct":   round(85 + 40*w + t*3 + np.random.uniform(-2,2), 2),
                "urban_subscribers_mn": round(0.4*w*(1+0.04*t) + np.random.uniform(0,0.05), 3),
                "wireless_subscribers_mn": round(1.8*w*(1+0.025*t) + np.random.uniform(0,0.1), 3),
                "table_type":"wireless_subscribers","source_label":"TRAI_PIR_Annual_DEMO",
            })

            # ── Digital Literacy ─────────────────────────────────────────────
            dl = min(base_dl * (1.12**t), 110)
            inet_pct = min(base_inet + t*4.5, 95)
            rows_dl.append({
                "city":city,"year":yr,
                "internet_subscribers_mn": round(1.2*w*(1+0.09*t), 3),
                "broadband_subscribers_mn": round(0.9*w*(1+0.10*t), 3),
                "fixed_broadband_mn":       round(0.15*w*(1+0.15*t), 3),
                "wireless_broadband_mn":    round(0.75*w*(1+0.09*t), 3),
                "avg_download_mbps":        round(dl, 2),
                "avg_upload_mbps":          round(dl * np.random.uniform(0.3, 0.45), 2),
                "hh_internet_pct":          round(inet_pct, 2),
                "table_type":"internet_subscribers","source_label":"TRAI_PIR_BB_DEMO",
            })

            # ── Infrastructure ───────────────────────────────────────────────
            rows_inf.append({
                "city":city,"year":yr,
                "total_telecom_towers": rows_nc[-1]["towers_total"],
                "towers_4g":     base_4g,
                "towers_5g":     base_5g,
                "ofc_route_km":  int(base_ofc * (1.07**t)),
                "ftth_connections_mn":   round(base_ftth*(1.20**t)/1e6, 4),
                "wireline_subscribers_mn": round(0.08*w*(0.95**t), 4),
                "pm_wani_hotspots": int(0 if yr<2021 else 800*w*(yr-2020)*np.random.uniform(0.8,1.2)),
                "table_type":"infrastructure","source_label":"TRAI_PIR_Infra_DEMO",
            })

            # ── Electricity ──────────────────────────────────────────────────
            rows_el.append({
                "city":city,"year":yr,
                "electricity_hours_per_day": round(min(base_elec + t*0.4, 24), 2),
                "saidi_hours_per_year":      round(max(base_saidi*(0.92**t), 1.5), 2),
                "electrification_pct":       round(min(90 + 8*w + t*0.8, 100), 2),
                "installed_capacity_mw":     round(800*w*(1.06**t), 1),
                "avg_tariff_rs_unit":        round(5.5 + 0.3*t + np.random.uniform(-0.3,0.3), 2),
                "table_type":"electricity_supply","source_label":"CEA_Annual_DEMO",
            })

            # ── Socio-Economic ───────────────────────────────────────────────
            gdp = int(base_gdp * (1.07 + 0.02*w)**t)
            rows_se.append({
                "city":city,"year":yr,
                "state":              STATE_MAP.get(city,"Other"),
                "gdp_per_capita_rs":  gdp,
                "urban_poverty_rate_pct": round(max(base_pov*(0.93**t), 2.5), 2),
                "female_lfpr_pct":    round(min(base_lfpr + t*1.2, 60), 2),
                "literacy_rate_pct":  round(min(78 + 15*w + t*0.6, 99), 2),
                "jan_dhan_accounts_lakh": round(8*w*(1.08**t) + np.random.uniform(0,1), 2),
                "pmjay_beneficiaries_lakh": round(3*w*(1.10**t) + np.random.uniform(0,0.5), 2),
                "gini_coefficient":   round(max(0.28, 0.42 - 0.005*t - 0.03*w), 3),
                "table_type":"socio_economic","source_label":"OGD_MoSPI_DEMO",
            })

    return {
        "network_connectivity": pd.DataFrame(rows_nc),
        "digital_literacy":     pd.DataFrame(rows_dl),
        "infrastructure":       pd.DataFrame(rows_inf),
        "electricity":          pd.DataFrame(rows_el),
        "socio_economic":       pd.DataFrame(rows_se),
    }


# =============================================================================
# SECTION 8 — EXCEL OUTPUT
# =============================================================================

def write_excel(dfs: Dict[str, pd.DataFrame], path: str = OUTPUT_FILE) -> None:
    """
    Write all 5 factor DataFrames to a single Excel workbook.

    Each factor becomes one sheet with:
      - Bold, blue-on-white frozen header row
      - Auto-width columns (capped at 35)
      - Alternating row shading for readability
      - City and year as first two columns always

    Args:
        dfs  : dict of factor_name → DataFrame
        path : output file path
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    SHEET_NAMES = {
        "network_connectivity": "📡 Network Connectivity",
        "digital_literacy":     "💻 Digital Literacy",
        "infrastructure":       "🔌 Infrastructure",
        "electricity":          "⚡ Electricity",
        "socio_economic":       "📊 Socio-Economic",
    }

    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    ALT_FILL    = PatternFill("solid", fgColor="EBF3FB")
    HEADER_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
    DATA_FONT   = Font(name="Calibri", size=10)
    CENTER      = Alignment(horizontal="center", vertical="center", wrap_text=True)
    THIN        = Side(style="thin", color="CCCCCC")
    BORDER      = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    wb = Workbook()
    wb.remove(wb.active)   # remove default blank sheet

    for factor_key, df in dfs.items():
        sheet_name = SHEET_NAMES.get(factor_key, factor_key[:31])
        ws = wb.create_sheet(title=sheet_name)

        if df.empty:
            ws["A1"] = f"No data available for {factor_key}"
            continue

        # ── Write headers ─────────────────────────────────────────────────────
        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name.replace("_", " ").title())
            cell.fill      = HEADER_FILL
            cell.font      = HEADER_FONT
            cell.alignment = CENTER
            cell.border    = BORDER

        # ── Write data rows ───────────────────────────────────────────────────
        for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
            fill = ALT_FILL if row_idx % 2 == 0 else None
            for col_idx, value in enumerate(row.values, start=1):
                # Convert numpy types for openpyxl
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value) if not np.isnan(value) else ""
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.font   = DATA_FONT
                cell.border = BORDER
                if fill:
                    cell.fill = fill

        # ── Auto column width ─────────────────────────────────────────────────
        for col_idx, col_name in enumerate(df.columns, start=1):
            col_letter = get_column_letter(col_idx)
            max_len    = max(
                len(str(col_name)),
                df[col_name].astype(str).map(len).max() if not df.empty else 0,
            )
            ws.column_dimensions[col_letter].width = min(max_len + 3, 35)

        ws.freeze_panes = "C2"   # freeze city + year columns

    wb.save(path)
    log.info("[EXCEL] Saved → %s  (%d sheets)", path, len(wb.sheetnames))


# =============================================================================
# SECTION 9 — PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(demo: bool = False,
                 factor: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Main entry point. Runs all 5 factor scrapers end-to-end.

    Args:
        demo   : if True, skip all network calls and return demo data
        factor : if set (e.g. "network"), run only that one factor

    Returns:
        dict of factor_name → DataFrame
    """
    log.info("=" * 68)
    log.info("TRAI 5-Factor Data Pipeline — %s mode",
             "DEMO" if demo else "LIVE SCRAPE")
    log.info("=" * 68)

    if demo:
        log.info("Generating realistic demo data (no network required)…")
        dfs = _make_demo_data()
    else:
        # ── Live path ─────────────────────────────────────────────────────────
        log.info("Step 1/3  Discovering & downloading TRAI PDFs…")
        parsed_data = _download_and_parse_all_pirs()

        if not parsed_data:
            log.warning("No PDFs downloaded — falling back to demo data")
            return run_pipeline(demo=True, factor=factor)

        log.info("Step 2/3  Building factor DataFrames…")
        builders = {
            "network_connectivity": lambda: build_factor_network_connectivity(parsed_data),
            "digital_literacy":     lambda: build_factor_digital_literacy(parsed_data),
            "infrastructure":       lambda: build_factor_infrastructure(parsed_data),
            "electricity":          lambda: build_factor_electricity(parsed_data),
            "socio_economic":       build_factor_socio_economic,
        }

        if factor:
            # Single factor mode
            if factor not in builders:
                raise ValueError(f"Unknown factor '{factor}'. "
                                 f"Choose from: {list(builders)}")
            dfs = {factor: builders[factor]()}
        else:
            dfs = {name: fn() for name, fn in builders.items()}

        # Fallback: replace empty DFs with demo data
        demo_dfs = _make_demo_data()
        for name, df in dfs.items():
            if df.empty:
                log.warning("Factor '%s' returned empty — using demo data", name)
                dfs[name] = demo_dfs[name]

    log.info("Step 3/3  Writing Excel output…")
    write_excel(dfs, OUTPUT_FILE)

    # ── Summary report ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 68)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 68)
    log.info("%-28s  %10s  %10s", "Factor", "Rows", "Columns")
    log.info("-" * 60)
    for name, df in dfs.items():
        log.info("%-28s  %10d  %10d", name, len(df), len(df.columns))
    if _FAILED_URLS:
        log.info("")
        log.info("Failed URLs (%d):", len(_FAILED_URLS))
        for u in _FAILED_URLS:
            log.info("  ✗  %s", u)
    log.info("")
    log.info("Output file: %s", OUTPUT_FILE)
    log.info("Log file:    trai_scraper.log")

    return dfs


# =============================================================================
# SECTION 10 — ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRAI 5-Factor Data Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trai_scraper_pipeline.py                     # full live scrape
  python trai_scraper_pipeline.py --demo              # offline demo mode
  python trai_scraper_pipeline.py --factor network    # single factor only
  python trai_scraper_pipeline.py --output my_out.xlsx
        """
    )
    parser.add_argument("--demo",   action="store_true",
                        help="Run in demo mode (no network required)")
    parser.add_argument("--factor", choices=[
                            "network_connectivity","digital_literacy",
                            "infrastructure","electricity","socio_economic",
                        ], help="Run a single factor only")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output Excel path (default: {OUTPUT_FILE})")
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    OUTPUT_FILE = args.output

    results = run_pipeline(demo=args.demo, factor=args.factor)

    # Print first 5 rows of each DataFrame for quick inspection
    print("\n" + "=" * 68)
    print("DATAFRAME PREVIEWS")
    print("=" * 68)
    for name, df in results.items():
        print(f"\n── {name.upper().replace('_',' ')} ({df.shape[0]} rows × {df.shape[1]} cols) ──")
        print(df.head(5).to_string(index=False))