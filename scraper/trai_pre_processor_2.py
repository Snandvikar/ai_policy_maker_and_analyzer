import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from pypdf import PdfReader, PdfWriter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# STEP 1: Local PDF Path
# ============================================================

PDF_PATH = "data/PIR_2025-07_pir_quarterly.pdf"
print(f"📄 Using PDF: {PDF_PATH}")


# ============================================================
# STEP 2: Table Page Mapping (0-based)
# ============================================================

TARGET_INDICES_RAW = {
    "1.1": [15],
    "1.3": [17],
    "1.4": [18],
    "1.5": [19],
    "1.6": [21],
    "1.7": [22],
    "1.8": [23, 24],
    "1.9": [23, 24],
    "1.11": [26],
    "1.12": [28],
    "1.13": [29],
    "1.14": [30],
    "1.15": [31],
    "1.16": [32],
    "1.17": [33],
    "1.18": [34],
}


# ============================================================
# STEP 3: Define 2-table groups
# ============================================================

TABLE_GROUPS = {
    "group_1": ["1.1", "1.3"],
    "group_2": ["1.4", "1.5"],
    "group_3": ["1.6", "1.7"],
    "group_4": ["1.8", "1.9"],
    "group_5": ["1.11", "1.12"],
    "group_6": ["1.13", "1.14"],
    "group_7": ["1.15", "1.16"],
    "group_8": ["1.17", "1.18"]
}


# ============================================================
# STEP 4: Create Mini PDFs per Group
# ============================================================

reader = PdfReader(PDF_PATH)

group_pdf_paths = {}

os.makedirs("mini_pdfs", exist_ok=True)

for group_name, tables in TABLE_GROUPS.items():

    pages = sorted(set(
        idx for t in tables for idx in TARGET_INDICES_RAW[t]
    ))

    writer = PdfWriter()

    for p in pages:
        writer.add_page(reader.pages[p])

    mini_path = f"mini_pdfs/{group_name}.pdf"

    with open(mini_path, "wb") as f:
        writer.write(f)

    group_pdf_paths[group_name] = {
        "path": mini_path,
        "tables": tables,
        "pages": pages
    }

    print(f"✅ {group_name} → pages {[p+1 for p in pages]} saved")


# ============================================================
# STEP 5: Docling Config
# ============================================================

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ============================================================
# STEP 6: Process Each Mini PDF
# ============================================================

def clean_df(df):
    df = df.copy()

    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    df.replace(["", "nan", "None", "-"], pd.NA, inplace=True)

    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    df.columns = [
        re.sub(r'[^a-z0-9]+', '_', str(c).lower()).strip('_')
        for c in df.columns
    ]

    return df.reset_index(drop=True)


all_dataframes = {}

for group_name, meta in group_pdf_paths.items():

    print(f"\n🚀 Processing {group_name}")

    result = converter.convert(meta["path"])
    doc = result.document

    print(f"   Found {len(doc.tables)} tables")

    # Assign tables sequentially
    for i, table in enumerate(doc.tables):

        if i >= len(meta["tables"]):
            continue

        tbl_id = meta["tables"][i]
        var_name = f"df_table_{tbl_id.replace('.', '_')}"

        try:
            df = table.export_to_dataframe()
            df = clean_df(df)

            all_dataframes[var_name] = df

            print(f"   ✅ {tbl_id} → {df.shape}")

        except Exception as e:
            print(f"   ❌ {tbl_id} failed: {e}")

# ============================================================
# STEP 8: APPLY SCHEMA (Table Names + Column Renaming)
# ============================================================

SCHEMA = {
    "1.1": {
        "table_name": "subscriber_base_teledensity_rural_urban",
        "columns": [
            "year_ending",
            "subscriber_base_million_rural",
            "subscriber_base_million_urban",
            "subscriber_base_million_total",
            "teledensity_percent_rural",
            "teledensity_percent_urban",
            "teledensity_percent_overall",
        ],
    },
    "1.3": {
        "table_name": "service_area_access_subscriber_base",
        "columns": [
            "service_area",
            "subscribers_million_mar_2024",
            "subscribers_million_mar_2025",
            "yearly_net_additions_million",
            "yearly_growth_rate_percent",
        ],
    },
    "1.4": {
        "table_name": "service_area_rural_urban_subscribers",
        "columns": [
            "service_area",
            "subscribers_total_mar_2025",
            "subscribers_rural_mar_2025",
            "subscribers_urban_mar_2025",
            "rural_subscriber_share_percent",
        ],
    },
    "1.5": {
        "table_name": "service_area_teledensity",
        "columns": [
            "service_area",
            "rural_teledensity_percent_mar_2024",
            "urban_teledensity_percent_mar_2024",
            "total_teledensity_percent_mar_2024",
            "rural_teledensity_percent_mar_2025",
            "urban_teledensity_percent_mar_2025",
            "total_teledensity_percent_mar_2025",
        ],
    },
    "1.6": {
        "table_name": "state_ut_subscriber_base",
        "columns": [
            "serial_no",
            "state_ut",
            "total_telephone_subscription",
            "rural_telephone_subscription",
            "urban_telephone_subscription",
        ],
    },
    "1.7": {
        "table_name": "state_ut_total_teledensity",
        "columns": [
            "serial_no",
            "state_ut",
            "total_teledensity_percent",
            "rural_teledensity_percent",
            "urban_teledensity_percent",
        ],
    },
    "1.8": {
        "table_name": "service_provider_subscriber_growth",
        "columns": [
            "service_provider",
            "subscriber_base_million_mar_2024",
            "subscriber_base_million_mar_2025",
            "yearly_net_addition_or_decline_million",
            "growth_rate_percent",
            "market_share_percent_mar_2024",
            "market_share_percent_mar_2025",
        ],
    },
    "1.9": {
        "table_name": "service_provider_rural_market_share",
        "columns": [
            "service_provider",
            "total_subscribers_million",
            "rural_subscribers_million",
            "rural_subscriber_share_percent",
            "rural_market_share_percent",
        ],
    },
    "1.11": {
        "table_name": "wireless_subscriber_growth",
        "columns": [
            "service_provider",
            "wireless_subscriber_base_million_mar_2024",
            "wireless_subscriber_base_million_mar_2025",
            "wireless_net_additions_million",
            "growth_rate_percent",
            "market_share_percent_mar_2024",
            "market_share_percent_mar_2025",
        ],
    },
    "1.12": {
        "table_name": "service_area_wireless_subscriber_base",
        "columns": [
            "service_area",
            "wireless_subscriber_base_million_mar_2024",
            "wireless_subscriber_base_million_mar_2025",
            "wireless_net_additions_million",
            "wireless_growth_rate_percent",
        ],
    },
    "1.13": {
        "table_name": "service_area_wireless_rural_urban_subscribers",
        "columns": [
            "service_area",
            "wireless_subscribers_total_mar_2025",
            "wireless_subscribers_rural_mar_2025",
            "wireless_subscribers_urban_mar_2025",
            "wireless_rural_subscriber_share_percent",
        ],
    },
    "1.14": {
        "table_name": "service_area_wireless_teledensity",
        "columns": [
            "service_area",
            "wireless_rural_teledensity_percent_mar_2024",
            "wireless_urban_teledensity_percent_mar_2024",
            "wireless_total_teledensity_percent_mar_2024",
            "wireless_rural_teledensity_percent_mar_2025",
            "wireless_urban_teledensity_percent_mar_2025",
            "wireless_total_teledensity_percent_mar_2025",
        ],
    },
    "1.15": {
        "table_name": "state_ut_wireless_subscriber_base",
        "columns": [
            "serial_no",
            "state_ut",
            "wireless_subscribers_total_million",
            "wireless_subscribers_rural_million",
            "wireless_subscribers_urban_million",
        ],
    },
    "1.16": {
        "table_name": "state_ut_wireless_teledensity",
        "columns": [
            "serial_no",
            "state_ut",
            "wireless_teledensity_total_percent",
            "wireless_teledensity_rural_percent",
            "wireless_teledensity_urban_percent",
        ],
    },
    "1.17": {
        "table_name": "rural_wireless_subscriber_market_share",
        "columns": [
            "service_provider",
            "total_wireless_subscribers_million",
            "rural_wireless_subscribers_million",
            "rural_subscriber_share_percent",
            "rural_market_share_percent",
        ],
    },
    "1.18": {
        "table_name": "rural_wireless_subscriber_market_share_alt",
        "columns": [
            "service_provider",
            "total_wireless_subscribers_million",
            "rural_wireless_subscribers_million",
            "rural_subscriber_share_percent",
            "rural_market_share_percent",
        ],
    },
}


print("\n🔧 Applying schema...")

canonical_dataframes = {}

for var_name, df in all_dataframes.items():

    tbl_id = var_name.replace("df_table_", "").replace("_", ".")
    schema = SCHEMA.get(tbl_id)

    if not schema:
        continue

    tname = schema["table_name"]
    expected_cols = schema["columns"]

    if df.empty:
        canonical_dataframes[tname] = df
        continue

    if len(df.columns) >= len(expected_cols):
        df = df.iloc[:, :len(expected_cols)]

    df.columns = expected_cols[:len(df.columns)]

    canonical_dataframes[tname] = df

    print(f"✅ {tbl_id} → {tname}")

# ============================================================
# STEP 7: Save Outputs
# ============================================================

os.makedirs("output", exist_ok=True)

for name, df in canonical_dataframes.items():
    path = f"output/{name}.csv"
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path}")


# ============================================================
# STEP 9: Dump into DuckDB
# ============================================================

import duckdb

con = duckdb.connect("database.duckdb")

print("\n🦆 Writing to DuckDB...\n")

for tname, df in canonical_dataframes.items():

    if df.empty:
        print(f"⚠️ Skipped {tname}")
        continue

    con.register("temp_df", df)

    con.execute(f'DROP TABLE IF EXISTS "{tname}"')
    con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM temp_df')

    count = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]

    print(f"✅ {tname} → {count} rows")

con.close()

print("\n🦆 DuckDB saved → database.duckdb")