import os
from pathlib import Path
import re as _re
import pandas as pd

# 🔥 Memory + threading control
os.environ["OMP_NUM_THREADS"] = "1"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# STEP 1: Define PDF Path
# ============================================================

pdf_path = "data/TELECOM_SUB_2025-07_pir_quarterly_1.pdf"
print(f"📄 Using file: {pdf_path}")


# ============================================================
# STEP 2: Configure Docling (Optimized)
# ============================================================

pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = False  # ⚠️ keep False unless scanned
pipeline_options.do_table_structure = True

# 🔥 Memory optimizations
pipeline_options.images_scale = 0.5
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ============================================================
# STEP 3: Convert PDF (Page-by-page)
# ============================================================

target_pages = [1, 2]
tables_found = []

print("🔄 Processing pages individually (memory-safe mode)...")

for page in target_pages:
    try:
        print(f"\n➡️ Processing page {page}...")

        result = converter.convert(
            pdf_path,
            page_range=(page, page)
        )

        doc = result.document

        for table_idx, table in enumerate(doc.tables):

            page_numbers = set()

            if hasattr(table, 'prov') and table.prov:
                for prov in table.prov:
                    page_numbers.add(prov.page_no)

            tables_found.append((table_idx, table, sorted(page_numbers)))

    except Exception as e:
        print(f"❌ Failed on page {page}: {e}")


print(f"\n✅ Total tables found: {len(tables_found)}")


# ============================================================
# STEP 4: Table Naming
# ============================================================

TABLE_NAME_MAP = {
    frozenset({1}): "wireline_subscriber_base",
    frozenset({2}): "wireless_mobile_subscriber_base",
}

def resolve_table_name(pages, table_idx):
    key = frozenset(pages)
    if key in TABLE_NAME_MAP:
        return TABLE_NAME_MAP[key]
    return f"table_page_{'_'.join(map(str, pages))}"


# ============================================================
# STEP 5: Convert Tables → DataFrames
# ============================================================

dataframes = {}

for table_idx, table, pages in tables_found:
    label = resolve_table_name(pages, table_idx)

    print(f"\n📊 Processing: {label} (pages: {pages})")

    try:
        df = table.export_to_dataframe()
        dataframes[label] = df

        print(f"Shape: {df.shape}")
        print(df.head())

    except Exception as e:
        print(f"⚠️ DataFrame conversion failed: {e}")


# ============================================================
# STEP 6: Cleaning Functions
# ============================================================

def slugify(text):
    s = str(text).strip().lower()
    s = s.replace('%', '_percent').replace('/', '_or_')
    s = _re.sub(r'[\s,\.\(\)\[\]\+\-&]+', '_', s)
    s = _re.sub(r'[^a-z0-9_]', '', s)
    s = _re.sub(r'_+', '_', s)
    return s.strip('_')


def clean_telecom_table(df):
    df = df.copy()

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(["", "nan", "None", "-", "NaN"], pd.NA, inplace=True)

    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Remove repeated header rows
    header_set = set(df.columns.astype(str))
    mask = df.apply(lambda r: set(r.astype(str)) == header_set, axis=1)
    df = df[~mask]

    # Clean columns
    df.columns = [slugify(c) for c in df.columns]

    # Deduplicate
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols

    # Numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""),
            errors="ignore"
        )

    return df.reset_index(drop=True)


# ============================================================
# STEP 7: Clean DataFrames
# ============================================================

cleaned_dataframes = {}

for label, df in dataframes.items():
    print(f"\n🧹 Cleaning: {label}")

    cleaned_df = clean_telecom_table(df)
    cleaned_dataframes[label] = cleaned_df

    print(f"Shape: {cleaned_df.shape}")
    print(f"Columns: {list(cleaned_df.columns)}")


# ============================================================
# STEP 8: Export
# ============================================================

os.makedirs("scraper/trai_output", exist_ok=True)

for label, df in cleaned_dataframes.items():
    path = f"scraper/trai_output/{label}.csv"
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


# Combined Excel
combined_path = "scraper/trai_output/TRAI_tables.xlsx"

with pd.ExcelWriter(combined_path) as writer:
    for label, df in cleaned_dataframes.items():
        df.to_excel(writer, sheet_name=label[:31], index=False)

print(f"\n✅ Combined Excel saved: {combined_path}")

# ============================================================
# STEP 9: Dump Cleaned Tables into DuckDB
# ============================================================

import duckdb

DUCK_DB_PATH = "database.duckdb"
con = duckdb.connect(DUCK_DB_PATH)

print(f"\n🦆 Writing {len(cleaned_dataframes)} tables to DuckDB → {DUCK_DB_PATH}\n")

written, skipped = [], []

for label, df in cleaned_dataframes.items():

    # Ensure SQL-safe table name
    tname = slugify(label)

    if df.empty:
        print(f"  ⚠️  {label} → {tname} — empty, skipped")
        skipped.append(tname)
        continue

    try:
        # Drop if exists
        con.execute(f'DROP TABLE IF EXISTS "{tname}"')

        # Register dataframe temporarily
        con.register("temp_df", df)

        # Create table
        con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM temp_df')

        # Row count
        row_count = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]

        written.append(tname)

        print(f"  ✅ {label}")
        print(f"      → table: {tname} | {row_count} rows × {len(df.columns)} cols")
        print(f"         cols: {list(df.columns)}\n")

    except Exception as e:
        print(f"  ❌ Failed to write {label}: {e}")
        skipped.append(tname)


# List all tables
all_tables = con.execute("SHOW TABLES").fetchall()
con.close()

print(f"{'='*60}")
print(f"🦆 DuckDB saved: {DUCK_DB_PATH}")
print(f"   Tables written : {len(written)}")

if skipped:
    print(f"   Skipped        : {skipped}")

print(f"   All tables in DB: {[t[0] for t in all_tables]}")

print(f"\n💡 Query example:")
print(f'import duckdb')
print(f'con = duckdb.connect("{DUCK_DB_PATH}")')

if written:
    print(f'con.execute("SELECT * FROM {written[0]} LIMIT 5").df()')