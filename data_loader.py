import pandas as pd
import duckdb
import re

excel_file = "data/India_10City_Consolidated_Tables.xlsx"
duckdb_file = "database.duckdb"

conn = duckdb.connect(duckdb_file)

def clean_table_name(sheet_name):
    
    # convert to lowercase
    name = sheet_name.lower()
    
    # replace & with and
    name = name.replace("&", "and")
    
    # replace special chars with underscore
    name = re.sub(r"[^\w]+", "_", name)
    
    # remove leading/trailing underscores
    name = name.strip("_")
    
    return name

def clean_column_name(col):
    
    # convert to string
    col = str(col)
    
    # lowercase
    col = col.lower()
    
    # remove newline characters
    col = col.replace("\n", " ")
    
    # replace special symbols
    col = col.replace("%", "percent")
    col = col.replace("₹", "rs")
    col = col.replace("≥", "ge")
    col = col.replace("+", "plus")
    col = col.replace("/", "_per_")
    
    # replace km²
    col = col.replace("km²", "km2")
    
    # remove brackets
    col = re.sub(r"[()]", "", col)
    
    # replace special chars with underscore
    col = re.sub(r"[^a-z0-9]+", "_", col)
    
    # remove multiple underscores
    col = re.sub(r"_+", "_", col)
    
    # remove leading/trailing underscores
    col = col.strip("_")
    
    return col

def make_unique_columns(columns):
    seen = {}
    new_cols = []
    
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    
    return new_cols

excel = pd.ExcelFile(excel_file)

for sheet in excel.sheet_names:
    
    print(f"Processing: {sheet}")
    
    df = pd.read_excel(excel_file, sheet_name=sheet)
    
    # Clean column names
    df.columns = [clean_column_name(c) for c in df.columns]
    df.columns = make_unique_columns(df.columns)
    
    # Clean table name
    table_name = clean_table_name(sheet)
    
    # Drop empty columns
    df = df.dropna(axis=1, how="all")
    
    # Dump to DuckDB
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

print("table creation completed")

infrastructure_cell_towers = conn.execute("SELECT * FROM infrastructure_cell_towers LIMIT 5").df()
print("cell towers columns:", infrastructure_cell_towers.columns)

infrastructure_fiber_and_ofc = conn.execute("SELECT * FROM infrastructure_fiber_and_ofc LIMIT 5").df()
print("fiber and OFC columns:", infrastructure_fiber_and_ofc.columns)

socio_economic_indicators = conn.execute("SELECT * FROM socio_economic_indicators LIMIT 5").df()
print("socio economic indicators columns:", socio_economic_indicators.columns)

digital_literacy = conn.execute("SELECT * FROM digital_literacy LIMIT 5").df()
print("digital literacy columns:", digital_literacy.columns)
