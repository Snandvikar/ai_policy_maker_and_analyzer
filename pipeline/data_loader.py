# import pandas as pd
# import duckdb
# import re

# excel_file = "data/India_54City_Master_Yearwise.xlsx"
# duckdb_file = "database.duckdb"

# conn = duckdb.connect(duckdb_file)

# def clean_table_name(sheet_name):
    
#     # convert to lowercase
#     name = sheet_name.lower()
    
#     # replace & with and
#     name = name.replace("&", "and")
    
#     # replace special chars with underscore
#     name = re.sub(r"[^\w]+", "_", name)
    
#     # remove leading/trailing underscores
#     name = name.strip("_")
    
#     return name

# def clean_column_name(col):
    
#     # convert to string
#     col = str(col)
    
#     # lowercase
#     col = col.lower()
    
#     # remove newline characters
#     col = col.replace("\n", " ")
    
#     # replace special symbols
#     col = col.replace("%", "percent")
#     col = col.replace("₹", "rs")
#     col = col.replace("≥", "ge")
#     col = col.replace("+", "plus")
#     col = col.replace("/", "_per_")
    
#     # replace km²
#     col = col.replace("km²", "km2")
    
#     # remove brackets
#     col = re.sub(r"[()]", "", col)
    
#     # replace special chars with underscore
#     col = re.sub(r"[^a-z0-9]+", "_", col)
    
#     # remove multiple underscores
#     col = re.sub(r"_+", "_", col)
    
#     # remove leading/trailing underscores
#     col = col.strip("_")
    
#     return col

# def make_unique_columns(columns):
#     seen = {}
#     new_cols = []
    
#     for col in columns:
#         if col not in seen:
#             seen[col] = 0
#             new_cols.append(col)
#         else:
#             seen[col] += 1
#             new_cols.append(f"{col}_{seen[col]}")
    
#     return new_cols

# def convert_numeric_columns(df):
    
#     for col in df.columns:
        
#         if df[col].dtype == "object":
            
#             cleaned = (
#                 df[col]
#                 .astype(str)
#                 .str.replace(",", "", regex=False)
#                 .str.replace("%", "", regex=False)
#                 .str.replace("₹", "", regex=False)
#                 .str.strip()
#             )
            
#             numeric = pd.to_numeric(cleaned, errors="coerce")
            
#             # Convert only if most values are numeric
#             if numeric.notna().sum() > len(df) * 0.5:
#                 df[col] = numeric
    
#     return df

# excel = pd.ExcelFile(excel_file)

# for sheet in excel.sheet_names:
    
#     print(f"Processing: {sheet}")
    
#     df = pd.read_excel(excel_file, sheet_name=sheet)
    
#     # Clean column names
#     df.columns = [clean_column_name(c) for c in df.columns]
#     df.columns = make_unique_columns(df.columns)
    
#     # Convert numeric values
#     df = convert_numeric_columns(df)
    
#     # Clean table name
#     table_name = clean_table_name(sheet)

#     if (table_name == "source_references") :
#         continue
    
#     # Drop empty columns
#     df = df.dropna(axis=1, how="all")
    
#     # Dump to DuckDB
#     conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

#     print(f"table - {table_name}: colums - {df.columns.tolist()}")

# print("table creation completed")

# # infrastructure_cell_towers = conn.execute("SELECT * FROM infrastructure_cell_towers LIMIT 5").df()
# # print("cell towers columns:", infrastructure_cell_towers.columns)

# # infrastructure_fiber_and_ofc = conn.execute("SELECT * FROM infrastructure_fiber_and_ofc LIMIT 5").df()
# # print("fiber and OFC columns:", infrastructure_fiber_and_ofc.columns)

# # socio_economic_indicators = conn.execute("SELECT * FROM socio_economic_indicators LIMIT 5").df()
# # print("socio economic indicators columns:", socio_economic_indicators.columns)

# # digital_literacy = conn.execute("SELECT * FROM digital_literacy LIMIT 5").df()
# # print("digital literacy columns:", digital_literacy.columns)



import pandas as pd
import duckdb
import re
import ast
from pathlib import Path

csv_file = "scraper/ndap_data/socio_economic_indicators_7150.csv"
duckdb_file = "database.duckdb"

conn = duckdb.connect(duckdb_file)


def clean_dataset_name(file_path):
    """
    socio_economic_indicators_7150.csv -> socio_economic_indicators
    """
    name = Path(file_path).stem
    name = re.sub(r"_\d+$", "", name)
    return name.lower()


def extract_avg(value):
    """
    Extract avg from dict-like string:
    {'avg':895.0, 'max':895.0, 'min':895.0}
    """
    if pd.isna(value):
        return value
    
    try:
        if isinstance(value, str) and "avg" in value:
            value_dict = ast.literal_eval(value)
            return value_dict.get("avg")
    except:
        pass
    
    return value


def extract_year(value):
    """
    Calendar Year (Jan - Dec) 2026 -> 2026
    """
    if pd.isna(value):
        return value
    
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match:
        return int(match.group())
    
    return value


def convert_numeric_columns(df):
    for col in df.columns:
        
        if df[col].dtype == "object":
            
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            
            numeric = pd.to_numeric(cleaned, errors="coerce")
            
            if numeric.notna().sum() > len(df) * 0.5:
                df[col] = numeric
    
    return df


print("Processing new dataset...")

df = pd.read_csv(csv_file)

# Drop CalendarDay if exists
if "CalendarDay" in df.columns:
    df = df.drop(columns=["CalendarDay"])


# Extract avg values
df = df.applymap(extract_avg)


# Extract year
year_columns = [c for c in df.columns if "year" in c.lower()]

for col in year_columns:
    df[col] = df[col].apply(extract_year)


# Convert numeric
df = convert_numeric_columns(df)


# Drop empty columns
df = df.dropna(axis=1, how="all")


# Clean table name
table_name = clean_dataset_name(csv_file)


# Store to DuckDB
# conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
print(df)
# print(f"Table created: {table_name}")
print("Columns:", df.columns.tolist())

print("Done.")