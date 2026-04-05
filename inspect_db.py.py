import duckdb

conn = duckdb.connect("trai_5factor.duckdb")

# Get all table names dynamically
tables = conn.execute("SHOW TABLES").fetchall()
for (table_name,) in tables:
    print(f"\n📊 Table: {table_name}")
    
    # Row count
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Rows: {count}")
    
    # Show sample data
    df = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
    
    if df.empty:
        print("⚠️ No data in this table")
    else:
        print(df.head())