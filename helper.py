import duckdb

excel_file = "data/India_10City_Consolidated_Tables.xlsx"
duckdb_file = "database.duckdb"

conn = duckdb.connect(duckdb_file)

print(conn.execute("SHOW TABLES").df())
# print(conn.execute("DESC TABLE infrastructure_fiber_and_ofc").df())

# infrastructure_cell_towers = conn.execute("SELECT * FROM infrastructure_cell_towers LIMIT 5").df()
# print("infrastructure_cell_towers  columns:", infrastructure_cell_towers.columns)

# infrastructure_fiber_and_ofc = conn.execute("SELECT * FROM infrastructure_fiber_and_ofc LIMIT 5").df()
# print("infrastructure_fiber_and_ofc columns:", infrastructure_fiber_and_ofc.columns)

# socio_economic_indicators = conn.execute("SELECT * FROM socio_economic_indicators LIMIT 5").df()
# print("socio_economic_indicators columns:", socio_economic_indicators.columns)

# digital_literacy = conn.execute("SELECT * FROM digital_literacy LIMIT 5").df()
# print("digital_literacy columns:", digital_literacy.columns)
