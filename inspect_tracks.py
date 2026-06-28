import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path)

print("=== Forme (lignes, colonnes) ===")
print(df.shape)

print("\n=== Liste des colonnes ===")
print(list(df.columns))

print("\n=== Aperçu structurel (df.info) ===")
df.info()     
print("\n=== 5 premières lignes ===")
print(df.head())
