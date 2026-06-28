import pandas as pd

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"


df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
)

print("Nombre d'oiseaux suivis :", df["Animal_ID"].nunique())
print("Période couverte      :",
      df["Location_Timestamp_UTC"].min(), "->", df["Location_Timestamp_UTC"].max())

print("\n--- Classes de précision Argos ---")
print(df["Location_Class"].value_counts(dropna=False))

print("\n--- Drapeau de plausibilité (0=plausible, 1=implausible) ---")
print(df["Location_DAF_Filter"].value_counts(dropna=False))

print("\n--- Statut de suivi ---")
print(df["Tracking_Status"].value_counts(dropna=False))

print("\n--- Vérif : les positions manquantes sont-elles des dead/shed ? ---")
print("Lignes sans Latitude :", df["Latitude"].isna().sum())
print(df.loc[df["Latitude"].isna(), "Tracking_Status"].value_counts(dropna=False))
