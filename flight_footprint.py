import sys; sys.path.insert(0, "src")
import pandas as pd
from godwit.features.kinematics import add_kinematics

FLIGHT_KMH = 15   # au-dessus : l'oiseau vole vraiment

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)

flights = df[df["ground_speed_kmh"] > FLIGHT_KMH].copy()
t = flights["Location_Timestamp_UTC"]

print(f"Pas en vol : {len(flights)} / {len(df)}  ({100*len(flights)/len(df):.0f} %)")
print(f"Oiseaux concernés : {flights['Animal_ID'].nunique()}")
print(f"Période couverte   : {t.min().date()} -> {t.max().date()}")

print(f"\nLatitude des vols  : {flights['Latitude'].min():.1f} -> {flights['Latitude'].max():.1f}")
print(f"Longitude (brute)  : {flights['Longitude'].min():.1f} -> {flights['Longitude'].max():.1f}")
print("  (^ si ça va de ~-180 à ~+180, c'est l'antiméridien : le Pacifique enjambe la ligne de date)")

print("\nPas de vol par mois (toutes années confondues) :")
print(flights.groupby(t.dt.month).size().to_string())

print("\nPas de vol par année :")
print(flights.groupby(t.dt.year).size().to_string())
