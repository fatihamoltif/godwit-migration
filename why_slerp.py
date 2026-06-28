import numpy as np
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def to_unit_vector(lat, lon):
    """Convertit lat/lon (degrés) en vecteur 3D unitaire sur la sphère."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.array([np.cos(lat)*np.cos(lon),
                     np.cos(lat)*np.sin(lon),
                     np.sin(lat)])

def slerp_midpoint(lat1, lon1, lat2, lon2):
    """Point milieu le long du grand cercle (interpolation sphérique, t=0.5)."""
    v1, v2 = to_unit_vector(lat1, lon1), to_unit_vector(lat2, lon2)
    omega = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    if omega < 1e-9:
        v = v1
    else:
        v = (np.sin(0.5*omega)/np.sin(omega)) * (v1 + v2)
    lat = np.degrees(np.arcsin(v[2]))
    lon = np.degrees(np.arctan2(v[1], v[0]))
    return lat, lon

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"]).sort_values(
    ["Animal_ID", "Location_Timestamp_UTC"]).reset_index(drop=True)

bird_id = clean["Animal_ID"].value_counts().index[0]
bird = clean[clean["Animal_ID"] == bird_id].copy().reset_index(drop=True)
gap_h = bird["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600
bird["leg"] = (gap_h > 48).cumsum()
biggest_leg = bird["leg"].value_counts().index[0]
leg = bird[bird["leg"] == biggest_leg].reset_index(drop=True)
leg_step_km = haversine_km(leg["Latitude"].shift(), leg["Longitude"].shift(),
                           leg["Latitude"], leg["Longitude"])
i = int(leg_step_km.idxmax())
lat1, lon1 = leg.loc[i-1, ["Latitude", "Longitude"]]
lat2, lon2 = leg.loc[i,   ["Latitude", "Longitude"]]

lin_lat, lin_lon = (lat1+lat2)/2, (lon1+lon2)/2          # milieu linéaire
sph_lat, sph_lon = slerp_midpoint(lat1, lon1, lat2, lon2) # milieu sphérique
ecart = haversine_km(lin_lat, lin_lon, sph_lat, sph_lon)

print(f"Oiseau {bird_id}, leg le plus gros — plus grand pas interne :")
print(f"  P1 = ({lat1:.3f}, {lon1:.3f})")
print(f"  P2 = ({lat2:.3f}, {lon2:.3f})")
print(f"  distance P1->P2 : {leg_step_km.max():.0f} km")
print(f"  milieu linéaire  : ({lin_lat:.3f}, {lin_lon:.3f})")
print(f"  milieu sphérique : ({sph_lat:.3f}, {sph_lon:.3f})")
print(f"  >>> écart entre les deux milieux : {ecart:.1f} km")
