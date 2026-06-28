from pathlib import Path
import urllib.request
import shutil
import zipfile

URL = ("https://www.sciencebase.gov/catalog/file/get/66f9a5f0d34ec4a45157a715"
       "?name=barTailedGodwit_USGS_ASC_argos_processedData%5BALL%5D.zip")

out_dir = Path("data/raw/usgs")
out_dir.mkdir(parents=True, exist_ok=True)
zip_path = out_dir / "processedData.zip"

print("Téléchargement en cours...")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
    shutil.copyfileobj(resp, f)
print(f"Zip téléchargé : {zip_path}  ({zip_path.stat().st_size/1e6:.1f} Mo)")

with zipfile.ZipFile(zip_path) as z:
    z.extractall(out_dir)
    names = z.namelist()
print(f"\nExtrait {len(names)} fichier(s) :")
for n in names:
    print("  -", n)
