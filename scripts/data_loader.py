import os
import zipfile
import urllib.request
from urllib.parse import urlparse
from pathlib import Path


URL = os.getenv(
    "DATASET_URL", "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
)
DATA_DIR = os.getenv("DATA_DIR", "data")

dataset_name = "ml-1m"
dataset_path = os.path.join(DATA_DIR, dataset_name)

if os.path.exists(dataset_path) and os.path.exists(
    os.path.join(dataset_path, "ratings.dat")
):
    print(f"Dataset already exists in {dataset_path}")
    print("Skipping download.")
else:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    zip_name = os.path.basename(urlparse(URL).path)
    zip_path = os.path.join(DATA_DIR, zip_name)

    print(f"Downloading dataset from {URL}...")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"Zip archive downloaded!")

    print(f"Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as arch:
        arch.extractall(DATA_DIR)
    print(f"Files extracted to the folder: {DATA_DIR}")

    os.remove(zip_path)
    print(f"Cleaned up {zip_path}")

print(f"\nDataset ready at: {dataset_path}")
