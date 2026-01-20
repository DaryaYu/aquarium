import os
import zipfile
import urllib.request
from urllib.parse import urlparse


URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
DATA_DIR = 'data'


# Retrieve zip archive name
zip_name = os.path.basename(urlparse(URL).path)
zip_path = os.path.join(DATA_DIR, zip_name)

# Download the archive
urllib.request.urlretrieve(URL, zip_path)
print(f"Zip archive downloaded!")

# Extract files from the archive
with zipfile.ZipFile(zip_path, "r") as arch:
    arch.extractall(DATA_DIR)
print(f"Files extracted to the folder: {DATA_DIR}")