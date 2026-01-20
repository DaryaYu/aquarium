# aquarium


1. Raw data files are not stored in the repo due to the files size. 
To load the file archive and unzip it locally run:

python scripts/data_loader.py

2. After the files are unzipped to open them, use:

from utils import read_movies_file, read_users_file, read_ratings_file
movies = read_movies_file()
ratings = read_ratings_file()
users = read_users_file()