# AQUARIUM

The project covers experiments with different recommendation system approaches on MovieLens dataset. It predicts user ratings and recommends movies.
The recommendation system approaches cover:
- Similarity-based:
  - Content-based filtering with different similarity functions;
  - Collaborative filtering (user-user and item-item) with different similarity functions;
- Matrix factorization:
  - FunkSVD;
  - Alternating Least Squares;
- Ranking Heuristics & Graph-Based Signals:
  - Popularity-based;
  - Recency-based;
  - PageRank;
- Learning-to-Rank with Pairwise Optimization:
  - BPR-Opt;
- Hybrid 
---
## Project Structure
```
project_root/
 ├── data/                               # loaded datasets
 │   └── ml-1m/
 │       ├── movies.dat
 │       ├── ratings.dat
 │       └── users.dat
 ├── eda/                                # exploratory notebook
 ├── experiments/                        # notebooks with experiments
 │   ├── content_based_filtering.ipynb   # experiments with similarity-based content-based approach
 │   ├── similarity_based_cf.ipynb       # experiments with similarity-based collaborative filtering
 │   ├── svd_experiment.ipynb            # experiments with matrix factorization FunkSVD
 │   ├── als_experiment.ipynb            # experiments with matrix factorization ALS algorithm
 │   ├── bpr_opt.ipynb                   # experiments with BPR-Opt model
 │   ├── hybrid_experiment.ipynb         # experiments with hybrid approaches
 │   ├── online_evaluation_ab.ipynb      # online evaluation simulation with A/B tests
 │   └── online_evaluation_bandits.ipynb # online evaluation simulation with multi-armed bandits 
 ├── src/                                # core source code
 │   ├── models/                         # algorithms implementations
 │   ├── data_reading.py                 # raw data reading functions
 │   └── evaluation.py                   # metrics and evaluation functions
 ├── scripts/                            # helper scripts
 │   └── data_loader.py                  # data loader
 ├── requirements.txt                    # pip dependencies
 ├── Dockerfile                          # container build instructions
 ├── docker-compose.yml                  # container orchestration configuration
 └── README.md                           # project overview
```

---
## Installation and Usage

### Option 1: Docker (Recommended)

1. Copy the environment file and adjust if needed:
```bash
cp .env.example .env
```

2. Start the services with Docker Compose:
```bash
docker-compose up
```

This will:
- Download and extract the MovieLens 1M dataset to the `data/` folder
- Start Jupyter Lab on http://localhost:8888
- Mount your local project folders for live editing

3. Access Jupyter Lab at http://localhost:8888 (no token required by default)

To stop the services:
```bash
docker-compose down
```

### Option 2: Local Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Raw data files are not stored in the repo due to the files size.
To load the file archive and unzip it locally run:
```bash
python scripts/data_loader.py
```

3. After the files are unzipped, to open them use:
```python
from src.data_reading import read_movies_file, read_users_file, read_ratings_file
movies = read_movies_file()
ratings = read_ratings_file()
users = read_users_file()
```