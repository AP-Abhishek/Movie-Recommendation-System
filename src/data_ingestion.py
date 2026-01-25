import os
from kaggle.api.kaggle_api_extended import KaggleApi

def fetch_data():
    api = KaggleApi()
    api.authenticate()

    dataset = 'tmdb/tmdb-movie-metadata'
    raw_path = 'data/raw/'

    api.dataset_download_files(
        dataset,
        path=raw_path,
        unzip=True
    )

if __name__ == "__main__":
    fetch_data()    