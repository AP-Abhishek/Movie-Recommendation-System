import os
import pandas as pd
from src.preprocessing import clean_data

def test_raw_files_exists():
    assert os.path.exists('data/raw/tmdb_5000_movies.csv')
    assert os.path.exists('data/raw/tmdb_5000_credits.csv')

def test_clean_data_output():
    dummy_data = pd.DataFrame({
        'movie_id': [1],
        'title': ['Avatar'],
        'overview': ['A 22nd century action film'],
        'genres': ['[{"name": "Action"}]'],
        'keywords': ['[{"name": "space"}]'],
        'cast': ['[{"name": "Sam Worthington"}]'],
        'crew': ['[{"job": "Director", "name": "James Cameron"}]']
    })

    processed = clean_data(dummy_data)

    assert 'tags' in processed.columns
    assert 'jamescameron' in processed['tags'].iloc[0]
    assert processed['tags'].iloc[0].islower()

def test_processed_file_exists():
    assert os.path.exists('data/processed/movies.csv')