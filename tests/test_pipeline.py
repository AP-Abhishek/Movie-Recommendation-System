import os
import joblib
import pandas as pd
from src.preprocessing import clean_data
from src.recommender import build_matrix

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

    processed = clean_data(dummy_data, save_to_disk=False)

    assert 'tags' in processed.columns
    assert 'jamescameron' in processed['tags'].iloc[0]
    assert processed['tags'].iloc[0].islower()


def test_processed_file_exists():
    if not os.path.exists('data/processed/final_movies.csv') or len(pd.read_csv('data/processed/final_movies.csv')) < 10:
        clean_data()
    assert os.path.exists('data/processed/final_movies.csv')


def test_model_generation():
    build_matrix()
    assert os.path.exists('models/similarity_matrix.joblib')

    matrix = joblib.load('models/similarity_matrix.joblib')
    df = pd.read_csv('data/processed/final_movies.csv')

    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == len(df)

def test_recommendation_output():
    df = pd.read_csv('data/processed/final_movies.csv')
    similarity = joblib.load('models/similarity_matrix.joblib')

    movie_title = 'Avatar'
    movie_index = df[df['title'] == movie_title].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_titles = [df.iloc[i[0]].title for i in movies_list]

    assert len(recommended_titles) == 5
    assert isinstance(recommended_titles[0], str)