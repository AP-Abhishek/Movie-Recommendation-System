import pandas as pd
import ast
import os

def clean_data(movies=None, credits=None, save_to_disk=True):
    is_custom_input = movies is not None
    if movies is None and credits is None:
        movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')
        credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')
        df = movies.merge(credits, on='title')
    elif isinstance(movies, pd.DataFrame) and credits is None and 'overview' in movies.columns:
        df = movies.copy()
    else:
        df = movies.merge(credits, on='title')
    
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    df.dropna(inplace=True)
    
    def convert(obj):
        if isinstance(obj, list):
            return [i['name'] if isinstance(i, dict) else str(i) for i in obj]
        return [i['name'] for i in ast.literal_eval(obj)]

    def collapse(L): return [i.replace(" ", "") for i in L]
    
    df['genres'] = df['genres'].apply(convert).apply(collapse)
    df['keywords'] = df['keywords'].apply(convert).apply(collapse)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] if isinstance(i, dict) else str(i) for i in (x if isinstance(x, list) else ast.literal_eval(x))[:3]]).apply(collapse)
    df['crew'] = df['crew'].apply(lambda x: [i['name'].replace(" ", "") for i in (x if isinstance(x, list) else ast.literal_eval(x)) if isinstance(i, dict) and i.get('job') == 'Director'])
    df['overview'] = df['overview'].apply(lambda x: x.split() if isinstance(x, str) else x)
    
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    
    if os.path.exists('data/processed/poster.csv'):
        poster_df = pd.read_csv('data/processed/poster.csv').drop_duplicates(subset=['title'])
        df = df.merge(poster_df[['title', 'poster']], on='title', how='left')
    else:
        df['poster'] = "data/processed/not-found.png"
    
    final_df = df[['movie_id', 'title', 'tags', 'poster']]
    if save_to_disk and not is_custom_input:
        os.makedirs('data/processed', exist_ok=True)
        final_df.to_csv('data/processed/final_movies.csv', index=False)
    return final_df


if __name__ == "__main__":
    clean_data()

