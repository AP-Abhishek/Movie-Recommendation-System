import pandas as pd
import ast
import os

def clean_data():
    movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    df.dropna(inplace=True)
    
    def convert(obj): return [i['name'] for i in ast.literal_eval(obj)]
    def collapse(L): return [i.replace(" ", "") for i in L]
    
    df['genres'] = df['genres'].apply(convert).apply(collapse)
    df['keywords'] = df['keywords'].apply(convert).apply(collapse)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]]).apply(collapse)
    df['crew'] = df['crew'].apply(lambda x: [i['name'].replace(" ", "") for i in ast.literal_eval(x) if i['job'] == 'Director'])
    df['overview'] = df['overview'].apply(lambda x: x.split())
    
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    
    if os.path.exists('data/processed/poster.csv'):
        poster_df = pd.read_csv('data/processed/poster.csv').drop_duplicates(subset=['title'])
        df = df.merge(poster_df[['title', 'poster']], on='title', how='left')
    else:
        df['poster'] = "data/processed/not-found.png"
    
    df[['movie_id', 'title', 'tags', 'poster']].to_csv('data/processed/final_movies.csv', index=False)

if __name__ == "__main__":
    clean_data()
