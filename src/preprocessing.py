import pandas as pd
import ast

def load_and_merge():
    movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')

    return movies.merge(credits, on='title')

def convert_json(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def clean_data(df):
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(convert_json)
    df['keywords'] = df['keywords'].apply(convert_json)

    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    
    df['crew'] = df['crew'].apply(fetch_director)

    def collapse(L):
        return [i.replace(" ", "") for i in L]
    
    df['cast'] = df['cast'].apply(collapse)
    df['crew'] = df['crew'].apply(collapse)
    df['genres'] = df['genres'].apply(collapse)
    df['keywords'] = df['keywords'].apply(collapse)

    df['overview'] = df['overview'].apply(lambda x: x.split())
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']

    new_df = df[['movie_id', 'title', 'tags']].copy()
    new_df.loc[:, 'tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

    return new_df

if __name__ == "__main__":
    raw_df = load_and_merge()
    processed_df = clean_data(raw_df)
    processed_df.to_csv('data/processed/movies.csv', index=False)
