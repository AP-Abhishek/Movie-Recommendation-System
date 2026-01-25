import streamlit as st
import pandas as pd
import joblib
import requests
import ast

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("Movie Recommendation System")

st.markdown("""
    <style>
    [data-testid="stImage"] img {
        height: 400px;
        object-fit: contain;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    df = pd.read_csv('data/processed/movies.csv')

    poster_df = pd.read_csv('data/processed/poster.csv')
    df = df.merge(poster_df[['title', 'poster']], on='title', how='left')

    similarity = joblib.load('models/similarity_matrix.joblib')
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df, similarity

movies, similarity = load_assets()

st.sidebar.header("Filters")
all_genres = sorted(list(set([g for sublist in movies['genres'] for g in sublist])))
selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)

def get_recommendations(title, genre_filter):
    try:
        index = movies[movies['title'] == title].index[0]
        distances = similarity[index]
        candidate_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in candidate_indices:
            movie_row = movies.iloc[i[0]]
            if not genre_filter or any(g in movie_row['genres'] for g in  genre_filter):
                recommended_movies.append(movie_row)
            if len(recommended_movies) == 5:
                break
        return recommended_movies
    except:
        return []

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button("Show Recommendations"):
    recommendations = get_recommendations(selected_movie, selected_genres)

    if recommendations:
        cols = st.columns(5)
        for idx, movie in enumerate(recommendations):
            with cols[idx]:
                poster_url = movie['poster'] if pd.notna(movie['poster']) else "data/processed/not-found.png"
                st.image(poster_url)
                st.caption(movie['title'])
    else:
        st.warning("No matches found for that genre combination.")