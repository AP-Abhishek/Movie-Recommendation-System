import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

st.markdown("""
    <style>
    [data-testid="stImage"] img {
        height: 400px;
        object-fit: cover;
        border-radius: 10px;
    }
    .movie-title {
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    df = pd.read_csv('data/processed/final_movies.csv')
    similarity = joblib.load('models/similarity_matrix.joblib')
    return df, similarity

movies, similarity = load_assets()

def get_recommendations(title):
    try:
        idx_list = movies[movies['title'] == title].index
        if idx_list.empty: return []
        
        index = idx_list[0]
        distances = similarity[index]
        top_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [movies.iloc[i[0]] for i in top_indices]
    except Exception:
        return []

selected_movie = st.selectbox("Type or select a movie", movies['title'].values)

if st.button("Show Recommendations"):
    recos = get_recommendations(selected_movie)
    if recos:
        cols = st.columns(5)
        for idx, movie in enumerate(recos):
            with cols[idx]:
                path = movie['poster'] if pd.notna(movie['poster']) else "data/processed/not-found.png"
                if not str(path).startswith('http') and not os.path.exists(str(path)):
                    path = "data/processed/not-found.png"
                st.image(path)
                st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
    else:
        st.error("No recommendations found.")
