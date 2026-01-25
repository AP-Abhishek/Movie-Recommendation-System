import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("Movie Recommendation System")

@st.cache_resource
def load_assets():
    df = pd.read_csv('data/processed/movies.csv')
    similarity = joblib.load('models/similarity_matrix.joblib')
    return df, similarity

movies, similarity = load_assets()

def get_recommendations(title):
    try:
        index = movies[movies['title'] == title].index[0]
        distances = similarity[index]
        top_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        return [movies.iloc[i[0]].title for i in top_indices]
    except IndexError:
        return ["Error: Movie not found or index mismatch."]

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button("Show Recommendations"):
    recommendations = get_recommendations(selected_movie)

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(recommendations[idx])