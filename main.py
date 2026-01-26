import streamlit as st
import pandas as pd
import joblib
import base64
import os
from PIL import Image

favicon_path = "src/favicon.png"
if os.path.exists(favicon_path):
    favicon = Image.open(favicon_path)
    st.set_page_config(page_title="Movie Recommendation System", page_icon=favicon, layout="wide")
    img_base64 = base64.b64encode(open(favicon_path, "rb").read()).decode()
else:
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    img_base64 = ""

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("src/favicon.png")

st.markdown(
    f"""
    <style>
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap; /* Allows wrapping on mobile */
        margin-top: -30px;
        margin-bottom: 20px;
        gap: 15px;
    }}
    
    .header-logo {{
        width: 80px; /* Base size for desktop */
        height: auto;
    }}
    
    .header-title {{
        text-align: center;
        font-size: clamp(1.5rem, 5vw, 3rem); /* Responsively scales between 1.5rem and 3rem */
        margin: 0;
    }}

    /* Mobile adjustments */
    @media (max-width: 640px) {{
        .header-container {{
            flex-direction: column; /* Stacks icon on top of text */
            margin-top: 0px;
        }}
        .header-logo {{
            width: 60px;
        }}
    }}
    </style>

    <div class="header-container">
        <img src="data:image/png;base64,{img_base64}" class="header-logo">
        <h1 class="header-title">Movie Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

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