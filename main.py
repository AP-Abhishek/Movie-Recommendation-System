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
        flex-wrap: wrap;
        margin-top: -30px;
        margin-bottom: 20px;
        gap: 15px;
    }}
    
    .header-logo {{
        width: 80px;
        height: auto;
    }}
    
    .header-title {{
        text-align: center;
        font-size: clamp(1.5rem, 5vw, 3rem);
        margin: 0;
    }}

    @media (max-width: 640px) {{
        .header-container {{
            flex-direction: column;
            margin-top: 0px;
        }}
        .header-logo {{
            width: 60px;
        }}
    }}
    
    [data-testid="stImage"] img {{
        height: 400px;
        object-fit: cover;
        border-radius: 10px;
    }}
    
    .movie-title {{
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
    }}
    </style>

    <div class="header-container">
        <img src="data:image/png;base64,{img_base64}" class="header-logo">
        <h1 class="header-title">Movie Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>

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
