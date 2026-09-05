import streamlit as st
import pandas as pd
import joblib
import base64
import os
from PIL import Image

from src.preprocessing import clean_data
from src.recommender import build_matrix



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
    .header-logo {{ width: 80px; height: auto; }}
    .header-title {{ text-align: center; font-size: clamp(1.5rem, 5vw, 3rem); margin: 0; }}

    .main-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        width: 100%;
    }}

    .movie-card {{
        flex: 0 1 calc(20% - 20px);
        min-width: 200px;
        text-align: center;
    }}

    .movie-card img {{
        width: 100%;
        height: 400px;
        object-fit: cover;
        border-radius: 10px;
    }}

    .movie-title {{
        font-weight: bold;
        margin-top: 10px;
        word-wrap: break-word;
    }}

    @media (max-width: 1000px) {{
        .movie-card {{ flex: 0 1 calc(33.33% - 20px); }}
    }}
    @media (max-width: 700px) {{
        .movie-card {{ flex: 0 1 calc(50% - 20px); }}
    }}
    @media (max-width: 450px) {{
        .movie-card {{ flex: 0 1 100%; }}
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
    csv_path = 'data/processed/final_movies.csv'
    matrix_path = 'models/similarity_matrix.joblib'
    
    if not os.path.exists(csv_path):
        clean_data()
    if not os.path.exists(matrix_path):
        os.makedirs('models', exist_ok=True)
        build_matrix()
        
    df = pd.read_csv(csv_path)
    similarity = joblib.load(matrix_path)
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
        html_content = '<div class="main-grid">'
        for movie in recos:
            path = movie['poster'] if pd.notna(movie['poster']) else "data/processed/not-found.png"
            
            if not str(path).startswith('http'):
                if os.path.exists(str(path)):
                    img_data = get_base64_image(path)
                    src = f"data:image/png;base64,{img_data}"
                else:
                    src = f"data:image/png;base64,{get_base64_image('data/processed/not-found.png')}"
            else:
                src = path

            html_content += f'<div class="movie-card"><img src="{src}"><div class="movie-title">{movie["title"]}</div></div>'
        
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.error("No recommendations found.")