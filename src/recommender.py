import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_matrix():
    df = pd.read_csv('data/processed/final_movies.csv')
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vector)
    joblib.dump(similarity, 'models/similarity_matrix.joblib')

if __name__ == "__main__":
    build_matrix()
