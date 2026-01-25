import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix():
    df = pd.read_csv('data/processed/movies.csv')

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()

    similarity = cosine_similarity(vector)

    joblib.dump(similarity, 'models/similarity_matrix.joblib')

    df.to_csv('data/processed/movies_with_index.csv', index=False)

def recommend(movie_title):
    df = pd.read_csv('data/processed/movies_with_index.csv')
    similarity = joblib.load('models/similarity_matrix.joblib')

    try:
        movie_index = df[df['title'] == movie_title].index[0]
        distances = similarity[movie_index]

        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        for i in movies_list:
            print(df.iloc[i[0]].title)
    except IndexError:
        print("Movie not found in the database.")

if __name__ == "__main__":
    build_similarity_matrix()
    recommend("The Dark Knight")