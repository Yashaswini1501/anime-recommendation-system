import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Title
st.title("🎌 Anime Recommendation System")

# Load dataset
df = pd.read_csv("anime.csv")

# Preprocessing
df['genre'] = df['genre'].fillna('')
df['rating'] = df['rating'].fillna(df['rating'].mean())

df['episodes'] = df['episodes'].replace('Unknown', np.nan)
df['episodes'] = df['episodes'].fillna(df['episodes'].median())

df['members'] = df['members'].fillna(df['members'].median())

df['episodes'] = df['episodes'].astype(float)

# Feature extraction
cv = CountVectorizer(tokenizer=lambda x: x.split(','), token_pattern=None)
genre_matrix = cv.fit_transform(df['genre'])

scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[['rating', 'episodes', 'members']])

feature_matrix = hstack([genre_matrix, num_features])

# Cosine similarity
cosine_sim = cosine_similarity(feature_matrix)

# Recommendation function
def recommend_with_threshold(title, threshold=0.6):
    idx = df[df['name'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    filtered = [i for i in sim_scores if i[1] >= threshold]
    
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in filtered[1:]]
    
    return df['name'].iloc[anime_indices].values

# ================= UI ================= #

# Dropdown
anime_list = df['name'].values
selected_anime = st.selectbox("🎬 Select an Anime", anime_list)

# Slider
threshold = st.slider("🔍 Similarity Threshold", 0.0, 1.0, 0.6)

# Button
if st.button("🚀 Recommend"):
    results = recommend_with_threshold(selected_anime, threshold)
    
    if len(results) == 0:
        st.warning("No recommendations found. Try lowering threshold.")
    else:
        st.subheader("✨ Recommended Anime:")
        for anime in results:
            st.write("👉", anime)