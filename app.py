# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="🎧 Podcast Recommender", layout="wide")

# Load TSV dataset (from Zenodo)
@st.cache_data
def load_data(nrows=3000):
    url = "https://zenodo.org/records/5834061/files/deezer_podcast_dataset.tsv?download=1"
    df = pd.read_csv(url, sep="\t", nrows=nrows)
    df = df[['title', 'description']].dropna().drop_duplicates()
    return df[df['description'].str.strip().astype(bool)].reset_index(drop=True)

df = load_data()

# TF-IDF vectorizer
@st.cache_resource
def create_vectorizer(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    matrix = vectorizer.fit_transform(data['description'])
    return vectorizer, matrix

vectorizer, tfidf_matrix = create_vectorizer(df)

# Header
st.title("🎧 Podcast Recommendation System")
st.markdown("Enter a topic or interest and get podcast recommendations based on descriptions.")

# Input
user_input = st.text_input("What are you interested in?", placeholder="e.g. tech news, startup interviews, fitness...")

# Recommendation logic
def get_recommendations(query, top_n=5):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        title = df.iloc[idx]['title']
        desc = df.iloc[idx]['description'][:160] + "..."
        score = round(sim_scores[idx] * 100, 2)
        results.append((title, desc, score))
    return results

# Show recommendations
if user_input:
    with st.spinner("Finding podcasts..."):
        recommendations = get_recommendations(user_input, top_n=5)
        for title, desc, score in recommendations:
            st.subheader(f"🎙️ {title} ({score}%)")
            st.write(desc)
            st.markdown("---")
