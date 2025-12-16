import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import pickle

with open("model.pkl", "rb") as f:
    df, cosine_sim, indices, AVAILABLE_TITLES = pickle.load(f)

def get_recommendations(song_title, n=10):
    if song_title not in indices:
        return {"error": f"Song '{song_title}' not found"}

    idx = indices[song_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]

    song_indices = [i[0] for i in sim_scores]

    return df[["title", "artist"]].iloc[song_indices].to_dict("records")





print("1. Loading and Cleaning Data...")
try:
    
    df = pd.read_parquet('spotify_millsongdata.parquet') 
except FileNotFoundError:
    print("Error: 'spotify_millsongdata.parquet' not found. Ensure the file is in the current directory.")
    exit()


df = df.drop(columns=['link']).rename(columns={'song': 'title', 'text': 'lyrics'})
df['lyrics'] = df['lyrics'].fillna('')


SAMPLE_SIZE = 2500 
if len(df) > SAMPLE_SIZE:
    print(f"   -> Dataset reduced from {len(df)} to {SAMPLE_SIZE} songs (Memory Optimization)")
    
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

def clean_lyrics(text):
    text = str(text).replace('\n', ' ').replace('\r', ' ')
   
    return ' '.join(text.split()).lower()

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)


print("2. Applying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['cleaned_lyrics']) 


print(f"3. Calculating Cosine Similarity Matrix (Size: {len(df)}x{len(df)})...")

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) 


indices = pd.Series(df.index, index=df['title']).drop_duplicates()


AVAILABLE_TITLES = df['title'].tolist()

print("Data preparation complete. Ready to serve recommendations.")


def get_recommendations(title, n=10, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Function that takes a song title and returns the top 'n' recommended songs.
    """
    if title not in indices:
       
        return {"error": f"Song '{title}' not found in the reduced dataset of {len(df)} songs. Please search /available_songs for a valid song."}

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    song_indices = [i[0] for i in sim_scores]
    recommendations = df[['title', 'artist']].iloc[song_indices].to_dict('records')
    
    return recommendations