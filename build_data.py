import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Building recommendation model...")

df = pd.read_parquet("spotify_millsongdata.parquet")

df = df.drop(columns=["link"]).rename(columns={"song": "title", "text": "lyrics"})
df["lyrics"] = df["lyrics"].fillna("")

SAMPLE_SIZE = 2500
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

df["cleaned_lyrics"] = (
    df["lyrics"]
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.lower()
)

tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["cleaned_lyrics"])

cosine_sim = cosine_similarity(tfidf_matrix)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
available_titles = df["title"].tolist()

with open("model.pkl", "wb") as f:
    pickle.dump((df, cosine_sim, indices, available_titles), f)

print("✅ Model built & saved as model.pkl")
