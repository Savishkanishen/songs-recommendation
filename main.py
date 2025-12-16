from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

from data_prep import get_recommendations, AVAILABLE_TITLES

app = FastAPI(
    title="Song Recommendation System API",
    description="Content-Based Music Recommendation using TF-IDF & Cosine Similarity",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Song Recommendation API is running"}

@app.get("/available_songs", response_model=List[str])
def available_songs():
    return AVAILABLE_TITLES

@app.get("/recommend/{song_title}", response_model=List[Dict[str, str]])
def recommend(song_title: str, limit: int = 10):

    processed_title = " ".join(
        word.capitalize() for word in song_title.split()
    )

    recommendations = get_recommendations(
        processed_title,
        limit
    )

    if "error" in recommendations:
        raise HTTPException(
            status_code=404,
            detail=recommendations["error"]
        )

    return recommendations
