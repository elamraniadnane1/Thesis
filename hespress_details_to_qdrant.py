import pandas as pd
import uuid
import random
import datetime
import hashlib
import time
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import streamlit as st
# Set your OpenAI API key (alternatively, set OPENAI_API_KEY env variable)
openai.api_key = st.secrets["openai"]["api_key"]


# --- Qdrant Setup ---
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "hespress_politics_details"

if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        # text-embedding-ada-002 returns 1536-dimensional vectors
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("✅ Created collection:", collection_name)

# --- Helper Functions ---
def fake_date():
    """Generate a random date between Jan 1, 2018 and Dec 31, 2023."""
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d")

def chunk_text(text, max_words):
    """
    Splits text into chunks, each containing up to max_words words.
    This strategy approximates token count and prevents exceeding model limits.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def embed_text(text, model="text-embedding-ada-002", max_words_per_chunk=500):
    """
    Computes an embedding for the given text using OpenAI's API.
    If the text exceeds max_words_per_chunk, it is split into chunks and the embeddings are averaged.
    """
    words = text.split()
    if len(words) <= max_words_per_chunk:
        response = openai.Embedding.create(input=text, model=model)
        return response["data"][0]["embedding"]
    else:
        chunks = chunk_text(text, max_words_per_chunk)
        embeddings = []
        for chunk in chunks:
            response = openai.Embedding.create(input=chunk, model=model)
            embeddings.append(response["data"][0]["embedding"])
            time.sleep(0.2)  # delay to avoid rate limits
        avg_embedding = [sum(x)/len(x) for x in zip(*embeddings)]
        return avg_embedding

def create_point_from_row(row):
    """
    Creates a Qdrant point from a DataFrame row.
    The text used for embedding is a concatenation of 'title' and 'content'.
    """
    # Concatenate title and content to form the text for embedding.
    text = f"{row.get('title', '')} {row.get('content', '')}"
    embedding = embed_text(text)
    point_id = str(uuid.uuid4())
    payload = {
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "publication_date": row.get("publication_date", ""),
        "content": row.get("content", ""),
        "images": row.get("images", []),
        "date_added": fake_date()
    }
    return PointStruct(id=point_id, vector=embedding, payload=payload)

def process_csv_to_qdrant(csv_path, collection_name, chunk_size=100):
    """
    Reads a CSV file in chunks and upserts each chunk as points into Qdrant.
    """
    total_points = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        points = []
        for _, row in chunk.iterrows():
            point = create_point_from_row(row)
            points.append(point)
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                qdrant.upsert(collection_name=collection_name, points=points)
                success = True
            except Exception as e:
                print(f"Upsert failed (attempt {retries+1}): {e}")
                retries += 1
                time.sleep(2)
        if success:
            total_points += len(points)
            print(f"✅ Upserted chunk of {len(points)} points (Total so far: {total_points})")
        else:
            print("❌ Failed to upsert chunk after multiple retries.")
    print(f"✅ Finished upserting. Total points inserted: {total_points}")

# --- Example Usage ---
if __name__ == "__main__":
    csv_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\hespress_politics_details.csv"  # Update to your CSV file path
    process_csv_to_qdrant(csv_path, collection_name, chunk_size=100)
