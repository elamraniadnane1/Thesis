import csv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ← point to the real CSV file, not a .py
csv_file_path = r"C:\Users\Administrator\Desktop\Thesis\REMACTO Projects.csv"

# 1) Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "remacto_projects"

# 2) Instead of SentenceTransformer, use TF-IDF
# First, collect all texts to fit the vectorizer
all_texts = []
rows = []

with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Safely get each field; default to "" if missing
        title = row.get("Titles", "").strip()
        ct1 = row.get("CT1", "").strip()
        ct2 = row.get("CT2", "").strip()
        topics = row.get("Topics", "").strip()

        # Combine them into one string for embedding
        text_to_embed = " ".join([title, ct1, ct2, topics]).strip()
        if not text_to_embed:
            # Skip rows that are entirely blank
            continue
            
        all_texts.append(text_to_embed)
        rows.append(row)

# Create and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=384)  # Limit to 384 features for Qdrant compatibility
vectorizer.fit(all_texts)
vector_size = 384

# 3) (Re)create the collection with Cosine distance
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

points = []
for idx, (text, row) in enumerate(zip(all_texts, rows)):
    # Convert sparse matrix to dense numpy array and ensure fixed length
    vector = vectorizer.transform([text]).toarray()[0]
    
    # Pad or truncate vector to match vector_size
    if len(vector) < vector_size:
        vector = np.pad(vector, (0, vector_size - len(vector)))
    elif len(vector) > vector_size:
        vector = vector[:vector_size]
    
    point = {
        "id": idx,
        "vector": vector.tolist(),
        "payload": {
            "Titles": row.get("Titles", "").strip(),
            "CT1": row.get("CT1", "").strip(),
            "CT2": row.get("CT2", "").strip(),
            "Topics": row.get("Topics", "").strip()
        }
    }
    points.append(point)

# 4) Upsert points into Qdrant in one batch
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Inserted {len(points)} points into '{collection_name}' collection using TF-IDF embeddings.")