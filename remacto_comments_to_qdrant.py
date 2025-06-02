import csv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the CSV file path
csv_file_path = r"C:\Users\Administrator\Desktop\Thesis\REMACTO Comments.csv"

# Initialize the Qdrant client (adjust host/port if needed)
client = QdrantClient(host="localhost", port=6333)
collection_name = "remacto_comments"

# First, collect all texts to fit the vectorizer
all_texts = []
rows = []

with open(csv_file_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Combine fields to generate an embedding
        text_to_embed = " ".join([
            row.get("Theme", ""),
            row.get("What are the challenges/issues raised?", ""),
            row.get("What is the proposed solution?", "")
        ]).strip()
        
        if not text_to_embed:
            # Skip rows that are entirely blank
            continue
            
        all_texts.append(text_to_embed)
        rows.append(row)

# Create and fit the TF-IDF vectorizer
vector_size = 384  # Choose a reasonable dimension
vectorizer = TfidfVectorizer(max_features=vector_size)
vectorizer.fit(all_texts)

# Create or recreate the collection with the desired vector parameters
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

points = []
for idx, (text, row) in enumerate(zip(all_texts, rows)):
    # Convert sparse matrix to dense numpy array
    vector = vectorizer.transform([text]).toarray()[0]
    
    # Pad or truncate vector to match vector_size
    if len(vector) < vector_size:
        vector = np.pad(vector, (0, vector_size - len(vector)))
    elif len(vector) > vector_size:
        vector = vector[:vector_size]
    
    try:
        # Try to get the ID from the row, with fallback to the index
        id_value = int(row.get("Idea Number", idx))
    except (ValueError, TypeError):
        # If conversion fails, use the index as ID
        id_value = idx
    
    # Prepare a point with a unique id, embedding vector, and payload
    point = {
        "id": id_value,
        "vector": vector.tolist(),
        "payload": {
            "Idea Number": row.get("Idea Number", ""),
            "Channel": row.get("Channel", ""),
            "Theme": row.get("Theme", ""),
            "Challenges": row.get("What are the challenges/issues raised?", ""),
            "Solution": row.get("What is the proposed solution?", "")
        }
    }
    points.append(point)

# Insert (upsert) the points into the collection
client.upsert(collection_name=collection_name, points=points)

print(f"Inserted {len(points)} points into the '{collection_name}' collection using TF-IDF embeddings.")