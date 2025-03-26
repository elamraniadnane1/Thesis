import csv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer

# Define the CSV file path
csv_file_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Projects.csv"

# Initialize the Qdrant client (update host/port as necessary)
client = QdrantClient(host="localhost", port=6333)
collection_name = "remacto_projects"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = model.get_sentence_embedding_dimension()

# (Re)create the collection with the desired vector configuration
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

points = []
with open(csv_file_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
        # Concatenate fields to form a text input for the embedding model
        text_to_embed = " ".join([
            row["Titles"],
            row["CT1"],
            row["CT2"],
            row["Topics"]
        ])
        vector = model.encode(text_to_embed).tolist()
        
        # Create a point with a unique id, the vector, and the original fields as payload
        point = {
            "id": idx,
            "vector": vector,
            "payload": {
                "Titles": row["Titles"],
                "CT1": row["CT1"],
                "CT2": row["CT2"],
                "Topics": row["Topics"]
            }
        }
        points.append(point)

# Insert (upsert) the points into the collection
client.upsert(collection_name=collection_name, points=points)

print(f"Inserted {len(points)} points into the '{collection_name}' collection.")
