import csv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer

# Define the CSV file path
csv_file_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Comments.csv"

# Initialize the Qdrant client (adjust host/port if needed)
client = QdrantClient(host="localhost", port=6333)
collection_name = "remacto_comments"

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = model.get_sentence_embedding_dimension()

# Create or recreate the collection with the desired vector parameters
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

points = []
with open(csv_file_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Combine fields to generate an embedding (customize the concatenation as desired)
        text_to_embed = " ".join([
            row["Theme"],
            row["What are the challenges/issues raised?"],
            row["What is the proposed solution?"]
        ])
        vector = model.encode(text_to_embed).tolist()
        
        # Prepare a point with a unique id, embedding vector, and payload
        point = {
            "id": int(row["Idea Number"]),
            "vector": vector,
            "payload": {
                "Idea Number": row["Idea Number"],
                "Channel": row["Channel"],
                "Theme": row["Theme"],
                "Challenges": row["What are the challenges/issues raised?"],
                "Solution": row["What is the proposed solution?"]
            }
        }
        points.append(point)

# Insert (upsert) the points into the collection
client.upsert(collection_name=collection_name, points=points)

print(f"Inserted {len(points)} points into the '{collection_name}' collection.")
