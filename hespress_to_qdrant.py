import pandas as pd
import uuid
import random
import datetime
import hashlib
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct

# --- Qdrant Setup ---
qdrant = QdrantClient(host="localhost", port=6333)
comments_collection = "hespress_politics_comments"

if not qdrant.collection_exists(comments_collection):
    qdrant.create_collection(
        collection_name=comments_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created collection:", comments_collection)

# --- Helper Functions ---
def fake_embedding():
    """Generate a dummy 384-dimensional vector with values in [-1, 1]."""
    return [random.uniform(-1, 1) for _ in range(384)]

def fake_date():
    """Generate a random date between Jan 1, 2018 and Dec 31, 2023."""
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d")

def generate_project_id(article_url):
    """
    Generate a project ID from the article URL.
    Using an MD5 hash ensures that comments on the same article share the same project ID.
    """
    return hashlib.md5(article_url.encode('utf-8')).hexdigest()

# --- Process CSV in Chunks and Upsert into Qdrant ---
def process_csv_to_qdrant(csv_path, collection_name, chunk_size=100):
    # A dictionary to ensure that the same article URL produces the same project_id
    project_id_map = {}
    total_points = 0
    
    # Use pandas to read CSV in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        points = []
        # Ensure required columns exist
        required_cols = {"article_url", "commenter", "comment_date", "comment"}
        if not required_cols.issubset(chunk.columns):
            raise ValueError(f"CSV file must contain columns: {required_cols}")

        for _, row in chunk.iterrows():
            article_url = row["article_url"]
            if article_url not in project_id_map:
                project_id_map[article_url] = generate_project_id(article_url)
            project_id = project_id_map[article_url]
            
            payload = {
                "article_url": article_url,
                "project_id": project_id,
                "commenter": row["commenter"],
                "comment_date": row["comment_date"],  # Use the provided date
                "comment": row["comment"],
                "date_added": fake_date()  # Timestamp when this data was ingested
            }
            
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=fake_embedding(),
                payload=payload
            )
            points.append(point)
        
        # Upsert this chunk into Qdrant
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                qdrant.upsert(collection_name=collection_name, points=points)
                success = True
            except Exception as e:
                print(f"Upsert failed (attempt {retries+1}): {e}")
                retries += 1
                time.sleep(2)  # Wait before retrying
        
        if success:
            total_points += len(points)
            print(f"✅ Upserted chunk of {len(points)} points (Total so far: {total_points})")
        else:
            print("❌ Failed to upsert chunk after multiple retries.")
    
    print(f"✅ Finished upserting. Total points inserted: {total_points}")

# --- Example Usage ---
if __name__ == "__main__":
    csv_path = r"C:\Users\Administrator\Desktop\Thesis\hespress_politics_comments.csv"  # Update to your file path
    process_csv_to_qdrant(csv_path, comments_collection, chunk_size=100)
