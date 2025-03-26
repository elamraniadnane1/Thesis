import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd
import re
import uuid
import random
import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct

# Optional: Set Tesseract command path if needed (adjust for your system)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Qdrant Setup ---
qdrant = QdrantClient(host="localhost", port=6333)
pdf_collection = "pdf_tables"

if not qdrant.collection_exists(pdf_collection):
    qdrant.create_collection(
        collection_name=pdf_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created Qdrant collection:", pdf_collection)

# --- Helper Functions ---

def fake_embedding():
    """Generate a dummy 384-dimensional vector."""
    return [random.uniform(-1, 1) for _ in range(384)]

def fake_date():
    """Generate a random date between Jan 1, 2018 and Dec 31, 2023."""
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d")

def ocr_extract_table_from_page(image):
    """
    Use pytesseract to extract text from an image.
    Attempt to parse the text into table rows and columns.
    This simple approach splits lines by newline and columns by multiple spaces.
    """
    # Configure Tesseract for Arabic (you can add more config if needed)
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(image, lang='ara', config=custom_config)
    rows = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            # Split columns by two or more spaces or tabs
            cols = re.split(r'\s{2,}|\t', line)
            rows.append(cols)
    return rows

def extract_tables_with_ocr(pdf_path):
    """
    Open the PDF using PyMuPDF, render each page as an image, apply OCR,
    and attempt to detect table rows.
    Returns a list of DataFrames (one per page with detected table rows).
    """
    doc = fitz.open(pdf_path)
    dfs = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        # Extract rows from the page using OCR
        table_rows = ocr_extract_table_from_page(image)
        if table_rows:
            # Assume the first row is header if it has more than one column
            if len(table_rows[0]) > 1:
                df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
            else:
                df = pd.DataFrame(table_rows)
            dfs.append(df)
    return dfs

def process_tables_to_points(tables):
    """
    Converts a list of DataFrames (tables) into Qdrant points.
    Each row is treated as a point; the row data is stored in the payload.
    """
    points = []
    for df in tables:
        # Skip empty dataframes
        if df.empty:
            continue
        # Ensure column names are strings
        df.columns = [str(col) for col in df.columns]
        for _, row in df.iterrows():
            payload = row.to_dict()
            # Add unique id and extraction date
            payload["row_id"] = str(uuid.uuid4())
            payload["date_extracted"] = fake_date()
            point = PointStruct(
                id=payload["row_id"],
                vector=fake_embedding(),
                payload=payload
            )
            points.append(point)
    return points

def pdf_to_qdrant(pdf_path, collection_name):
    # First, attempt to extract tables using OCR
    tables = extract_tables_with_ocr(pdf_path)
    if not tables:
        print("No tables found in the PDF using OCR.")
        return
    points = process_tables_to_points(tables)
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"✅ Inserted {len(points)} points into collection '{collection_name}'.")
    else:
        print("No valid data points to insert.")

# --- Example Usage ---
if __name__ == "__main__":
    pdf_path = "DGCT.pdf"  # Replace with your PDF file path containing Arabic tables
    pdf_to_qdrant(pdf_path, pdf_collection)
