import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uvicorn

QDRANT_URL = "http://qdrant_db:6333"
COLLECTION = "semantic_search"
DOCS_PATH = "data/docs.txt"

app = FastAPI()

# Allow all CORS for testing / local use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Embedding Model (768-dim, high accuracy)
print("Loading Embedding Model: BGE Base...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Connect to Qdrant
client = QdrantClient(QDRANT_URL)

def ingest_documents():
    print("Reading documents from:", DOCS_PATH)

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"docs.txt not found at {DOCS_PATH}")

    # Load dataset
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(documents)} documents.")

    # Create/recreate collection with 768 dims
    print("Creating Qdrant Collection...")
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedder.encode(documents, normalize_embeddings=True)

    # Upload points
    print("Uploading to Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"text": documents[i]}
        )
        for i in range(len(documents))
    ]

    client.upload_points(collection_name=COLLECTION, points=points)

    print("✅ Ingestion Completed Successfully!")

@app.get("/")
def home():
    return {"message": "Semantic Search API is running!"}

@app.get("/ingest")
def trigger_ingest():
    """Manually call this to ingest docs.txt into Qdrant."""
    ingest_documents()
    return {"status": "success", "message": "Documents ingested!"}

@app.get("/search")
def search(query: str, top_k: int = 5):
    """Semantic Search Endpoint"""
    print("Received Query:", query)

    # Embed query
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]

    # Search in Qdrant
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k
    )

    return [
        {
            "score": float(res.score),
            "text": res.payload["text"]
        }
        for res in results
    ]
