import os
import time
import json
import redis
import pymupdf4llm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# ==========================================
# 1. INFRASTRUCTURE SETUP
# ==========================================
# Connect to Redis Message Broker
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Initialize Qdrant (Persistent Local Storage)
# Using a relative path so it works bare-metal or mapped via Docker volumes
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_storage")
print(f"Hey, setting up Qdrant at {QDRANT_PATH}...")
qdrant = QdrantClient(path=QDRANT_PATH)

COLLECTION_NAME = "sec_filings"

# Create collection if it doesn't exist
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Load FastEmbed Dense Model (Runs locally, $0 API cost)
print("Loading up the FastEmbed model (BAAI/bge-small-en-v1.5)...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")

# ==========================================
# 2. CORE PROCESSING LOGIC
# ==========================================
def process_document(filepath):
    """Parses complex PDFs to clean Markdown and embeds them."""
    print(f"\n📄 Let's process this document: {filepath}")
    try:
        # pymupdf4llm natively preserves data tables as Markdown grids
        md_text = pymupdf4llm.to_markdown(filepath)
        
        # Chunking Strategy
        # Note: For production, upgrade this to Langchain's MarkdownHeaderTextSplitter
        chunks = md_text.split("\n\n")
        
        points = []
        # Generate a unique base ID based on timestamp
        point_id = int(time.time() * 1000) 
        
        for chunk in chunks:
            if len(chunk.strip()) > 20: # Ignore tiny artifacts
                # Generate embedding vector
                vector = list(embedding_model.embed([chunk]))[0]
                
                # Create Qdrant payload
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "text": chunk, 
                            "source": os.path.basename(filepath)
                        }
                    )
                )
                point_id += 1
        
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✅ Nice! Got {len(points)} chunks into Qdrant.")
            return True
            
    except Exception as e:
        print(f"❌ Oops, something went wrong with {filepath}: {str(e)}")
        return False

# ==========================================
# 3. BACKGROUND WORKER LOOP
# ==========================================
def run_worker():
    print("\n🛡️ Hey there! Ingestion worker is up and running.")
    print("Checking for any docs that might be waiting...")
    
    # 1. Initial Scan: Process anything currently sitting in the docs/ folder
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith(".pdf"):
                process_document(os.path.join(DOCS_DIR, filename))
                
    print("\n🎧 All ears on the Redis queue 'ingestion_tasks' for new files...")
    
    # 2. Polling Loop: Wait for the UI to tell us a new file arrived
    while True:
        # blpop blocks the thread until a message arrives (0 = wait forever)
        task = redis_client.blpop("ingestion_tasks", timeout=0)
        
        if task:
            _, message = task
            data = json.loads(message)
            file_name = data.get("file_name")
            
            if file_name:
                file_path = os.path.join(DOCS_DIR, file_name)
                if os.path.exists(file_path):
                    process_document(file_path)
                else:
                    print(f"⚠️ Hey, got a task but {file_path} seems to have vanished from disk.")

if __name__ == "__main__":
    run_worker()
