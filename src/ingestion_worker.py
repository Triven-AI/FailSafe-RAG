import os
import time
import json
import redis
import requests
import fitz  # PyMuPDF
import pymupdf4llm
import pytesseract
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# ==========================================
# 1. INFRASTRUCTURE SETUP
# ==========================================
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_storage")
print(f"Initializing Qdrant locally at {QDRANT_PATH}...")
qdrant = QdrantClient(path=QDRANT_PATH)
COLLECTION_NAME = "medical_records"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

print("Loading FastEmbed Dense Model (BAAI/bge-small-en-v1.5)...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ==========================================
# 2. THE FORMAT ROUTER & LLM HELPERS
# ==========================================
def generate_child_summary(raw_text: str) -> str:
    """Uses a cheap LLM call to summarize raw, messy text for clean vector search."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a medical data summarizer. Extract the core entities, patient stats, and key facts from this text into a clean 2-sentence summary. Do not omit numbers."},
            {"role": "user", "content": raw_text}
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def vision_transcribe_handwriting(image_path: str) -> str:
    """Uses Llama 3.2 Vision to transcribe terrible doctor handwriting."""
    import base64
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3.2-90b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an expert pharmacist. Transcribe this doctor's handwritten note exactly. Maintain any tabular structure. If a word is completely illegible, output [ILLEGIBLE]."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def determine_document_type(filepath: str) -> str:
    """Basic Format Router heuristic."""
    ext = filepath.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png']:
        return "handwriting"
    elif ext == 'pdf':
        doc = fitz.open(filepath)
        page = doc[0]
        if page.get_text():
            return "clean_pdf"
        return "scanned_pdf"
    return "unknown"

# ==========================================
# 3. CORE PROCESSING LOGIC (PARENT-CHILD)
# ==========================================
def process_document(filepath: str):
    print(f"\n📄 Routing Document: {filepath}")
    doc_type = determine_document_type(filepath)
    raw_text = ""

    try:
        if doc_type == "clean_pdf":
            print("➡️ Route: PyMuPDF4LLM (Preserving Medical Tables)")
            raw_text = pymupdf4llm.to_markdown(filepath)
            
        elif doc_type == "scanned_pdf":
            print("➡️ Route: Tesseract OCR (Extracting Scanned Text)")
            # Note: For hackathon simplicity, reading just the first page image
            doc = fitz.open(filepath)
            pix = doc[0].get_pixmap()
            pix.save("temp.png")
            raw_text = pytesseract.image_to_string(Image.open("temp.png"))
            os.remove("temp.png")
            
        elif doc_type == "handwriting":
            print("➡️ Route: Vision LLM (Transcribing Handwriting)")
            raw_text = vision_transcribe_handwriting(filepath)
            
        else:
            print("❌ Error: Unsupported file format.")
            return False

        # --- PARENT-CHILD EMBEDDING STRATEGY ---
        print("🧠 Generating Child Summary for Vectorization...")
        chunks = raw_text.split("\n\n")
        points = []
        point_id = int(time.time() * 1000)

        for chunk in chunks:
            if len(chunk.strip()) > 30:
                # 1. Generate the clean summary (The Child)
                child_summary = generate_child_summary(chunk)
                
                # 2. Embed the summary for high-accuracy search
                vector = list(embedding_model.embed([child_summary]))[0]
                
                # 3. Store the RAW messy text (The Parent) in the payload for the generator
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "child_summary": child_summary,
                            "parent_raw_text": chunk, 
                            "source": os.path.basename(filepath),
                            "ingestion_route": doc_type
                        }
                    )
                )
                point_id += 1

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✅ Success: Ingested {len(points)} Parent-Child chunks into Qdrant.")
            return True

    except Exception as e:
        print(f"❌ Error processing {filepath}: {str(e)}")
        return False

# ==========================================
# 4. BACKGROUND WORKER LOOP
# ==========================================
def run_worker():
    print("\n🛡️ AegisAudit (Medical) Ingestion Worker Started.")
    
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                process_document(file_path)
                
    print("\n🎧 Listening to Redis queue 'ingestion_tasks' for new files...")
    while True:
        task = redis_client.blpop("ingestion_tasks", timeout=0)
        if task:
            _, message = task
            data = json.loads(message)
            file_name = data.get("file_name")
            if file_name:
                file_path = os.path.join(DOCS_DIR, file_name)
                if os.path.exists(file_path):
                    process_document(file_path)

if __name__ == "__main__":
    run_worker()
