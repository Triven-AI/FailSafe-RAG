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
from src.logger import get_logger
logger = get_logger("AegisWorker")
# ==========================================
# 1. INFRASTRUCTURE SETUP
# ==========================================
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant_server:6333")

logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
while True:
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        qdrant.get_collections()
        logger.info("✅ Successfully connected to Qdrant Server!")
        break
    except Exception:
        logger.info("⏳ Qdrant database starting up, retrying in 2 seconds...")
        time.sleep(2)

COLLECTION_NAME = "enterprise_records"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

logger.info("Loading FastEmbed Dense Model...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ==========================================
# 2. RATE-LIMIT SAFE LLM HELPERS
# ==========================================
def make_groq_request_with_retry(payload: dict, max_retries: int = 5) -> dict:
    """Helper to handle Groq API calls with exponential backoff on Rate Limits (429)."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        resp_json = response.json()
        
        if response.status_code == 200 and 'choices' in resp_json:
            return resp_json
        
        # Check if rate limited
        if 'error' in resp_json and resp_json['error'].get('code') == 'rate_limit_exceeded':
            wait_time = (attempt + 1) * 5  # 5s, 10s, 15s...
            logger.info(f"  ⏳ Groq Rate Limit hit. Pausing {wait_time}s before retry (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
        else:
            logger.info(f"  ⚠️ Groq API Error: {resp_json}")
            time.sleep(3)
            
    raise Exception("Max retries exceeded for Groq API call.")

def generate_child_summary(raw_text: str) -> str:
    """Uses Groq to summarize enterprise contracts/invoices for clean vector search."""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a BPO data summarizer. Extract core pricing tiers, SLA downtime penalties, and refund conditions into a clean 2-sentence summary. NEVER omit dollar amounts, dates, or percentages."},
            {"role": "user", "content": raw_text}
        ]
    }
    try:
        resp_json = make_groq_request_with_retry(payload)
        return resp_json['choices'][0]['message']['content']
    except Exception:
        return raw_text[:200]  # Fallback to truncated raw text if all retries fail

def vision_transcribe_handwriting(image_path: str) -> str:
    """Uses Groq's Qwen Vision with dynamic image compression & rate-limit retries."""
    import base64
    import io
    
    # Resize image to keep payload small
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((1024, 1024))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this handwritten note or scanned invoice exactly. Maintain any tabular structure. If a number or fee is completely illegible, output [ILLEGIBLE]."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    
    resp_json = make_groq_request_with_retry(payload)
    return resp_json['choices'][0]['message']['content']

def determine_document_type(filepath: str) -> str:
    ext = filepath.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png']:
        return "vision_scan"
    elif ext == 'pdf':
        doc = fitz.open(filepath)
        page = doc[0]
        if page.get_text():
            return "clean_contract"
        return "scanned_pdf"
    return "unknown"

# ==========================================
# 3. CORE PROCESSING LOGIC (PARENT-CHILD)
# ==========================================
def process_document(filepath: str):
    logger.info(f"\n📄 Routing Document: {filepath}")
    doc_type = determine_document_type(filepath)
    raw_text = ""

    try:
        if doc_type == "clean_contract":
            logger.info("➡️ Route: PyMuPDF4LLM (Preserving SLA Tables)")
            raw_text = pymupdf4llm.to_markdown(filepath)
            
        elif doc_type == "scanned_pdf":
            logger.info("➡️ Route: Tesseract OCR (Extracting Scanned Text)")
            doc = fitz.open(filepath)
            pix = doc[0].get_pixmap()
            pix.save("temp.png")
            raw_text = pytesseract.image_to_string(Image.open("temp.png"))
            if os.path.exists("temp.png"): os.remove("temp.png")
            
        elif doc_type == "vision_scan":
            logger.info("➡️ Route: Vision LLM (Invoices & Overrides)")
            raw_text = vision_transcribe_handwriting(filepath)
            
        else:
            logger.info("❌ Error: Unsupported file format.")
            return False

        logger.info("🧠 Generating Child Summary for Vectorization...")
        chunks = raw_text.split("\n\n")
        points = []
        point_id = int(time.time() * 1000)

        for chunk in chunks:
            if len(chunk.strip()) > 30:
                child_summary = generate_child_summary(chunk)
                vector = list(embedding_model.embed([child_summary]))[0]
                
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
                time.sleep(0.5)  # Slight delay between chunks to prevent burst limit

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"✅ Success: Ingested {len(points)} Parent-Child chunks into Qdrant.")
            return True

    except Exception as e:
        logger.info(f"❌ Error processing {filepath}: {str(e)}")
        return False

# ==========================================
# 4. BACKGROUND WORKER LOOP
# ==========================================
def run_worker():
    logger.info("\n🛡️ AegisAudit (Enterprise) Ingestion Worker Started.")
    
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                process_document(file_path)
                time.sleep(2)  # 2s cooldown between files to respect 8k TPM limit
                
    logger.info("\n🎧 Listening to Redis queue 'ingestion_tasks' for new files...")
    while True:
        task = redis_client.blpop("ingestion_tasks", timeout=1)
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