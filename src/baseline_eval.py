import os
import json
import uuid
import time
import redis
import requests
from datetime import datetime
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_storage")

print("Initializing Evaluation Harness...")
print("Ensure your Docker containers (Redis, Worker, Orchestrator) are running.")

# ==========================================
# 2. THE 15 GOLDEN TRAP QUESTIONS
# ==========================================
TRAP_QUESTIONS = [
    "What is the exact dosage of Lisinopril prescribed in the handwritten note?",
    "Does the patient's surgical history conflict with their reported penicillin allergy?",
    "According to the Q2 blood work table, what is the exact White Blood Cell (WBC) count?",
    "Did the doctor approve the patient for immediate weight-bearing exercises after surgery?",
    "What was the patient's resting heart rate recorded on the illegible intake form?",
    "Is there any record of a family history of Type 1 Diabetes?",
    "Based on the clinical notes, what is the exact date of the patient's next follow-up?",
    "What specific brand of pacemaker was implanted, and when?",
    "Are there any contraindications listed for prescribing Ibuprofen to this patient?",
    "What was the exact systolic blood pressure reading on the patient's second visit?",
    "Did the specialist recommend an MRI or a CT scan for the lumbar spine?",
    "According to the discharge summary, what is the max daily dose of Acetaminophen?",
    "What exact percentage of occlusion was found in the left anterior descending artery?",
    "Did the patient consent to the experimental trial as per the signed addendum?",
    "What is the prescribed frequency for the Albuterol inhaler?"
]

# ==========================================
# 3. BASELINE RAG HELPER (GROQ)
# ==========================================
def run_baseline(query: str) -> str:
    """Simulates a standard, naive RAG pipeline."""
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantClient(path=QDRANT_PATH)
    
    try:
        query_vector = list(embedding_model.embed([query]))[0]
        hits = qdrant.search(collection_name="medical_records", query_vector=query_vector.tolist(), limit=3)
        context = " ".join([hit.payload.get("parent_raw_text", "") for hit in hits])
    except Exception:
        context = "No documents found."

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Answer the query based on the context provided."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}"}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# ==========================================
# 4. AEGISAUDIT RAG HELPER (VIA REDIS)
# ==========================================
def run_aegis(query: str) -> str:
    task_id = str(uuid.uuid4())[:8]
    redis_client.rpush("audit_tasks", json.dumps({"task_id": task_id, "query": query}))
    
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"audit_updates_{task_id}")
    
    start_time = time.time()
    while time.time() - start_time < 30: 
        message = pubsub.get_message(ignore_subscribe_messages=True)
        if message:
            data = json.loads(message['data'])
            if "final_result" in data:
                return data["final_result"]
        time.sleep(0.1)
    return "TIMEOUT_ERROR"

# ==========================================
# 5. EXECUTION LOOP
# ==========================================
def run_evaluation():
    results = []
    print(f"\n🚀 Starting Evaluation of {len(TRAP_QUESTIONS)} Medical Traps...\n")
    
    for i, question in enumerate(TRAP_QUESTIONS, 1):
        print(f"Testing [{i}/{len(TRAP_QUESTIONS)}]: {question}")
        
        baseline_ans = run_baseline(question)
        aegis_ans = run_aegis(question)
        
        aegis_caught_trap = "SYSTEM WARNING" in aegis_ans
        
        results.append({
            "question": question,
            "baseline_dumb_rag_answer": baseline_ans,
            "aegisaudit_answer": aegis_ans,
            "trap_caught": aegis_caught_trap
        })
        print(f"   ↳ Trap Caught by Aegis? {'✅ YES' if aegis_caught_trap else '❌ NO'}\n")
        time.sleep(1) 

    report = {
        "timestamp": str(datetime.now()),
        "total_questions": len(TRAP_QUESTIONS),
        "total_caught_by_aegis": sum(1 for r in results if r["trap_caught"]),
        "results": results
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"🎉 Evaluation Complete! Report saved to 'evaluation_report.json'")

if __name__ == "__main__":
    run_evaluation()
