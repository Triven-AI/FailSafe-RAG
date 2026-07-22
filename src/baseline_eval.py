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
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

print("Initializing Enterprise Evaluation Harness...")

# ==========================================
# 2. THE 15 ENTERPRISE GOLDEN TRAP QUESTIONS
# ==========================================
TRAP_QUESTIONS = [
    "According to the legacy SLA contract, what is the exact downtime penalty percentage for the Pro Tier?",
    "Does the handwritten manager's note authorize a full refund, or just a 50% credit?",
    "What is the exact total amount due, including tax, on the Acme Corp invoice?",
    "Based on the illegible scanned form, what date was the cancellation request officially received?",
    "Did the account manager approve the customer to bypass the 30-day waiting period?",
    "Is there any record of a prior chargeback on the customer's Q2 billing statement?",
    "What specific hardware SKU was billed on the third line item of the damaged invoice?",
    "Are there any contraindications or specific clauses preventing us from upgrading this legacy account?",
    "What was the exact overage fee charged for exceeding the API limit?",
    "Did the client sign the waiver acknowledging the data-loss risks?",
    "According to the Q3 billing table, what is the exact cost per seat for the Enterprise plan?",
    "What is the maximum allowed response time (in hours) listed under the critical severity tier?",
    "What exact percentage of the service fee is non-refundable?",
    "Did the customer consent to the auto-renewal terms in the scanned addendum?",
    "Who is the authorized signatory listed at the bottom of the faded contract?"
]

# ==========================================
# 3. BASELINE RAG HELPER (NAIVE LLM)
# ==========================================
def run_baseline(query: str) -> str:
    """Simulates a standard RAG pipeline (High Liability)."""
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantClient(url=QDRANT_URL)
    
    try:
        query_vector = list(embedding_model.embed([query]))[0]
        response = qdrant.query_points(
            collection_name="enterprise_records", 
            query=query_vector.tolist(), 
            limit=3
        )
        context = " ".join([hit.payload.get("parent_raw_text", "") for hit in response.points])
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
# 4. VOICEGUARD RAG HELPER (VIA REDIS)
# ==========================================
def run_voiceguard(query: str) -> str:
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
    print(f"\n🚀 Starting Evaluation of {len(TRAP_QUESTIONS)} Enterprise Traps...\n")
    
    for i, question in enumerate(TRAP_QUESTIONS, 1):
        print(f"Testing [{i}/{len(TRAP_QUESTIONS)}]: {question}")
        
        baseline_ans = run_baseline(question)
        aegis_ans = run_voiceguard(question)
        
        # Check if VoiceGuard successfully blocked a hallucination with a fallback
        aegis_caught_trap = "SYSTEM WARNING" in aegis_ans
        
        results.append({
            "question": question,
            "baseline_dumb_rag_answer": baseline_ans,
            "voiceguard_answer": aegis_ans,
            "liability_prevented": aegis_caught_trap
        })
        print(f"   ↳ Liability Prevented by VoiceGuard? {'✅ YES' if aegis_caught_trap else '❌ NO'}\n")
        time.sleep(2)  # Respect Groq Rate Limits

    report = {
        "timestamp": str(datetime.now()),
        "total_questions": len(TRAP_QUESTIONS),
        "total_liabilities_prevented": sum(1 for r in results if r["liability_prevented"]),
        "results": results
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"🎉 Evaluation Complete! Report saved to 'evaluation_report.json'")

if __name__ == "__main__":
    run_evaluation()