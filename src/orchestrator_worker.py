import os
import json
import redis
import time
import re
from datetime import datetime
from typing import TypedDict, List, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

# ==========================================
# 1. INFRASTRUCTURE & TELEMETRY SETUP
# ==========================================
# Tracing (Fulfills AEGIS-OBS-001)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AegisAudit-Medical"

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Look for Qdrant across the Docker network, not localhost
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant_server:6333")

print(f"Connecting to Qdrant at {QDRANT_URL}...")
while True:
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        # Actually ping the database to force a connection test
        qdrant.get_collections()
        print("✅ Successfully connected to Qdrant Server!")
        break
    except Exception as e:
        print("⏳ Qdrant database starting up, retrying in 2 seconds...")
        time.sleep(2)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# The Groq API (Blistering Fast Llama 3.3 70B)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

# --- NEW: Initialize the Semantic Cache Collection ---
CACHE_COLLECTION = "semantic_cache"
if not qdrant.collection_exists(CACHE_COLLECTION):
    qdrant.create_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class GraphState(TypedDict):
    task_id: str
    query: str
    documents: List[str]
    retry_count: int
    # operator.add ensures logs append rather than overwrite
    status_log: Annotated[List[str], operator.add] 
    final_answer: str

# Helper to stream state back to UI via Redis Pub/Sub
def publish_update(task_id: str, message: str):
    redis_client.publish(f"audit_updates_{task_id}", json.dumps({"status": message}))
    print(f"[{task_id}] {message}")

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def retrieve(state: GraphState):
    query_vector = list(embedding_model.embed([state["query"]]))[0]
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=3
    )
    
    # Extract Parent Raw Text (Overcoming the Table Destruction Flaw)
    docs = [hit.payload.get("parent_raw_text", hit.payload.get("text", "")) for hit in hits]
    
    log = "Vector Search: Retrieved top parent context chunks."
    publish_update(state["task_id"], log)
    return {"documents": docs, "status_log": [log]}

def python_code_gate(state: GraphState):
    """Deterministic check to ensure numbers/dosages exist if queried."""
    text = " ".join(state["documents"])
    
    # If the user asks for dosage, mg, or dates, ensure the context actually has numbers
    needs_numbers = bool(re.search(r'(mg|dosage|date|born|weight)', state["query"], re.IGNORECASE))
    has_numbers = bool(re.search(r'\d+', text))
    
    if needs_numbers and not has_numbers:
        log = "Code Gate FAILED: Query implies numerical data, but context lacks figures."
        publish_update(state["task_id"], log)
        return {"status_log": [log], "validation": "failed"}
        
    log = "Code Gate PASSED: Contextual heuristics match."
    publish_update(state["task_id"], log)
    return {"status_log": [log], "validation": "passed"}

def llm_critic(state: GraphState):
    """Llama 3.3 Contradiction & Sufficiency Check."""
    prompt = f"""
    Act as a strictly objective Medical Auditor. 
    Does the following context contain sufficient information to answer the query, and is it free of internal contradictions?
    Query: '{state['query']}'
    Context: {state['documents']}
    Answer strictly TRUE (if it is safe and sufficient) or FALSE (if it contradicts or lacks data).
    """
    response = llm.invoke(prompt).content.strip().upper()
    
    if "FALSE" in response:
        log = "Critic FAILED: Found conflicting data or insufficient medical context."
        publish_update(state["task_id"], log)
        return {"status_log": [log], "validation": "failed"}
        
    log = "Critic PASSED: Data verified for Medical generation."
    publish_update(state["task_id"], log)
    return {"status_log": [log], "validation": "passed"}

def rewrite_query(state: GraphState):
    new_count = state["retry_count"] + 1
    prompt = f"Rewrite this medical audit query to be slightly broader to aid vector search: {state['query']}"
    new_query = llm.invoke(prompt).content
    
    log = f"Rewriting Query (Attempt {new_count}): {new_query}"
    publish_update(state["task_id"], log)
    return {"query": new_query, "retry_count": new_count, "status_log": [log]}

def generate_answer(state: GraphState):
    prompt = f"Answer the medical query strictly based on the context. If the context does not hold the answer, refuse to answer. Query: {state['query']}\nContext: {state['documents']}"
    answer = llm.invoke(prompt).content
    
    log = "Generation complete."
    publish_update(state["task_id"], log)
    return {"final_answer": answer, "status_log": [log]}

def circuit_breaker(state: GraphState):
    log = "Circuit Breaker Tripped: Halting autonomous loops to prevent hallucination."
    publish_update(state["task_id"], log)
    
    # NEW: Generate a clarifying question instead of just failing
    prompt = f"""
    The user asked: '{state['query']}'. 
    However, our retrieved medical context is insufficient or contradictory. 
    Generate a polite, 1-sentence clarifying question asking the user for a specific detail 
    (e.g., date of visit, specific doctor's name, or clarifying a dosage) to help us search better.
    """
    clarifying_q = llm.invoke(prompt).content.strip()
    
    final_message = f"🛑 **SYSTEM WARNING:** Insufficient or contradictory medical data found. Traceability mandate failed.\n\n**Next Step:** {clarifying_q}"
    
    return {"final_answer": final_message, "status_log": [log]}

def export_audit_trail(state: GraphState):
    os.makedirs("audit_logs", exist_ok=True)
    log_data = {
        "timestamp": str(datetime.now()),
        "task_id": state["task_id"],
        "original_query": state["query"],
        "retries": state["retry_count"],
        "final_output": state["final_answer"],
        "system_logs": state["status_log"]
    }
    filepath = f"audit_logs/audit_{state['task_id']}.json"
    
    # AEGIS-SEC-001 Placeholder: In production, AES-256 encryption occurs here
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)
        
    log = "Fiduciary Audit Trail Exported."
    publish_update(state["task_id"], log)
    redis_client.publish(f"audit_updates_{state['task_id']}", json.dumps({"final_result": state["final_answer"]}))
    return {"status_log": [log]}

# ==========================================
# 4. ROUTING LOGIC & GRAPH COMPILATION
# ==========================================
def evaluate_gates(state: GraphState):
    # LangGraph routing based on validation keys
    if state.get("validation") == "failed":
        if state["retry_count"] >= 2: # PRD requests up to 2 retries
            return "circuit_breaker"
        return "rewrite_query"
    return "pass"

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("python_code_gate", python_code_gate)
workflow.add_node("llm_critic", llm_critic)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("circuit_breaker", circuit_breaker)
workflow.add_node("export_audit_trail", export_audit_trail)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "python_code_gate")

workflow.add_conditional_edges("python_code_gate", evaluate_gates, {
    "rewrite_query": "rewrite_query",
    "circuit_breaker": "circuit_breaker",
    "pass": "llm_critic"
})

workflow.add_conditional_edges("llm_critic", evaluate_gates, {
    "rewrite_query": "rewrite_query",
    "circuit_breaker": "circuit_breaker",
    "pass": "generate_answer"
})

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", "export_audit_trail")
workflow.add_edge("circuit_breaker", "export_audit_trail")
workflow.add_edge("export_audit_trail", END)

aegis_app = workflow.compile()

# ==========================================
# 5. BACKGROUND WORKER LOOP
# ==========================================
def run_worker():
    print("\n🧠 AegisAudit LangGraph Orchestrator Started (Groq Engine Active).")
    print("🎧 Listening to Redis queue 'audit_tasks'...")
    
    while True:
        task = redis_client.blpop("audit_tasks", timeout=1)
        if task:
            _, message = task
            data = json.loads(message)
            task_id = data.get("task_id")
            query = data.get("query")
            
            if task_id and query:
                print(f"\n🚀 Processing Task: {task_id}")
                
                # ==========================================
                # THE SEMANTIC CACHE INTERCEPTOR
                # ==========================================
                query_vector = list(embedding_model.embed([query]))[0]
                
                # Check Qdrant for a 95% semantic match
                cache_hits = qdrant.search(
                    collection_name=CACHE_COLLECTION,
                    query_vector=query_vector.tolist(),
                    limit=1,
                    score_threshold=0.95 
                )
                
                if cache_hits:
                    # CACHE HIT! Bypass the LLM entirely.
                    fast_answer = cache_hits[0].payload["final_answer"]
                    publish_update(task_id, "⚡ SEMANTIC CACHE HIT: 95%+ Similarity Found. Bypassing AI generation.")
                    redis_client.publish(f"audit_updates_{task_id}", json.dumps({"final_result": fast_answer}))
                    print("✅ Answer served from Cache in < 100ms.")
                    continue 
                # ==========================================
                
                # If no cache hit, run the full LangGraph Path B loop
                publish_update(task_id, "🔍 No cache match. Initiating Agentic Loop...")
                inputs = {
                    "task_id": task_id,
                    "query": query, 
                    "documents": [], 
                    "retry_count": 0, 
                    "status_log": [], 
                    "final_answer": ""
                }
                
                # Execute the Graph
                final_state = aegis_app.invoke(inputs)
                
                # Save the new verified answer to the Semantic Cache for next time!
                if "SYSTEM WARNING" not in final_state["final_answer"]:
                    qdrant.upsert(
                        collection_name=CACHE_COLLECTION,
                        points=[
                            PointStruct(
                                id=int(time.time() * 1000),
                                vector=query_vector.tolist(),
                                payload={"query": query, "final_answer": final_state["final_answer"]}
                            )
                        ]
                    )

if __name__ == "__main__":
    run_worker()
