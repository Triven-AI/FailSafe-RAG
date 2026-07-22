import os
import json
import redis
import time
import re
from datetime import datetime
from typing import TypedDict, List, Annotated
import operator
from src.logger import get_logger
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

# ==========================================
# 1. INFRASTRUCTURE & TELEMETRY SETUP
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AegisAudit-VoiceGuard"

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant_server:6333")
COLLECTION_NAME = "enterprise_records"
CACHE_COLLECTION = "semantic_cache"

logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
while True:
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        qdrant.get_collections()
        logger.info("✅ Connected to Qdrant Server!")
        break
    except Exception:
        logger.info("⏳ Waiting for Qdrant Server...")
        time.sleep(2)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Groq Llama 3.3 70B Engine
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=1  # THE FIX: Prevents LangChain from freezing the UI for 60 seconds on rate limits
)

# Initialize Cache Collection
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
    validation: str
    status_log: Annotated[List[str], operator.add]
    final_answer: str

def publish_update(task_id: str, message: str):
    redis_client.publish(f"audit_updates_{task_id}", json.dumps({"status": message}))
    logger.info(f"[{task_id}] {message}")

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def retrieve(state: GraphState):
    query_vector = list(embedding_model.embed([state["query"]]))[0]
    
    # NEW QDRANT 1.18 SYNTAX
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3
    )
    
    docs = [hit.payload.get("parent_raw_text", hit.payload.get("text", "")) for hit in response.points]
    log = "Vector Search: Retrieved top parent context chunks."
    publish_update(state["task_id"], log)
    return {"documents": docs, "status_log": [log]}

def python_code_gate(state: GraphState):
    """Deterministic check ensuring numerical values (fees, %, dates) exist if queried."""
    text = " ".join(state["documents"])
    
    # Check if query asks for monetary amounts, SLA tiers, percentages, or dates
    implies_figures = bool(re.search(r'(\$|fee|cost|percentage|%|downtime|tier|date|total|tax|refund)', state["query"], re.IGNORECASE))
    has_figures = bool(re.search(r'(\$\d+|\d+%|\d+\s*hours?|\d+\s*days?|\d+)', text))
    
    if implies_figures and not has_figures:
        log = "Code Gate FAILED: Query implies specific terms/figures, but context lacks numerical data."
        publish_update(state["task_id"], log)
        return {"status_log": [log], "validation": "failed"}
        
    log = "Code Gate PASSED: Contextual heuristics match."
    publish_update(state["task_id"], log)
    return {"status_log": [log], "validation": "passed"}

def llm_critic(state: GraphState):
    """Llama 3.3 Contradiction & Sufficiency Check."""
    prompt = f"""
    Act as a strict Enterprise Call Center Auditor for OneInbox Voice AI.
    Does the context contain clear, accurate information to answer the query without hallucinating, and is it free of contradictions?
    Query: '{state['query']}'
    Context: {state['documents']}
    
    Respond strictly with TRUE if safe and fully grounded, or FALSE if missing critical details or contradictory.
    """
    response = llm.invoke(prompt).content.strip().upper()
    
    if "FALSE" in response:
        log = "Critic FAILED: Found conflicting or incomplete enterprise context."
        publish_update(state["task_id"], log)
        return {"status_log": [log], "validation": "failed"}
        
    log = "Critic PASSED: Data verified for Voice AI generation."
    publish_update(state["task_id"], log)
    return {"status_log": [log], "validation": "passed"}

def decompose_query(state: GraphState):
    """Voice-Safe Targeted Query Rewriter (1-Loop Max)."""
    new_count = state["retry_count"] + 1
    prompt = f"""
    Break down or rephrase this call center query to isolate the specific contract term, SLA, or invoice item missing:
    Query: {state['query']}
    Output only the revised short search query.
    """
    new_query = llm.invoke(prompt).content.strip()
    
    log = f"Decomposing Query (Attempt {new_count}): {new_query}"
    publish_update(state["task_id"], log)
    return {"query": new_query, "retry_count": new_count, "status_log": [log]}

def generate_answer(state: GraphState):
    prompt = f"""
    You are a Voice AI Agent representing OneInbox enterprise support.
    Answer the query strictly based on the provided context. Speak clearly and concisely.
    Query: {state['query']}
    Context: {state['documents']}
    """
    answer = llm.invoke(prompt).content.strip()
    
    log = "Generation complete."
    publish_update(state["task_id"], log)
    return {"final_answer": answer, "status_log": [log]}

def circuit_breaker(state: GraphState):
    """Voice-Ready Graceful Fallback for TTS engines."""
    log = "Circuit Breaker Tripped: Halting loop to prevent Voice AI hallucination."
    publish_update(state["task_id"], log)
    
    tts_fallback = (
        "SYSTEM WARNING: Insufficient or contradictory contract/policy data.\n\n"
        "🎙️ [Voice Agent Fallback]: 'I want to make sure I give you the exact right information "
        "regarding your account terms. Let me double-check our secondary records, or I can transfer "
        "you directly to a supervisor who can assist.'"
    )
    return {"final_answer": tts_fallback, "status_log": [log]}

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
    
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)
        
    log = "Enterprise Audit Trail Exported."
    publish_update(state["task_id"], log)
    redis_client.publish(f"audit_updates_{state['task_id']}", json.dumps({"final_result": state["final_answer"]}))
    return {"status_log": [log]}

# ==========================================
# 4. ROUTING LOGIC & GRAPH COMPILATION
# ==========================================
def evaluate_gates(state: GraphState):
    if state.get("validation") == "failed":
        if state["retry_count"] >= 1:  # Voice AI 1-Loop limit
            return "circuit_breaker"
        return "decompose_query"
    return "pass"

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("python_code_gate", python_code_gate)
workflow.add_node("llm_critic", llm_critic)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("circuit_breaker", circuit_breaker)
workflow.add_node("export_audit_trail", export_audit_trail)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "python_code_gate")

workflow.add_conditional_edges("python_code_gate", evaluate_gates, {
    "decompose_query": "decompose_query",
    "circuit_breaker": "circuit_breaker",
    "pass": "llm_critic"
})

workflow.add_conditional_edges("llm_critic", evaluate_gates, {
    "decompose_query": "decompose_query",
    "circuit_breaker": "circuit_breaker",
    "pass": "generate_answer"
})

workflow.add_edge("decompose_query", "retrieve")
workflow.add_edge("generate_answer", "export_audit_trail")
workflow.add_edge("circuit_breaker", "export_audit_trail")
workflow.add_edge("export_audit_trail", END)

aegis_app = workflow.compile()

# ==========================================
# 5. BACKGROUND WORKER LOOP
# ==========================================
def run_worker():
    logger.info("\n🧠 Voice-Guard LangGraph Orchestrator Active.")
    logger.info("🎧 Listening to Redis queue 'audit_tasks'...")
    
    while True:
        task = redis_client.blpop("audit_tasks", timeout=1)
        if task:
            _, message = task
            data = json.loads(message)
            task_id = data.get("task_id")
            query = data.get("query")
            
            if task_id and query:
                logger.info(f"\n🚀 Processing Task: {task_id}")
                
               # Semantic Cache Interceptor
                query_vector = list(embedding_model.embed([query]))[0]
                cache_response = qdrant.query_points(
                    collection_name=CACHE_COLLECTION,
                    query=query_vector.tolist(),
                    limit=1,
                    score_threshold=0.95 
                )
                
                if cache_response.points:
                    fast_answer = cache_response.points[0].payload["final_answer"]
                    publish_update(task_id, "⚡ SEMANTIC CACHE HIT: <100ms response.")
                    redis_client.publish(f"audit_updates_{task_id}", json.dumps({"final_result": fast_answer}))
                    continue
                
                publish_update(task_id, "🔍 Initiating Voice-Guard Agentic Loop...")
                inputs = {
                    "task_id": task_id,
                    "query": query, 
                    "documents": [], 
                    "retry_count": 0, 
                    "validation": "pending",
                    "status_log": [], 
                    "final_answer": ""
                }
                
                final_state = aegis_app.invoke(inputs)
                
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

def rewrite_query_node(state: dict) -> dict:
    """
    CRAG Node: Rewrites ambiguous or low-confidence user queries 
    to optimize vector database retrieval on retry.
    """
    logger.info("🔄 [CRAG] Triggering Query Rewriter Node...")
    original_query = state.get("query", "")
    attempt = state.get("attempt", 1)
    
    # Prompt LLM to expand/disambiguate query keywords
    prompt = (
        f"You are a search query optimizer for enterprise SLA contracts.\n"
        f"Original User Query: '{original_query}'\n"
        f"The previous search yielded low-confidence or missing context.\n"
        f"Rewrite this query into a concise, keyword-rich search phrase focusing on core legal terms, SLA penalties, or contract overrides."
    )
    
    try:
        # Call your existing LLM client (Groq)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        rewritten_query = response.choices[0].message.content.strip()
        logger.info(f"✨ [CRAG] Query Rewritten: '{rewritten_query}'")
    except Exception as e:
        logger.error(f"Failed to rewrite query: {e}")
        rewritten_query = original_query

    return {
        **state,
        "query": rewritten_query,
        "attempt": attempt + 1,
        "crag_triggered": True
    }

def route_after_critic(state: dict) -> str:
    confidence = state.get("confidence", 0.0)
    attempt = state.get("attempt", 1)
    
    if confidence >= 0.85:
        return "generate_safe_output"
    elif attempt == 1:
        # First failure: Trigger CRAG query re-writer instead of instantly quitting
        return "rewrite_query"
    else:
        # Second failure: Trip the Circuit Breaker
        return "trip_circuit_breaker"

if __name__ == "__main__":
    run_worker()