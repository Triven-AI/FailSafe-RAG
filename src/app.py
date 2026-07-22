import os
import time
import json
import uuid
import redis
import requests
import streamlit as st
from qdrant_client import QdrantClient

# ==========================================
# 1. INFRASTRUCTURE SETUP
# ==========================================
st.set_page_config(page_title="AegisAudit Medical", layout="wide", page_icon="🛡️")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(url=QDRANT_URL)

# ==========================================
# 2. HELPER: THE "DUMB" BASELINE RAG (GROQ)
# ==========================================
def run_baseline_rag(query: str) -> str:
    """A standard RAG implementation with NO guardrails. Designed to fail."""
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantClient(url=QDRANT_URL)
    
    try:
        query_vector = list(embedding_model.embed([query]))[0]
        hits = qdrant.search(
            collection_name="medical_records",
            query_vector=query_vector.tolist(),
            limit=3
        )
        context = " ".join([hit.payload.get("parent_raw_text", "") for hit in hits])
    except Exception:
        context = "No documents ingested yet."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Answer the query based on the context."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}"}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# ==========================================
# 3. UI LAYOUT & TABS
# ==========================================
st.title("🛡️ FailSafe-RAG: Voice AI Knowledge Guard")

tab_production, tab_matrix = st.tabs(["Call Center Copilot", "Adversarial Matrix (Stress Test)"])

# ------------------------------------------
# TAB 1: PRODUCTION INTERFACE
# ------------------------------------------
with tab_production:
    st.markdown("Ensure Voice Agents never hallucinate refund policies or SLA terms mid-conversation.")
    
    uploaded_file = st.file_uploader("Upload Messy Knowledge Base (Scanned Policies, Handwritten Notes)", type=["pdf", "png", "jpg"])
    if uploaded_file is not None:
        file_path = os.path.join("./docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        redis_client.rpush("ingestion_tasks", json.dumps({"file_name": uploaded_file.name}))
        st.success(f"File '{uploaded_file.name}' pushed to Vision/OCR Ingestion Queue.")

    query = st.text_input("Enter Voice Agent Context Query:")
    
    if st.button("Execute Safe Audit"):
        if not query:
            st.warning("Please enter a query.")
        else:
            task_id = str(uuid.uuid4())[:8]
            redis_client.rpush("audit_tasks", json.dumps({"task_id": task_id, "query": query}))
            status_container = st.status("Agentic Routing...", expanded=True)
            pubsub = redis_client.pubsub()
            pubsub.subscribe(f"audit_updates_{task_id}")
            
            final_answer = None
            with st.spinner("Processing via LangGraph Path B..."):
                start_time = time.time()
                while time.time() - start_time < 60: 
                    message = pubsub.get_message(ignore_subscribe_messages=True)
                    if message:
                        data = json.loads(message['data'])
                        if "status" in data:
                            status_container.write(f"→ {data['status']}")
                        if "final_result" in data:
                            final_answer = data["final_result"]
                            break
                    time.sleep(0.2)
            
            status_container.update(label="Audit Complete", state="complete", expanded=False)
            
            if final_answer:
                if "SYSTEM WARNING" in final_answer:
                    st.error(final_answer)
                else:
                    st.success(final_answer)
            else:
                st.error("Task timed out.")

# ------------------------------------------
# TAB 2: LIVE HALLUCINATION MATRIX
# ------------------------------------------
with tab_matrix:
    st.markdown("### Real-Time Architecture Evaluation")
    st.write("Comparing standard RAG (Blind Trust) vs. FailSafe-RAG State Machine (Self-Correcting)")
    
    trap_question = st.selectbox("Select a Golden Trap Question:", [
        "According to the handwritten manager's note, is the customer eligible for a full refund after 45 days?",
        "Does the updated SLA policy contradict the legacy downtime guarantees in the scanned table?",
        "What is the exact penalty fee listed on the illegible customer invoice?"
    ])
    
    if st.button("Execute Side-by-Side Matrix"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("🛑 Baseline RAG")
            with st.spinner("Standard processing..."):
                start_time = time.time()
                baseline_answer = run_baseline_rag(trap_question)
                latency = round(time.time() - start_time, 2)
                st.write(f"**Answer:** {baseline_answer}")
                st.caption(f"Latency: {latency}s | Evaluation: Blind Trust")
                
        with col2:
            st.success("🛡️ FailSafe-RAG")
            task_id = str(uuid.uuid4())[:8]
            redis_client.rpush("audit_tasks", json.dumps({"task_id": task_id, "query": trap_question}))
            
            status_matrix = st.status("Agentic Loop Running...", expanded=True)
            pubsub = redis_client.pubsub()
            pubsub.subscribe(f"audit_updates_{task_id}")
            
            final_answer = None
            start_time = time.time()
            
            while time.time() - start_time < 60:
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = json.loads(message['data'])
                    if "status" in data:
                        status_matrix.write(f"→ {data['status']}")
                    if "final_result" in data:
                        final_answer = data["final_result"]
                        break
                time.sleep(0.2)
                
            latency = round(time.time() - start_time, 2)
            status_matrix.update(label="Loop Complete", state="complete", expanded=False)
            
            if "SYSTEM WARNING" in final_answer:
                st.error(f"**Answer:** {final_answer}")
            else:
                st.write(f"**Answer:** {final_answer}")
            st.caption(f"Latency: {latency}s | Evaluation: Guardrails Active")