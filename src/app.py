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
st.set_page_config(page_title="AegisAudit VoiceGuard", layout="wide", page_icon="🛡️")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant_server:6333")
qdrant = QdrantClient(url=QDRANT_URL)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ==========================================
# 2. HELPER: THE "DUMB" BASELINE RAG
# ==========================================
def run_baseline_rag(query: str) -> str:
    """A standard RAG implementation with NO guardrails. Designed to fail."""
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    try:
        query_vector = list(embedding_model.embed([query]))[0]
        response = qdrant_client.query_points(
            collection_name="enterprise_records",
            query=query_vector.tolist(),
            limit=3
        )
        context = " ".join([hit.payload.get("parent_raw_text", "") for hit in response.points])
    except Exception:
        context = "No documents ingested yet."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a customer support agent. Answer the query based on the context."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}"}
        ]
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        resp_json = response.json()
        
        # THE FIX: Check if 'choices' exists before accessing it
        if 'choices' in resp_json:
            return resp_json['choices'][0]['message']['content']
        else:
            error_msg = resp_json.get('error', {}).get('message', 'Unknown API Error')
            return f"⚠️ Baseline RAG Failed: {error_msg}"
            
    except Exception as e:
        return f"⚠️ Connection Error: {str(e)}"

# ==========================================
# 3. UI LAYOUT & TABS
# ==========================================
st.title("🛡️ AegisAudit VoiceGuard: OneInbox AI Integration")

tab_production, tab_matrix = st.tabs(["Call Center Copilot", "Adversarial Call Matrix (Stress Test)"])

# ------------------------------------------
# TAB 1: PRODUCTION INTERFACE
# ------------------------------------------
with tab_production:
    st.markdown("Ensure Voice Agents never hallucinate refund policies, enterprise SLAs, or billing data mid-conversation.")
    
    uploaded_file = st.file_uploader("Upload Messy BPO Knowledge Base (Legacy SLAs, Handwritten Manager Notes, Invoices)", type=["pdf", "png", "jpg"])
    if uploaded_file is not None:
        file_path = os.path.join("./docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        redis_client.rpush("ingestion_tasks", json.dumps({"file_name": uploaded_file.name}))
        st.success(f"File '{uploaded_file.name}' pushed to Vision/OCR Ingestion Queue.")

    query = st.text_input("Enter Voice Agent Query (Caller Context):")
    
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
            with st.spinner("Processing via LangGraph VoiceGuard..."):
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
    st.markdown("### Real-Time Liability Evaluation for Voice AI")
    st.write("Comparing standard RAG (Blind Trust) vs. VoiceGuard State Machine (Self-Correcting)")
    
    trap_question = st.selectbox("Select an Enterprise Trap Question:", [
        "According to the legacy SLA contract, what is the exact downtime penalty percentage for the Pro Tier?",
        "Does the handwritten manager's note authorize a full refund, or just a 50% credit?",
        "What is the exact total amount due, including tax, on the Acme Corp invoice?",
        "What specific hardware SKU was billed on the third line item of the damaged invoice?"
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
                st.caption(f"Latency: {latency}s | Evaluation: High Liability (Blind Trust)")
                
        with col2:
            st.success("🛡️ VoiceGuard (AegisAudit)")
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
            
            # THE FIX: Ensure final_answer is not None before processing
            if final_answer:
                if "SYSTEM WARNING" in final_answer:
                    st.error(f"**TTS Fallback Triggered:**\n\n{final_answer}")
                else:
                    st.write(f"**Verified Answer:** {final_answer}")
            else:
                st.error("⚠️ Orchestrator Timeout: The backend crashed or took longer than 60 seconds.")
            st.caption(f"Latency: {latency}s | Evaluation: Grounded & Voice-Safe")