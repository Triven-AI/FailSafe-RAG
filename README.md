# 🛡️ AegisAudit VoiceGuard: OneInbox Enterprise AI Integration

> **The Circuit Breaker for Enterprise Voice AI.** Stop conversational agents from hallucinating high-liability SLA penalties, billing data, and manager overrides mid-call.

---

## 🚨 The Problem
In enterprise BPO and customer support, a standard Retrieval-Augmented Generation (RAG) agent suffers from **blind trust**. When pressured by a caller about an SLA penalty, refund policy, or contract override, standard LLMs will confidently hallucinate an answer. In enterprise contexts, a single hallucination translates directly to regulatory non-compliance, unauthorized payouts, and thousand-dollar legal liabilities.

## 💡 The Solution: AegisAudit VoiceGuard
**AegisAudit VoiceGuard** is a self-correcting, state-machine-driven architecture built with **LangGraph, Qdrant, and Redis**. Instead of blindly generating responses, it treats generation as a high-stakes transaction requiring strict validation:
1. **Parent-Child Vector Retrieval:** Splits massive, messy enterprise documents (scans, handwritten overrides, legacy SLAs) into dense child vectors for matching while preserving the full parent context for grounding.
2. **Deterministic Code Gates & LLM Critics:** Evaluates retrieved chunks against strict contextual heuristics before letting an agent speak.
3. **Automatic Circuit Breaking & TTS Fallbacks:** If data is ambiguous, contradictory, or missing, the system instantly trips the circuit breaker and outputs a pre-formatted, human-ready **Text-to-Speech (TTS) safety script**.
4. **Semantic Caching:** Delivers sub-100ms responses for repeated queries to ensure low latency in live voice environments.

---

## 🏗️ System Architecture & Microservices

Our architecture is fully containerized and decoupled into independent microservices:
* **`failsafe-rag-ui` (Streamlit):** The real-time call center copilot and **Adversarial Call Matrix** dashboard for live auditing.
* **`failsafe-rag-orchestrator` (LangGraph & FastAPI):** The deterministic state machine managing the audit loop, critic nodes, and circuit breaking.
* **`failsafe-rag-ingestion` (Redis & Python Workers):** Asynchronous document routing supporting clean contracts, OCR tesseract, and vision-based document processing.
* **`qdrant_server`:** High-performance vector database storing parent-child chunks and semantic cache entries.
* **`redis`:** Message broker managing asynchronous task queues and pub/sub live status updates.

---

## 🚀 Quickstart & Installation

You can spin up the entire production-grade multi-container stack with a single command:

### Prerequisites
* Docker & Docker Compose installed.
* A Groq API Key (`GROQ_API_KEY`).

### 1. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here