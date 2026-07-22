# 🛡️ AegisAudit VoiceGuard: Enterprise AI Circuit Breaker

> **The Fail-Safe Guardrail for Enterprise Voice AI.** Intercept and mitigate high-liability LLM hallucinations—such as unverified SLA penalties, unauthorized manager overrides, and inaccurate billing data—mid-call before they reach the caller.

---

## 🚨 The Enterprise Risk

In call center environments (BPO and Enterprise Support), standard Retrieval-Augmented Generation (RAG) agents operate under **blind trust**. When subjected to adversarial prompting or missing documentation:
* **Standard RAG** hallucinated answers under pressure, generating legal and financial liabilities (e.g., promising non-existent refunds or incorrect SLA payout rates).
* **Compliance Failure:** A single unverified AI commitment can translate directly to regulatory penalties, unauthorized payouts, and legal breach of contract.

---

## 💡 The Solution: AegisAudit VoiceGuard

**AegisAudit VoiceGuard** introduces a state-machine-driven, self-correcting validation architecture powered by **LangGraph, Qdrant, and Redis**. Instead of outputting unverified text, VoiceGuard treats every model execution as a high-stakes transaction requiring deterministic verification:

| Feature | Execution Mechanism | Business Impact |
| :--- | :--- | :--- |
| **Parent-Child Retrieval** | Dense child matching linked to full parent context | Preserves full legal context while maintaining high match precision |
| **Deterministic Code Gates** | Regex heuristics + Contextual boundary checks | Instantly drops contextually irrelevant or risky document chunks |
| **LLM Grounding Critics** | Adversarial validation nodes evaluating factuality | Blocks ungrounded claims before text-to-speech generation |
| **Circuit Breaker Engine** | Deterministic fallback triggering standard TTS scripts | Guarantees 0% liability when data is ambiguous, missing, or contradictory |
| **Sub-100ms Semantic Cache** | Vector-similarity caching in Qdrant | Delivers instantaneous responses for repeated high-frequency queries |

---

## 🏗️ Microservice Architecture

AegisAudit is built as a fully containerized, microservices-based application:

* **`failsafe-rag-ui` (Streamlit):** Real-time operator copilot and **Adversarial Call Matrix** for live audit monitoring.
* **`failsafe-rag-orchestrator` (LangGraph & FastAPI):** The deterministic state machine managing query flow, critic evaluation, and circuit breaking.
* **`failsafe-rag-ingestion` (Redis Workers & Python):** Asynchronous document worker supporting PDF parsing, OCR tesseract, and vision-based document processing.
* **`qdrant_server`:** Vector database managing parent-child embeddings and semantic cache entries.
* **`redis`:** Message broker managing pub/sub updates and asynchronous task execution queues.

---

## 🚀 Quickstart & Installation

Deploy the complete multi-container environment locally using Docker Compose.

### Prerequisites
* [Docker Desktop](https://www.docker.com/) & Docker Compose installed.
* A valid [Groq API Key](https://console.groq.com/).

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://qdrant_server:6333
REDIS_URL=redis://redis:6379/0