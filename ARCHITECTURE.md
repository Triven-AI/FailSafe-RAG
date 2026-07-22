```markdown
# AegisAudit Architecture Specification

## 1. System Topology & Event Flow

AegisAudit is a microservices-based, event-driven state machine designed to eliminate hallucinations in real-time enterprise voice workflows.

[ Incoming Query ]
│
▼
[ Redis Queue ] ──► [ LangGraph Orchestrator ]
│
┌───────────────────┴───────────────────┐
▼                                       ▼
[ Qdrant Vector DB ]                   [ Deterministic Code Gates ]
(Parent-Child Context)                 (Heuristic Regex Filter)
│                                       │
└───────────────────┬───────────────────┘
▼
[ LLM Critic Node ]
│
┌────────────┴────────────┐
│ Pass                    │ Fail / Low Confidence
▼                         ▼
[ Safe Output Node ]     [ Circuit Breaker Triggered ]
│                         │
▼                         ▼
[ Final Voice Script ]     [ Pre-Approved TTS Safety Script ]


---

## 2. State Machine Execution Pipeline (LangGraph)

The core orchestration service (`failsafe-rag-orchestrator`) executes queries through a strictly controlled state machine rather than a linear chain:

### Node Breakdown

1. **`Ingest_and_Cache_Lookup`**: Checks Qdrant semantic cache for cosine similarity $> 0.95$. On cache hit, short-circuits execution and returns standard response ($< 100\text{ms}$).
2. **`Parent_Child_Retrieval`**: Queries Qdrant dense vector index for child chunk matches ($k=5$), retrieving associated parent context blocks to preserve structural grounding.
3. **`Code_Gate_Filter`**: Executes deterministic Python checks (regex rules, date validity, missing fields, conflicting terms) against retrieved chunks. Chunks failing heuristics are dropped before LLM execution.
4. **`LLM_Critic_Audit`**: Evaluates remaining context against user query for **Factuality** and **Groundedness**. Returns a confidence score ($0.0 - 1.0$).
5. **`Evaluate_Circuit_Breaker`**:
   * **If Confidence $\ge 0.85$:** Generates grounded answer and forwards to output buffer.
   * **If Confidence $< 0.85$ or Context Missing/Contradictory:** Trips the Circuit Breaker node, logging an adversarial flag to Redis and returning a pre-verified, standardized Text-to-Speech (TTS) safety script.

---

## 3. Data & Storage Specifications

### Qdrant Payload Schema
```json
{
  "child_id": "uuid-v4",
  "parent_id": "doc-ref-104",
  "child_vector": [0.012, -0.043, "..."],
  "payload": {
    "text_content": "Child chunk snippet...",
    "parent_context": "Full parent document section...",
    "document_type": "SLA_Contract",
    "metadata": {
      "effective_date": "2026-01-01",
      "override_allowed": false
    }
  }
}