# FailSafe-RAG
We built a self-healing, low-latency CRAG pipeline. Our architecture uses LLM as a strict Critic. If the retrieved data is irrelevant or contradictory, the system rejects it, actively rewrites the user's search query, and tries again. By forcing the heaviest data processing, we neutralized network latency we built an enterprise-grade AI system.
