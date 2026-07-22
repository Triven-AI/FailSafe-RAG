import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Centralized, validated configuration for the AegisAudit microservices."""
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant_server:6333")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    def validate(self):
        """Ensures critical runtime tokens and endpoints are present."""
        if not self.GROQ_API_KEY:
            raise ValueError("CRITICAL CONFIG ERROR: GROQ_API_KEY environment variable is missing!")

settings = Settings()