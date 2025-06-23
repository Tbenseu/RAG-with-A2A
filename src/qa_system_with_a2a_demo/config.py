import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
VECTORSTORE_DIR = "vector_index"

# Ports for downstream agents
RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:8001/run")
WEBSEARCH_AGENT_URL = os.getenv("WEB_AGENT_URL", "http://localhost:8002/run")

# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1


