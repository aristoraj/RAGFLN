import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL: str = "text-embedding-3-small"
LLM_MODEL: str = "gpt-4o-mini"

# On Render: set CHROMA_DIR=/data/chroma_db (persistent disk mount path)
# Locally: defaults to ./chroma_db next to the project root
_default_chroma = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_DIR: str = os.environ.get("CHROMA_DIR", _default_chroma)

PDFS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdfs")
COLLECTION_NAME: str = "learning_docs"
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
TOP_K: int = 12
