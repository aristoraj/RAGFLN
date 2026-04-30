"""
Run once to ingest all PDFs from /pdfs into ChromaDB.
Usage: python backend/ingest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHROMA_DIR,
    PDFS_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def ingest_pdfs() -> None:
    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {PDFS_DIR}. Add PDFs and re-run.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF(s): {', '.join(pdf_files)}\n")

    Settings.embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # Disable the default LLM so ingestion never calls GPT
    Settings.llm = None

    print("Loading PDFs...")
    try:
        reader = SimpleDirectoryReader(
            input_dir=PDFS_DIR,
            required_exts=[".pdf"],
            recursive=False,
        )
        documents = reader.load_data()
    except Exception as exc:
        print(f"ERROR reading PDFs: {exc}")
        sys.exit(1)

    print(f"Loaded {len(documents)} document page(s) from {len(pdf_files)} PDF(s).\n")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection so re-runs start fresh
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Existing collection '{COLLECTION_NAME}' deleted; rebuilding.\n")
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Generating embeddings and storing in ChromaDB...")
    for i, pdf_name in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] Processing: {pdf_name}")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nIngestion complete. {len(documents)} chunks stored in '{COLLECTION_NAME}'.")
    print(f"ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_pdfs()
