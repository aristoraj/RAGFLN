"""
RAG query engine: loads ChromaDB index and answers questions grounded in stored documents.
"""

import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OPENAI_API_KEY,
    EMBED_MODEL,
    LLM_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
)

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


SYSTEM_PROMPT = (
    "You are a helpful learning assistant. Your job is to answer questions "
    "strictly based on the document context provided below.\n\n"
    "Rules:\n"
    "1. Only use information present in the context. Do not use prior knowledge.\n"
    "2. If the answer cannot be found in the context, respond with: "
    "\"I couldn't find this in the learning materials.\"\n"
    "3. Always mention the source document name or section when you use information from it.\n"
    "4. Be concise and accurate.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


class RAGResponse(TypedDict):
    answer: str
    sources: list[str]


class RAGEngine:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        Settings.embed_model = OpenAIEmbedding(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )
        Settings.llm = OpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1,
        )

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found in ChromaDB at '{CHROMA_DIR}'. "
                "Run 'python backend/ingest.py' first."
            )

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

        retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
        response_synthesizer = get_response_synthesizer(
            text_qa_template=PromptTemplate(SYSTEM_PROMPT),
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    def query(self, question: str) -> RAGResponse:
        if not question.strip():
            return RAGResponse(answer="Please enter a question.", sources=[])

        try:
            response = self._query_engine.query(question)
        except Exception as exc:
            return RAGResponse(
                answer=f"An error occurred while processing your question: {exc}",
                sources=[],
            )

        answer = str(response).strip()
        if not answer:
            answer = "I couldn't find this in the learning materials."

        sources: list[str] = []
        if response.source_nodes:
            seen: set[str] = set()
            for node in response.source_nodes:
                filename = node.metadata.get("file_name") or node.metadata.get("source", "")
                if filename and filename not in seen:
                    seen.add(filename)
                    sources.append(filename)

        return RAGResponse(answer=answer, sources=sources)
