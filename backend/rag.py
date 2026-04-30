"""
RAG engine with conversation memory and real-time streaming.
"""
import os
import sys
from typing import Iterator

sys.path.insert(0, os.path.dirname(__file__))

from config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


SYSTEM_PROMPT = (
    "You are a knowledgeable FLN (Foundational Literacy and Numeracy) learning assistant. "
    "Answer questions thoroughly using the provided document context. "
    "For summaries and overviews, synthesize information from all relevant passages. "
    "Use clear markdown formatting: bullet points, bold key terms, numbered steps where appropriate. "
    "Reference the source chapter when citing specific details. "
    "Only say you couldn't find something if the context is genuinely irrelevant to the question."
)


def _extract_sources(source_nodes) -> list[str]:
    if not source_nodes:
        return []
    seen: set[str] = set()
    sources: list[str] = []
    for node in source_nodes:
        fname = node.metadata.get("file_name") or node.metadata.get("source", "")
        if fname and fname not in seen:
            seen.add(fname)
            sources.append(fname)
    return sources


class RAGEngine:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
        Settings.llm = OpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

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
        self._retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)

    def _build_engine(self, messages: list[dict]) -> CondensePlusContextChatEngine:
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        for msg in messages:
            role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            memory.put(ChatMessage(role=role, content=msg.get("content", "")))
        return CondensePlusContextChatEngine.from_defaults(
            retriever=self._retriever,
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=False,
        )

    def query(self, question: str, messages: list[dict] | None = None) -> dict:
        if not question.strip():
            return {"answer": "Please enter a question.", "sources": []}
        try:
            engine = self._build_engine(messages or [])
            response = engine.chat(question)
        except Exception as exc:
            return {"answer": f"An error occurred: {exc}", "sources": []}
        return {
            "answer": str(response).strip() or "I couldn't find this in the learning materials.",
            "sources": _extract_sources(getattr(response, "source_nodes", [])),
        }

    def stream_query(self, question: str, messages: list[dict] | None = None) -> Iterator[dict]:
        """Yields {type:'token', content:str} events then a final {type:'done', sources:[]}."""
        if not question.strip():
            yield {"type": "token", "content": "Please enter a question."}
            yield {"type": "done", "sources": []}
            return
        try:
            engine = self._build_engine(messages or [])
            streaming_response = engine.stream_chat(question)
        except Exception as exc:
            yield {"type": "token", "content": f"Error: {exc}"}
            yield {"type": "done", "sources": []}
            return

        for token in streaming_response.response_gen:
            if token:
                yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "sources": _extract_sources(getattr(streaming_response, "source_nodes", [])),
        }
