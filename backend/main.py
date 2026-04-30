import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from config import OPENAI_API_KEY

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set. /api/chat will return errors.")
    else:
        try:
            from rag import RAGEngine
            rag_engine = RAGEngine()
            print("RAG engine loaded successfully.")
        except RuntimeError as exc:
            print(f"WARNING: RAG engine could not be loaded: {exc}")
            print("Run 'python backend/ingest.py' to index your PDFs, then restart.")
    yield


app = FastAPI(
    title="PDF Chatbot API",
    description="RAG-powered document Q&A using LlamaIndex + ChromaDB + GPT-4o mini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return JSONResponse(
        {"message": "Frontend not found. Place index.html in the frontend/ directory."},
        status_code=404,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag_ready": rag_engine is not None,
        "openai_key_set": bool(OPENAI_API_KEY),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The RAG engine is not ready. "
                "Run 'python backend/ingest.py' to index your PDFs, then restart the server."
            ),
        )

    result = rag_engine.query(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
