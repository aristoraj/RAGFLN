import json
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
    title="FLN Learning Assistant",
    description="RAG-powered FLN document Q&A with conversation memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    messages: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def _check_engine():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured on the server.")
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not ready. Run 'python backend/ingest.py' to index your PDFs.",
        )


@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return JSONResponse({"message": "Frontend not found. Place index.html in frontend/."}, status_code=404)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag_ready": rag_engine is not None,
        "openai_key_set": bool(OPENAI_API_KEY),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    _check_engine()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    result = rag_engine.query(req.question, history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    _check_engine()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.messages]

    async def event_gen():
        try:
            for event in rag_engine.stream_query(req.question, history):
                if event["type"] == "token":
                    yield f"data: {json.dumps({'token': event['content']})}\n\n"
                elif event["type"] == "done":
                    yield f"data: {json.dumps({'done': True, 'sources': event['sources']})}\n\n"
                await asyncio.sleep(0)
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
