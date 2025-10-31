import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import conf
from .models import import_model, import_tokenizer
from .preprocess import baseline_chunking, generate_chunk_summaries
from .utils import ensure_nltk_resources


class SummarizeRequest(BaseModel):
    text: str
    min_length: Optional[int] = 20
    max_length: Optional[int] = 60


app = FastAPI(title="Medical Article Summarizer API")

# Allow simple local testing via browser/Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Globals initialized at startup
_tokenizer = None
_model = None


@app.on_event("startup")
def _startup_load() -> None:
    global _tokenizer, _model
    ensure_nltk_resources()
    _tokenizer = import_tokenizer(conf)
    _model = import_model(conf)
    _model.to(conf["device"]) 


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string")

    chunks = baseline_chunking(req.text, tokenizer=_tokenizer, chunk_size=1024)
    summary = generate_chunk_summaries(
        chunks,
        _tokenizer,
        _model,
        device=conf["device"],
        min_length=req.min_length or 20,
        max_length=req.max_length or 60,
    )
    return {"summary": summary}


# Enable: python -m src.server
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=False)


