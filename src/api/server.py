import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.config import conf
from ..core.models import import_model, import_tokenizer
from ..core.preprocess import baseline_chunking, generate_chunk_summaries
from ..core.utils import ensure_nltk_resources
import gradio as gr
from gradio.routes import mount_gradio_app


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


# --- Gradio UI mounted at /ui ---
def _build_gradio_app():
    def summarize_fn(text: str, min_length: int, max_length: int) -> str:
        if not text or not text.strip():
            return "Please enter some text."
        # Call the local FastAPI endpoint directly
        try:
            import requests

            resp = requests.post(
                "http://127.0.0.1:8000/summarize",
                json={
                    "text": text,
                    "min_length": int(min_length),
                    "max_length": int(max_length),
                },
                timeout=120,
            )
            if resp.status_code != 200:
                return f"Error {resp.status_code}: {resp.text}"
            return resp.json().get("summary", "")
        except Exception as e:
            return f"Request failed: {e}"

    with gr.Blocks(title="Medical Article Summarizer") as demo:
        gr.Markdown("**Minimal Summarizer UI** — enter an article, get an abstract.")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Article Text", lines=12, placeholder="Paste your article here...")
                min_len = gr.Slider(5, 200, value=20, step=1, label="Min length")
                max_len = gr.Slider(20, 300, value=60, step=1, label="Max length")
                btn = gr.Button("Summarize")
            with gr.Column():
                out = gr.Textbox(label="Generated Abstract", lines=12)
        btn.click(fn=summarize_fn, inputs=[inp, min_len, max_len], outputs=out)
    return demo


# Mount Gradio under /ui
_demo = _build_gradio_app()
mount_gradio_app(app, _demo, path="/ui")


# Enable: python -m src.api.server
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=port, reload=False)


