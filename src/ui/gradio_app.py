import os
import requests
import gradio as gr


API_URL = os.environ.get("MAS_API_URL", "http://127.0.0.1:8000")


def extract_text_from_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts).strip()
    except Exception:
        return ""


def ingest_and_summarize(text: str, pdf_file, length_percent: int) -> str:
    provided_text = (text or "").strip()
    if not provided_text and not pdf_file:
        return "Please enter text or upload a PDF."

    if not provided_text and pdf_file:
        file_path = None
        if isinstance(pdf_file, (str, os.PathLike)):
            file_path = str(pdf_file)
        elif isinstance(pdf_file, dict):
            file_path = pdf_file.get("path") or pdf_file.get("name")
        else:
            file_path = getattr(pdf_file, "name", None)

        if not file_path:
            return "Could not read uploaded PDF file."
        extracted = extract_text_from_pdf(file_path)
        if not extracted:
            return "Could not extract text from PDF (is 'pypdf' installed?)."
        provided_text = extracted

    return summarize_fn(provided_text, length_percent)


def summarize_fn(text: str, length_percent: int) -> str:
    if not text or not text.strip():
        return "Please enter some text."
    try:
        # Map 0-100 slider to 20-500 tokens
        target_tokens = int(20 + (500 - 20) * (max(0, min(100, length_percent)) / 100.0))
        resp = requests.post(
            f"{API_URL}/summarize",
            json={
                "text": text,
                # Use the same value for min and max to target a specific length
                "min_length": target_tokens,
                "max_length": target_tokens,
            },
            timeout=600,
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
            pdf = gr.File(label="Or upload a PDF", file_count="single", file_types=[".pdf"])
            length_pct = gr.Slider(0, 100, value=20, step=1, label="Abstract length (0-100)")
            btn = gr.Button("Summarize")
        with gr.Column():
            out = gr.Textbox(label="Generated Abstract", lines=12)

    btn.click(fn=ingest_and_summarize, inputs=[inp, pdf, length_pct], outputs=out)


if __name__ == "__main__":
    # Run Gradio on http://127.0.0.1:7860
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")))


