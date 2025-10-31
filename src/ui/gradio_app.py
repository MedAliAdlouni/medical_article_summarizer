import os
import requests
import gradio as gr


API_URL = os.environ.get("MAS_API_URL", "http://127.0.0.1:8000")


def summarize_fn(text: str, min_length: int, max_length: int) -> str:
    if not text or not text.strip():
        return "Please enter some text."
    try:
        resp = requests.post(
            f"{API_URL}/summarize",
            json={"text": text, "min_length": min_length, "max_length": max_length},
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


if __name__ == "__main__":
    # Run Gradio on http://127.0.0.1:7860
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")))


