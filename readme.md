# Medical Article Summarizer

A comprehensive system for automatically summarizing medical research articles, focusing on two main study types:
- **Observational Studies (OBS)**
- **Randomized Controlled Trials (RCTs)**

The project employs fine-tuned BART (Bidirectional and Auto-Regressive Transformer) models to generate concise summaries, evaluated using ROUGE metrics.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Preprocessing & Chunking](#preprocessing--chunking)
- [Model Inference](#model-inference)
- [FastAPI Server](#fastapi-server)
- [Gradio UI](#gradio-ui)
- [Docker Deployment](#docker-deployment)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Evaluation](#evaluation)

---

## Repository Structure

```
medical_article_summarizer/
├── src/                          # Main application code
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Configuration and settings
│   │   ├── dataset.py           # Dataset loading and management
│   │   ├── inference.py         # Model inference pipeline
│   │   ├── models.py            # Model and tokenizer loading
│   │   ├── preprocess.py        # Text chunking strategies
│   │   └── utils.py             # Utility functions (ROUGE, NLTK, etc.)
│   ├── api/                      # FastAPI server
│   │   └── server.py            # API endpoints and Gradio integration
│   ├── ui/                       # User interface
│   │   └── gradio_app.py        # Standalone Gradio application
│   └── model/                    # Local model cache
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
├── data/                         # Dataset directory
│   ├── train/                    # Training data
│   │   ├── OBS/
│   │   │   ├── articles_OBS/    # 402 observational articles
│   │   │   └── abstracts_OBS/   # 429 observational abstracts
│   │   └── RCT/
│   │       ├── articles_RCT/    # 491 RCT articles
│   │       └── abstracts_RCT/   # 378 RCT abstracts
│   └── test/                     # Test data
│       ├── OBS_test/
│       │   └── articles_OBS_test/
│       └── RCT_test/
│           └── articles_RCT_test/
├── scripts/                      # Utility scripts
│   └── warmup.py                # Model warmup script for Docker
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   └── rendu_final.ipynb
├── Dockerfile                    # Docker configuration
├── .dockerignore                 # Docker ignore patterns
├── requirements.txt              # Python dependencies
├── requirements-serving.txt      # Minimal serving dependencies
└── README.md                     # This file
```

---

## Dataset

### Overview

The dataset consists of medical research articles with corresponding abstracts, split into two categories:

- **Observational Studies (OBS)**: 402 article-abstract pairs in training set, 26 in test set
- **Randomized Controlled Trials (RCT)**: 370 article-abstract pairs in training set, 27 in test set

### Dataset Features

- **Articles**: Full-length medical research papers extracted as plain text
- **Abstracts**: Manually written reference summaries for evaluation
- **File Format**: Each article/abstract stored in separate `.txt` files
- **File Naming**: `article-{PUBMED_ID}.txt` and `abstract-{PUBMED_ID}.txt`

### Dataset Loading

The `ArticlesAndAbstracts` class in `src/core/dataset.py` handles loading and preprocessing:

```python
from src.core.config import conf
from src.core.dataset import ArticlesAndAbstracts

dataset = ArticlesAndAbstracts(conf)
index, article, abstract = dataset[0]  # Get first sample
```

**Key Features:**
- Automatic pairing of articles with abstracts using PubMed ID
- Filters out articles without corresponding abstracts
- Returns article text, reference abstract, and index

---

## Preprocessing & Chunking

Medical articles often exceed model token limits (typically 1024-2048 tokens). The project implements two chunking strategies:

### 1. Baseline Chunking (`baseline_chunking`)

Simple sliding window approach with overlap:

```python
from src.core.preprocess import baseline_chunking

chunks = baseline_chunking(
    article=article_text,
    tokenizer=tokenizer,
    chunk_size=1024,      # Maximum tokens per chunk
    overlap_size=100      # Tokens to overlap between chunks
)
```

**How it works:**
- Tokenizes the entire article
- Splits into fixed-size chunks with configurable overlap
- Prevents information loss at chunk boundaries

### 2. Smart Chunking (`smart_chunking`)

Sentence-aware chunking that respects sentence boundaries:

```python
from src.core.preprocess import smart_chunking

chunks = smart_chunking(
    article=article_text,
    tokenizer=tokenizer,
    chunk_size=512,
    overlap_sentences=1    # Number of sentences to overlap
)
```

**How it works:**
- Splits article into sentences using NLTK
- Groups sentences into chunks without breaking them
- Applies sentence-level overlap for coherence
- More semantically coherent than baseline chunking

### Chunk Summarization

After chunking, each chunk is summarized independently and concatenated:

```python
from src.core.preprocess import generate_chunk_summaries

summary = generate_chunk_summaries(
    chunked_article_tokens=chunks,
    tokenizer=tokenizer,
    model=model,
    device="cuda",
    min_length=20,          # Minimum summary length
    max_length=60,          # Maximum summary length
    length_penalty=2.0      # Encourages longer summaries
)
```

---

## Model Inference

### Model Architecture

The project uses **Facebook's BART-Large-CNN** (`facebook/bart-large-cnn`):
- **Architecture**: Encoder-decoder transformer
- **Specialization**: Pre-trained on CNN/DailyMail for summarization
- **Parameters**: 400M parameters
- **Input**: Medical article text
- **Output**: Concise abstract

### Inference Pipeline

Complete inference workflow in `src/core/inference.py`:

```python
from src.core.inference import inference

summary = inference(article_text)
```

**Steps:**
1. Load tokenizer and model (with local caching)
2. Chunk long articles using baseline or smart chunking
3. Generate summary for each chunk
4. Concatenate chunk summaries into final output
5. Log results for debugging

### Model Loading & Caching

Models are automatically cached in `src/model/` to avoid repeated downloads:

```python
from src.core.models import import_model, import_tokenizer
from src.core.config import conf

tokenizer = import_tokenizer(conf)  # Cached in src/model/
model = import_model(conf)          # Cached in src/model/
model.to(conf["device"])            # Move to GPU if available
```

---

## FastAPI Server

### Server Setup

The FastAPI server (`src/api/server.py`) provides a production-ready API:

```bash
python -m src.api.server
```

**Configuration:**
- **Port**: 8000 (configurable via `PORT` environment variable)
- **Host**: 0.0.0.0 (accessible from all interfaces)
- **CORS**: Enabled for browser-based clients

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{"status": "ok"}
```

#### 2. Summarize
```bash
POST /summarize
Content-Type: application/json

{
  "text": "Article text here...",
  "min_length": 20,
  "max_length": 60
}
```

**Response:**
```json
{
  "summary": "Generated summary text..."
}
```

**Parameters:**
- `text` (required): Medical article to summarize
- `min_length` (optional, default=20): Minimum summary length
- `max_length` (optional, default=60): Maximum summary length

### Features

- **Automatic Model Loading**: Models loaded once at startup
- **Gradio Integration**: UI mounted at `/ui` endpoint
- **Error Handling**: Proper HTTP status codes and error messages
- **CORS Support**: Cross-origin requests enabled

---

## Gradio UI

### Standalone UI

Run the Gradio interface independently:

```bash
python -m src.ui.gradio_app
```

Access at `http://127.0.0.1:7860`

### Integrated UI

When running the FastAPI server, Gradio is automatically available at:
```
http://localhost:8000/ui
```

### UI Features

- **Text Input**: Large text area for pasting articles (12 lines)
- **Length Controls**: Sliders for min/max summary length
  - Min length: 5-200 tokens (default: 20)
  - Max length: 20-300 tokens (default: 60)
- **Real-time Summarization**: Click "Summarize" to generate
- **Error Display**: Shows API errors if request fails

### Configuration

Set custom API URL via environment variable:
```bash
export MAS_API_URL=http://your-api-server:8000
python -m src.ui.gradio_app
```

---

## Docker Deployment

### Dockerfile Overview

The `Dockerfile` creates a production-ready container:

```dockerfile
FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --prefer-binary --no-build-isolation -r requirements.txt

# Copy application code
COPY src ./src
COPY scripts ./scripts

# Optional: Copy pre-downloaded model cache
COPY model ./model

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "-m", "src.api.server"]
```

### Building the Docker Image

```bash
# Basic build
docker build -t medical-summarizer .

# Build with model warmup (downloads model during build)
docker build --build-arg DO_WARMUP=1 -t medical-summarizer .

# Build with pre-cached model (faster startup)
docker build -t medical-summarizer .
# Make sure src/model/ contains downloaded model files
```

### Running the Container

```bash
# Run with default settings (port 8000)
docker run -p 8000:8000 medical-summarizer

# Run with custom port
docker run -p 8080:8000 medical-summarizer

# Run with GPU support (if available)
docker run --gpus all -p 8000:8000 medical-summarizer

# Run with volume for model persistence
docker run -p 8000:8000 \
  -v $(pwd)/model:/app/src/model \
  medical-summarizer
```

### Docker Ignore Patterns

The `.dockerignore` file excludes:
- Development files (`.git`, `.venv`, `*.ipynb`)
- Data directories
- Large model files (unless pre-cached)
- Archive folders
- Log files

### Model Caching Strategy

Two approaches for Docker deployment:

**Option 1: Download at runtime** (faster builds)
- Model downloads on first API call
- Slightly slower first request (~2-5 minutes)

**Option 2: Pre-cache during build** (faster startup)
- Use `--build-arg DO_WARMUP=1` during build
- Model included in image
- Faster container startup

**Option 3: Copy local cache** (best balance)
- Copy `src/model/` into container during build
- Instant model loading
- Requires pre-downloading model

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip or conda
- GPU recommended (CUDA compatible) for faster inference

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd medical_article_summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `transformers>=4.43.0`: HuggingFace models and tokenizers
- `fastapi>=0.115.0`: REST API framework
- `uvicorn>=0.30.0`: ASGI server
- `gradio>=4.44.0`: Web UI framework
- `rouge-score>=0.1.2`: ROUGE metric calculation
- `nltk>=3.8.0`: Natural language processing
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations

### Verify Installation

```bash
# Test model loading
python -c "from src.core.models import import_tokenizer, import_model; from src.core.config import conf; import_tokenizer(conf); import_model(conf)"

# Download NLTK resources (auto-downloads, but can pre-download)
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage

### Command Line Inference

```bash
python -m src.core.inference
```

This runs inference on the first sample in the dataset and prints:
- Input article preview
- Generated summary
- ROUGE scores vs. reference

### FastAPI Server

```bash
# Start server
python -m src.api.server

# Server available at http://localhost:8000
# API docs at http://localhost:8000/docs
# Gradio UI at http://localhost:8000/ui
```

### Gradio UI (Standalone)

```bash
python -m src.ui.gradio_app

# Access at http://localhost:7860
```

### Docker Deployment

```bash
# Build and run
docker build -t medical-summarizer .
docker run -p 8000:8000 medical-summarizer
```

### API Usage Examples

**Using curl:**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your medical article text here...",
    "min_length": 20,
    "max_length": 60
  }'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "text": "Article text...",
        "min_length": 20,
        "max_length": 60
    }
)
summary = response.json()["summary"]
print(summary)
```

---

## Evaluation

### ROUGE Metrics

The system is evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

- **ROUGE-1**: Overlap of unigrams between generated and reference
- **ROUGE-2**: Overlap of bigrams between generated and reference
- **ROUGE-L**: Longest common subsequence between generated and reference

**Formula:**
$$\text{ROUGE} = \text{Measure of overlap between generated and reference summaries}$$

### Calculate ROUGE Scores

```python
from src.core.utils import calculate_rouge_scores

scores = calculate_rouge_scores(generated_summary, true_abstract)
print(scores)
# Output:
# {'rouge1': Score(precision=0.85, recall=0.82, fmeasure=0.83),
#  'rouge2': Score(precision=0.75, recall=0.72, fmeasure=0.73),
#  'rougeL': Score(precision=0.81, recall=0.79, fmeasure=0.80)}
```

### Batch Evaluation

```python
from src.core.utils import summarize_and_evaluate_random_articles

results = summarize_and_evaluate_random_articles(
    dataset=dataset,
    tokenizer=tokenizer,
    model=model,
    chunker=baseline_chunking,
    summarizer=generate_chunk_summaries,
    num_articles=10
)
```

---

## Additional Notes

### Configuration

Edit `src/core/config.py` to customize:

- **Model**: Change `"model"` to use different BART variants
- **Device**: Auto-detects CUDA, manually set if needed
- **Data Path**: Set `MAS_DATA_PATH` environment variable

### Logging

Inference logs are saved to `src/inference_run.log` with timestamps and details.
