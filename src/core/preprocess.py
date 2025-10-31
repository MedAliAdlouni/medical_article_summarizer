from typing import List

import torch
from nltk import sent_tokenize

from .config import conf


def baseline_chunking(article: str, tokenizer, chunk_size: int = 1024, overlap_size: int = 100) -> List[List[int]]:
    tokenized_article = tokenizer(article, truncation=False, padding=False)["input_ids"]
    chunks: List[List[int]] = []
    for i in range(0, len(tokenized_article), chunk_size - overlap_size):
        chunk = tokenized_article[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


def smart_chunking(article: str, tokenizer, chunk_size: int = 512, overlap_sentences: int = 1) -> List[List[int]]:
    sentences = sent_tokenize(article)
    tokenized_sentences = [tokenizer(sentence, truncation=False, padding=False)["input_ids"] for sentence in sentences]
    token_lengths = [len(t) for t in tokenized_sentences]

    current_len = 0
    chunks: List[List[int]] = []
    current_chunk: List[int] = []

    for i, sent_tokens in enumerate(tokenized_sentences):
        if current_len + token_lengths[i] <= chunk_size:
            current_chunk.extend(sent_tokens)
            current_len += token_lengths[i]
        else:
            chunks.append(current_chunk)
            overlap_tokens: List[int] = []
            for j in range(max(0, i - overlap_sentences), i):
                overlap_tokens.extend(tokenized_sentences[j])
            current_chunk = overlap_tokens + sent_tokens
            current_len = sum(token_lengths[j] for j in range(max(0, i - overlap_sentences), i + 1))

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_chunk_summaries(
    chunked_article_tokens: List[List[int]],
    tokenizer,
    model,
    device: str | None = None,
    chunk_size: int = 1024,
    overlap_size: int = 100,
    min_length: int = 20,
    max_length: int = 60,
    length_penalty: float = 2.0,
    verbose: int = 0,
) -> str:
    summaries: List[str] = []
    device = device or conf["device"]
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, chunk_tokens in enumerate(chunked_article_tokens):
            if verbose:
                print(f"Processing chunk {i+1}/ {len(chunked_article_tokens)}")

            inputs = torch.tensor([chunk_tokens]).to(device)

            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,
                early_stopping=True,
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

    final_summary = " ".join(summaries)
    return final_summary



