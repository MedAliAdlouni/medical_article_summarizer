import warnings
import os
import logging
import traceback

import numpy as np
import torch

from .config import conf
from .dataset import ArticlesAndAbstracts
from .models import import_model, import_tokenizer
from .preprocess import baseline_chunking, smart_chunking, generate_chunk_summaries
from .utils import (
    calculate_rouge_scores,
    ensure_nltk_resources,
    summarize_and_evaluate_random_articles,
)


warnings.filterwarnings("ignore")


def _setup_logging() -> None:
    log_path = os.path.join(os.path.dirname(__file__), "inference_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )


def inference(article: str) -> str:
    _setup_logging()
    logging.info("Starting inference with configuration: %s", conf)

    ensure_nltk_resources()

    tokenizer = import_tokenizer(conf)
    model = import_model(conf)
    model.to(conf["device"])

    chunks = baseline_chunking(article, tokenizer=tokenizer, chunk_size=1024)
    logging.info("Number of chunks: %s", len(chunks))

    summary = generate_chunk_summaries(
        chunks,
        tokenizer,
        model,
        device=conf["device"],
    )
    logging.info("summary:\n%s", summary)
    return summary


if __name__ == "__main__":
    _setup_logging()
    data_path = conf["dataset"]["path"]

    dataset = ArticlesAndAbstracts(conf)
    ind, article, abstract = dataset[0]
    logging.info("index of sample: %s", ind)
    logging.info("sample of article:\n%s", article[: min(500, len(article))])

    generated_abstract = inference(article)
    logging.info("generated abstract:\n%s", generated_abstract)

    rouge_scores = calculate_rouge_scores(generated_abstract, abstract)
    for key, value in rouge_scores.items():
        logging.info("%s: %s", key, value)



