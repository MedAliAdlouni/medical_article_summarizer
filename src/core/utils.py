import os
import re
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

import nltk


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    # Some environments need this auxiliary table
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def extract_data(path: str, tp: str, num_rows: int = 10) -> pd.DataFrame:
    x = os.listdir(f"{path}/{tp}/articles_{tp}")
    y = os.listdir(f"{path}/{tp}/abstracts_{tp}")

    x_id = {"".join(re.findall(r"\d+", xi)): xi for xi in x if "article" in xi}
    y_id = {"".join(re.findall(r"\d+", yi)): yi for yi in y if "abstract" in yi}

    ids = list(set(x_id) | set(y_id))

    only_in_x = 0
    only_in_y = 0
    in_both = 0

    selected = pd.DataFrame(columns=["id", "article", "abstract"])

    for id_ in ids[:num_rows]:
        if id_ in x_id and id_ in y_id:
            in_both += 1
            with open(f"{path}/{tp}/articles_{tp}/{x_id[id_]}", "r", encoding="utf-8", errors="ignore") as f:
                article = f.read()
            with open(f"{path}/{tp}/abstracts_{tp}/{y_id[id_]}", "r", encoding="utf-8", errors="ignore") as f:
                abstract = f.read()
            selected.loc[len(selected)] = [id_, article, abstract]
        elif id_ in x_id:
            only_in_x += 1
        else:
            only_in_y += 1

    print(f"\nIn {tp}:")
    print(f"Only in x : {only_in_x} -> non-used")
    print(f"Only in y : {only_in_y} -> non-used")
    print(f"In both : {in_both}")

    return selected


def calculate_rouge_scores(generated_summary: str, true_abstract: str) -> Dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(true_abstract, generated_summary)
    return scores


def summarize_and_evaluate_random_articles(
    dataset,
    tokenizer,
    model,
    chunker,
    summarizer,
    num_articles: int = 5,
    chunker_kwargs: Dict[str, Any] | None = None,
    summarizer_kwargs: Dict[str, Any] | None = None,
):
    results = []
    dataset_size = len(dataset)

    if num_articles > dataset_size:
        print(
            f"Warning: Requested {num_articles} articles, but dataset only contains {dataset_size}. Processing all."
        )
        num_articles = dataset_size

    random_indices = random.sample(range(dataset_size), num_articles)

    chunker_kwargs = chunker_kwargs or {}
    summarizer_kwargs = summarizer_kwargs or {}

    for i, index in enumerate(random_indices):
        print(f"\nProcessing article {i+1}/{num_articles} (Index: {index})")
        ind, article, abstract = dataset[index]

        chunks = chunker(article, tokenizer=tokenizer, **chunker_kwargs)
        summary = summarizer(chunks, tokenizer, model, **summarizer_kwargs)

        rouge_scores = calculate_rouge_scores(summary, abstract)
        results.append({
            "index": ind,
            "summary": summary,
            "abstract": abstract,
            "rouge_scores": rouge_scores,
        })

        print(f"ROUGE Scores for Article {ind}:")
        for key, value in rouge_scores.items():
            print(f"  {key}: {value}")

    return results



