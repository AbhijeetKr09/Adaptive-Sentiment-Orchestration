"""
setup_and_data.py
=================
Dataset loading, cleaning, and preprocessing utilities for the
Adaptive Sentiment Orchestration (ASO) pipeline.

Supports:
    - HuggingFace `datasets` (SST-2 / tweet_eval) — no Kaggle credentials needed
    - Optional: Sentiment140 from Kaggle (CSV path passed via argument)

Author : ASO Research Team
Paper  : "Adaptive Sentiment Orchestration (ASO): A Hybrid Framework
          for Real-Time Sentiment Analysis"
"""

import re
import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lightweight text normalisation:
      1. Lower-case
      2. Remove URLs
      3. Remove Twitter handles (@user)
      4. Remove non-ASCII / special characters (keep apostrophes)
      5. Collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # URLs
    text = re.sub(r"@\w+", "", text)                    # handles
    text = re.sub(r"#", "", text)                       # hash symbols (keep word)
    text = re.sub(r"[^a-z0-9\s']", " ", text)          # non-alphanum
    text = re.sub(r"\s+", " ", text).strip()            # whitespace
    return text


def clean_texts(texts: List[str]) -> List[str]:
    """Vectorised wrapper around clean_text."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_sst2(
    max_samples: Optional[int] = 4000,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load Stanford Sentiment Treebank v2 (binary) via HuggingFace datasets.

    Labels:
        0 → Negative
        1 → Positive

    Returns
    -------
    train_texts, test_texts, train_labels, test_labels
    """
    logger.info("Loading SST-2 dataset …")
    ds = load_dataset("glue", "sst2", split="train")  # 67k rows

    texts  = ds["sentence"]
    labels = ds["label"]

    # Sub-sample for faster Colab experiments
    if max_samples and max_samples < len(texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=max_samples, replace=False)
        texts  = [texts[i]  for i in idx]
        labels = [labels[i] for i in idx]
        logger.info(f"Sub-sampled to {max_samples} examples.")

    texts = clean_texts(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    logger.info(
        f"Train: {len(X_train)} | Test: {len(X_test)} "
        f"| Positive ratio (train): {np.mean(y_train):.2%}"
    )
    return X_train, X_test, y_train, y_test


def load_tweet_eval(
    max_samples: Optional[int] = 4000,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load tweet_eval/sentiment (3-class → binarised: negative=0, positive=1,
    neutral samples dropped) via HuggingFace.

    Labels in tweet_eval: 0=Negative, 1=Neutral, 2=Positive
    Binarised          :  0=Negative, 1=Positive  (neutral dropped)
    """
    logger.info("Loading tweet_eval/sentiment dataset …")
    ds = load_dataset("tweet_eval", "sentiment", split="train")

    texts  = list(ds["text"])
    labels = list(ds["label"])

    # Binarise: keep only 0 (neg) and 2 (pos), map 2→1
    binary_texts, binary_labels = [], []
    for t, l in zip(texts, labels):
        if l == 0:
            binary_texts.append(t);  binary_labels.append(0)
        elif l == 2:
            binary_texts.append(t);  binary_labels.append(1)
    # neutral dropped for binary sentiment task

    if max_samples and max_samples < len(binary_texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(binary_texts), size=max_samples, replace=False)
        binary_texts  = [binary_texts[i]  for i in idx]
        binary_labels = [binary_labels[i] for i in idx]

    binary_texts = clean_texts(binary_texts)

    X_train, X_test, y_train, y_test = train_test_split(
        binary_texts, binary_labels,
        test_size=test_size, random_state=seed, stratify=binary_labels
    )
    logger.info(
        f"Train: {len(X_train)} | Test: {len(X_test)} "
        f"| Positive ratio (train): {np.mean(y_train):.2%}"
    )
    return X_train, X_test, y_train, y_test


def load_sentiment140_csv(
    csv_path: str,
    max_samples: Optional[int] = 4000,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load Sentiment140 from a local CSV file (Kaggle download).

    Sentiment140 column layout (no header):
        col 0 = polarity (0=neg, 4=pos)
        col 5 = tweet text

    Labels: 0→0 (negative), 4→1 (positive)
    """
    logger.info(f"Loading Sentiment140 from {csv_path} …")
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        usecols=[0, 5],
        names=["polarity", "text"],
    )
    df["label"] = (df["polarity"] == 4).astype(int)
    df["text"]  = clean_texts(df["text"].tolist())

    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["label"].tolist(),
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Default loader (uses SST-2 → safest for Colab without Kaggle)
# ---------------------------------------------------------------------------

def get_data(
    source: str = "sst2",
    max_samples: int = 4000,
    test_size: float = 0.20,
    seed: int = 42,
    csv_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Unified data loader.

    Parameters
    ----------
    source      : "sst2" | "tweet_eval" | "sentiment140"
    max_samples : cap on total samples (for fast Colab runs)
    test_size   : fraction reserved for testing
    seed        : reproducibility seed
    csv_path    : required when source == "sentiment140"
    """
    if source == "sst2":
        return load_sst2(max_samples, test_size, seed)
    elif source == "tweet_eval":
        return load_tweet_eval(max_samples, test_size, seed)
    elif source == "sentiment140":
        assert csv_path, "csv_path must be provided for Sentiment140"
        return load_sentiment140_csv(csv_path, max_samples, test_size, seed)
    else:
        raise ValueError(f"Unknown source: {source}. Choose sst2 | tweet_eval | sentiment140")
