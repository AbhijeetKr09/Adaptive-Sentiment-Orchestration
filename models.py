"""
models.py
=========
Model wrappers for the ASO pipeline. Each wrapper exposes a consistent
interface:

    predict(texts: List[str]) -> Tuple[List[int], List[float]]
        returns (predictions, confidences)

Models implemented
------------------
1. LogisticRegressionModel  — TF-IDF + sklearn LogisticRegression
2. DistilBERTModel          — Tier-1 fast transformer (HuggingFace)
3. BERTModel                — Tier-2 strong transformer (HuggingFace)
   (or RoBERTa — configurable via `model_name` argument)

Author : ASO Research Team
Paper  : "Adaptive Sentiment Orchestration (ASO): A Hybrid Framework
          for Real-Time Sentiment Analysis"
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Device selection: GPU > CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class SentimentModelBase(ABC):
    """
    Unified interface for all sentiment analysis models in the ASO framework.
    All subclasses must implement `predict` and `fit` (if trainable).
    """

    @abstractmethod
    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Predict sentiment for a list of texts.

        Returns
        -------
        predictions : list of int (0 = Negative, 1 = Positive)
        confidences : list of float (probability of predicted class)
        """
        ...

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Override in trainable models."""
        raise NotImplementedError(f"{self.__class__.__name__} has no fit() method.")


# ---------------------------------------------------------------------------
# 1. Logistic Regression Baseline
# ---------------------------------------------------------------------------

class LogisticRegressionModel(SentimentModelBase):
    """
    TF-IDF + Logistic Regression — classic NLP baseline.

    Hyperparameters
    ---------------
    max_features : vocabulary size cap for TF-IDF
    C            : regularisation strength for LR
    ngram_range  : n-gram window for TF-IDF features
    """

    def __init__(
        self,
        max_features: int = 50_000,
        C: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=True,        # apply log(1+tf) scaling
                        min_df=2,                 # filter very rare tokens
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"\w{1,}",
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="saga",            # fast for large datasets
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> None:
        logger.info("Training Logistic Regression …")
        t0 = time.perf_counter()
        self.pipeline.fit(texts, labels)
        self._fitted = True
        logger.info(f"LR training done in {time.perf_counter() - t0:.2f}s")

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        preds      = self.pipeline.predict(texts).tolist()
        proba      = self.pipeline.predict_proba(texts)
        confidences = proba[np.arange(len(preds)), preds].tolist()
        return preds, confidences


# ---------------------------------------------------------------------------
# 2. HuggingFace Transformer Wrapper (shared by Tier-1 and Tier-2)
# ---------------------------------------------------------------------------

class HuggingFaceTransformerModel(SentimentModelBase):
    """
    Generic wrapper for HuggingFace sequence-classification models.

    Parameters
    ----------
    model_name  : HuggingFace model identifier
    max_length  : tokeniser max sequence length
    batch_size  : inference batch size (tune for GPU VRAM)
    label_map   : maps model output labels to {0, 1}.
                  SST-2 fine-tuned models often output LABEL_0/LABEL_1;
                  sentiment-specific models may use NEGATIVE/POSITIVE.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 128,
        batch_size: int = 32,
        label_map: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading tokeniser: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

        # Build label map: model output id → {0, 1}
        id2label = self.model.config.id2label  # e.g. {0: 'LABEL_0', 1: 'LABEL_1'}
        if label_map:
            self._label_map = label_map
        else:
            # Auto-detect: assume lower-id label == negative
            self._label_map = self._auto_detect_label_map(id2label)

        logger.info(f"Label map: {self._label_map}")

    @staticmethod
    def _auto_detect_label_map(id2label: dict) -> dict:
        """
        Heuristically map model output ids to {0=negative, 1=positive}.
        Works for models trained on SST-2, TweetEval, etc.
        """
        mapping = {}
        for idx, name in id2label.items():
            name_lower = name.lower()
            if "neg" in name_lower or name_lower in ("label_0", "0", "false"):
                mapping[idx] = 0
            elif "pos" in name_lower or name_lower in ("label_1", "1", "true"):
                mapping[idx] = 1
            else:
                # Fallback: sort by idx — smaller idx = negative
                mapping[idx] = 0 if idx == min(id2label.keys()) else 1
        return mapping

    @torch.no_grad()
    def _forward_batch(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Tokenise and run a single mini-batch through the model."""
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(DEVICE)

        logits = self.model(**encoding).logits          # [B, num_labels]
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

        raw_preds = np.argmax(probs, axis=-1)           # model output ids
        preds     = [self._label_map[int(p)] for p in raw_preds]
        confs     = [float(probs[i, raw_preds[i]]) for i in range(len(texts))]
        return preds, confs

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        all_preds, all_confs = [], []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            preds, confs = self._forward_batch(batch)
            all_preds.extend(preds)
            all_confs.extend(confs)
        return all_preds, all_confs


# ---------------------------------------------------------------------------
# 3. Concrete model factories
# ---------------------------------------------------------------------------

def build_tier1_model(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    max_length: int = 128,
    batch_size: int = 32,
) -> HuggingFaceTransformerModel:
    """
    Tier-1 (Fast) Model — DistilBERT fine-tuned on SST-2.

    ~40% smaller and ~60% faster than BERT-base with ~97% of its accuracy.
    Reference: Sanh et al., 2019.
    """
    logger.info("[Tier-1] Loading DistilBERT …")
    return HuggingFaceTransformerModel(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )


def build_tier2_model(
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
    max_length: int = 256,
    batch_size: int = 16,
    label_map: Optional[dict] = None,
) -> HuggingFaceTransformerModel:
    """
    Tier-2 (Strong) Model — BERT or RoBERTa with higher capacity.

    Default: 'textattack/bert-base-uncased-SST-2' is a solid SST-2
    fine-tuned BERT-base checkpoint.

    Alternative RoBERTa option:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    """
    logger.info("[Tier-2] Loading BERT/RoBERTa …")
    return HuggingFaceTransformerModel(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        label_map=label_map,
    )
