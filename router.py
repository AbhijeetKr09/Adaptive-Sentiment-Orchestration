"""
router.py
=========
Adaptive Router — the core logic of the ASO (Adaptive Sentiment
Orchestration) framework.

Decision rule
-------------
    confidence(Tier-1) >= threshold  →  accept Tier-1 prediction
    confidence(Tier-1) <  threshold  →  escalate to Tier-2

The threshold is configurable and can be tuned on a held-out
validation set to trade off Tier-2 invocation rate vs. accuracy.

Author : ASO Research Team
Paper  : "Adaptive Sentiment Orchestration (ASO): A Hybrid Framework
          for Real-Time Sentiment Analysis"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    """Holds the final decision produced by the ASO router for one sample."""
    text          : str
    prediction    : int       # 0 = Negative, 1 = Positive
    confidence    : float     # Tier-1 softmax confidence
    tier_used     : int       # 1 or 2
    tier1_latency : float = 0.0   # seconds
    tier2_latency : float = 0.0   # seconds (0 if Tier-2 not invoked)

    @property
    def total_latency(self) -> float:
        return self.tier1_latency + self.tier2_latency


@dataclass
class RouterStats:
    """Aggregate statistics collected over a batch run."""
    total_samples      : int   = 0
    tier1_count        : int   = 0   # samples answered by Tier-1
    tier2_count        : int   = 0   # samples escalated to Tier-2
    total_latency      : float = 0.0
    tier1_total_lat    : float = 0.0
    tier2_total_lat    : float = 0.0

    @property
    def tier2_rate(self) -> float:
        """Fraction of samples escalated to Tier-2."""
        return self.tier2_count / max(self.total_samples, 1)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.total_samples, 1)

    def __str__(self) -> str:
        return (
            f"RouterStats | samples={self.total_samples} "
            f"| Tier-1 answered={self.tier1_count} ({1-self.tier2_rate:.1%}) "
            f"| Tier-2 escalated={self.tier2_count} ({self.tier2_rate:.1%}) "
            f"| Avg latency={self.avg_latency*1000:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

class AdaptiveRouter:
    """
    Implements the two-tier adaptive routing mechanism.

    Parameters
    ----------
    tier1_predict_fn  : callable(texts) → (predictions, confidences)
                        Must accept a list of strings and return two
                        equal-length arrays/lists.
    tier2_predict_fn  : callable(texts) → (predictions, confidences)
                        Same contract as tier1_predict_fn.
    threshold         : float in [0, 1].  Samples with Tier-1 confidence
                        below this value are escalated to Tier-2.
    batch_escalation  : if True (recommended for GPU efficiency), collect
                        all low-confidence samples and call Tier-2 once
                        per batch; if False, call Tier-2 per sample.
    """

    def __init__(
        self,
        tier1_predict_fn: Callable,
        tier2_predict_fn: Callable,
        threshold: float = 0.85,
        batch_escalation: bool = True,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1.")

        self.tier1_predict_fn  = tier1_predict_fn
        self.tier2_predict_fn  = tier2_predict_fn
        self.threshold         = threshold
        self.batch_escalation  = batch_escalation

        self._stats = RouterStats()
        logger.info(
            f"AdaptiveRouter initialised | threshold={threshold} "
            f"| batch_escalation={batch_escalation}"
        )

    # ------------------------------------------------------------------
    # Core routing logic
    # ------------------------------------------------------------------

    def route(
        self,
        texts: List[str],
        verbose: bool = False,
    ) -> Tuple[List[RouterDecision], RouterStats]:
        """
        Route a list of texts through the two-tier system.

        Returns
        -------
        decisions : list of RouterDecision (one per input text)
        stats     : RouterStats for this batch
        """
        n = len(texts)
        decisions: List[Optional[RouterDecision]] = [None] * n
        stats = RouterStats(total_samples=n)

        # ── Step 1: Run all samples through Tier-1 ─────────────────────
        t0 = time.perf_counter()
        t1_preds, t1_confs = self.tier1_predict_fn(texts)
        tier1_batch_latency = time.perf_counter() - t0
        tier1_per_sample    = tier1_batch_latency / max(n, 1)

        stats.tier1_total_lat += tier1_batch_latency

        # ── Step 2: Partition by confidence ────────────────────────────
        escalate_indices: List[int] = []

        for i, (pred, conf) in enumerate(zip(t1_preds, t1_confs)):
            if conf >= self.threshold:
                # Tier-1 is confident → accept prediction
                decisions[i] = RouterDecision(
                    text=texts[i],
                    prediction=int(pred),
                    confidence=float(conf),
                    tier_used=1,
                    tier1_latency=tier1_per_sample,
                    tier2_latency=0.0,
                )
                stats.tier1_count += 1
                stats.total_latency += tier1_per_sample
            else:
                escalate_indices.append(i)
                stats.tier2_count += 1

        # ── Step 3: Escalate low-confidence samples to Tier-2 ──────────
        if escalate_indices:
            escalate_texts = [texts[i] for i in escalate_indices]

            if self.batch_escalation:
                t0 = time.perf_counter()
                t2_preds, t2_confs = self.tier2_predict_fn(escalate_texts)
                tier2_batch_latency = time.perf_counter() - t0
                tier2_per_sample    = tier2_batch_latency / len(escalate_indices)
            else:
                # Per-sample escalation (slower but may be needed for APIs)
                t2_preds, t2_confs = [], []
                tier2_per_sample_list = []
                for et in escalate_texts:
                    t0 = time.perf_counter()
                    p, c = self.tier2_predict_fn([et])
                    tier2_per_sample_list.append(time.perf_counter() - t0)
                    t2_preds.append(p[0])
                    t2_confs.append(c[0])
                tier2_per_sample  = float(np.mean(tier2_per_sample_list))
                tier2_batch_latency = sum(tier2_per_sample_list)

            stats.tier2_total_lat += tier2_batch_latency

            for rank, i in enumerate(escalate_indices):
                lat2 = (
                    tier2_per_sample
                    if self.batch_escalation
                    else tier2_per_sample_list[rank]
                )
                decisions[i] = RouterDecision(
                    text=texts[i],
                    prediction=int(t2_preds[rank]),
                    confidence=float(t2_confs[rank]),
                    tier_used=2,
                    tier1_latency=tier1_per_sample,
                    tier2_latency=lat2,
                )
                stats.total_latency += tier1_per_sample + lat2

        if verbose:
            logger.info(str(stats))

        # Accumulate global stats
        self._stats.total_samples   += stats.total_samples
        self._stats.tier1_count     += stats.tier1_count
        self._stats.tier2_count     += stats.tier2_count
        self._stats.total_latency   += stats.total_latency
        self._stats.tier1_total_lat += stats.tier1_total_lat
        self._stats.tier2_total_lat += stats.tier2_total_lat

        return decisions, stats  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Simplified interface: returns (predictions, latencies_per_sample).
        Useful for plugging into evaluation.py.
        """
        decisions, _ = self.route(texts)
        preds     = [d.prediction     for d in decisions]
        latencies = [d.total_latency  for d in decisions]
        return preds, latencies

    @property
    def global_stats(self) -> RouterStats:
        """Cumulative stats across all route() calls."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset global counters (e.g. between experiments)."""
        self._stats = RouterStats()

    def set_threshold(self, new_threshold: float) -> None:
        """Update threshold in place (for sweep experiments)."""
        if not 0.0 < new_threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        logger.info(f"Threshold updated: {self.threshold} → {new_threshold}")
        self.threshold = new_threshold
