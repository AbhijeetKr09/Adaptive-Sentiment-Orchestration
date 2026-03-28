"""
evaluation.py
=============
Evaluation utilities for the ASO framework.

Provides:
  - run_evaluation()     : full eval loop with latency measurement
  - build_results_table(): pandas DataFrame comparison of all models
  - plot_comparison()    : matplotlib bar charts for the paper

Metrics
-------
  - Accuracy
  - Macro F1-score
  - Average per-sample latency (ms)
  - Total inference time (s)

Author : ASO Research Team
Paper  : "Adaptive Sentiment Orchestration (ASO): A Hybrid Framework
          for Real-Time Sentiment Analysis"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Stores evaluation results for a single model."""
    model_name      : str
    accuracy        : float
    f1_macro        : float
    avg_latency_ms  : float    # per-sample average latency in milliseconds
    total_time_s    : float    # total wall-clock inference time in seconds
    predictions     : List[int] = field(default_factory=list)
    true_labels     : List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    model_name       : str,
    predict_fn       : Callable[[List[str]], Tuple[List[int], ...]],
    texts            : List[str],
    true_labels      : List[int],
    batch_size       : int = 64,
    warmup_batches   : int = 1,
) -> EvalResult:
    """
    Evaluate a model (or the ASO router) and return an EvalResult.

    Parameters
    ----------
    model_name     : display name for the model
    predict_fn     : callable accepting a list of strings; must return
                     (predictions, ...) where predictions are ints {0,1}.
                     Additional return values (e.g. confidences) are ignored.
    texts          : test texts (already cleaned)
    true_labels    : gold labels
    batch_size     : number of samples per inference call
    warmup_batches : number of initial batches to discard from latency
                     measurement (GPU cache warm-up)

    Returns
    -------
    EvalResult with all metrics populated.
    """
    logger.info(f"Evaluating: {model_name} on {len(texts)} samples …")

    all_preds   : List[int]   = []
    latencies   : List[float] = []
    total_start = time.perf_counter()

    n = len(texts)
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        batch = texts[start : start + batch_size]

        t0     = time.perf_counter()
        result = predict_fn(batch)
        elapsed = time.perf_counter() - t0

        # predict_fn may return (preds, confs) or (preds, lat_list) etc.
        preds = list(result[0])
        all_preds.extend(preds)

        # Discard warmup batches from latency stats
        if batch_idx >= warmup_batches:
            per_sample_lat = elapsed / len(batch)
            latencies.extend([per_sample_lat] * len(batch))

    total_time = time.perf_counter() - total_start

    # Guard: if warmup consumed all batches, use the full latency
    if len(latencies) == 0:
        latencies = [total_time / max(n, 1)] * n

    acc     = accuracy_score(true_labels, all_preds)
    f1_mac  = f1_score(true_labels, all_preds, average="macro", zero_division=0)
    avg_lat = float(np.mean(latencies)) * 1000   # convert to ms

    result_obj = EvalResult(
        model_name     = model_name,
        accuracy       = round(acc,    4),
        f1_macro       = round(f1_mac, 4),
        avg_latency_ms = round(avg_lat, 3),
        total_time_s   = round(total_time, 3),
        predictions    = all_preds,
        true_labels    = true_labels,
    )

    logger.info(
        f"  Accuracy : {acc:.4f} | F1 (macro): {f1_mac:.4f} "
        f"| Avg latency: {avg_lat:.2f} ms | Total: {total_time:.2f}s"
    )
    return result_obj


def run_evaluation_aso(
    router,
    texts      : List[str],
    true_labels: List[int],
    batch_size : int = 64,
) -> EvalResult:
    """
    Specialised evaluation wrapper for the AdaptiveRouter.

    The router.predict() returns (predictions, per_sample_latencies).
    We handle that explicitly to get accurate latency tracking via
    the router's internal measurement.
    """
    logger.info(f"Evaluating: ASO (threshold={router.threshold}) …")
    router.reset_stats()

    all_preds   : List[int]   = []
    all_latencies: List[float] = []
    total_start  = time.perf_counter()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        preds, lats = router.predict(batch)
        all_preds.extend(preds)
        all_latencies.extend(lats)

    total_time = time.perf_counter() - total_start

    acc    = accuracy_score(true_labels, all_preds)
    f1_mac = f1_score(true_labels, all_preds, average="macro", zero_division=0)
    avg_lat = float(np.mean(all_latencies)) * 1000

    name = f"ASO (τ={router.threshold:.2f})"
    result_obj = EvalResult(
        model_name     = name,
        accuracy       = round(acc,    4),
        f1_macro       = round(f1_mac, 4),
        avg_latency_ms = round(avg_lat, 3),
        total_time_s   = round(total_time, 3),
        predictions    = all_preds,
        true_labels    = true_labels,
    )

    stats = router.global_stats
    logger.info(
        f"  Accuracy: {acc:.4f} | F1: {f1_mac:.4f} "
        f"| Avg latency: {avg_lat:.2f} ms | Tier-2 rate: {stats.tier2_rate:.1%}"
    )
    return result_obj


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(results: List[EvalResult]) -> pd.DataFrame:
    """
    Construct a formatted pandas DataFrame summarising all model results.

    Columns: Model | Accuracy | F1 Score (Macro) | Avg Latency (ms) | Total Time (s)
    """
    rows = []
    for r in results:
        rows.append(
            {
                "Model"              : r.model_name,
                "Accuracy"           : f"{r.accuracy:.4f}",
                "F1 Score (Macro)"   : f"{r.f1_macro:.4f}",
                "Avg Latency (ms)"   : f"{r.avg_latency_ms:.3f}",
                "Total Time (s)"     : f"{r.total_time_s:.3f}",
            }
        )
    df = pd.DataFrame(rows)
    return df


def print_results_table(results: List[EvalResult]) -> None:
    """Pretty-print results table to stdout."""
    df = build_results_table(results)
    sep = "─" * 90
    header = (
        f"{'Model':<40} {'Accuracy':>10} {'F1 Score':>12} "
        f"{'Avg Lat (ms)':>14} {'Total (s)':>10}"
    )
    print(f"\n{sep}")
    print("  ASO FRAMEWORK — FINAL COMPARISON TABLE")
    print(sep)
    print(f"  {header}")
    print(sep)
    for r in results:
        print(
            f"  {r.model_name:<40} {r.accuracy:>10.4f} {r.f1_macro:>12.4f} "
            f"{r.avg_latency_ms:>14.3f} {r.total_time_s:>10.3f}"
        )
    print(f"{sep}\n")


def print_classification_reports(results: List[EvalResult]) -> None:
    """Print per-class classification reports for each model."""
    target_names = ["Negative", "Positive"]
    for r in results:
        print(f"\n── Classification Report: {r.model_name} ──")
        print(
            classification_report(
                r.true_labels,
                r.predictions,
                target_names=target_names,
                zero_division=0,
            )
        )


# ---------------------------------------------------------------------------
# Threshold sweep (for ASO analysis)
# ---------------------------------------------------------------------------

def threshold_sweep(
    router,
    texts       : List[str],
    true_labels : List[int],
    thresholds  : List[float] = None,
    batch_size  : int = 64,
) -> pd.DataFrame:
    """
    Evaluate ASO at multiple confidence thresholds.
    Returns a DataFrame with columns:
        Threshold | Accuracy | F1 | Avg Latency (ms) | Tier-2 Rate (%)
    """
    if thresholds is None:
        thresholds = [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    rows = []
    for tau in thresholds:
        router.set_threshold(tau)
        result = run_evaluation_aso(router, texts, true_labels, batch_size)
        rate   = router.global_stats.tier2_rate * 100
        rows.append(
            {
                "Threshold"       : tau,
                "Accuracy"        : result.accuracy,
                "F1 Macro"        : result.f1_macro,
                "Avg Latency (ms)": result.avg_latency_ms,
                "Tier-2 Rate (%)" : round(rate, 1),
            }
        )
        router.reset_stats()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    results     : List[EvalResult],
    save_path   : Optional[str] = None,
    figsize     : Tuple[int, int] = (14, 5),
) -> None:
    """
    Generate a three-panel bar chart comparing all models on:
      (a) Accuracy
      (b) F1 Score (Macro)
      (c) Average Latency (ms, log scale)
    """
    PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    names  = [r.model_name for r in results]
    acc    = [r.accuracy        for r in results]
    f1     = [r.f1_macro        for r in results]
    lat    = [r.avg_latency_ms  for r in results]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        "Adaptive Sentiment Orchestration (ASO) — Model Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )

    def _bar_plot(ax, values, title, ylabel, logy=False, fmt=".4f"):
        bars = ax.bar(range(len(names)), values, color=PALETTE[: len(names)], width=0.55)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        if logy:
            ax.set_yscale("log")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * (1.05 if not logy else 1.15),
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8.5,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar_plot(axes[0], acc, "Accuracy",            "Accuracy",       fmt=".4f")
    _bar_plot(axes[1], f1,  "F1 Score (Macro)",    "F1 Score",       fmt=".4f")
    _bar_plot(axes[2], lat, "Avg Latency (ms)",    "Latency (ms)",   logy=True, fmt=".2f")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def plot_threshold_sweep(
    sweep_df  : pd.DataFrame,
    save_path : Optional[str] = None,
) -> None:
    """
    Plot ASO accuracy, F1, and Tier-2 invocation rate
    as a function of the confidence threshold τ.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_acc  = "#4C72B0"
    color_f1   = "#55A868"
    color_rate = "#DD8452"

    ax1.plot(
        sweep_df["Threshold"], sweep_df["Accuracy"],
        marker="o", color=color_acc, label="Accuracy", linewidth=2
    )
    ax1.plot(
        sweep_df["Threshold"], sweep_df["F1 Macro"],
        marker="s", color=color_f1,  label="F1 Macro", linewidth=2
    )
    ax1.set_xlabel("Confidence Threshold (τ)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0.5, 1.05)
    ax1.set_title("ASO Threshold Sweep", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        sweep_df["Threshold"], sweep_df["Tier-2 Rate (%)"],
        marker="^", color=color_rate, linestyle="--",
        label="Tier-2 Rate (%)", linewidth=2,
    )
    ax2.set_ylabel("Tier-2 Invocation Rate (%)", color=color_rate, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_rate)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Threshold sweep plot saved to {save_path}")
    plt.show()
