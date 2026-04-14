# Adaptive Sentiment Orchestration (ASO) Pipeline
## Technical Documentation

> **Paper:** *"Adaptive Sentiment Orchestration (ASO): A Hybrid Framework for Real-Time Sentiment Analysis"*
> **Author:** ASO Research Team

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Module: setup_and_data.py](#3-module-setup_and_datapy)
4. [Module: models.py](#4-module-modelspy)
5. [Module: router.py](#5-module-routerpy)
6. [Module: evaluation.py](#6-module-evaluationpy)
7. [Dependencies](#7-dependencies)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Label Convention](#9-label-convention)

---

## 1. Project Overview

The ASO pipeline implements a **two-tier hybrid sentiment analysis system** designed for efficient, real-time binary sentiment classification (Positive / Negative). The core idea is:

- A **fast, lightweight Tier-1 model** (DistilBERT) handles the majority of predictions.
- Only samples where Tier-1 is **not confident** (below a threshold τ) are **escalated** to a **stronger but slower Tier-2 model** (BERT / RoBERTa).
- This reduces average latency while maintaining high accuracy.

**Baseline comparisons include:**
- Logistic Regression (TF-IDF)
- Tier-1 Only (DistilBERT)
- Tier-2 Only (BERT/RoBERTa)
- ASO Hybrid (Tier-1 + Tier-2 with adaptive routing)

---

## 2. System Architecture

```
Raw Text Input
      |
      v
[ setup_and_data.py ]  <- Load, clean, and split dataset
      |
      v  (X_train, X_test, y_train, y_test)
[ models.py ]          <- Instantiate and train models
  - LogisticRegression   (Baseline: TF-IDF + sklearn)
  - DistilBERT           (Tier-1: Fast transformer)
  - BERT / RoBERTa       (Tier-2: Strong transformer)
      |
      v  (predict_fn callables)
[ router.py ]          <- Adaptive routing logic
  conf >= threshold  ->  Accept Tier-1 answer
  conf <  threshold  ->  Escalate to Tier-2
      |
      v  (RouterDecision list)
[ evaluation.py ]      <- Metrics, tables, and plots
```

---

## 3. Module: setup_and_data.py

**Purpose:** Handles all dataset loading, text cleaning, and train/test splitting. Acts as the entry point for data preparation before any model training or evaluation.

**Supported datasets:**

| Dataset | Source | Labels |
|---|---|---|
| SST-2 | HuggingFace `glue/sst2` | 0=Negative, 1=Positive |
| tweet_eval | HuggingFace `tweet_eval/sentiment` | 0=Neg, 1=Neutral (dropped), 2=Pos |
| Sentiment140 | Local CSV (Kaggle download) | 0=Negative, 4 mapped to 1=Positive |

---

### Function: `clean_text`

```python
def clean_text(text: str) -> str
```

**Purpose:** Normalises a single raw text string for downstream NLP processing.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Any raw text (tweet, movie review, etc.) |

**Output:**

| Type | Description |
|---|---|
| `str` | Cleaned, normalised lowercase string |

**Processing Steps (in order):**
1. **Lowercase** — converts entire string to lowercase via `str.lower()`.
2. **Remove URLs** — strips `http://...`, `https://...`, and `www....` links using regex.
3. **Remove Twitter handles** — strips `@username` mentions.
4. **Remove hash symbols** — removes `#` character but keeps the word following it.
5. **Remove non-alphanumeric characters** — keeps only `[a-z0-9 ']`; removes punctuation and special chars.
6. **Collapse whitespace** — replaces multiple consecutive spaces with a single space and strips leading/trailing whitespace.

**Example:**
```
Input:  "I LOVE this movie!! @johndoe check http://example.com #amazing"
Output: "i love this movie  check  amazing"
```

---

### Function: `clean_texts`

```python
def clean_texts(texts: List[str]) -> List[str]
```

**Purpose:** Vectorised (list-level) wrapper that applies `clean_text` to each item in a list.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `texts` | `List[str]` | A list of raw text strings |

**Output:**

| Type | Description |
|---|---|
| `List[str]` | List of cleaned strings, same length and order as input |

**Processing:** Iterates over `texts` and calls `clean_text(t)` for each element using a list comprehension.

---

### Function: `load_sst2`

```python
def load_sst2(max_samples=4000, test_size=0.20, seed=42)
    -> Tuple[List[str], List[str], List[int], List[int]]
```

**Purpose:** Loads the Stanford Sentiment Treebank v2 (SST-2) binary sentiment dataset from HuggingFace.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_samples` | `int` or `None` | `4000` | Cap on total samples. Set to `None` to use all ~67k rows |
| `test_size` | `float` | `0.20` | Fraction of samples held out for testing |
| `seed` | `int` | `42` | Random seed for reproducibility |

**Output:**

| Return Value | Type | Description |
|---|---|---|
| `X_train` | `List[str]` | Cleaned training texts |
| `X_test` | `List[str]` | Cleaned test texts |
| `y_train` | `List[int]` | Training labels (0 or 1) |
| `y_test` | `List[int]` | Test labels (0 or 1) |

**Processing:**
1. Downloads (or uses cached) SST-2 training split via `load_dataset("glue", "sst2")`.
2. Extracts `"sentence"` and `"label"` columns.
3. Randomly sub-samples `max_samples` rows using a NumPy RNG (if `max_samples < total`).
4. Applies `clean_texts()` to all text.
5. Stratified train/test split with `sklearn.train_test_split` (preserves class balance).
6. Logs split sizes and positive class ratio.

---

### Function: `load_tweet_eval`

```python
def load_tweet_eval(max_samples=4000, test_size=0.20, seed=42)
    -> Tuple[List[str], List[str], List[int], List[int]]
```

**Purpose:** Loads the `tweet_eval/sentiment` dataset (3-class) from HuggingFace and **binarises** it by dropping neutral samples.

**Input:** Same parameters as `load_sst2`.

**Output:** Same 4-tuple as `load_sst2`.

**Processing:**
1. Downloads `tweet_eval sentiment` training split (labels: 0=Negative, 1=Neutral, 2=Positive).
2. **Binarisation:** Only keeps samples where label is `0` or `2`. Label `2` is remapped to `1`. Neutral (`1`) samples are **discarded entirely**.
3. Sub-samples, cleans, and splits same as `load_sst2`.

---

### Function: `load_sentiment140_csv`

```python
def load_sentiment140_csv(csv_path, max_samples=4000, test_size=0.20, seed=42)
    -> Tuple[List[str], List[str], List[int], List[int]]
```

**Purpose:** Loads the Sentiment140 dataset (1.6M tweets) from a **local Kaggle CSV file**.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | *(required)* | Absolute path to the local `.csv` file |
| `max_samples` | `int` | `4000` | Sample cap |
| `test_size` | `float` | `0.20` | Test fraction |
| `seed` | `int` | `42` | Reproducibility seed |

**Output:** Same 4-tuple as `load_sst2`.

**Processing:**
1. Reads CSV with `latin-1` encoding (Sentiment140 is not UTF-8). Only columns `0` (polarity) and `5` (text) are loaded.
2. Renames columns to `"polarity"` and `"text"`.
3. **Label mapping:** polarity `4` → `1` (Positive); polarity `0` → `0` (Negative).
4. Cleans texts with `clean_texts()`.
5. Random sample (if needed), then stratified train/test split.

---

### Function: `get_data` — Unified Entry Point

```python
def get_data(source="sst2", max_samples=4000, test_size=0.20, seed=42, csv_path=None)
    -> Tuple[List[str], List[str], List[int], List[int]]
```

**Purpose:** Single dispatcher function — the recommended way to load data. Internally calls the appropriate loader based on `source`.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | `"sst2"` | Dataset name: `"sst2"`, `"tweet_eval"`, or `"sentiment140"` |
| `max_samples` | `int` | `4000` | Sample cap (passed to the selected loader) |
| `test_size` | `float` | `0.20` | Test fraction |
| `seed` | `int` | `42` | Reproducibility seed |
| `csv_path` | `str` or `None` | `None` | Required only when `source="sentiment140"` |

**Output:** Same 4-tuple `(X_train, X_test, y_train, y_test)` as any individual loader.

**Processing:** Routes to `load_sst2`, `load_tweet_eval`, or `load_sentiment140_csv` based on `source`. Raises `ValueError` on unknown source name.

---

## 4. Module: models.py

**Purpose:** Defines all model wrappers used in the ASO pipeline. Every model exposes a **unified `predict(texts)` interface** returning `(predictions, confidences)`, making them interchangeable in the router and evaluator.

**Device Selection:** Automatically uses CUDA (GPU) if available; falls back to CPU.

---

### Abstract Base Class: `SentimentModelBase`

```python
class SentimentModelBase(ABC)
```

**Purpose:** Enforces a consistent interface for all sentiment models. Any class extending this must implement `predict()`.

**Abstract Methods:**

#### `predict(texts)`

```python
@abstractmethod
def predict(texts: List[str]) -> Tuple[List[int], List[float]]
```

- **Input:** `texts` — list of cleaned text strings
- **Output:** `(predictions, confidences)`
  - `predictions`: list of `int`, each `0` (Negative) or `1` (Positive)
  - `confidences`: list of `float`, probability of the predicted class

#### `fit(texts, labels)` — optional override

```python
def fit(texts: List[str], labels: List[int]) -> None
```

- Raises `NotImplementedError` by default. Override in trainable models (e.g., `LogisticRegressionModel`).

---

### Class: `LogisticRegressionModel`

```python
class LogisticRegressionModel(SentimentModelBase)
```

**Purpose:** Classical NLP baseline. Uses TF-IDF features fed into a Logistic Regression classifier. Fast to train, no GPU needed.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_features` | `int` | `50_000` | Vocabulary cap for TF-IDF |
| `C` | `float` | `1.0` | Inverse regularisation strength (LR hyperparameter) |
| `ngram_range` | `Tuple[int,int]` | `(1, 2)` | Unigrams + bigrams |
| `max_iter` | `int` | `1000` | Max solver iterations |

**Internal Pipeline:**
1. **`TfidfVectorizer`** — converts texts to sparse TF-IDF matrix
   - `sublinear_tf=True`: applies `log(1 + tf)` scaling
   - `min_df=2`: ignores tokens appearing fewer than 2 times
   - Word-level tokenisation with `token_pattern=r"\w{1,}"`
2. **`LogisticRegression`** — classifier with `solver="saga"` (fast for large datasets), parallelised with `n_jobs=-1`

#### Method: `fit`

```python
def fit(texts: List[str], labels: List[int]) -> None
```

| Parameter | Type | Description |
|---|---|---|
| `texts` | `List[str]` | Training texts |
| `labels` | `List[int]` | Corresponding binary labels (0 or 1) |

**Output:** None. Trains the internal pipeline in-place and sets `_fitted = True`.

**Processing:** Calls `self.pipeline.fit(texts, labels)`. Times the training and logs duration.

#### Method: `predict`

```python
def predict(texts: List[str]) -> Tuple[List[int], List[float]]
```

**Input:** `texts` — list of text strings (should be cleaned)

**Output:** `(predictions, confidences)`

**Processing:**
1. Calls `self.pipeline.predict(texts)` for hard predictions.
2. Calls `self.pipeline.predict_proba(texts)` for probability matrix `[n_samples, 2]`.
3. Extracts the probability for the **predicted class** using NumPy advanced indexing.
4. Raises `RuntimeError` if called before `fit()`.

---

### Class: `HuggingFaceTransformerModel`

```python
class HuggingFaceTransformerModel(SentimentModelBase)
```

**Purpose:** Generic HuggingFace transformer wrapper reused for both Tier-1 (DistilBERT) and Tier-2 (BERT/RoBERTa). Handles tokenisation, batching, and label mapping.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | HuggingFace model identifier (e.g., `"distilbert-base-uncased-finetuned-sst-2-english"`) |
| `max_length` | `int` | `128` | Tokeniser max sequence length (longer sequences are truncated) |
| `batch_size` | `int` | `32` | Mini-batch size for GPU inference |
| `label_map` | `dict` or `None` | `None` | Custom mapping from model output IDs to `{0, 1}`. If `None`, auto-detected. |

**Initialisation Steps:**
1. Loads `AutoTokenizer` from HuggingFace Hub.
2. Loads `AutoModelForSequenceClassification` from HuggingFace Hub.
3. Moves model to `DEVICE` (GPU/CPU) and sets `model.eval()`.
4. Detects or uses the provided `label_map`.

#### Static Method: `_auto_detect_label_map`

```python
@staticmethod
def _auto_detect_label_map(id2label: dict) -> dict
```

**Purpose:** Heuristically creates the mapping from model output IDs to binary labels without requiring manual specification.

**Input:** `id2label` — model config's `id2label` dict (e.g., `{0: 'LABEL_0', 1: 'LABEL_1'}`)

**Output:** `dict` mapping output id to `0` or `1`

**Processing:**
- Label name contains `"neg"` — maps to `0`
- Label name contains `"pos"` — maps to `1`
- Patterns `"label_0"`, `"0"`, `"false"` — mapped to `0`; `"label_1"`, `"1"`, `"true"` — mapped to `1`
- Fallback: smallest ID → `0`, all others → `1`

#### Method: `_forward_batch`

```python
@torch.no_grad()
def _forward_batch(texts: List[str]) -> Tuple[List[int], List[float]]
```

**Purpose:** Runs a single mini-batch through the transformer model. Decorated with `@torch.no_grad()` to skip gradient computation (inference only).

**Input:** `texts` — a list of strings (one mini-batch, length ≤ `batch_size`)

**Output:** `(predictions, confidences)` for the batch

**Processing:**
1. Tokenises the batch: padding, truncation, PyTorch tensors, moved to `DEVICE`.
2. Runs forward pass → `logits` tensor of shape `[B, num_labels]`.
3. Applies `softmax` along dim=-1 → `probs` moved to CPU as NumPy array.
4. Takes `argmax` per row → raw model output IDs.
5. Maps raw IDs through `_label_map` → binary predictions `{0, 1}`.
6. Extracts confidence = probability of the predicted class for each sample.

#### Method: `predict`

```python
def predict(texts: List[str]) -> Tuple[List[int], List[float]]
```

**Input:** `texts` — full list of text strings to classify

**Output:** `(all_predictions, all_confidences)` — lists of the same length as `texts`

**Processing:** Splits `texts` into chunks of `batch_size` and calls `_forward_batch` on each. Concatenates results with `list.extend()`.

---

### Factory Function: `build_tier1_model`

```python
def build_tier1_model(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    max_length=128,
    batch_size=32
) -> HuggingFaceTransformerModel
```

**Purpose:** Convenience factory that creates the Tier-1 (fast) model instance.

**Default model:** `distilbert-base-uncased-finetuned-sst-2-english`
- ~40% smaller than BERT-base
- ~60% faster inference
- ~97% of BERT accuracy on SST-2 (Sanh et al., 2019)

**Output:** A ready-to-use `HuggingFaceTransformerModel` instance.

---

### Factory Function: `build_tier2_model`

```python
def build_tier2_model(
    model_name="nlptown/bert-base-multilingual-uncased-sentiment",
    max_length=256,
    batch_size=16,
    label_map=None
) -> HuggingFaceTransformerModel
```

**Purpose:** Convenience factory that creates the Tier-2 (strong, slower) model instance.

**Default model:** `nlptown/bert-base-multilingual-uncased-sentiment`
- Higher capacity than Tier-1
- Larger `max_length=256` for longer texts
- Smaller `batch_size=16` due to higher VRAM usage

**Alternative model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (recommended for tweet-domain data)

**Output:** A ready-to-use `HuggingFaceTransformerModel` instance.

---

## 5. Module: router.py

**Purpose:** Implements the core adaptive routing logic of the ASO framework. Decides, for each input sample, whether Tier-1's prediction is confident enough to accept, or whether the sample must be escalated to Tier-2.

**Decision Rule:**
```
Tier-1 confidence >= threshold  →  Accept Tier-1 answer
Tier-1 confidence <  threshold  →  Escalate to Tier-2
```

---

### Dataclass: `RouterDecision`

```python
@dataclass
class RouterDecision
```

**Purpose:** Stores the final result for a **single text sample** after routing.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The original input text |
| `prediction` | `int` | Final binary label (0=Negative, 1=Positive) |
| `confidence` | `float` | Softmax confidence from whichever tier made the decision |
| `tier_used` | `int` | `1` if Tier-1 answered, `2` if escalated |
| `tier1_latency` | `float` | Tier-1 inference time for this sample (seconds) |
| `tier2_latency` | `float` | Tier-2 inference time (seconds); `0.0` if not escalated |

**Property:**
- `total_latency` → `tier1_latency + tier2_latency`

---

### Dataclass: `RouterStats`

```python
@dataclass
class RouterStats
```

**Purpose:** Aggregates statistics over a **batch of samples** processed by the router.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `total_samples` | `int` | Total number of samples processed |
| `tier1_count` | `int` | Samples answered by Tier-1 directly |
| `tier2_count` | `int` | Samples escalated to Tier-2 |
| `total_latency` | `float` | Sum of per-sample total latencies (seconds) |
| `tier1_total_lat` | `float` | Total time spent in Tier-1 for this batch |
| `tier2_total_lat` | `float` | Total time spent in Tier-2 for this batch |

**Properties:**
- `tier2_rate` → fraction of samples escalated to Tier-2
- `avg_latency` → average per-sample latency in seconds

**`__str__`:** Returns a formatted one-line summary string suitable for logging.

---

### Class: `AdaptiveRouter`

```python
class AdaptiveRouter
```

**Purpose:** Orchestrates the two-tier prediction pipeline using the configured confidence threshold.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tier1_predict_fn` | `Callable` | *(required)* | Any callable: `List[str]` → `(preds, confs)` |
| `tier2_predict_fn` | `Callable` | *(required)* | Same contract as `tier1_predict_fn` |
| `threshold` | `float` | `0.85` | Confidence threshold τ, must be in (0, 1) |
| `batch_escalation` | `bool` | `True` | If `True`, all low-confidence samples are sent to Tier-2 as a single batch (GPU-efficient). If `False`, each is escalated individually. |

**Validation:** Raises `ValueError` if `threshold` is not strictly between 0 and 1.

---

#### Method: `route`

```python
def route(texts: List[str], verbose: bool = False)
    -> Tuple[List[RouterDecision], RouterStats]
```

**Purpose:** Main routing method. Processes an entire batch of texts through the two-tier system.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `texts` | `List[str]` | Batch of cleaned text strings to classify |
| `verbose` | `bool` | If `True`, logs batch stats via the logger |

**Output:**

| Value | Type | Description |
|---|---|---|
| `decisions` | `List[RouterDecision]` | One `RouterDecision` per input text, in original order |
| `stats` | `RouterStats` | Aggregate statistics for this batch |

**Processing — 3 Steps:**

**Step 1 — Tier-1 Inference:**
- Calls `tier1_predict_fn(texts)` on the entire batch at once.
- Measures total Tier-1 batch latency and divides by `n` to get per-sample latency.

**Step 2 — Partition by Confidence:**
- Iterates over all `(prediction, confidence)` pairs from Tier-1.
- If `confidence >= threshold`: creates a `RouterDecision(tier_used=1)` immediately.
- If `confidence < threshold`: adds index to `escalate_indices` list, increments `tier2_count`.

**Step 3 — Tier-2 Escalation (only if any sample needs it):**
- Collects escalated texts into a sub-list.
- If `batch_escalation=True`: calls `tier2_predict_fn(escalate_texts)` once for the whole group.
- If `batch_escalation=False`: calls `tier2_predict_fn([et])` for each text individually.
- Creates `RouterDecision(tier_used=2)` entries for each escalated sample.
- Accumulates these stats into the instance-level `self._stats` for global tracking across multiple calls.

---

#### Method: `predict`

```python
def predict(texts: List[str]) -> Tuple[List[int], List[float]]
```

**Purpose:** Simplified interface for use with `evaluation.py`. Wraps `route()`.

**Input:** `texts` — list of text strings

**Output:** `(predictions, latencies_per_sample)`
- `predictions`: list of `int` (0 or 1)
- `latencies_per_sample`: list of `float` (total seconds per sample, including both Tier-1 and Tier-2 if escalated)

---

#### Method: `set_threshold`

```python
def set_threshold(new_threshold: float) -> None
```

**Purpose:** Updates the confidence threshold in-place without re-creating the router. Primary use: threshold sweep experiments.

**Input:** `new_threshold` — float in `(0, 1)`. Raises `ValueError` otherwise.

**Output:** None. Logs the old and new threshold values.

---

#### Method: `reset_stats`

```python
def reset_stats() -> None
```

**Purpose:** Clears the accumulated `_stats` object. Must be called between separate experiments to avoid mixing cumulative statistics across runs.

---

#### Property: `global_stats`

```python
@property
def global_stats -> RouterStats
```

**Purpose:** Returns the cumulative `RouterStats` across all `route()` calls since the last `reset_stats()` call.

---

## 6. Module: evaluation.py

**Purpose:** Provides all evaluation utilities — metric computation, results formatting (tables), and visualisation (plots). Designed to work with any model that exposes a `predict(texts)` interface, as well as directly with the `AdaptiveRouter`.

---

### Dataclass: `EvalResult`

```python
@dataclass
class EvalResult
```

**Purpose:** Stores the complete evaluation output for **one model**.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Display name (e.g., `"Logistic Regression"`, `"ASO (τ=0.85)"`) |
| `accuracy` | `float` | Overall classification accuracy (4 decimal places) |
| `f1_macro` | `float` | Macro-averaged F1 score (4 decimal places) |
| `avg_latency_ms` | `float` | Average per-sample latency in **milliseconds** |
| `total_time_s` | `float` | Total wall-clock inference time in **seconds** |
| `predictions` | `List[int]` | All predicted labels (0 or 1) |
| `true_labels` | `List[int]` | Ground truth labels |

---

### Function: `run_evaluation`

```python
def run_evaluation(
    model_name, predict_fn, texts, true_labels,
    batch_size=64, warmup_batches=1
) -> EvalResult
```

**Purpose:** Evaluates any model or callable on a test set. Handles batching, GPU warmup, and timing.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | Label for the model in output tables and reports |
| `predict_fn` | `Callable` | *(required)* | Function accepting `List[str]`, returning `(predictions, ...)` |
| `texts` | `List[str]` | *(required)* | Test texts (already cleaned) |
| `true_labels` | `List[int]` | *(required)* | Ground truth labels |
| `batch_size` | `int` | `64` | Inference batch size |
| `warmup_batches` | `int` | `1` | Number of initial batches excluded from latency stats (allows GPU cache warm-up) |

**Output:** An `EvalResult` object with all metrics populated.

**Processing:**
1. Iterates over `texts` in batches of `batch_size`.
2. Times each batch call to `predict_fn` using `time.perf_counter()`.
3. Discards the first `warmup_batches` batches from latency measurement.
4. Computes `accuracy_score` and `f1_score(average="macro")` from sklearn.
5. Converts mean latency from seconds to milliseconds.
6. Fallback: if all batches were warmup batches, uses total time as latency estimate.

---

### Function: `run_evaluation_aso`

```python
def run_evaluation_aso(router, texts, true_labels, batch_size=64) -> EvalResult
```

**Purpose:** Specialised evaluator for the `AdaptiveRouter`. Uses the router's internal per-sample latency tracking (more accurate than external timing) because the router already measures Tier-1 and Tier-2 latencies separately.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `router` | `AdaptiveRouter` | A configured and loaded router instance |
| `texts` | `List[str]` | Test texts |
| `true_labels` | `List[int]` | Ground truth labels |
| `batch_size` | `int` | Batch size for `router.predict()` calls |

**Output:** `EvalResult` with `model_name` set to `"ASO (τ=<threshold>)"`.

**Processing:**
1. Calls `router.reset_stats()` to clear previous run state.
2. Batches texts through `router.predict()`, which returns `(preds, per_sample_latencies)`.
3. Collects all predictions and their per-sample latencies.
4. Computes accuracy, macro F1, average latency (ms), and total wall-clock time.
5. Logs Tier-2 invocation rate from `router.global_stats`.

---

### Function: `build_results_table`

```python
def build_results_table(results: List[EvalResult]) -> pd.DataFrame
```

**Purpose:** Converts a list of `EvalResult` objects into a clean pandas DataFrame for display or export.

**Input:** `results` — list of `EvalResult` instances

**Output:** `pd.DataFrame` with columns: `Model`, `Accuracy`, `F1 Score (Macro)`, `Avg Latency (ms)`, `Total Time (s)`. All numeric values pre-formatted as strings with consistent decimal places.

---

### Function: `print_results_table`

```python
def print_results_table(results: List[EvalResult]) -> None
```

**Purpose:** Pretty-prints the comparison table to stdout with a horizontal rule border.

**Input:** `results` — list of `EvalResult`

**Output:** None (prints to stdout)

---

### Function: `print_classification_reports`

```python
def print_classification_reports(results: List[EvalResult]) -> None
```

**Purpose:** Prints per-class precision, recall, and F1 for each model using sklearn's `classification_report`.

**Input:** `results` — list of `EvalResult` (must have `predictions` and `true_labels` populated)

**Output:** None (prints to stdout)

**Class names used:** `["Negative", "Positive"]`

---

### Function: `threshold_sweep`

```python
def threshold_sweep(
    router, texts, true_labels,
    thresholds=None, batch_size=64
) -> pd.DataFrame
```

**Purpose:** Evaluates the ASO router at multiple values of the confidence threshold τ. Used to find the optimal trade-off between accuracy and Tier-2 invocation rate.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `router` | `AdaptiveRouter` | *(required)* | Router instance to sweep |
| `texts` | `List[str]` | *(required)* | Test texts |
| `true_labels` | `List[int]` | *(required)* | Ground truth labels |
| `thresholds` | `List[float]` or `None` | `[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]` | Threshold values to evaluate |
| `batch_size` | `int` | `64` | Batch size per evaluation call |

**Output:** `pd.DataFrame` with columns: `Threshold`, `Accuracy`, `F1 Macro`, `Avg Latency (ms)`, `Tier-2 Rate (%)`

**Processing:** For each τ in `thresholds`:
1. Calls `router.set_threshold(tau)`.
2. Calls `run_evaluation_aso(router, texts, true_labels, batch_size)`.
3. Reads `tier2_rate` from `router.global_stats` (as a percentage).
4. Calls `router.reset_stats()` before the next iteration.

---

### Function: `plot_comparison`

```python
def plot_comparison(results, save_path=None, figsize=(14, 5)) -> None
```

**Purpose:** Creates a three-panel bar chart comparing all models.

**Input:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `List[EvalResult]` | *(required)* | Models to compare |
| `save_path` | `str` or `None` | `None` | If set, saves the figure at this path at 150 DPI |
| `figsize` | `Tuple[int,int]` | `(14, 5)` | Matplotlib figure size in inches |

**Output:** None (displays plot; optionally saves file)

**Panels generated:**
- **Panel (a) Accuracy** — standard bar chart
- **Panel (b) F1 Score (Macro)** — standard bar chart
- **Panel (c) Avg Latency (ms)** — bar chart with **log scale** (to accommodate the wide performance gap between LR and transformers)

**Styling:** 4-color palette, rotated x-axis labels (25°), no top/right spines, value annotations on bars.

---

### Function: `plot_threshold_sweep`

```python
def plot_threshold_sweep(sweep_df, save_path=None) -> None
```

**Purpose:** Visualises the threshold sweep results with a dual-axis line chart showing the accuracy / F1 trade-off against Tier-2 invocation rate.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `sweep_df` | `pd.DataFrame` | Output of `threshold_sweep()` |
| `save_path` | `str` or `None` | Optional file path to save the figure at 150 DPI |

**Output:** None (displays plot; optionally saves file)

**Left Y-axis (ax1):** Accuracy and F1 Macro vs. threshold (solid lines)

**Right Y-axis (ax2):** Tier-2 invocation rate (%) vs. threshold (dashed line)

**Rationale for dual axis:** Tier-2 rate is on a 0–100% scale while accuracy/F1 are on 0–1, requiring two different y-axis scales on the same chart. A combined legend displays all three lines.

---

## 7. Dependencies

| Package | Min Version | Role |
|---|---|---|
| `torch` | >= 2.0.0 | Deep learning framework; GPU inference for transformer models |
| `transformers` | >= 4.38.0 | HuggingFace model hub; `AutoTokenizer`, `AutoModelForSequenceClassification` |
| `datasets` | >= 2.18.0 | HuggingFace dataset loader for SST-2 and tweet_eval |
| `accelerate` | >= 0.27.0 | Required internally by `transformers` for efficient model loading |
| `sentencepiece` | >= 0.1.99 | Tokeniser dependency for some transformer models (e.g., RoBERTa) |
| `scikit-learn` | >= 1.4.0 | `LogisticRegression`, `TfidfVectorizer`, `train_test_split`, accuracy/F1 metrics |
| `numpy` | >= 1.26.0 | Array operations; used throughout all modules |
| `pandas` | >= 2.2.0 | Results DataFrame construction and tabulation |
| `matplotlib` | >= 3.8.0 | Plotting comparison bar charts and threshold sweep line charts |
| `tqdm` | >= 4.66.0 | Progress bars (used internally by HuggingFace libraries) |

---

## 8. Data Flow Diagram

```
get_data("sst2")
    |
    +---> load_sst2()
    |         +--> clean_texts()
    |                  +--> clean_text() × N
    v
(X_train, X_test, y_train, y_test)
    |
    +--- LR Model -------> lr.fit(X_train, y_train)
    |                           lr.predict(X_test) ------------------+
    |                                                                 |
    +--- Tier-1 -----------------> tier1.predict(X_test) -----------+|
    |                                                                ||
    +--- Tier-2 -----------------> tier2.predict(X_test) ----------+||
    |                                                               |||
    +--- AdaptiveRouter ---->                                       |||
            |                                                       |||
            +--> tier1.predict(batch)   conf >= tau -> accept -----+||
            +--> tier2.predict(low_conf) conf < tau -> escalate ---+||
                                                                    |||
                                                                    vvv
                                                    run_evaluation() x 4 models
                                                                     |
                                                    build_results_table()
                                                    print_results_table()
                                                    plot_comparison()
                                                    threshold_sweep()
                                                    plot_threshold_sweep()
```

---

## 9. Label Convention

Used consistently across all modules:

| Integer Label | Sentiment |
|---|---|
| `0` | Negative |
| `1` | Positive |

Neutral class (where applicable, e.g., tweet_eval) is **always dropped** during preprocessing. All models output and accept only binary labels 0 and 1.

---

*Documentation generated: March 2026 | ASO Research Team*
