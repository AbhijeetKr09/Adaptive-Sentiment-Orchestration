# Lumina Analytics — Backend Implementation Plan

## Goal
Create a standalone Python backend folder (`lumina-backend/`) inside the current workspace that implements the full AI-driven sentiment analysis pipeline described in `documentation.md`. The backend will ingest Reddit data for a user-specified product, orchestrate dual-LLM processing, and write a UI-ready `Data.json` file consumed by the existing Lumina Analytics React frontend.

---

## Proposed Folder Location

```
c:\Users\ADMIN\Desktop\Major Project\project\
├── documentation.md
├── lumina-frontend/           (existing)
├── stitch_product_sentiment_dashboard/   (existing)
└── lumina-backend/            ← NEW (created by this plan)
    ├── main.py
    ├── reddit_scraper.py
    ├── preprocessor.py
    ├── aso_router.py
    ├── json_synthesizer.py
    ├── templates.py          ← prompts as a Python dict (no file I/O)
    ├── data_manager.py       ← Pythonic class wrappers for Data.json & memory.json
    ├── settings.json
    ├── Data.json             ← file persists (React frontend reads this)
    ├── memory.json           ← file persists (survives server restarts)
    └── requirements.txt
```

---

## User Review Required

> [!IMPORTANT]
> The backend calls **Reddit's API (PRAW)** and the **GroqCloud API**. You will need to supply:
> - A **Reddit API App** (`client_id`, `client_secret`, `user_agent`) — created at https://www.reddit.com/prefs/apps
> - A **GroqCloud API key** — created at https://console.groq.com
> These will be stored in `settings.json` (gitignored). I will add placeholder values to be replaced.

> [!NOTE]
> **Review Change (Templates):** The `templates/` directory has been replaced with a single `templates.py` Python module holding all prompts as a `PROMPTS` dict. This eliminates file I/O, keeps prompts syntax-highlighted and IDE-friendly, and makes them directly importable — `from templates import PROMPTS`.

> [!NOTE]
> **Review Change (Data/Memory):** `Data.json` must remain a file — the React frontend reads it from disk. `memory.json` must also remain a file — Python dicts die on server restart. Both are wrapped by `data_manager.py` which provides a clean class interface (`DataStore`, `MemoryStore`) so no raw `json.load/dump` is scattered across modules.

> [!WARNING]
> The backend runs as a **FastAPI** server (not Flask) for async support. The frontend's Start Page will need to call `POST /analyze` on `localhost:8000`. If the existing frontend already points to a different port/endpoint, that will need a minor adjustment (out of scope for this plan but worth noting).

> [!NOTE]
> **Review Change (Data.json persistence):** `Data.json` now stores a **persistent array of all past analysis sessions** — not a single overwritten payload. Each completed analysis is appended as a new session object, making the full history available to the `PastResults` frontend page. The frontend's `PastResults.jsx` renders a table of rows with fields: `product`, `date`, `score`, `sentiment`, `status` — the backend will write exactly this shape per session. The current/live session is always the last entry in the array.

---

## Proposed Changes

### `lumina-backend/` — New Python Backend Directory

---

#### [NEW] `requirements.txt`
All Python dependencies pinned to stable versions:

| Package | Purpose |
|---|---|
| `fastapi` | REST API server |
| `uvicorn` | ASGI runner for FastAPI |
| `praw` | Reddit API wrapper |
| `groq` | Official GroqCloud Python SDK |
| `pydantic` | JSON schema validation (catches LLM hallucinations) |
| `python-dotenv` | Load secrets safely |
| `httpx` | Async HTTP for internal calls |

---

#### [NEW] `settings.json`
Centralized config file. Contains:

**API Credentials (placeholders)**
- `groq_api_key`
- `reddit_client_id`, `reddit_client_secret`, `reddit_user_agent`

**Model Configuration**
- `model_heavy`: `"llama-3.1-70b-versatile"` — used for JSON synthesis
- `model_light`: `"llama3-8b-8192"` — used for classification, sarcasm detection, and competitor name discovery
- `token_limit_light`: `1024`
- `token_limit_heavy`: `4096`

**Scraping Configuration**
- `posts_per_product`: `10` — Number of Reddit posts fetched for the **main product** and **each competitor** (same value applies to both)
- `max_competitors`: `4` — Maximum number of similar/competitor products the LLM may identify for market comparison
- `rate_limit_sleep_seconds`: `5` — Mandatory sleep duration (seconds) injected **between every external API call** (Reddit PRAW calls + GroqCloud calls) to prevent rate limiting and respect API quotas

**Analysis Configuration**
- `score_out_of`: `10` — All sentiment scores are normalized to a 0–10 scale
- `community_clusters`: Static list of predefined cluster names (e.g., `["Tech", "Students", "Gamers", "Professionals", "Casual Users"]`)
- `design_colors`: Locked design system hex/rgba values injected into LLM prompts (`#4a40e0`, `#00675e`, `#b90035`, etc.)

---

#### [NEW] `main.py` — FastAPI Entry Point
**Responsibilities:**
1. Expose `POST /analyze` endpoint accepting `{ "product": "string" }`.
2. Coordinate the full pipeline in sequence:
   - Call `reddit_scraper.py` → raw posts
   - Call `preprocessor.py` → cleaned text
   - Call `aso_router.py` → sentiment-tagged data
   - Call `json_synthesizer.py` → final JSON payload
3. Validate output with Pydantic before writing to `Data.json`.
4. Log pipeline context to `memory.json`.
5. Return `{ "status": "done" }` or `{ "status": "error", "message": "..." }` to the frontend.
6. Expose `GET /status` for polling (frontend loading screen support).
7. CORS configured to allow the frontend's origin (localhost dev ports).

**Flow Diagram:**
```
POST /analyze
     │
     ▼
reddit_scraper  ──►  preprocessor  ──►  aso_router  ──►  json_synthesizer
                                                               │
                                              Pydantic Validation
                                                    │         │
                                               ✅ PASS    ❌ FAIL → retry
                                                    │
                                              Write Data.json
                                              Write memory.json
                                                    │
                                           Return { status: "done" }
```

---

#### [NEW] `reddit_scraper.py` — Data Ingestion
**Responsibilities:**

**Step 0 — Competitor Discovery (LLM Call):**
- Before any Reddit scraping starts, make a single `model_light` call asking the LLM to name up to `max_competitors` real similar/competing products for the given `product_name`.
- This is the **only task** for this LLM call — output is a simple JSON array of product name strings.
- A `rate_limit_sleep_seconds` sleep is applied after this call before scraping begins.

**Step 1 — Main Product Scraping:**
- Use **PRAW** to search Reddit for the main product.
- Fetch exactly `posts_per_product` posts (from `settings.json`).
- Apply `rate_limit_sleep_seconds` sleep after the Reddit call.

**Step 2 — Competitor Scraping:**
- Repeat the same Reddit search for **each competitor name** returned by Step 0.
- Fetch exactly `posts_per_product` posts per competitor (same setting as main product).
- Apply `rate_limit_sleep_seconds` sleep **between each competitor's Reddit call**.

**Return value:**
```python
{
  "target": { "name": "iPhone 15", "posts": [...10 posts] },
  "competitors": {
    "Samsung Galaxy S24": { "posts": [...10 posts] },
    "Google Pixel 8":     { "posts": [...10 posts] },
    "OnePlus 12":         { "posts": [...10 posts] }
  }
}
```

> **Total Reddit calls:** `1 (main) + max_competitors` calls, each sleeping 5s between them.

---

#### [NEW] `preprocessor.py` — Text Normalization
**Responsibilities:**
1. Strip HTML tags, URLs, markdown formatting, and emoji.
2. Remove duplicate/near-duplicate comments (deduplication).
3. Truncate individual comment length to a max character limit.
4. Merge all cleaned text into a compact, token-efficient string per product.
5. **Goal:** Reduce token footprint by ~40–60% before LLM inference.

---

#### [NEW] `aso_router.py` — Adaptive Sentiment Orchestration
This is the custom routing "ASO" model described in the doc.

**Responsibilities:**
1. Receive cleaned text chunks from `preprocessor.py`.
2. **Intent Classification:** Call `model_light` using `templates/intent_classification.txt` to label each chunk as `POSITIVE`, `NEGATIVE`, `NEUTRAL`, or `COMPLEX`.
3. **Sarcasm Detection:** For `COMPLEX` chunks, call `model_light` using `templates/sarcasm_detection.txt`. If sarcasm is confirmed, route to `model_heavy` for deeper analysis.
4. Return a list of sentiment-tagged data objects for synthesis.

**Routing Logic:**
```
chunk → intent_classification (model_light)
            │
    ┌───────┼────────────────┐
 POSITIVE  NEGATIVE        COMPLEX
 NEUTRAL                      │
    │                  sarcasm_detection (model_light)
 fast path                    │
    │               ┌─────────┴──────────┐
    │           SARCASM             NOT SARCASM
    │           (model_heavy)       (model_light)
    └───────────────┴────────────────────┘
                    │
             sentiment_tagged_data
```

---

#### [NEW] `json_synthesizer.py` — LLM JSON Assembly
**Responsibilities:**
1. Receive the complete sentiment-tagged data from `aso_router.py`.
2. Load the master template from `templates/json_payload_construction.txt`.
3. Call **Llama 3.1 70B via GroqCloud** with the assembled prompt.
4. Parse and validate the response using **Pydantic models** matching the exact frontend schema (see doc §4 & §5).
5. On Pydantic validation failure: issue **one automatic retry** with an error-correction addendum in the prompt.
6. On second failure: raise an error to `main.py`.
7. Return the validated Python dict for `main.py` to write to `Data.json`.

**Pydantic Models to implement:**
- `KeywordItem` → `word`, `size`, `color`, `weight`
- `DemographicItem` → `label`, `pct`, `color`
- `ClusterItem` → `label`, `users`, `size`, `color`, `border`, `textColor`, `top`, `left`
- `SentimentBrand` → `name`, `score`, `color`, `pct`
- `TrendingTopic` → `badge`, `badgeColor`, `title`, `desc`, `positive`, `icon`, `iconColor`, `discussions`, `bg`
- `PerformanceMetrics` → `totalMentions`, `avgResponseTime`, `analysisAccuracy`
- `LuminaPayload` → root model combining all of the above

---

#### [NEW] `templates.py` — Prompt Dictionary Module
Replaces the `templates/` directory entirely. A single importable Python file holding all prompts in a `PROMPTS` dict.

```python
# templates.py  (structure preview)
PROMPTS = {
    "intent_classification": """
        You are a sentiment classifier. Output ONLY one token: POSITIVE, NEGATIVE, NEUTRAL, or COMPLEX.
        ...
    """,
    "sarcasm_detection": """
        You are a sarcasm detector. Output ONLY: SARCASM or LITERAL.
        ...
    """,
    "json_payload_construction": """
        You are an expert market analyst AI (Ethereal Analyst).
        You will receive sentiment data for the product "{{PRODUCT_NAME}}".
        Output ONLY valid JSON. Use ONLY these design colors: {{DESIGN_COLORS}}.
        Required JSON Structure: {{JSON_SCHEMA}}
        ...
    """
}
```

**Benefits over `.txt` files:**
- Direct import: `from templates import PROMPTS` — no `open()`, no path resolution
- IDE syntax highlighting and linting on the prompt strings
- `{{PLACEHOLDER}}` values are injected via Python `.format()` or `str.replace()` at runtime
- All prompts version-controlled as real Python code
- Locked design colors from `settings.json` are injected once at startup into the dict

---

#### [NEW] `data_manager.py` — Pythonic File Interface
Provides two clean classes so no raw `json.load/dump` is scattered across modules:

```python
# data_manager.py  (structure preview)
class DataStore:
    """
    Manages Data.json — persistent array of ALL past analysis sessions.
    Powers the PastResults frontend page.
    """
    def append_session(self, session: dict) -> None: ...  # appends new analysis to sessions[]
    def get_all_sessions(self) -> list: ...               # returns full history list
    def get_latest(self) -> dict: ...                     # returns the most recent session
    def clear(self) -> None: ...                          # resets to { "sessions": [] }

class MemoryStore:
    """Manages memory.json — raw AI pipeline context surviving server restarts."""
    def append_context(self, context: dict) -> None: ...  # appends raw LLM reasoning/state
    def get_history(self, n: int = 10) -> list: ...       # last N context blobs
    def clear(self) -> None: ...                          # full reset
```

**Data.json session shape (matches PastResults.jsx table row exactly):**
```json
{
  "sessions": [
    {
      "product": "iPhone 15",
      "date": "Apr 10, 2026",
      "source": "Reddit",
      "sourceIcon": "forum",
      "iconColor": "#4a40e0",
      "score": 8.4,
      "sentiment": "positive",
      "status": "Completed",
      "payload": { ...full Intelligence/MarketPulse data... }
    }
  ]
}
```

**Why files still persist:**
- `Data.json` → React frontend reads this file directly from disk. In-memory dict dies on restart — all history would be lost.
- `memory.json` → Same reason; raw LLM context must survive restarts for cross-session reasoning.

**Why the class wrapper:**
- Single point of truth for file paths — no hardcoded strings in other modules
- Pydantic validation happens inside `DataStore.append_session()` before writing
- `MemoryStore` can later be swapped for SQLite with zero changes to calling code

---

#### [NEW] `Data.json`
Initialized as `{}`. **Only written via `DataStore.write()`** — never touched directly by business logic modules. This is the file the React frontend reads after each analysis run.

---

#### [NEW] `memory.json`
Initialized as `{ "sessions": [] }`. **Only accessed via `MemoryStore`** — never read/written directly. Appended on each successful run: `{ "timestamp", "product", "competitor_names", "summary_stats" }`.

---

## Implementation Order

| Step | Task |
|---|---|
| 1 | Create `lumina-backend/` folder |
| 2 | Write `requirements.txt` |
| 3 | Write `settings.json` with all config and placeholders |
| 4 | Write empty `Data.json` and `memory.json` |
| 5 | Write `templates.py` — `PROMPTS` dict with all 3 prompt templates |
| 6 | Write `data_manager.py` — `DataStore` and `MemoryStore` classes |
| 7 | Write `preprocessor.py` |
| 8 | Write `reddit_scraper.py` (imports `data_manager`, `settings`) |
| 9 | Write `aso_router.py` (imports `templates`, `settings`) |
| 10 | Write `json_synthesizer.py` — Pydantic models + Groq call (imports `templates`, `data_manager`) |
| 11 | Write `main.py` — FastAPI server, wires all modules together |
| 12 | Verify: `pip install -r requirements.txt` runs cleanly |
| 13 | Verify: `uvicorn main:app --reload` starts without errors |

---

## Verification Plan

### Automated
- Run `pip install -r requirements.txt` — no dependency errors.
- Run `uvicorn main:app --reload` — server starts on port 8000.
- `GET /status` returns `{ "status": "idle" }`.

### Manual (After API keys added)
- `POST /analyze` with `{ "product": "iPhone 15" }` → should return `{ "status": "done" }`.
- Confirm `Data.json` is populated with all required fields matching the Pydantic schema.
- Confirm `memory.json` has a new session entry appended.

### Integration (Frontend)
- Start the React frontend dev server alongside the backend.
- Submit a product on the Start Page — loading screen appears, then dashboard populates from `Data.json`.
