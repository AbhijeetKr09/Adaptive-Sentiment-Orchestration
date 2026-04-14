# Lumina Analytics Backend Architecture & Integration Document

## Overview
Lumina Analytics requires a robust, AI-native backend to drive its ethereal dashboard interface. The purpose of this backend is to take a target product (entered by the user on the Start Page), orchestrate social media data collection, and utilize Large Language Models (LLMs) to perform extensive market performance and sentiment analysis.

The backend will serve this compiled intelligence to the frontend using structured JSON files (`Data.json` and `memory.json`), adhering strictly to the schemas expected by the React components.

## 1. Core Technology Stack
- **Backend Language:** Python (Chosen for robust text preprocessing, web scraping, and seamless AI pipeline integrations).
- **AI Inference Engine:** GroqCloud API (for high-speed, cost-effective generation).
- **Dual LLM Architecture:** 
  - **High-Fidelity Model:** Llama 3.1 70B (Used strictly for high-quality JSON synthesis and complex reasoning).
  - **Lightweight/Sentiment Model:** Smaller LLM (e.g., Llama 3 8B) for direct classification and token optimization.
- **Specialized Processing (ASO):** Adaptive Sentiment Orchestration to dynamically route data between models based on context (e.g., detecting sarcasm).
- **Data Persistence (Primary State):** `Data.json` (Stores the final UI-ready state for the current session/product analysis)
- **Context Management:** `memory.json` (Stores historical context, agent memory, or intermediate states across pipeline reasoning steps)

## 2. Backend Workflow

1. **User Initiation:** The user enters a target product or brand name on the Lumina Start Page.
2. **Data Ingestion & Preprocessing:** The backend programmatically searches Reddit regarding the primary product. This raw data undergoes aggressive Python-based text preprocessing (removing HTML, filtering noise, and standardizing formatting) to significantly reduce the token footprint and enhance the LLM's analytical performance.
3. **Adaptive Sentiment Orchestration (ASO):** The unstructured data is processed by the Custom ASO model, which determines the input intent and routes it appropriately:
   - **Direct Classification:** Clean data goes to the lightweight LLM for fast sentiment analysis, saving tokens.
   - **Sarcasm/Nuance Handling:** Complex or sarcastic text is automatically detected and routed to the advanced LLM for higher-accuracy interpretation.
4. **Clustering & Segmentation:** Analyzed data points are mapped to statically predefined community names (e.g., "Tech," "Students").
5. **Dynamic Competitive Analysis:** The AI evaluates the base product and independently identifies 3 to 4 direct competitors. The backend then searches Reddit for these competitors to assemble comparative market performance.
6. **LLM JSON Synthesis (Llama 3.1 70B):** The finalized insights are passed strictly through Llama 3.1 70B via Groq using templates to convert the orchestration insights into highly specific JSON payloads to render the frontend.
7. **Data Storage:** The generated payloads are written securely to `Data.json`, while conversational state/memory is logged permanently to `memory.json`.
8. **Frontend Rendering:** The Lumina frontend ingests `Data.json` dynamically to populate its charts, bento-box layouts, and word clouds.

## 3. Storage Layer Specifications

### `Data.json`
Functioning as a lightweight "flat file database," this file holds the immediate application state. This eliminates the necessity for complex DB polling on the MVP while allowing the frontend to quickly pull localized state.

### `memory.json`
This maintains the persistent analysis context. As the system performs market checks, the AI can refer back to `memory.json` to establish how sentiment changed over time, maintain past searches, or store longer-context reasonings that don't need UI rendering.

## 4. LLM Templates & Prompt Engineering

To guarantee that Llama 3.1 70B returns the exact data structures the frontend visually expects, we must enforce rigid output templates.

**Example Custom System Prompt Design:**
```text
You are an expert market analyst AI (Ethereal Analyst).
You will receive an aggregation of recent social media comments regarding the product "{{PRODUCT_NAME}}".
Your task is to analyze sentiment, identify demographics, evaluate competitors, and structure your findings EXACTLY as the requested JSON object.
Output ONLY valid JSON. Do not include markdown formatting, conversational text, or explanations.

Required JSON Structure:
{
  "brandName": "{{PRODUCT_NAME}}",
  "keywords": [ 
    {"word": "string", "size": "string (e.g. 2.25rem)", "color": "hex", "weight": number} 
  ],
  "demographics": [ 
    {"label": "string", "pct": number, "color": "hex"} 
  ],
  "clusters": [ 
    {"label": "string", "users": "string", "size": number, "color": "rgba", "border": "rgba", "textColor": "hex", "top": "string", "left": "string"} 
  ],
  "platformPosts": ["string", "string", "string"],
  "sentimentBrands": [ 
    {"name": "string", "score": number, "color": "string", "pct": number} 
  ]
}
```

By combining unstructured social data with these static prompts via Groq's high-speed API, the backend enforces API-contract safety.

## 5. Expected Data Payloads (JSON Schema Requirements)

The frontend currently utilizes static arrays internally. To make the dashboard fully dynamic, the backend must dynamically reconstruct these data arrays inside `Data.json` utilizing the LLM:

### A. Intelligence Dashboard
* **Keywords (Word Cloud):** Popular terms appearing alongside the product.
  * Fields: `word`, `size` (rem), `color` (hex), `weight` (font weight).
* **User Demographics:** Segment breakdowns of the user base discussing the product.
  * Fields: `label` (e.g., "Gen Z (18-24)"), `pct` (integer 0-100), `color`.
* **Community Clusters:** Bubble components representing user groupings or pain points.
  * Fields: `label` (e.g., "Tech"), `users` (e.g., "12.4k"), `size` (px), `color` (rgba), `border` (rgba), `textColor` (hex), and layout coordinates (`top`, `left`, `bottom`, `right`).
* **Platform Posts:** Extracted notable positive or negative verbatims from platforms.
  * Fields: Array of strings.

### B. Market Pulse Dashboard
* **Comparative Brand Sentiment:** How the target product measures up against key rivals.
  * Fields: `name`, `score` (decimal 0-10), `color`, `pct` (integer 0-100).
* **Trending Topics:** Actionable strategic shifts occurring right now in the ecosystem.
  * Fields: `badge` (e.g., "High Velocity"), `badgeColor`, `title`, `desc`, `positive` (e.g., "84% Positive"), `icon` (Google material icon name), `iconColor`, `discussions`, `bg` (CSS background gradient string).
* **Performance Metrics:** 
  * Total Mentions, Average Response Time, and Analysis Accuracy percentage.

## 6. Implementation Strategies & Data Formatting Restrictions
1. **Loading State Flow:** Since AI ingestion and reasoning takes brief time even on Groq, the frontend Start Page should transition to a "Synthesizing Market Data" loading screen while the backend populates `Data.json`.
2. **Schema Validation Strategy (Crucial):** LLMs may occasionally hallucinate JSON syntax. You must use a validation library (like Pydantic in Python or Zod in Node.js) to catch malformed JSON before it overwrites `Data.json`. If a structure failure is detected, the backend should issue a fast retry prompt.
3. **Design System Adherence:** The frontend relies heavily on specific core variable colors for aesthetics (e.g., `#4a40e0`, `#00675e`, `#b90035`). The LLM's system prompt must instruct the model to use these specific hex/rgba color codes appropriately for Positive vs Negative sentiments to maintain UI integrity, instead of hallucinating random CSS colors.

## 7. Backend Project Structure & Files

The backend will be constructed in Python to leverage powerful token preprocessing capabilities. To ensure modularity and ease of customization, the project must follow this exact structured layout:

### Core Application
* `main.py`: The central backend server (e.g., FastAPI or Flask) handling frontend requests and orchestrating the analysis pipeline.
* `reddit_scraper.py`: Manages the ingestion of raw posts and comments from Reddit for the target product and its competitors.
* `preprocessor.py`: Cleans and normalizes the ingested social media data, stripping noise to drastically reduce the LLM token size and improve inference accuracy.
* `aso_router.py`: Houses the Adaptive Sentiment Orchestration logic, executing the lightweight LLM for direct classification and routing complex intent/sarcasm to the heavier LLM.
* `json_synthesizer.py`: Manages the Llama 3.1 70B GroqCloud API calls, forcing the orchestration insights into the final UI-ready schema.

### Configuration & Templates
* `settings.json`: A centralized backend configuration file. This will handle API keys, specific LLM model names to use, token limits, and hold the static lists of predefined community clustering names.
* `/templates`: A dedicated directory to store prompt design modularly. Keeping templates out of the Python code allows for easy customization without breaking backend logic.
  * `templates/intent_classification.txt` (For the lightweight LLM)
  * `templates/sarcasm_detection.txt` (For ASO routing)
  * `templates/json_payload_construction.txt` (For Llama 3.1 70B generation)

### Persistence
* `Data.json`: The live flat-file database populated by the backend to feed the frontend UI.
* `memory.json`: The persistent storage file maintaining historical context and application state.
