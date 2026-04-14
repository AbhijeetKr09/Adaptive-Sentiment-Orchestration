# Lumina Analytics: Data Implementation Report (Real vs. Mock Data)

This report provides a comprehensive page-by-page breakdown of the Lumina Analytics platform, detailing what data is driven by the backend artificial intelligence pipeline (real) and what data is powered by hardcoded, static variables (mock/fake).

---

## Complete Backend Data Flow & Logic

The Lumina Backend contains a fully working sentiment analysis pipeline powered by the `Groq` API (using models like Llama 3.1 70B for JSON synthesis and a smaller model for ASO tagging). 
1. **Scraping**: `reddit_scraper.py` fetches live Reddit posts for the target product.
2. **Preprocessing & ASO Routing**: Text is cleaned, then sent to the LLM to tag each post with sentiments (Positive, Neutral, Critical).
3. **Synthesis**: `json_synthesizer.py` prompts a heavy Llama 70B model to synthesize the tagged data into a structured Pydantic schema (`LuminaPayload`).
4. **Persistence**: The results are appended to a local `Data.json` file. 

The React Frontend consumes this data via the `AnalysisContext.jsx`. If no analysis is loaded (`payload === null`), the frontend gracefully falls back to displaying `MOCK_` constants defined at the top of each page component.

---

## Page-By-Page Breakdown

### 1. Start Page (`StartPage.jsx`)
This is primarily a landing and input page. It does not display analytical data.
*   **Real Data Source:** Connects to the backend to kick off the analysis via `triggerAnalysis(productName)` and tracks if the pipeline is actively `isRunning`.
*   **Fake/Static Data:** 
    *   **Trust Stats** (98.2% Accuracy, 1.2B Sources, < 3s Latency, 45k+ Analysts) are entirely static marketing text.
    *   **Feature Modules** ("Emotional Mapping," "Community Synthesis") and visual gradient banners are hardcoded.
*   **Graphical Representation:** CSS radial gradients (Orbs) and a styled input box with a spinning loader, animated entirely by React state and CSS modules, utilizing no backend analysis as its input.

### 2. Intelligence Dashboard (`IntelligenceDashboard.jsx`)
The core overview page. It makes extensive use of both the real LLM object and several static fallback charts.

*   **Overall Sentiment Chart (`<DonutChart />`)**
    *   **FAKE:** The `IntelligenceDashboard` mounts `<DonutChart />` without passing any `data` prop. Inside `DonutChart.jsx`, the graphic defaults to a hardcoded array (`Positive 74, Neutral 15, Negative 11`). The UI does not populate this from the backend.
    *   **Graphical Logic:** Built using Recharts `<PieChart>` and `<Pie>`. CSS gradients and tooltips are configured, but data is static.
*   **Sentiment Evolution Area Chart (`<SentimentAreaChart />`)**
    *   **FAKE:** Similarly, no props are passed to this Recharts component. It relies on internal mock data spanning 'Oct 01' to 'Oct 30'. 
    *   **Graphical Logic:** Uses `<AreaChart>` with custom SVG `<linearGradient>` `<defs>` to render glowing fill colors beneath the plot lines.
*   **Keywords (Word Cloud)**
    *   **REAL Source:** Backend payload `keywords` array. 
    *   **Backend Logic:** The Llama 3.3 model generates the word text, along with dynamic UI values like CSS `fontSize`, hex `color`, and `fontWeight` based on the perceived importance of the topic in the scraped Reddit data.
    *   **FAKE Source:** `MOCK_KEYWORDS`.
*   **User Demographics Progress Bars**
    *   **REAL Source:** Backend payload `demographics`. Renders custom CSS-driven percentage bars mapped from the LLM.
    *   **FAKE Source:** `MOCK_DEMOGRAPHICS`.
*   **Community Clusters (Bubble Canvas)**
    *   **REAL Source:** Backend payload `clusters`. 
    *   **Graphical Logic:** Renders absolute-positioned DOM nodes. The diameter, color, and x/y screen coordinates (top/left) are generated entirely by LLM logic. The SVG connection lines behind the bubbles, however, are static hardcoded `<line>` elements.
    *   **FAKE Source:** `MOCK_CLUSTERS`.
*   **Community Insights Post Grid**
    *   **REAL Source:** Backend payload `platformPosts` (raw ingested Reddit strings).
    *   **FAKE Source:** `MOCK_POSTS`.
    *   *Note:* The UI tabs ("Reddit", "X", "YouTube") are static HTML. The backend only contains a Reddit scraper; switching tabs currently does nothing.

### 3. Market Pulse (`MarketPulse.jsx`)
Focuses on market positioning and trending topics.

*   **Market Share of Voice (`<MarketShareTreemap />`)**
    *   **FAKE:** The Recharts component is mounted without props. Inside `MarketShareTreemap.jsx`, the map hardcodes "LUMINA 42.8%", "NEXUS AI", "FLUX CORE", bypassing the backend payload.
    *   **Graphical Logic:** A Recharts `<Treemap>` calculates rectangular areas based on the `size` property of the dummy JSON tree.
*   **Comparative Sentiment Progress Bars**
    *   **REAL Source:** Backend payload `sentimentBrands`. Contains competitor names and sentiment scores (0.0-10.0), which power the width of the custom CSS filled-bar graphics.
    *   **FAKE Source:** `MOCK_SENTIMENT_BRANDS`.
*   **Trending Market Verticals**
    *   **REAL Source:** Backend payload `trendingTopics`. The LLM dynamically constructs badges, titles, descriptions, metrics (e.g. "+12% Growth"), and even selects the Google Material icons to use.
    *   **FAKE Source:** `MOCK_TRENDING_TOPICS`.
*   **Pulse Indicator / Market Vitality**
    *   **FAKE:** The glowing heart-beat visual and accompanying text ("Consumer optimism is currently at an 18-month high") is entirely hardcoded in the frontend.
*   **Footer Stats**
    *   **REAL Source:** Backend payload `performanceMetrics` (Total Mentions, Avg Response, Accuracy). Total Mentions accurately reflects the exact volume scraped from Reddit by the Python backend algorithm.
    *   **FAKE Source:** Static fallbacks (244.1k, 1.2s, 99.2%).

### 4. Community Analysis (`CommunityAnalysis.jsx`)
Focuses heavily on text-driven narrative clustering.

*   **Header Stats**
    *   **Mixed:** The `Sentiment` score uses the REAL `payload.overallScore` generated by the backend LLM. However, the `Engagement (+24.8%)` statistic is FAKE and hardcoded into the component.
*   **Sentiment Spectrum (`<SentimentBarChart />`)**
    *   **FAKE:** The Recharts bar chart receives no data props, defaulting to the internal `mockData` mapping 'Critical' through 'Euphoric' using fake frequencies.
    *   **Graphical Logic:** Maps categorical labels to `<Bar>` heights, colored using Recharts generic `<Cell>` nodes.
*   **Key Highlights**
    *   **FAKE:** The right-hand column displaying "Ecosystem Lock-in" and "Price Elasticity" uses a hardcoded `highlights` constant array. 
*   **Market Velocity**
    *   **FAKE:** The gradient box claiming "8.4x (+12% WoW)" is static HTML.
*   **Influential Narratives**
    *   **REAL Source:** Uses a slice of the first 3 `payload.platformPosts` fetched by the backend Reddit scraper.
    *   *Twist:* The component artificially assigns a FAKE sentiment label to these real posts via a static cycling array (`SENTIMENT_CYCLE = ['positive', 'critical', 'neutral']`).

### 5. Past Results (`PastResults.jsx`)
A historical log of previous analyses.

*   **Data Table & Summary Stats**
    *   **REAL Output Structure:** This page utilizes a different backend API flow (`getHistory()` calling `/history`). It derives its table directly from `Data.json`. 
    *   **Data Logic:** Total Analyses, Avg Sentiment, Table rows (Product name, Date, Icon, Score) are all accurately calculated and drawn from real backend persistence. There are no fallback mock constants on this page; if no history exists, it displays an empty state.
    *   **Graphical Logic:** The Score progress bar track in the table maps `(score * 10)` to a CSS inline `width` property percentage and interpolates its background color via a local `scoreColor()` Javascript helper.
*   **Intelligence Pulse Bubble Info**
    *   **Mixed:** Uses a basic ternary statement based on `sessions.length`. It accurately counts the real number of tracked analyses, but wraps it in static formatted strings ("Data patterns reveal sentiment trends..."). 

---

## Summary of Front-End Limitations
While the backend's AI pipeline successfully returns a highly complex, 100% LLM-driven structured JSON output for texts, keywords, topics, and metrics, **none of the four Recharts visualization components** (`DonutChart`, `SentimentAreaChart`, `SentimentBarChart`, `MarketShareTreemap`) are currently configured to accept or display dynamic data. They act strictly as visually appealing placeholders. To make this fully "real", the Recharts `<Pie>`, `<Area>`, `<Bar>`, and `<Treemap>` nodes need their `data={...}` props hooked into properly formatted sub-arrays constructed from the live `payload`.
