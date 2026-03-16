# Semantic Market Prediction

An LLM-powered sentiment analysis tool for financial news that quantifies recession fears across US and European markets, correlates sentiment with real market indices, and predicts next-day market direction.

Built for the BNP Paribas Equity & Derivatives Strategy team.

## What It Does

The tool collects financial news articles, has an LLM read and score each one for recession fear (0-10 scale), then builds a daily sentiment timeseries that's compared against actual market performance (S&P 500 for US, Euro Stoxx 50 for Europe). An XGBoost model trained on 2 years of historical VIX data predicts whether the market will go up or down tomorrow.

## How Scoring Works

I tried three approaches and settled on what worked best:

**Approach 1 (Simple averaging):** Score each article individually, average all scores for the day. Problem: a single "Fed emergency rate cut" article gets diluted by 15 routine earnings reports. The daily score doesn't reflect the true severity.

**Approach 2 (LLM daily assessment):** Give the LLM ALL articles for a given day plus the previous days' market data, and ask it to produce ONE daily score weighing article importance. Much better — the LLM can recognize that one major event matters more than many minor ones.

**Approach 3 (Ensemble — what we use):** Run the daily assessment with BOTH GPT-5.4 and Claude 4.6, then average their daily fear scores. This reduces model-specific biases and produces more stable, reliable sentiment readings.

### No Data Leakage

Each day's assessment only sees market data from previous days — never the current day or future. Day 1 gets the prior week's S&P 500 closes. Day 2 gets Day 1's assessment plus market data. This is critical for honest backtesting.

### Per-Article vs Per-Day Scoring

- **Per-article scores** (GPT-5.4): Used for sector breakdown, article browser, stock mentions
- **Per-day ensemble** (GPT-5.4 + Claude 4.6): Used for the main sentiment timeseries and predictions
- **Sector views** still use per-article averages (works well at that granularity)

## Prediction

Two independent prediction methods, shown side by side:

**XGBoost model:** Trained on 2 years of VIX (CBOE Volatility Index) and S&P 500 data (~500 trading days). Features: daily fear score, 3-day and 7-day rolling averages, volatility, article volume, momentum. Time-based 70/30 train/test split (no random shuffle — financial data requires chronological splits to avoid leakage). Outputs next-day direction (bullish/bearish) with confidence.

**LLM forecast:** The LLM gets the last 7 days of fear scores and S&P 500 prices, then reasons about what comes next. Provides direction, confidence, and a written explanation.

The panel can compare both forecasts and see where they agree or disagree.

## The Pipeline

Each run goes through these steps (US and Europe run in parallel):

1. **Fetch articles** — NewsAPI for recent headlines, Google News RSS for historical context. Articles cached for 24h so re-runs don't waste API calls.
2. **Score articles** — GPT-5.4 reads each article and returns: recession_fear (0-10), market_sentiment (bearish/neutral/bullish), confidence, rationale, and sector tags. 10 articles scored in parallel. Results cached per model.
3. **Fetch market data** — S&P 500 or Euro Stoxx 50 via yfinance, going back 14 days before the earliest article so the daily assessment has prior market context.
4. **Ensemble daily assessment** — For each day, both GPT-5.4 and Claude 4.6 see all articles + previous market data. Their daily fear scores are averaged. Results cached so subsequent runs skip this step.
5. **Build timeseries** — Daily scores with 3-day and 7-day rolling averages, momentum, and volatility.
6. **Merge with market** — Align sentiment dates with trading days.
7. **Lag correlation** — Pearson and Spearman correlation at T+0, T+1, T+2, T+3, T+5 to measure predictive power.
8. **XGBoost prediction** — Train on historical VIX data, predict with live sentiment features.
9. **LLM forecast** — Claude reasons about next-day direction from recent data.

## Caching Strategy

Everything is cached to minimize API costs and speed up re-runs:

- **News articles** — Cached 24h. Won't refetch until stale.
- **Per-model article scores** — Cached per model name (e.g., `gpt54_us`, `claude46_us`). Switching models in the UI and clicking "Re-score" only scores articles not already cached for that model.
- **Ensemble daily assessment** — Cached as `ensemble_daily_us`. On restart, loads instantly.
- **Market data** — Fetched fresh each run (yfinance is free and fast).

On a warm cache, the dashboard launches in seconds.

## Multi-Region Support

The tool analyzes US and Europe independently:

- **US:** NewsAPI US business headlines + Google News, scored with US-specific prompt (Fed rate, unemployment, consumer confidence), compared against S&P 500 and US sector ETFs (XLK, XLE, XLF, XLV).
- **Europe:** Same sources with European keywords (ECB policy, eurozone, energy crisis, PMI), scored with EU-specific prompt, compared against Euro Stoxx 50 and European sector ETFs.

Region-specific few-shot examples in the prompts ensure the LLM understands that "the Fed held rates steady" is a US signal while "ECB hawkishness" is a European one.

## Dashboard Features

- **Region toggle** — Switch between US and Europe
- **Fear gauge** — Semicircular indicator showing overall market fear
- **Dual forecast** — XGBoost prediction and LLM analysis side by side with confidence scores
- **Sentiment vs market chart** — Recession fear (inverted axis) against S&P 500 or sector ETFs. When you select a sector, the chart swaps to that sector's ETF (e.g., XLE for Energy)
- **Lag correlation heatmap** — Shows whether sentiment at day T predicts market movement at T+1, T+3, etc.
- **Sector heatmap** — Fear levels across sectors (Technology, Finance, Energy, Healthcare, etc.) over time
- **Sector vs ETF comparison** — Each sector's fear plotted against its corresponding ETF
- **Stock mentions** — Companies extracted from article titles with their fear scores
- **Article browser** — Newspaper-style cards (cream on dark), sorted by fear. Hover to see sector tags, sentiment badge, and LLM rationale
- **Chatbot** — Powered by Claude 4.6, has full access to article data. Ask "What caused the sentiment drop on March 13?" and it references specific articles with scores
- **Model selector** — GPT-5.4, Claude Sonnet 4.6, Llama 3.3 70B, DeepSeek R1, or local Ollama. Click "Re-score" to re-run the pipeline with a different model
- **Time window** — Last 8 days (default) or 3 months (triggers extended fetch from Google News)

## LLM Backends

All accessed through OpenRouter (one API key, all models):

| Model | Use | Cost |
|-------|-----|------|
| GPT-5.4 | Default article scoring | ~$0.20/run |
| Claude Sonnet 4.6 | Ensemble daily assessment + chatbot | ~$0.30/run |
| Llama 3.3 70B | Free alternative for scoring | Free |
| DeepSeek R1 | Free alternative (slower, reasoning model) | Free |
| Ollama | Local scoring if installed | Free |

Typical first run costs ~$0.50 (GPT-5.4 articles + Claude ensemble). Re-runs with cached scores cost $0.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Add API keys to .env
cp .env.example .env
# Edit: NEWSAPI_KEY, OPENROUTER_API_KEY (required)
# Optional: NEWSDATA_API_KEY, HUGGINGFACE_TOKEN

python run.py
# Open http://127.0.0.1:8050
```

First run takes ~2 minutes (fetching + scoring + ensemble assessment). Subsequent runs with cache take ~10 seconds.

## Project Structure

```
src/
├── config/          # Settings (YAML + .env), logging
├── ingestion/       # NewsAPI, Google News RSS, NewsData.io, 24h caching
├── analysis/        # LLM scorers (GPT, Claude, Llama, Ollama), daily assessor, ensemble
├── timeseries/      # Daily aggregation, market data (yfinance), lag correlation
├── prediction/      # XGBoost (VIX-trained), feature engineering, LLM forecast
├── visualization/   # Plotly charts (sentiment, sectors, gauge, stocks, ETF comparison)
├── chatbot/         # Claude 4.6 Q&A with article context
├── dashboard/       # Dash app, layout, callbacks, article browser
└── pipeline.py      # Orchestrator (parallel regions, caching, ensemble)

prompts/             # Versioned LLM prompts (US, EU, daily assessment, chat, sector)
tests/               # 110+ unit tests
```

## Tests

```bash
python -m pytest tests/ -v
```

## Deployment

Deployed on Render: https://semantic-market-prediction.onrender.com

Free tier — first load after idle takes ~30 seconds to wake up, then ~2 minutes for the pipeline.
