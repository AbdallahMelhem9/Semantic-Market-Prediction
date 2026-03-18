# Semantic Market Prediction

An LLM-powered sentiment analysis tool for financial news that quantifies recession fears across US and European markets, correlates sentiment with real market indices, and predicts next-day market direction.

Built for the BNP Paribas Equity & Derivatives Strategy team.

## What It Does

The tool collects financial news articles from multiple sources, has an LLM read and score each one for recession fear on a 0 to 10 scale, then builds a daily sentiment timeseries. That timeseries is compared against actual market performance (S&P 500 for the US, Euro Stoxx 50 for Europe). An XGBoost model trained on 2 years of historical VIX data predicts whether the market will go up or down tomorrow.

There's also a standalone accuracy evaluation script (`metric.py`) that backtests the predictions against real market data and reports direction accuracy, confidence-weighted hit rates, and lag correlations.

## Data Sources

News comes from three APIs that complement each other:

**NewsAPI** is the primary source. It returns full articles (title, description, body text) for the last 30 days. The free tier caps at 100 articles per request, so the keywords are tuned to focus on recession-relevant topics: "recession", "Fed rate", "unemployment", "inflation", "GDP", "consumer confidence".

**Finnhub** fills in company-specific coverage. It fetches headlines and summaries across 16 major US tickers (AAPL, MSFT, JPM, XOM, etc.) spanning Tech, Finance, Energy, Healthcare, Consumer, and Industrials. This gives sector-level depth that general news queries miss.

**Google News RSS** is a free fallback for historical gaps. It only returns titles (no article body), so it's lower quality. The pipeline only uses Google News articles older than 7 days, where NewsAPI coverage is thinnest.

All three sources are deduplicated by URL before scoring. The final dataset is capped at 300 articles across a maximum of 35 days.

## Text Cleaning

Before any article reaches the LLM, it goes through a cleaning pipeline (`src/ingestion/text_cleaner.py`) that strips HTML tags, removes NewsAPI's `[+1234 chars]` truncation markers, cleans encoding artifacts (non-breaking spaces, zero-width characters, BOM), removes URLs, and collapses whitespace. Articles shorter than 15 characters after cleaning are flagged as unusable. This ensures the LLM scores clean, consistent text regardless of which API the article came from.

## How Scoring Works

I tried three approaches and settled on what worked best.

**Simple averaging** was the first attempt. Score each article individually, average all scores for the day. The problem: a single "Fed emergency rate cut" article gets diluted by 15 routine earnings reports. The daily score doesn't reflect the true severity.

**LLM daily assessment** was the second attempt. Give the LLM all articles for a given day plus the previous days' market data, and ask it to produce one daily score weighing article importance. Much better. The LLM can recognize that one major event matters more than many minor ones.

**Ensemble** is what we actually use. Run the daily assessment with both GPT-5.4 and Claude 4.6, then average their daily fear scores. This reduces model-specific biases and produces more stable, reliable sentiment readings.

### No Data Leakage

Each day's assessment only sees market data from previous days, never the current day or future. Day 1 gets the prior week's S&P 500 closes. Day 2 gets Day 1's assessment plus market data. This is critical for honest backtesting.

### Per-Article vs Per-Day Scoring

**Per-article scores** (GPT-5.4) are used for the sector breakdown, article browser, and stock mentions. **Per-day ensemble** (GPT-5.4 + Claude 4.6) powers the main sentiment timeseries and predictions. Sector views still use per-article averages, which works well at that granularity.

## Prediction

Two independent prediction methods shown side by side in the dashboard.

**XGBoost model** is trained on 2 years of VIX (CBOE Volatility Index) and S&P 500 data, roughly 500 trading days. Features include the daily fear score, 3-day and 7-day rolling averages, volatility, article volume, and momentum. The train/test split is time-based 70/30 with no random shuffle, because financial data requires chronological splits to avoid leakage. Outputs next-day direction (bullish/bearish) with a confidence score.

**LLM forecast** gives the LLM the last 7 days of fear scores and S&P 500 prices, then asks it to reason about what comes next. It provides direction, confidence, and a written explanation.

The dashboard shows both forecasts together so you can see where they agree or disagree.

## Accuracy Evaluation

`metric.py` runs a backtest against actual S&P 500 data. It converts each day's fear score into a directional prediction (fear above 5 means predict down, below 5 means predict up), then checks if the market actually moved that way the next trading day. The report includes:

**Direction accuracy** shows a day-by-day table of predictions vs actual market movements, with a hit/miss count and comparison against a 50% coin-flip baseline.

**Confidence-weighted accuracy** splits predictions into bands based on how far the fear score was from neutral (5). Extreme readings (fear below 2 or above 8) are "high confidence" signals, while readings near 4-6 are "low confidence". This reveals whether the model is more reliable when it's making a strong call.

**Lag correlations** (Pearson and Spearman) measure whether fear at day T predicts returns at T+1, T+2, T+3. The script explains each correlation, whether the signal is predictive or contrarian, and flags statistically significant results.

## The Pipeline

Each run goes through these steps. US and Europe run in parallel via ThreadPoolExecutor, so both regions score simultaneously.

1. **Fetch articles** from NewsAPI for recent headlines, Finnhub for company-specific coverage, and Google News RSS for historical context. Articles are cached for 24 hours so re-runs don't waste API calls.
2. **Clean text** by stripping HTML, truncation markers, encoding artifacts, and URLs from all article fields before they reach the LLM.
3. **Score articles** with GPT-5.4 reading each article and returning recession_fear (0-10), market_sentiment (bearish/neutral/bullish), confidence, rationale, and sector tags. 10 articles are scored concurrently via ThreadPoolExecutor. Results are cached per model.
4. **Fetch market data** for S&P 500 or Euro Stoxx 50 via yfinance, going back 14 days before the earliest article so the daily assessment has prior market context.
5. **Ensemble daily assessment** where both GPT-5.4 and Claude 4.6 see all articles plus previous market data for each day. Their daily fear scores are averaged. Results are cached so subsequent runs skip this step.
6. **Build timeseries** with daily scores, 3-day and 7-day rolling averages, momentum, and volatility.
7. **Merge with market** by aligning sentiment dates with trading days.
8. **Lag correlation** computing Pearson and Spearman correlation at T+0, T+1, T+2, T+3, T+5 to measure predictive power.
9. **XGBoost prediction** trained on historical VIX data, predicting with live sentiment features.
10. **LLM forecast** where Claude reasons about next-day direction from recent data.

## Rescoring

The dashboard lets you switch LLM models and rescore articles. When you hit "Re-score", it respects the currently selected time window. If you're viewing the last 3 days, only those articles get rescored while older scores are preserved. This saves both time and API costs compared to rescoring the full 30-day dataset.

## Caching Strategy

Everything is cached to minimize API costs and speed up re-runs.

**News articles** are cached for 24 hours and won't be refetched until stale. **Per-model article scores** are cached per model name (e.g., `gpt54_us`, `claude46_us`). Switching models in the UI and clicking "Re-score" only scores articles not already cached for that model. **Ensemble daily assessment** is cached as `ensemble_daily_us` and loads instantly on restart. **Market data** is fetched fresh each run since yfinance is free and fast.

On a warm cache, the dashboard launches in seconds.

## Multi-Region Support

The tool analyzes US and Europe independently.

**US** uses NewsAPI US business headlines plus Finnhub company news, scored with a US-specific prompt focused on Fed rate decisions, unemployment, and consumer confidence. Results are compared against the S&P 500 and US sector ETFs (XLK, XLE, XLF, XLV).

**Europe** uses the same sources with European keywords like ECB policy, eurozone, energy crisis, and PMI. Articles are scored with an EU-specific prompt and compared against Euro Stoxx 50 and European sector ETFs.

Region-specific few-shot examples in the prompts ensure the LLM understands that "the Fed held rates steady" is a US signal while "ECB hawkishness" is a European one.

## Dashboard Features

**Region toggle** to switch between US and Europe. **Fear gauge** showing overall market fear as a semicircular indicator. **Dual forecast** with XGBoost prediction and LLM analysis side by side with confidence scores.

**Sentiment vs market chart** plotting recession fear (inverted axis) against S&P 500 or sector ETFs. When you select a sector, the chart swaps to that sector's corresponding ETF (Energy to XLE, Technology to XLK, Finance to XLF, Healthcare to XLV). The overall "All" view uses the full ensemble daily assessment while sector views use per-article score averaging.

**Lag correlation heatmap** showing whether sentiment at day T predicts market movement at T+1, T+3, etc. **Sector heatmap** displaying fear levels across sectors over time, with dedicated LLM assessments for Financials and Energy overriding simple averages. **Sector vs ETF comparison** plotting each sector's fear against its corresponding ETF. **Stock mentions** showing companies extracted from article titles with their fear scores.

**Article browser** with newspaper-style cards sorted by fear. Hover to see sector tags, sentiment badge, and LLM rationale. **Chatbot** powered by Claude 4.6 with full access to article data, so you can ask things like "What caused the sentiment drop on March 13?" and get answers referencing specific articles.

**Model selector** supporting GPT-5.4, Claude Sonnet 4.6, Llama 3.3 70B, DeepSeek R1, or local Ollama. **Time window filter** for 3, 7, 14, or 30 days, which also controls rescoring scope.

## LLM Backends

All accessed through OpenRouter with a single API key:

| Model | Use | Cost |
|-------|-----|------|
| GPT-5.4 | Default article scoring | ~$0.20/run |
| Claude Sonnet 4.6 | Ensemble daily assessment + chatbot | ~$0.30/run |
| Llama 3.3 70B | Free alternative for scoring | Free |
| DeepSeek R1 | Free alternative (slower, reasoning model) | Free |
| Ollama | Local scoring if installed | Free |

A typical first run costs about $0.50 for GPT-5.4 article scoring plus Claude ensemble. Re-runs with cached scores cost nothing.

## Performance and Parallelism

**Region-level parallelism** runs the US and Europe pipelines simultaneously in separate threads. **Article-level parallelism** scores 10 articles concurrently via ThreadPoolExecutor, handling 70 articles in about 45 seconds instead of 12 minutes sequentially. **Caching** means all API results are saved to disk and warm restarts skip LLM calls entirely. **Rate limiting** uses exponential backoff with retries on API failures from OpenRouter and yfinance.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Add API keys to .env
cp .env.example .env
# Edit: NEWSAPI_KEY, OPENROUTER_API_KEY (required)
# Optional: FINNHUB_API_KEY, HUGGINGFACE_TOKEN

python run.py
# Open http://127.0.0.1:8050

# Run accuracy evaluation
python metric.py
```

First run takes about 2 minutes for fetching, scoring, and ensemble assessment. Subsequent runs with cache take about 10 seconds.

## Project Structure

```
src/
  config/          Settings (YAML + .env), logging
  ingestion/       NewsAPI, Finnhub, Google News RSS, text cleaning, 24h caching
  analysis/        LLM scorers (GPT, Claude, Llama, Ollama), daily assessor, ensemble
  timeseries/      Daily aggregation, market data (yfinance), lag correlation
  prediction/      XGBoost (VIX-trained), feature engineering, LLM forecast
  visualization/   Plotly charts (sentiment, sectors, gauge, stocks, ETF comparison)
  chatbot/         Claude 4.6 Q&A with article context
  dashboard/       Dash app, layout, callbacks, article browser
  pipeline.py      Orchestrator (parallel regions, caching, ensemble)

prompts/           Versioned LLM prompts (US, EU, daily assessment, chat, sector)
metric.py          Backtest predictions against actual S&P 500 movements
```
