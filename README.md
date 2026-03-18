# Semantic Market Prediction

LLM-powered sentiment analysis for financial news. Scores recession fear across US and European markets, correlates it with real market indices, and predicts next-day direction.

Built for the BNP Paribas Equity & Derivatives Strategy team.

## What It Does

Collects financial news from multiple APIs, has GPT-5.4 read and score each article for recession fear (0 to 10), then builds a daily sentiment timeseries compared against the S&P 500 (US) and Euro Stoxx 50 (Europe). An XGBoost model trained on 2 years of VIX data predicts whether the market goes up or down tomorrow.

Also includes a chatbot (Claude 4.6) that can answer questions about the data, and a standalone accuracy evaluation script (`metric.py`) for backtesting predictions against real market movements.

## Data Sources

News comes from three APIs that complement each other.

**NewsAPI** is the primary source. Full articles (title, description, body) for the last 30 days. Free tier caps at 100 per request, so keywords focus on recession-relevant topics: "recession", "Fed rate", "unemployment", "inflation", "GDP", "consumer confidence".

**Finnhub** fills company-specific coverage. Fetches headlines and summaries across 16 major US tickers (AAPL, MSFT, JPM, XOM, etc.) spanning Tech, Finance, Energy, Healthcare, Consumer, and Industrials. Gives sector-level depth that general queries miss.

**Google News RSS** is a free fallback for historical gaps. Title-only (no body), so lower quality. The pipeline only uses Google News articles older than 7 days, where NewsAPI coverage thins out.

All sources are deduplicated by URL before scoring. Final dataset caps at 300 articles across 35 days max.

## Text Cleaning

Before anything reaches the LLM, articles go through `src/ingestion/text_cleaner.py`. Strips HTML tags, removes NewsAPI's `[+1234 chars]` truncation markers, cleans encoding artifacts (non-breaking spaces, zero-width characters, BOM), removes URLs, collapses whitespace. Articles shorter than 15 characters after cleaning get flagged unusable. This keeps LLM input clean regardless of which API the article came from.

## How Scoring Works

Tried three approaches and settled on what worked best.

**Simple averaging** was first. Score each article, average all scores for the day. Problem: a single "Fed emergency rate cut" article gets diluted by 15 routine earnings reports.

**LLM daily assessment** was second. Give the LLM all articles for a day plus previous market data, ask for one daily score weighing importance. Much better. The LLM recognizes that one major event matters more than many minor ones.

**Ensemble** is what we use. Run the daily assessment with both GPT-5.4 and Claude 4.6, average their fear scores. Reduces model-specific bias and produces more stable readings.

### No Data Leakage

Each day's assessment only sees market data from previous days. Never the current day or future. Day 1 gets the prior week's S&P 500 closes. Day 2 gets Day 1's assessment plus market data. Critical for honest backtesting.

### Per-Article vs Per-Day

Per-article scores (GPT-5.4) feed the sector breakdown, article browser, and stock mentions. The per-day ensemble (GPT-5.4 + Claude 4.6) powers the main sentiment timeseries and predictions. Financials and Energy get their own dedicated GPT-5.4 daily assessments with sector-specific prompts, while other sectors use per-article averages.

## Prediction

Two independent methods shown side by side.

**XGBoost** trained on 2 years of VIX and S&P 500 data (~500 trading days). Features: daily fear, 3/7-day rolling averages, volatility, article count, momentum. Time-based 70/30 train/test split, no random shuffle (financial data needs chronological splits). Outputs next-day direction with confidence.

**LLM forecast** gives Claude the last 7 days of fear scores and S&P prices, asks it to reason about what happens next. Returns direction, confidence, and a written explanation.

Dashboard shows both so you can see where they agree or disagree.

## Accuracy Evaluation

`metric.py` backtests against actual S&P 500 data. Converts each day's fear score into a directional prediction (above 5 = predict down, below 5 = predict up), checks if the market moved that way next trading day.

Reports direction accuracy with day-by-day table, confidence-weighted accuracy split by signal strength (extreme readings vs neutral), and lag correlations (Pearson and Spearman) measuring whether fear at day T predicts returns at T+1, T+2, T+3.

## The Pipeline

Each run goes through these steps. US and Europe run in parallel via ThreadPoolExecutor.

1. **Fetch articles** from NewsAPI, Finnhub, and Google News RSS. Cached 24 hours.
2. **Clean text** by stripping HTML, truncation markers, encoding artifacts, URLs.
3. **Score articles** with GPT-5.4. 10 concurrent workers via ThreadPoolExecutor. Cached per model.
4. **Fetch market data** for S&P 500 or Euro Stoxx 50 via yfinance, going back 14 days before earliest article.
5. **Daily assessment** runs 4 parallel streams: GPT-5.4 overall, Claude 4.6 overall (averaged for ensemble), GPT-5.4 Financials, GPT-5.4 Energy. Each stream processes days sequentially for sentiment continuity. Cached.
6. **Build timeseries** with daily scores, rolling averages, momentum, volatility.
7. **Merge with market** aligning sentiment dates with trading days.
8. **Lag correlation** at T+0 through T+5.
9. **XGBoost prediction** trained on VIX history, predicting with live features.
10. **LLM forecast** where Claude reasons about next-day direction.

## Rescoring

The dashboard lets you switch models and rescore. Hit "Re-score" and it respects the selected time window. Viewing last 3 days? Only those articles get rescored, older scores stay. Also re-runs Financials and Energy sector assessments for the window. Saves time and API costs vs rescoring the full 30 days.

## Caching

Everything cached to minimize costs and speed up restarts.

**News articles** cached 24 hours. **Article scores** cached per model name (`gpt54_us`, `claude46_us`). Switching models only scores articles not already cached. **Ensemble daily** and **sector daily** cached separately, load instantly on restart. **Market data** fetched fresh (yfinance is free and fast).

Warm cache = dashboard launches in seconds.

## Multi-Region

US and Europe analyzed independently.

**US** uses NewsAPI headlines plus Finnhub company news, scored with US-specific prompts (Fed, unemployment, consumer confidence). Compared against S&P 500 and sector ETFs (XLK, XLE, XLF, XLV, XLY, XLI, XLP, XLB, XLU, XLRE, XLC).

**Europe** uses same sources with European keywords (ECB, eurozone, energy crisis, PMI). EU-specific prompts, compared against Euro Stoxx 50 and European sector ETFs.

Few-shot examples in prompts teach the LLM that "Fed held rates steady" is US while "ECB hawkishness" is European.

## Chatbot

3-step architecture powered by Claude 4.6.

**Step 1** resolves the query using chat history and current dashboard filters. If you're viewing US/Energy and ask "what's the sentiment?", it knows you mean US Energy. If you explicitly ask about Europe, it overrides.

**Step 2** classifies into one of 8 task types (factual, sector, prediction, correlation, comparison, summary, methodology, other) and fetches only the relevant data slice.

**Step 3** sends a specialized prompt with the sliced data to Claude, streamed back in real-time.

Methodology questions are answered instantly from hardcoded responses, no LLM call needed.

## Dashboard

Region toggle (US/Europe). Fear gauge. Dual forecast panel (XGBoost + LLM side by side).

Sentiment vs market chart with inverted fear axis against S&P 500 or sector ETFs. Selecting a sector swaps to its ETF. Lag correlation heatmap. Sector heatmap with dedicated LLM scores for Financials and Energy. Sector vs ETF comparison. Stock mentions extracted from titles.

Article browser with newspaper-style cards sorted by fear. Time window filter (3 days, 1 week, 2 weeks, 1 month) controls both display and rescoring scope.

Sectors use the 11 GICS standard: Technology, Healthcare, Financials, Energy, Consumer Discretionary, Consumer Staples, Industrials, Materials, Utilities, Real Estate, Communication Services.

## LLM Backends

All through OpenRouter (one API key):

| Model | Use | Cost |
|-------|-----|------|
| GPT-5.4 | Article scoring + sector daily assessment | ~$0.20/run |
| Claude Sonnet 4.6 | Ensemble daily + chatbot | ~$0.30/run |
| Llama 3.3 70B | Free alternative for scoring | Free |
| DeepSeek R1 | Free alternative, slower | Free |
| Ollama | Local scoring if installed | Free |

First run ~$0.50 (GPT articles + Claude ensemble). Cached re-runs cost nothing.

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

# Backtest accuracy
python metric.py
```

First run takes ~2 minutes (fetch + score + ensemble). Cached runs ~10 seconds.

## Project Structure

```
src/
  config/          Settings (YAML + .env), logging
  ingestion/       NewsAPI, Finnhub, Google News RSS, text cleaning, 24h caching
  analysis/        LLM scorers (GPT, Claude, Llama, Ollama), daily assessor, ensemble
  timeseries/      Daily aggregation, market data (yfinance), lag correlation
  prediction/      XGBoost (VIX-trained), feature engineering, LLM forecast
  visualization/   Plotly charts (sentiment, sectors, gauge, stocks, ETF comparison)
  chatbot/         Claude 4.6 chat with 3-step routing, streaming, 8 prompt types
  dashboard/       Dash app, layout, callbacks, article browser
  pipeline.py      Orchestrator (parallel regions, caching, ensemble, sector assessment)

prompts/           Versioned LLM prompts (US, EU, daily assessment, chat, sector)
metric.py          Backtest predictions against actual S&P 500 movements
```

## Deployment

Deployed on Render: https://semantic-market-prediction.onrender.com

Free tier. First load after idle takes ~30 seconds to wake, then ~2 minutes for the pipeline.
