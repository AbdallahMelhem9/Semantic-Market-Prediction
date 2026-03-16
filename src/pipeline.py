import logging
from datetime import date

import pandas as pd
import yaml

from src.config.settings import load_settings, Settings
from src.config.logging_setup import setup_logging
from src.ingestion.news_client import fetch_news
from src.ingestion.cache import CacheManager
from src.analysis.scorer_factory import create_scorer
from src.analysis.batch_processor import score_articles_in_batches
from src.timeseries.aggregator import aggregate_daily_sentiment, export_timeseries
from src.timeseries.market_data import fetch_sp500, merge_sentiment_and_market
from src.timeseries.correlation import compute_lag_correlations
from src.prediction.feature_engineer import engineer_features
from src.prediction.model import SentimentPredictor

logger = logging.getLogger(__name__)


def _load_regions(settings: Settings) -> dict:
    from pathlib import Path
    config_path = Path(settings.paths.root) / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("regions", {})


def run_pipeline_for_region(settings: Settings, region_key: str, region_config: dict, cache: CacheManager) -> dict:
    region_name = region_config.get("name", region_key)
    keywords = region_config.get("keywords", [])
    market_index = region_config.get("market_index", "^GSPC")
    market_name = region_config.get("market_name", "S&P 500")
    prompt_suffix = region_config.get("prompt_suffix", "")
    country = region_config.get("news_country", "")

    logger.info(f"{'='*50}")
    logger.info(f"REGION: {region_name} ({region_key})")

    settings.news.keywords = keywords

    logger.info(f"[{region_key}] Step 1: News Ingestion")

    if cache.has_fresh_cached_news(settings):
        articles_df = cache.load_news(settings)
    else:
        articles_df = fetch_news(settings)

    # Supplement with additional sources (only if we need more date coverage)
    MAX_RECENT = 70     # last 7 days
    MAX_HISTORICAL = 100  # weeks 2-5
    MAX_TOTAL = 170
    MAX_DAYS = 35       # ~5 weeks to catch more Google News

    needs_supplement = not cache.has_fresh_cached_news(settings) or (not articles_df.empty and articles_df["date"].nunique() < 7)

    if needs_supplement:
        if "api_source" not in articles_df.columns:
            articles_df["api_source"] = "newsapi"

        # Google News RSS — ONLY for historical (>7 days old, title-only = lower quality)
        try:
            from src.ingestion.google_news_client import fetch_google_news
            gn_df = fetch_google_news(keywords, max_articles=MAX_HISTORICAL)
            if not gn_df.empty:
                gn_df["date"] = pd.to_datetime(gn_df["date"]).dt.date
                cutoff_7d = date.today() - timedelta(days=7)
                gn_df = gn_df[gn_df["date"] < cutoff_7d]
                if not gn_df.empty:
                    before = len(articles_df)
                    articles_df = pd.concat([articles_df, gn_df], ignore_index=True)
                    articles_df = articles_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
                    added = len(articles_df) - before
                    if added > 0:
                        logger.info(f"[{region_key}] Google News added {added} historical articles (>7d old only)")
        except Exception as e:
            logger.warning(f"Google News failed: {e}")

    # Enforce: only last 28 days, 70 recent (last 7d) + 100 historical (weeks 2-4)
    if not articles_df.empty and "date" in articles_df.columns:
        articles_df["date"] = pd.to_datetime(articles_df["date"]).dt.date
        from datetime import timedelta

        cutoff_old = date.today() - timedelta(days=MAX_DAYS)
        cutoff_7d = date.today() - timedelta(days=7)

        articles_df = articles_df[articles_df["date"] >= cutoff_old]

        recent = articles_df[articles_df["date"] >= cutoff_7d].sort_values("published_at", ascending=False).head(MAX_RECENT)
        historical = articles_df[articles_df["date"] < cutoff_7d].sort_values("published_at", ascending=False).head(MAX_HISTORICAL)
        articles_df = pd.concat([recent, historical], ignore_index=True)

        n_days = articles_df["date"].nunique()
        logger.info(f"[{region_key}] Final: {len(recent)} recent + {len(historical)} historical = {len(articles_df)} articles across {n_days} days")

    cache.save_news(articles_df, settings)

    if articles_df.empty:
        logger.warning(f"[{region_key}] No articles available")
        return _empty_result(region_key, region_name, market_name)

    logger.info(f"[{region_key}] Articles ready: {len(articles_df)}")

    logger.info(f"[{region_key}] Step 2: LLM Scoring with {prompt_suffix} prompt")
    scorer = create_scorer(settings, region=prompt_suffix)
    scored_df = score_articles_in_batches(articles_df, scorer, settings)

    # Go back 14 days before earliest article so even day 1 gets prior S&P context
    logger.info(f"[{region_key}] Step 3: Market Data ({market_name})")
    market_df = pd.DataFrame()
    scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.date
    if not scored_df.empty:
        from datetime import timedelta
        dates = scored_df["date"].tolist()
        min_date = min(dates) - timedelta(days=14)
        max_date = max(dates)
        market_df = _fetch_market_index(market_index, min_date, max_date)

    logger.info(f"[{region_key}] Step 4: LLM Daily Assessment")
    daily_assessments = []
    try:
        from src.analysis.daily_assessor import assess_daily_sentiment
        daily_assessments = assess_daily_sentiment(settings, scored_df, market_df)
        if daily_assessments:
            logger.info(f"[{region_key}] LLM assessed {len(daily_assessments)} days")
    except Exception as e:
        logger.warning(f"[{region_key}] Daily assessment failed, using averages: {e}")

    logger.info(f"[{region_key}] Step 5: Timeseries")
    daily_df = aggregate_daily_sentiment(scored_df, settings.visualization.rolling_windows)

    # Override daily fear with LLM assessment scores (smarter than averaging)
    if daily_assessments:
        assess_df = pd.DataFrame(daily_assessments)
        assess_df["date"] = pd.to_datetime(assess_df["date"]).dt.date
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        daily_df = pd.merge(daily_df, assess_df[["date", "daily_fear", "key_driver", "reasoning"]], on="date", how="left")
        mask = daily_df["daily_fear"].notna()
        daily_df.loc[mask, "avg_recession_fear"] = daily_df.loc[mask, "daily_fear"]
        # Recompute rolling averages
        for window in settings.visualization.rolling_windows:
            daily_df[f"rolling_{window}d"] = daily_df["avg_recession_fear"].rolling(window, min_periods=1).mean()
        daily_df["momentum"] = daily_df["avg_recession_fear"].diff()

    export_timeseries(daily_df, f"{settings.paths.processed}/daily_sentiment_{region_key}.csv")

    logger.info(f"[{region_key}] Step 6: Merge Sentiment + Market")
    merged_df = daily_df
    if not daily_df.empty and not market_df.empty:
        merged_df = merge_sentiment_and_market(daily_df, market_df)

    logger.info(f"[{region_key}] Step 6: Correlation")
    corr_df = compute_lag_correlations(merged_df)

    logger.info(f"[{region_key}] Step 7: XGBoost Prediction")
    prediction = _run_prediction(settings, merged_df, region_key)

    logger.info(f"[{region_key}] Step 8: LLM Forecast")
    llm_forecast = {}
    try:
        from src.prediction.llm_forecast import get_llm_forecast
        llm_forecast = get_llm_forecast(settings, merged_df, market_name)
        if llm_forecast:
            logger.info(f"[{region_key}] LLM Forecast: {llm_forecast.get('direction')} ({llm_forecast.get('confidence', 0):.0%})")
    except Exception as e:
        logger.warning(f"LLM forecast failed: {e}")

    return {
        "region_key": region_key,
        "region_name": region_name,
        "market_name": market_name,
        "articles_df": articles_df,
        "scored_df": scored_df,
        "merged_df": merged_df,
        "corr_df": corr_df,
        "prediction": prediction,
        "llm_forecast": llm_forecast,
    }


def _fetch_market_index(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    import time as _time
    from datetime import timedelta

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=str(start - timedelta(days=5)), end=str(end + timedelta(days=1)))
            if hist.empty:
                if attempt < 2:
                    _time.sleep(3)
                    continue
                return pd.DataFrame()
            df = hist[["Close"]].reset_index()
            df.columns = ["date", "sp500_close"]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["sp500_return"] = df["sp500_close"].pct_change()
            df["sp500_direction"] = (df["sp500_return"] > 0).astype(int)
            return df
        except Exception as e:
            if attempt < 2:
                _time.sleep(3)
                continue
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()


def _run_prediction(settings: Settings, merged_df: pd.DataFrame, region_key: str) -> dict:
    prediction = {}
    try:
        from src.prediction.historical_data import fetch_historical_training_data

        hist_df = fetch_historical_training_data(years=2)
        if hist_df.empty or len(hist_df) < 50:
            logger.warning(f"[{region_key}] Not enough historical data for training")
            return prediction

        train_features, train_target = engineer_features(hist_df)
        if len(train_features) < 50:
            return prediction

        predictor = SentimentPredictor(settings)
        predictor.train(train_features, train_target)

        if not merged_df.empty and "avg_recession_fear" in merged_df.columns:
            latest = merged_df.iloc[-1]
            live_row = {}
            for col in train_features.columns:
                if col in merged_df.columns:
                    live_row[col] = latest.get(col, 0)
                else:
                    live_row[col] = 0

            live_row["avg_recession_fear"] = latest.get("avg_recession_fear", 5.0)
            if "rolling_3d" in train_features.columns:
                live_row["rolling_3d"] = latest.get("rolling_3d", latest.get("avg_recession_fear", 5.0))
            if "rolling_7d" in train_features.columns:
                live_row["rolling_7d"] = latest.get("rolling_7d", latest.get("avg_recession_fear", 5.0))
            if "volatility" in train_features.columns:
                live_row["volatility"] = latest.get("volatility", 0.5)
            if "article_count" in train_features.columns:
                live_row["article_count"] = latest.get("article_count", 10)
            if "momentum" in train_features.columns:
                live_row["momentum"] = latest.get("momentum", 0)

            live_features = pd.DataFrame([live_row])[train_features.columns]
            prediction = predictor.predict_next_day(live_features)
            logger.info(f"[{region_key}] Prediction: {prediction.get('direction')} ({prediction.get('confidence', 0):.0%})")

    except Exception as e:
        logger.error(f"[{region_key}] Prediction failed: {e}")
    return prediction


def _empty_result(region_key: str, region_name: str, market_name: str) -> dict:
    return {
        "region_key": region_key,
        "region_name": region_name,
        "market_name": market_name,
        "articles_df": pd.DataFrame(),
        "scored_df": pd.DataFrame(),
        "merged_df": pd.DataFrame(),
        "corr_df": pd.DataFrame(),
        "prediction": {},
    }


def run_pipeline(settings: Settings = None) -> dict:
    """Run pipeline for all configured regions. Returns dict keyed by region."""
    if settings is None:
        settings = load_settings()
        setup_logging(settings)

    cache = CacheManager(settings)
    regions = _load_regions(settings)

    if not regions:
        # No regions defined — fall back to single US pipeline
        regions = {"us": {
            "name": "United States", "market_index": "^GSPC", "market_name": "S&P 500",
            "keywords": settings.news.keywords, "prompt_suffix": "us", "news_country": "us",
        }}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import copy

    all_results = {}

    if len(regions) > 1:
        logger.info(f"Running {len(regions)} regions in parallel...")
        with ThreadPoolExecutor(max_workers=len(regions)) as pool:
            futures = {}
            for region_key, region_config in regions.items():
                region_settings = copy.deepcopy(settings)  # avoid keyword overwrite across threads
                f = pool.submit(run_pipeline_for_region, region_settings, region_key, region_config, cache)
                futures[f] = region_key

            for future in as_completed(futures):
                region_key = futures[future]
                try:
                    all_results[region_key] = future.result()
                except Exception as e:
                    logger.error(f"Region {region_key} failed: {e}")
                    all_results[region_key] = _empty_result(region_key, regions[region_key].get("name", ""), regions[region_key].get("market_name", ""))
    else:
        for region_key, region_config in regions.items():
            all_results[region_key] = run_pipeline_for_region(settings, region_key, region_config, cache)

    logger.info("=" * 50)
    logger.info("All regions complete!")
    return all_results
