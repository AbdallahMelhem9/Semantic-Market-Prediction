import ast
import logging
from collections import Counter
from datetime import date, timedelta

import pandas as pd
from dash import Input, Output, State, html, no_update, callback_context

from src.visualization.sentiment_chart import create_sentiment_vs_sp500_chart
from src.visualization.correlation_heatmap import create_correlation_heatmap
from src.visualization.sector_heatmap import create_sector_heatmap
from src.visualization.sector_timeseries import create_sector_timeseries
from src.visualization.stock_mentions import create_stock_mentions_chart
from src.visualization.fear_gauge import create_fear_gauge
from src.visualization.sector_vs_etf import create_sector_vs_etf_chart
from src.dashboard.components.article_browser import create_article_browser

logger = logging.getLogger(__name__)

SECTOR_ETFS_US = {
    "Technology": "XLK", "Finance": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Consumer": "XLY", "Industrial": "XLI",
}
SECTOR_ETFS_EU = {
    "Technology": "EXV8.DE", "Finance": "EXV1.DE",
    "Energy": "EXH1.DE", "Healthcare": "EXV4.DE",
}


def _fetch_sector_etf(symbol: str, start_date, end_date) -> pd.DataFrame:
    import yfinance as yf
    import time as _time
    from datetime import timedelta

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=str(start_date - timedelta(days=5)),
                end=str(end_date + timedelta(days=1)),
            )
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
            logger.warning(f"Failed to fetch ETF {symbol}: {e}")
            return pd.DataFrame()


def register_callbacks(app, pipeline_data: dict):
    """pipeline_data is keyed by region: {"us": {...}, "europe": {...}}"""

    @app.callback(
        [
            Output("store-active-region", "data"),
            Output("btn-region-us", "color"), Output("btn-region-us", "outline"),
            Output("btn-region-europe", "color"), Output("btn-region-europe", "outline"),
            Output("btn-region-all", "color"), Output("btn-region-all", "outline"),
        ],
        [Input("btn-region-us", "n_clicks"), Input("btn-region-europe", "n_clicks"), Input("btn-region-all", "n_clicks")],
        prevent_initial_call=True,
    )
    def toggle_region(us_clicks, eu_clicks, all_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return "us", "primary", False, "secondary", True, "secondary", True

        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn == "btn-region-us":
            return "us", "primary", False, "secondary", True, "secondary", True
        elif btn == "btn-region-europe":
            return "europe", "secondary", True, "primary", False, "secondary", True
        else:
            return "all", "secondary", True, "secondary", True, "primary", False

    @app.callback(
        [
            Output("fear-gauge", "figure"),
            Output("main-chart", "figure"),
            Output("correlation-heatmap", "figure"),
            Output("sector-heatmap", "figure"),
            Output("sector-timeseries", "figure"),
            Output("sector-vs-etf", "figure"),
            Output("stock-mentions", "figure"),
            Output("prediction-card", "children"),
            Output("filter-sector", "options"),
        ],
        [Input("filter-days", "value"), Input("filter-sector", "value"), Input("store-active-region", "data"), Input("store-rescore-trigger", "data")],
    )
    def update_dashboard(days_filter, sector_filter, active_region, rescore_trigger):
        merged_df, scored_df, corr_df, prediction, llm_forecast, market_name = _get_region_data(pipeline_data, active_region)

        # If 3 months selected and we don't have enough data, fetch more from Google News
        if days_filter == 90 and active_region != "all":
            data = pipeline_data.get(active_region, {})
            scored_dates = scored_df["date"].nunique() if not scored_df.empty and "date" in scored_df.columns else 0
            if scored_dates < 20:
                try:
                    from src.config.settings import load_settings
                    from src.ingestion.google_news_client import fetch_google_news
                    from src.analysis.batch_processor import score_articles_in_batches
                    from src.timeseries.aggregator import aggregate_daily_sentiment
                    import yaml
                    from pathlib import Path

                    settings = load_settings()
                    config_path = Path(settings.paths.root) / "config.yaml"
                    with open(config_path) as f:
                        raw = yaml.safe_load(f)
                    region_config = raw.get("regions", {}).get(active_region, {})
                    keywords = region_config.get("keywords", ["recession", "economy"])
                    prompt_suffix = "us" if active_region == "us" else "eu"

                    gn_df = fetch_google_news(keywords, max_articles=170)
                    if not gn_df.empty:
                        # Merge with existing articles
                        articles_df = data.get("articles_df", pd.DataFrame())
                        if not articles_df.empty:
                            articles_df = pd.concat([articles_df, gn_df], ignore_index=True)
                            articles_df = articles_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
                        else:
                            articles_df = gn_df

                        # Only score NEW articles (skip already scored URLs)
                        existing_scored = data.get("scored_df", pd.DataFrame())
                        already_scored_urls = set(existing_scored["url"].tolist()) if not existing_scored.empty and "url" in existing_scored.columns else set()
                        new_articles = articles_df[~articles_df["url"].isin(already_scored_urls)]

                        if not new_articles.empty:
                            scorer = create_scorer(settings, region=prompt_suffix)
                            new_scored = score_articles_in_batches(new_articles, scorer, settings)
                            scored_df = pd.concat([existing_scored, new_scored], ignore_index=True)
                            scored_df = scored_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
                        else:
                            scored_df = existing_scored

                        pipeline_data[active_region]["scored_df"] = scored_df
                        pipeline_data[active_region]["articles_df"] = articles_df

                        # Re-aggregate — sentiment only, no market data for 3-month view
                        daily = aggregate_daily_sentiment(scored_df, settings.visualization.rolling_windows)
                        if not daily.empty:
                            daily["date"] = pd.to_datetime(daily["date"]).dt.date
                            merged_df = daily
                            # Don't overwrite main pipeline data — this is a temporary extended view

                        logger.info(f"3-month fetch: {len(new_articles)} new articles scored, {scored_df['date'].nunique()} total days")
                except Exception as e:
                    logger.warning(f"3-month extended fetch failed: {e}")

        if not merged_df.empty and "date" in merged_df.columns:
            merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.date
        if not scored_df.empty and "date" in scored_df.columns:
            scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.date

        if days_filter:
            cutoff = date.today() - timedelta(days=days_filter)
            if not merged_df.empty:
                merged_df = merged_df[merged_df["date"] >= cutoff]
            if not scored_df.empty:
                scored_df = scored_df[scored_df["date"] >= cutoff]

        sector_options = _extract_sector_options(scored_df)

        chart_market_name = market_name

        if sector_filter and not scored_df.empty and "sectors" in scored_df.columns:
            scored_df = scored_df[scored_df["sectors"].apply(lambda x: _contains_sector(x, sector_filter))]
            if not scored_df.empty:
                from src.timeseries.aggregator import aggregate_daily_sentiment
                filtered_daily = aggregate_daily_sentiment(scored_df)
                if not filtered_daily.empty:
                    filtered_daily["date"] = pd.to_datetime(filtered_daily["date"]).dt.date

                    # Keep S&P 500 on merged_df, fetch sector ETF separately
                    if not merged_df.empty:
                        market_cols = [c for c in merged_df.columns if c.startswith("sp500")]
                        if market_cols:
                            merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.date
                            market_data = merged_df[["date"] + market_cols].drop_duplicates(subset=["date"])
                            merged_df = pd.merge(filtered_daily, market_data, on="date", how="left")
                        else:
                            merged_df = filtered_daily
                    else:
                        merged_df = filtered_daily

                    # Fetch sector ETF and REPLACE S&P 500 (don't show both)
                    etf_map = SECTOR_ETFS_US if active_region != "europe" else SECTOR_ETFS_EU
                    etf_symbol = etf_map.get(sector_filter)
                    if etf_symbol:
                        etf_data = _fetch_sector_etf(etf_symbol, filtered_daily["date"].min(), filtered_daily["date"].max())
                        if not etf_data.empty:
                            # Replace sp500 columns with sector ETF data
                            merged_df = merged_df.drop(columns=[c for c in merged_df.columns if c.startswith("sp500")], errors="ignore")
                            merged_df = pd.merge(merged_df, etf_data, on="date", how="left")
                            chart_market_name = f"{sector_filter} ETF ({etf_symbol})"

        avg_fear = merged_df["avg_recession_fear"].mean() if not merged_df.empty and "avg_recession_fear" in merged_df.columns else 5.0
        gauge_fig = create_fear_gauge(avg_fear)

        main_fig = create_sentiment_vs_sp500_chart(
            merged_df,
            market_label=chart_market_name,
        )

        corr_fig = create_correlation_heatmap(corr_df)
        sector_fig = create_sector_heatmap(scored_df)
        sector_ts_fig = create_sector_timeseries(scored_df)
        sector_etf_fig = create_sector_vs_etf_chart(scored_df, region=active_region if active_region != "all" else "us")
        stock_fig = create_stock_mentions_chart(scored_df)
        pred_card = _build_dual_forecast(prediction, llm_forecast)

        return gauge_fig, main_fig, corr_fig, sector_fig, sector_ts_fig, sector_etf_fig, stock_fig, pred_card, sector_options

    @app.callback(
        [Output("rescore-status", "children"), Output("store-rescore-trigger", "data")],
        [Input("btn-rescore", "n_clicks")],
        [State("filter-llm", "value"), State("store-active-region", "data"), State("store-rescore-trigger", "data")],
        prevent_initial_call=True,
    )
    def rescore_articles(n_clicks, model_value, active_region, current_trigger):
        if not model_value:
            return "Select a model first", current_trigger or 0

        import copy
        import yaml
        from src.config.settings import load_settings
        from src.ingestion.cache import CacheManager
        from src.analysis.batch_processor import score_articles_in_batches
        from src.analysis.daily_assessor import assess_daily_sentiment
        from src.timeseries.aggregator import aggregate_daily_sentiment
        from src.timeseries.market_data import merge_sentiment_and_market
        from src.timeseries.correlation import compute_lag_correlations
        from src.prediction.llm_forecast import get_llm_forecast
        from src.pipeline import _run_prediction

        settings = load_settings()

        if model_value == "ollama":
            settings.llm.backend = "ollama"
            settings.llm.model = "llama3.1"
        else:
            settings.llm.backend = "openrouter"
            settings.llm.model = model_value

        regions_to_score = [active_region] if active_region != "all" else list(pipeline_data.keys())

        for region_key in regions_to_score:
            data = pipeline_data.get(region_key, {})
            articles_df = data.get("articles_df", pd.DataFrame())
            if articles_df.empty:
                continue

            prompt_suffix = "us" if region_key == "us" else "eu"

            scorer = create_scorer(settings, region=prompt_suffix)
            scored_df = score_articles_in_batches(articles_df, scorer, settings)
            pipeline_data[region_key]["scored_df"] = scored_df

            # Get ORIGINAL market data (saved once, never modified)
            if "_original_market_df" not in data:
                old_merged = data.get("merged_df", pd.DataFrame()).copy()
                if not old_merged.empty:
                    old_merged["date"] = pd.to_datetime(old_merged["date"]).dt.date
                    market_cols = [c for c in old_merged.columns if c.startswith("sp500")]
                    if market_cols:
                        pipeline_data[region_key]["_original_market_df"] = old_merged[["date"] + market_cols].drop_duplicates(subset=["date"]).copy()

            market_df = data.get("_original_market_df", pd.DataFrame()).copy()

            daily_assessments = []
            try:
                daily_assessments = assess_daily_sentiment(settings, scored_df, market_df)
            except Exception:
                pass

            daily = aggregate_daily_sentiment(scored_df, settings.visualization.rolling_windows)
            if not daily.empty and daily_assessments:
                assess_df = pd.DataFrame(daily_assessments)
                assess_df["date"] = pd.to_datetime(assess_df["date"]).dt.date
                daily["date"] = pd.to_datetime(daily["date"]).dt.date
                daily = pd.merge(daily, assess_df[["date", "daily_fear", "key_driver", "reasoning"]], on="date", how="left")
                mask = daily["daily_fear"].notna()
                daily.loc[mask, "avg_recession_fear"] = daily.loc[mask, "daily_fear"]
                for w in settings.visualization.rolling_windows:
                    daily[f"rolling_{w}d"] = daily["avg_recession_fear"].rolling(w, min_periods=1).mean()
                daily["momentum"] = daily["avg_recession_fear"].diff()

            # Step 6: Merge — sentiment onto market dates (market data is the anchor, never changes)
            if not daily.empty and not market_df.empty:
                daily["date"] = pd.to_datetime(daily["date"]).dt.date
                market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
                # Inner join: only keep dates where BOTH sentiment and market exist
                merged = pd.merge(market_df, daily, on="date", how="inner")
                pipeline_data[region_key]["merged_df"] = merged
            elif not daily.empty:
                pipeline_data[region_key]["merged_df"] = daily

            merged = pipeline_data[region_key].get("merged_df", pd.DataFrame())
            pipeline_data[region_key]["corr_df"] = compute_lag_correlations(merged)

            pipeline_data[region_key]["prediction"] = _run_prediction(settings, merged, region_key)

            try:
                llm_fc = get_llm_forecast(settings, merged, data.get("market_name", "S&P 500"))
                pipeline_data[region_key]["llm_forecast"] = llm_fc
            except Exception:
                pass

        model_name = model_value.split("/")[-1] if "/" in model_value else model_value
        new_trigger = (current_trigger or 0) + 1
        return html.Div(f"Re-scored with {model_name}", style={"color": "#22c55e", "fontSize": "0.8rem"}), new_trigger

    from src.analysis.scorer_factory import create_scorer

    @app.callback(
        Output("article-browser-container", "children"),
        [Input("store-active-region", "data"), Input("filter-days", "value")],
    )
    def update_articles(active_region, days_filter):
        _, scored_df, _, _, _, _ = _get_region_data(pipeline_data, active_region)
        if not scored_df.empty and "date" in scored_df.columns:
            scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.date
            if days_filter:
                cutoff = date.today() - timedelta(days=days_filter)
                scored_df = scored_df[scored_df["date"] >= cutoff]
        return create_article_browser(scored_df)

    chat_engine = None

    @app.callback(
        [Output("chat-messages", "children"), Output("chat-input", "value"), Output("chat-loading", "children"), Output("store-chat-history", "data")],
        [Input("btn-chat", "n_clicks")],
        [State("chat-input", "value"), State("store-chat-history", "data"), State("store-active-region", "data")],
        prevent_initial_call=True,
    )
    def handle_chat(n_clicks, user_input, chat_history, active_region):
        nonlocal chat_engine

        if not user_input or not user_input.strip():
            return no_update, no_update, no_update, no_update

        if chat_history is None:
            chat_history = []

        chat_history.append({"role": "user", "text": user_input})

        region_data = pipeline_data.get(active_region, {}) if active_region != "all" else _merge_all_regions(pipeline_data)

        if chat_engine is None:
            try:
                from src.chatbot.chat_engine import ChatEngine
                from src.config.settings import load_settings
                chat_engine = ChatEngine(load_settings(), region_data)
            except Exception:
                chat_engine = None

        if chat_engine:
            chat_engine.pipeline_data = region_data
            response = chat_engine.ask(user_input)
        else:
            response = "Chat engine not available."

        chat_history.append({"role": "assistant", "text": response})

        messages_ui = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages_ui.append(html.Div(msg["text"], style={
                    "padding": "8px 14px", "margin": "6px 0", "borderRadius": "8px",
                    "maxWidth": "75%", "marginLeft": "auto",
                    "background": "#253045", "color": "#d4dce8", "fontSize": "0.85rem",
                }))
            else:
                # Format assistant response with line breaks
                lines = msg["text"].split("\n")
                formatted = []
                for line in lines:
                    if line.strip().startswith("- ") or line.strip().startswith("• "):
                        formatted.append(html.Div(line, style={"paddingLeft": "12px", "fontSize": "0.8rem"}))
                    elif line.strip().startswith("References") or line.strip().startswith("**"):
                        formatted.append(html.Div(line, style={"fontWeight": "600", "marginTop": "6px", "fontSize": "0.8rem"}))
                    elif line.strip():
                        formatted.append(html.Div(line, style={"fontSize": "0.8rem", "marginBottom": "3px"}))

                messages_ui.append(html.Div(formatted, style={
                    "padding": "10px 14px", "margin": "6px 0", "borderRadius": "8px",
                    "maxWidth": "85%", "background": "#1a2d42", "color": "#b8d0e8",
                    "borderLeft": "3px solid #D4A843", "lineHeight": "1.4",
                }))

        return messages_ui, "", "", chat_history


def _get_region_data(pipeline_data: dict, region: str) -> tuple:
    if region == "all":
        m, s, c, p, n = _merge_all_regions_data(pipeline_data)
        return m, s, c, p, {}, n

    data = pipeline_data.get(region, {})
    return (
        data.get("merged_df", pd.DataFrame()).copy(),
        data.get("scored_df", pd.DataFrame()).copy(),
        data.get("corr_df", pd.DataFrame()),
        data.get("prediction", {}),
        data.get("llm_forecast", {}),
        data.get("market_name", "S&P 500"),
    )


def _merge_all_regions_data(pipeline_data: dict) -> tuple:
    all_merged = []
    all_scored = []

    for key, data in pipeline_data.items():
        m = data.get("merged_df", pd.DataFrame())
        s = data.get("scored_df", pd.DataFrame())
        if not m.empty:
            m = m.copy()
            m["region"] = key
            all_merged.append(m)
        if not s.empty:
            s = s.copy()
            s["region"] = key
            all_scored.append(s)

    merged = pd.concat(all_merged, ignore_index=True) if all_merged else pd.DataFrame()
    scored = pd.concat(all_scored, ignore_index=True) if all_scored else pd.DataFrame()

    return merged, scored, pd.DataFrame(), {}, "Global Markets"


def _merge_all_regions(pipeline_data: dict) -> dict:
    merged, scored, _, _, _ = _merge_all_regions_data(pipeline_data)
    return {"merged_df": merged, "scored_df": scored, "corr_df": pd.DataFrame(), "prediction": {}}


def _contains_sector(val, target):
    if isinstance(val, list):
        return target in val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return target in parsed
        except (ValueError, SyntaxError):
            pass
        return target in val
    return False


def _extract_sector_options(scored_df: pd.DataFrame) -> list[dict]:
    if scored_df.empty or "sectors" not in scored_df.columns:
        return []

    all_sectors = []
    for val in scored_df["sectors"].dropna():
        if isinstance(val, list):
            all_sectors.extend(val)
        elif isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    all_sectors.extend(parsed)
            except (ValueError, SyntaxError):
                all_sectors.extend(s.strip() for s in val.split(",") if s.strip())

    counts = Counter(s for s in all_sectors if s)
    valid = {s for s, c in counts.items() if c >= 2}
    return [{"label": f"{s} ({counts[s]})", "value": s} for s in sorted(valid)]


def _build_dual_forecast(prediction: dict, llm_forecast: dict) -> list:
    cards = []

    if prediction:
        d = prediction.get("direction", "neutral")
        arrow = "↑" if d == "bullish" else "↓"
        color = "#22c55e" if d == "bullish" else "#ef4444"
        cards.append(html.Div([
            html.Div("XGBoost", style={"color": "#94a3b8", "fontSize": "0.75rem"}),
            html.Div(f"{d.capitalize()} {arrow}", style={"color": color, "fontSize": "1.4rem", "fontWeight": "700"}),
            html.Div(f"Conf: {prediction.get('confidence', 0):.0%} · Acc: {prediction.get('accuracy', 0):.0%}",
                     style={"color": "#64748b", "fontSize": "0.7rem"}),
        ], style={"flex": "1", "textAlign": "center"}))
    else:
        cards.append(html.Div([
            html.Div("XGBoost", style={"color": "#94a3b8", "fontSize": "0.75rem"}),
            html.Div("—", style={"color": "#64748b", "fontSize": "1.2rem"}),
        ], style={"flex": "1", "textAlign": "center"}))

    cards.append(html.Div(style={"width": "1px", "background": "#334155", "margin": "0 8px", "alignSelf": "stretch"}))

    if llm_forecast:
        d = llm_forecast.get("direction", "neutral")
        arrow = "↑" if d == "bullish" else "↓"
        color = "#22c55e" if d == "bullish" else "#ef4444"
        reasoning = llm_forecast.get("reasoning", "")
        cards.append(html.Div([
            html.Div("LLM Analysis", style={"color": "#94a3b8", "fontSize": "0.75rem"}),
            html.Div(f"{d.capitalize()} {arrow}", style={"color": color, "fontSize": "1.4rem", "fontWeight": "700"}),
            html.Div(f"Conf: {llm_forecast.get('confidence', 0):.0%}", style={"color": "#64748b", "fontSize": "0.7rem"}),
            html.Div(reasoning, style={"color": "#94a3b8", "fontSize": "0.65rem", "marginTop": "4px", "fontStyle": "italic"})
            if reasoning else None,
        ], style={"flex": "1", "textAlign": "center"}))
    else:
        cards.append(html.Div([
            html.Div("LLM Analysis", style={"color": "#94a3b8", "fontSize": "0.75rem"}),
            html.Div("—", style={"color": "#64748b", "fontSize": "1.2rem"}),
        ], style={"flex": "1", "textAlign": "center"}))

    return [html.Div([
        html.Div("Next Day Forecast", style={"color": "#f1f5f9", "fontSize": "0.85rem", "fontWeight": "600", "marginBottom": "8px", "textAlign": "center"}),
        html.Div(cards, style={"display": "flex", "alignItems": "flex-start"}),
    ])]


def _build_prediction_card(prediction: dict) -> list:
    if not prediction:
        return [
            html.Div("No prediction available", style={"color": "#94a3b8", "fontSize": "1.1rem"}),
            html.Div("Run pipeline to generate", className="meta"),
        ]

    direction = prediction.get("direction", "neutral")
    arrow = "↑" if direction == "bullish" else "↓"
    color = "#22c55e" if direction == "bullish" else "#ef4444"
    conf = prediction.get("confidence", 0)
    acc = prediction.get("accuracy", 0)

    return [
        html.Div("Next Day Forecast", style={"color": "#94a3b8", "fontSize": "0.85rem"}),
        html.Div(f"{direction.capitalize()} {arrow}", className="direction", style={"color": color}),
        html.Div(f"Confidence: {conf:.0%}  |  Model Accuracy: {acc:.0%}", className="meta"),
    ]
