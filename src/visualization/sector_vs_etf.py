import ast
import logging
from datetime import date, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_ETFS_US = {
    "Technology": "XLK",
    "Finance": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer": "XLY",
    "Industrial": "XLI",
}

SECTOR_ETFS_EU = {
    "Technology": "EXV8.DE",
    "Finance": "EXV1.DE",
    "Energy": "EXH1.DE",
    "Healthcare": "EXV4.DE",
}


def create_sector_vs_etf_chart(scored_df: pd.DataFrame, region: str = "us") -> go.Figure:
    if scored_df.empty or "sectors" not in scored_df.columns:
        return _empty_figure("No sector data for ETF comparison")

    etf_map = SECTOR_ETFS_US if region == "us" else SECTOR_ETFS_EU

    # Parse sectors and compute daily sector fear
    df = scored_df.copy()
    df["sectors"] = df["sectors"].apply(_parse_sectors)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.explode("sectors").dropna(subset=["sectors"])

    # Only sectors that have ETF mappings
    df = df[df["sectors"].isin(etf_map.keys())]
    if df.empty:
        return _empty_figure("No sectors with ETF mappings found")

    sector_daily = df.pivot_table(values="recession_fear", index="date", columns="sectors", aggfunc="mean")

    # Fetch ETF data
    import yfinance as yf
    min_date = min(sector_daily.index)
    max_date = max(sector_daily.index)
    start = (min_date - timedelta(days=5)).isoformat()
    end = (max_date + timedelta(days=1)).isoformat()

    fig = make_subplots(
        rows=len(sector_daily.columns), cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{s} Fear vs {etf_map.get(s, '?')}" for s in sector_daily.columns],
        vertical_spacing=0.06,
    )

    row = 1
    for sector in sector_daily.columns:
        etf_symbol = etf_map.get(sector)
        if not etf_symbol:
            continue

        # Sector fear line (inverted)
        fig.add_trace(go.Scatter(
            x=sector_daily.index, y=sector_daily[sector],
            name=f"{sector} Fear", line=dict(color="#ef4444", width=2),
            mode="lines+markers", showlegend=(row == 1),
        ), row=row, col=1)

        # ETF price line (with retry for rate limits)
        import time as _time
        hist = pd.DataFrame()
        for _attempt in range(3):
            try:
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(start=start, end=end)
                if not hist.empty:
                    break
                _time.sleep(2)
            except Exception:
                _time.sleep(2)
        try:
            if not hist.empty:
                etf_df = hist[["Close"]].reset_index()
                etf_df.columns = ["date", "close"]
                etf_df["date"] = pd.to_datetime(etf_df["date"]).dt.date

                # Normalize to 0-10 scale to overlay with fear
                min_p, max_p = etf_df["close"].min(), etf_df["close"].max()
                if max_p > min_p:
                    etf_df["normalized"] = 10 - (etf_df["close"] - min_p) / (max_p - min_p) * 10
                else:
                    etf_df["normalized"] = 5

                fig.add_trace(go.Scatter(
                    x=etf_df["date"], y=etf_df["normalized"],
                    name=f"{etf_symbol}", line=dict(color="#3b82f6", width=2, dash="dash"),
                    mode="lines", showlegend=(row == 1),
                ), row=row, col=1)
        except Exception as e:
            logger.warning(f"Failed to fetch {etf_symbol}: {e}")

        fig.update_yaxes(range=[0, 10], row=row, col=1)
        row += 1

    fig.update_layout(
        title=dict(text="Sector Fear vs Related ETF", font=dict(size=13)),
        template="plotly_dark",
        height=220 * len(sector_daily.columns) + 60,
        legend=dict(orientation="h", y=1.02, font=dict(size=9)),
        margin=dict(l=50, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=10),
    )

    return fig


def _parse_sectors(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [val]
        except (ValueError, SyntaxError):
            return [s.strip() for s in val.split(",") if s.strip()]
    return []


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_dark", height=300)
    return fig
