import ast
import re

import plotly.graph_objects as go
import pandas as pd


# Well-known companies/tickers to look for in titles
KNOWN_STOCKS = {
    "apple": ("AAPL", "Technology"),
    "microsoft": ("MSFT", "Technology"),
    "google": ("GOOGL", "Technology"),
    "alphabet": ("GOOGL", "Technology"),
    "amazon": ("AMZN", "Technology"),
    "meta": ("META", "Technology"),
    "nvidia": ("NVDA", "Technology"),
    "tesla": ("TSLA", "Automotive"),
    "jpmorgan": ("JPM", "Finance"),
    "goldman": ("GS", "Finance"),
    "bank of america": ("BAC", "Finance"),
    "adobe": ("ADBE", "Technology"),
    "netflix": ("NFLX", "Technology"),
    "boeing": ("BA", "Industrial"),
    "lockheed": ("LMT", "Industrial"),
    "exxon": ("XOM", "Energy"),
    "chevron": ("CVX", "Energy"),
    "pfizer": ("PFE", "Healthcare"),
    "novo nordisk": ("NVO", "Healthcare"),
    "honda": ("HMC", "Automotive"),
    "lucid": ("LCID", "Automotive"),
    "ford": ("F", "Automotive"),
    "gm": ("GM", "Automotive"),
    "general motors": ("GM", "Automotive"),
    "walmart": ("WMT", "Consumer"),
    "costco": ("COST", "Consumer"),
    "atlassian": ("TEAM", "Technology"),
    "live nation": ("LYV", "Consumer"),
    "opec": ("OIL", "Energy"),
}


def extract_stock_mentions(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()

    mentions = []
    for _, row in scored_df.iterrows():
        title = str(row.get("title", "")).lower()
        fear = row.get("recession_fear", 5)
        sentiment = row.get("market_sentiment", "neutral")
        d = row.get("date", "")

        for keyword, (ticker, sector) in KNOWN_STOCKS.items():
            if keyword in title:
                mentions.append({
                    "company": keyword.title(),
                    "ticker": ticker,
                    "sector": sector,
                    "recession_fear": fear,
                    "sentiment": sentiment,
                    "date": d,
                    "headline": str(row.get("title", ""))[:80],
                })

    if not mentions:
        return pd.DataFrame()

    return pd.DataFrame(mentions)


def create_stock_mentions_chart(scored_df: pd.DataFrame) -> go.Figure:
    mentions_df = extract_stock_mentions(scored_df)

    if mentions_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No specific stock mentions found in articles",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_dark", height=350)
        return fig

    # Aggregate: avg fear per company
    agg = mentions_df.groupby(["company", "ticker", "sector"]).agg(
        avg_fear=("recession_fear", "mean"),
        mentions=("recession_fear", "count"),
        latest_sentiment=("sentiment", "last"),
    ).reset_index().sort_values("avg_fear", ascending=True)

    colors = []
    for _, row in agg.iterrows():
        f = row["avg_fear"]
        if f >= 6:
            colors.append("#ef4444")
        elif f >= 4:
            colors.append("#f97316")
        else:
            colors.append("#22c55e")

    fig = go.Figure(go.Bar(
        x=agg["avg_fear"],
        y=[f"{r['company']} ({r['ticker']})" for _, r in agg.iterrows()],
        orientation="h",
        marker_color=colors,
        text=[f"{r['avg_fear']:.1f} — {r['latest_sentiment']} ({r['mentions']} articles)"
              for _, r in agg.iterrows()],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text="Stock & Company Mentions", font=dict(size=13)),
        template="plotly_dark",
        height=max(250, len(agg) * 40 + 80),
        xaxis=dict(range=[0, 10], title="Recession Fear"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=100, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=10),
    )

    return fig
