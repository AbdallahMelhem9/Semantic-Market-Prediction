import ast

from dash import html
import pandas as pd


def create_article_browser(scored_df: pd.DataFrame) -> html.Div:
    if scored_df.empty:
        return html.Div("No articles loaded", style={"color": "#7a8fa3", "textAlign": "center", "padding": "40px"})

    cards = []
    df = scored_df.sort_values("recession_fear", ascending=False).reset_index(drop=True)

    for _, row in df.iterrows():
        fear = float(row.get("recession_fear", 5))
        title = str(row.get("title", ""))[:140]
        source = str(row.get("source", "Unknown"))
        sentiment = str(row.get("market_sentiment", "neutral"))
        rationale = str(row.get("rationale", ""))
        d = str(row.get("date", ""))
        confidence = str(row.get("confidence", "")).capitalize()
        sectors = row.get("sectors", [])

        if isinstance(sectors, str):
            try:
                sectors = ast.literal_eval(sectors)
            except (ValueError, SyntaxError):
                sectors = [s.strip() for s in sectors.split(",") if s.strip()]
        if not isinstance(sectors, list):
            sectors = []

        # Left border color by fear level
        if fear >= 7:
            border_color = "#C62828"
        elif fear >= 4:
            border_color = "#D4A843"
        else:
            border_color = "#2E7D32"

        # Sentiment pill
        sent_colors = {
            "bearish": ("#C62828", "#FDECEA"),
            "bullish": ("#2E7D32", "#E8F5E9"),
            "neutral": ("#B8860B", "#FFF8E1"),
        }
        sent_fg, sent_bg = sent_colors.get(sentiment, ("#666", "#eee"))

        # Fear pill
        if fear < 4:
            fear_fg, fear_bg = "#2E7D32", "#E8F5E9"
        elif fear < 7:
            fear_fg, fear_bg = "#B8860B", "#FFF8E1"
        else:
            fear_fg, fear_bg = "#C62828", "#FDECEA"

        # Sector tags
        sector_tags = [
            html.Span(s, style={
                "background": "#E8DCC8", "color": "#5D4E37", "padding": "1px 7px",
                "borderRadius": "3px", "fontSize": "0.62rem", "marginRight": "4px",
                "display": "inline-block", "marginTop": "3px", "fontWeight": "500",
            }) for s in sectors[:4]
        ]

        card = html.Div([
            # Source + date (always visible)
            html.Div([
                html.Span(source.upper(), style={
                    "color": "#8B7355", "fontSize": "0.62rem", "fontWeight": "700",
                    "letterSpacing": "0.08em",
                }),
                html.Span(f"  ·  {d}", style={"color": "#A89880", "fontSize": "0.62rem"}),
            ], style={"marginBottom": "5px"}),

            # Title (always visible, serif font for newspaper feel)
            html.Div(title, style={
                "color": "#2C1810", "fontSize": "0.9rem", "fontWeight": "600",
                "lineHeight": "1.3", "marginBottom": "4px",
                "fontFamily": "'Georgia', 'Times New Roman', serif",
            }),

            # Hover details
            html.Div([
                html.Div([
                    html.Span(f"{fear:.1f}", style={
                        "background": fear_bg, "color": fear_fg,
                        "padding": "2px 8px", "borderRadius": "3px",
                        "fontSize": "0.68rem", "fontWeight": "700", "marginRight": "6px",
                    }),
                    html.Span(sentiment.capitalize(), style={
                        "background": sent_bg, "color": sent_fg,
                        "padding": "2px 8px", "borderRadius": "3px",
                        "fontSize": "0.68rem", "fontWeight": "500", "marginRight": "6px",
                    }),
                    html.Span(confidence, style={"color": "#8B7355", "fontSize": "0.62rem"}),
                ], style={"marginBottom": "4px"}),

                html.Div(sector_tags) if sector_tags else None,

                html.Div(rationale, style={
                    "color": "#6B5B4D", "fontSize": "0.7rem",
                    "marginTop": "5px", "fontStyle": "italic", "lineHeight": "1.35",
                    "borderTop": "1px solid #E8DCC8", "paddingTop": "5px",
                }) if rationale and "default" not in rationale.lower() else None,
            ], className="article-detail", style={
                "maxHeight": "0", "overflow": "hidden", "transition": "max-height 0.3s ease",
            }),

        ], className="article-card", style={
            "background": "#FDF6EC",
            "border": "1px solid #E8DCC8",
            "borderLeft": f"3px solid {border_color}",
            "borderRadius": "6px",
            "padding": "12px 14px",
            "marginBottom": "8px",
            "cursor": "pointer",
        })

        cards.append(card)

    return html.Div([
        html.Div([
            html.H5("Articles", style={
                "color": "#D4A843", "margin": "0", "fontWeight": "600", "fontSize": "1rem",
            }),
            html.Span(f"{len(df)} articles", style={"color": "#7a8fa3", "fontSize": "0.75rem"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px"}),
        html.Div(cards, style={
            "maxHeight": "700px", "overflowY": "auto", "paddingRight": "6px",
        }),
    ], style={"padding": "16px"})
