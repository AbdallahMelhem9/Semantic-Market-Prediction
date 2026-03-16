from dash import html, dcc
import dash_bootstrap_components as dbc


def create_layout() -> dbc.Container:
    return dbc.Container([
        # Header
        html.Div([
            html.Div([
                html.H1("Semantic Market Prediction"),
                html.Div("Sentiment Alpha Discovery Platform", className="subtitle"),
            ]),
        ], className="main-header"),

        # Region toggle
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("US", id="btn-region-us", color="primary", outline=False, size="sm", className="me-1"),
                dbc.Button("Europe", id="btn-region-europe", color="secondary", outline=True, size="sm"),
            ]),
            html.Div(dbc.Button(id="btn-region-all", style={"display": "none"})),
        ], className="region-toggle"),

        dcc.Store(id="store-active-region", data="us"),

        # Tabs
        dbc.Tabs([
            dbc.Tab(label="Dashboard", tab_id="tab-dashboard", children=[
                dbc.Row([
                    # Sidebar
                    dbc.Col([
                        html.Div([
                            html.Label("Time Window"),
                            dcc.Dropdown(
                                id="filter-days",
                                options=[
                                    {"label": "Past few days", "value": 8},
                                    {"label": "3 months", "value": 90},
                                ],
                                value=8,
                                clearable=False,
                            ),
                        ], className="sidebar-section"),

                        html.Div([
                            html.Label("Sector"),
                            dcc.Dropdown(
                                id="filter-sector",
                                options=[],
                                value=None,
                                placeholder="All",
                            ),
                        ], className="sidebar-section"),

                        html.Div([
                            html.Label("LLM Model"),
                            dcc.Dropdown(
                                id="filter-llm",
                                options=[
                                    {"label": "GPT-4o", "value": "openai/gpt-4o"},
                                    {"label": "Claude Sonnet", "value": "anthropic/claude-sonnet-4.6"},
                                    {"label": "Llama 3.3 70B", "value": "meta-llama/llama-3.3-70b-instruct"},
                                    {"label": "DeepSeek R1", "value": "deepseek/deepseek-r1"},
                                    {"label": "Ollama (local)", "value": "ollama"},
                                ],
                                value="openai/gpt-4o",
                                clearable=False,
                            ),
                        ], className="sidebar-section"),

                        dbc.Button(
                            "Re-score", id="btn-rescore",
                            color="warning", outline=True,
                            className="w-100", size="sm",
                        ),
                        dbc.Spinner(html.Div(id="rescore-status"), color="warning", size="sm"),

                    ], width=2, className="sidebar"),

                    # Main content
                    dbc.Col([
                        # Top row: gauge + prediction + correlation
                        dbc.Row([
                            dbc.Col([
                                html.Div(dcc.Graph(id="fear-gauge", config={"displayModeBar": False}), className="chart-card"),
                            ], width=4),
                            dbc.Col([
                                html.Div(id="prediction-card", className="prediction-card"),
                            ], width=4),
                            dbc.Col([
                                html.Div(dcc.Graph(id="correlation-heatmap", config={"displayModeBar": False}), className="chart-card"),
                            ], width=4),
                        ], className="mb-2"),

                        # Main chart
                        html.Div(
                            dcc.Graph(id="main-chart", config={"displayModeBar": True, "scrollZoom": True}),
                            className="chart-card",
                        ),

                        # Sector row
                        dbc.Row([
                            dbc.Col([
                                html.Div(dcc.Graph(id="sector-heatmap"), className="chart-card"),
                            ], width=6),
                            dbc.Col([
                                html.Div(dcc.Graph(id="sector-timeseries"), className="chart-card"),
                            ], width=6),
                        ]),

                        # Sector vs ETF
                        html.Div(dcc.Graph(id="sector-vs-etf"), className="chart-card"),

                        # Stock mentions
                        html.Div(dcc.Graph(id="stock-mentions"), className="chart-card"),

                        # Chat
                        html.Div([
                            html.H5("Chat", style={"color": "#e2e8f0"}),
                            html.Div(id="chat-messages", className="chat-messages"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="chat-input",
                                    placeholder="Ask about sentiment, sectors, predictions...",
                                    type="text",
                                    style={"background": "#1e3448", "border": "1px solid #2a4a6a", "color": "#d4dce8"},
                                ),
                                dbc.Button("Send", id="btn-chat", color="primary", size="sm"),
                            ], size="sm"),
                            dbc.Spinner(html.Div(id="chat-loading"), color="primary", size="sm"),
                        ], className="chat-container"),

                        # Disclaimer
                        html.Div(
                            "For research and demonstration purposes only. Not investment advice.",
                            className="disclaimer",
                        ),

                    ], width=10, style={"padding": "12px 16px"}),
                ], className="g-0"),
            ]),

            dbc.Tab(label="Articles", tab_id="tab-articles", children=[
                html.Div(id="article-browser-container", style={"padding": "16px 24px"}),
            ]),
        ], id="tabs", active_tab="tab-dashboard"),

        # Hidden stores
        dcc.Store(id="store-chat-history", data=[]),
        dcc.Store(id="store-rescore-trigger", data=0),

    ], fluid=True, style={"background": "#0d1b2a", "minHeight": "100vh", "padding": "0"})
