import dash
import dash_bootstrap_components as dbc

from src.dashboard.layout import create_layout
from src.dashboard.callbacks import register_callbacks
from src.dashboard.theme import CUSTOM_CSS


def create_app(pipeline_data: dict = None) -> dash.Dash:
    """Create and configure the Dash application."""
    if pipeline_data is None:
        pipeline_data = {}

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title="Semantic Market Prediction",
    )

    # Inject custom CSS
    app.index_string = app.index_string.replace(
        "</head>",
        f"<style>{CUSTOM_CSS}</style></head>",
    )

    app.layout = create_layout()
    register_callbacks(app, pipeline_data)

    return app


def run_dashboard(pipeline_data: dict, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    """Create app and run the server."""
    app = create_app(pipeline_data)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
