"""
Semantic Market Prediction — Entry Point
Usage: python run.py
"""

import logging

from src.config.settings import load_settings
from src.config.logging_setup import setup_logging
from src.pipeline import run_pipeline
from src.dashboard.app import run_dashboard


def main() -> None:
    settings = load_settings()
    setup_logging(settings)
    logger = logging.getLogger(__name__)

    logger.info("Semantic Market Prediction — Starting pipeline...")
    pipeline_data = run_pipeline(settings)

    import os
    host = os.getenv("HOST", settings.server.host)
    port = int(os.getenv("PORT", settings.server.port))
    debug = os.getenv("DEBUG", "true").lower() == "true" if os.getenv("DEBUG") else settings.server.debug

    logger.info(f"Launching dashboard at http://{host}:{port}")
    run_dashboard(pipeline_data, host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
