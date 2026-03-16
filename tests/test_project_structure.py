"""Tests to verify project structure is correctly initialized."""

from pathlib import Path


def get_root() -> Path:
    return Path(__file__).parent.parent


def test_source_packages_exist():
    """All source packages must exist with __init__.py."""
    root = get_root()
    packages = [
        "src",
        "src/config",
        "src/ingestion",
        "src/analysis",
        "src/timeseries",
        "src/prediction",
        "src/visualization",
        "src/chatbot",
        "src/dashboard",
        "src/dashboard/components",
    ]
    for pkg in packages:
        pkg_dir = root / pkg
        assert pkg_dir.exists(), f"Package directory missing: {pkg}"
        assert (pkg_dir / "__init__.py").exists(), f"__init__.py missing in: {pkg}"


def test_data_directories_exist():
    """All data directories must exist."""
    root = get_root()
    dirs = [
        "data/cache/news_raw",
        "data/cache/scored",
        "data/processed",
        "data/models",
    ]
    for d in dirs:
        assert (root / d).exists(), f"Data directory missing: {d}"


def test_config_files_exist():
    """Configuration files must exist."""
    root = get_root()
    assert (root / "config.yaml").exists(), "config.yaml missing"
    assert (root / "requirements.txt").exists(), "requirements.txt missing"
    assert (root / ".gitignore").exists(), ".gitignore missing"
    assert (root / ".env.example").exists(), ".env.example missing"
    assert (root / "run.py").exists(), "run.py missing"


def test_prompts_directory_exists():
    """Prompts directory must exist."""
    root = get_root()
    assert (root / "prompts").exists(), "prompts/ directory missing"


def test_requirements_contains_dependencies():
    """requirements.txt must list all required packages."""
    root = get_root()
    content = (root / "requirements.txt").read_text()
    required = [
        "dash", "dash-bootstrap-components", "plotly", "pandas",
        "numpy", "scikit-learn", "xgboost", "yfinance",
        "requests", "python-dotenv", "pyyaml",
    ]
    for dep in required:
        assert dep in content, f"Dependency missing from requirements.txt: {dep}"


def test_gitignore_excludes_secrets():
    """gitignore must exclude sensitive files."""
    root = get_root()
    content = (root / ".gitignore").read_text()
    assert ".env" in content, ".gitignore must exclude .env"
    assert "__pycache__" in content or ".eggs/" in content, ".gitignore must exclude Python artifacts"


def test_config_loads():
    """Config should load without errors."""
    from src.config.settings import load_config
    config = load_config()
    assert "news" in config
    assert "llm" in config
    assert "visualization" in config
    assert "prediction" in config
    assert "secrets" in config
    assert "paths" in config
