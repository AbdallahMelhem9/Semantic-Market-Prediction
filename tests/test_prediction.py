import numpy as np
import pandas as pd
from datetime import date


def _make_daily_df(n=25):
    """Synthetic daily sentiment + market data."""
    np.random.seed(42)
    from datetime import timedelta
    base = date(2026, 1, 5)
    dates = [base + timedelta(days=i) for i in range(n)]
    fear = np.cumsum(np.random.randn(n) * 0.3) + 5
    returns = np.random.randn(n) * 0.01

    return pd.DataFrame({
        "date": dates,
        "avg_recession_fear": fear,
        "rolling_3d": pd.Series(fear).rolling(3, min_periods=1).mean(),
        "rolling_7d": pd.Series(fear).rolling(7, min_periods=1).mean(),
        "volatility": pd.Series(fear).rolling(7, min_periods=2).std(),
        "article_count": np.random.randint(5, 20, n),
        "momentum": pd.Series(fear).diff(),
        "avg_sentiment_numeric": np.random.uniform(-0.5, 0.5, n),
        "sp500_return": returns,
        "sp500_direction": (returns > 0).astype(int),
    })


# --- Feature engineering ---

def test_engineer_features_returns_tuple():
    from src.prediction.feature_engineer import engineer_features
    df = _make_daily_df()
    features, target = engineer_features(df)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert len(features) == len(target)


def test_engineer_features_has_expected_columns():
    from src.prediction.feature_engineer import engineer_features
    df = _make_daily_df()
    features, _ = engineer_features(df)
    assert "avg_recession_fear" in features.columns
    assert "rolling_3d" in features.columns
    assert "momentum" in features.columns


def test_engineer_features_no_nans():
    from src.prediction.feature_engineer import engineer_features
    df = _make_daily_df()
    features, target = engineer_features(df)
    assert not features.isna().any().any()
    assert not target.isna().any()


def test_engineer_features_target_is_binary():
    from src.prediction.feature_engineer import engineer_features
    df = _make_daily_df()
    _, target = engineer_features(df)
    assert set(target.unique()).issubset({0, 1})


def test_engineer_features_empty():
    from src.prediction.feature_engineer import engineer_features
    features, target = engineer_features(pd.DataFrame())
    assert len(features) == 0


# --- Model training ---

def test_model_train_returns_metrics():
    from src.config.settings import load_settings
    from src.prediction.feature_engineer import engineer_features
    from src.prediction.model import SentimentPredictor

    settings = load_settings()
    df = _make_daily_df(25)
    features, target = engineer_features(df)

    predictor = SentimentPredictor(settings)
    metrics = predictor.train(features, target)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_model_predict_next_day():
    from src.config.settings import load_settings
    from src.prediction.feature_engineer import engineer_features
    from src.prediction.model import SentimentPredictor

    settings = load_settings()
    df = _make_daily_df(25)
    features, target = engineer_features(df)

    predictor = SentimentPredictor(settings)
    predictor.train(features, target)

    # Use last row as "latest"
    latest = features.iloc[[-1]]
    prediction = predictor.predict_next_day(latest)

    assert prediction["direction"] in ("bullish", "bearish")
    assert 0 < prediction["confidence"] <= 1
    assert isinstance(prediction["feature_names"], list)
    assert isinstance(prediction["feature_importances"], list)


def test_model_feature_importances():
    from src.config.settings import load_settings
    from src.prediction.feature_engineer import engineer_features
    from src.prediction.model import SentimentPredictor

    settings = load_settings()
    df = _make_daily_df(25)
    features, target = engineer_features(df)

    predictor = SentimentPredictor(settings)
    predictor.train(features, target)

    importances = predictor.get_feature_importances()
    assert len(importances) == len(features.columns)
    assert all(i >= 0 for i in importances)


# --- Evaluator ---

def test_evaluator_returns_all_metrics():
    from src.prediction.evaluator import evaluate_model
    from xgboost import XGBClassifier

    np.random.seed(42)
    X = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
    y = pd.Series(np.random.randint(0, 2, 20))

    model = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["confusion_matrix"], list)
