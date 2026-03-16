import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SentimentPredictor:

    def __init__(self, settings: Settings) -> None:
        self.test_split = settings.prediction.test_split
        self.model_path = Path(settings.paths.models) / "xgboost_model.json"
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """Train XGBoost with time-based split. Returns metrics dict."""
        if len(features) < 10:
            logger.warning(f"Only {len(features)} samples — too few for reliable training")

        n = len(features)
        split_idx = int(n * (1 - self.test_split))

        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        logger.info(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

        self.feature_names = list(features.columns)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        from src.prediction.evaluator import evaluate_model
        self.metrics = evaluate_model(self.model, X_test, y_test)

        # Save model
        self._save_model()

        logger.info(f"Model trained — accuracy: {self.metrics.get('accuracy', 0):.2%}")
        return self.metrics

    def predict_next_day(self, latest_features: pd.DataFrame) -> dict:
        """Predict next trading day direction from latest feature row."""
        if self.model is None:
            self.model = self._load_model()
            if self.model is None:
                return {"direction": "unknown", "confidence": 0, "accuracy": 0}

        proba = self.model.predict_proba(latest_features)
        prediction = self.model.predict(latest_features)[0]

        direction = "bullish" if prediction == 1 else "bearish"
        confidence = float(proba[0].max())

        return {
            "direction": direction,
            "confidence": confidence,
            "accuracy": self.metrics.get("accuracy", 0),
            "feature_names": self.feature_names,
            "feature_importances": self.get_feature_importances(),
        }

    def get_feature_importances(self) -> list[float]:
        if self.model is None:
            return []
        return self.model.feature_importances_.tolist()

    def _save_model(self) -> None:
        if self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_path))

        # Save metadata alongside
        meta_path = self.model_path.with_suffix(".meta.json")
        meta = {
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self) -> XGBClassifier | None:
        if not self.model_path.exists():
            logger.warning("No saved model found")
            return None

        model = XGBClassifier()
        model.load_model(str(self.model_path))

        meta_path = self.model_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.metrics = meta.get("metrics", {})

        logger.info("Loaded saved model")
        return model
