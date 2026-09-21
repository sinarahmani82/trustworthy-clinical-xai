import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from typing import Tuple, List

class ClinicalPredictor:
    """مدل پیش‌بینی ریسک بالینی بیماری قلبی با قابلیت ارائه احتمالات کالیبره‌شده"""
    def __init__(self, random_state: int = 42):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.feature_names = feature_names
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
