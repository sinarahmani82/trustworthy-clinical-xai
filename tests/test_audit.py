import numpy as np
import pytest
from src.model import ClinicalPredictor
from src.explainer import ClinicalExplainer
from src.fairness import FairnessAuditor

def test_clinical_predictor_probabilities():
    X = np.random.randn(50, 4)
    y = np.random.binomial(1, 0.5, 50)
    features = ["F1", "F2", "F3", "F4"]

    predictor = ClinicalPredictor()
    predictor.fit(X, y, features)
    probs = predictor.predict_proba(X)

    assert len(probs) == 50
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be strictly bounded in [0, 1]"

def test_feature_attributions_sum():
    X = np.random.randn(50, 3)
    y = np.random.binomial(1, 0.5, 50)
    features = ["A", "B", "C"]

    predictor = ClinicalPredictor()
    predictor.fit(X, y, features)
    explainer = ClinicalExplainer(predictor)

    importances = explainer.explain_global_importance()
    total_importance = sum(item["importance"] for item in importances)
    assert 0.99 <= total_importance <= 1.01, "Feature importances should sum to ~1.0"

def test_fairness_demographic_parity():
    # ساخت دو گروه با نرخ‌های پیش‌بینی مشخص
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    group = np.array([0, 0, 1, 1])

    auditor = FairnessAuditor(y_true, y_pred, group)
    diff = auditor.demographic_parity_difference()
    # نرخ گروه ۰ برابر ۱.۰ و نرخ گروه ۱ برابر ۰.۰ است؛ اختلاف باید ۱.۰ باشد
    assert diff == 1.0
