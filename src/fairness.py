import numpy as np
from typing import Dict

class FairnessAuditor:
    """سنجش معیارهای عدالت الگوریتمی مطابق چارچوب‌های استاندارد اخلاق در AI"""
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attribute: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        self.group = sensitive_attribute  # مثلاً ۰ برای زنان و ۱ برای مردان

    def demographic_parity_difference(self) -> float:
        """بررسی برابری نرخ پذیرش بین دو گروه (Demographic Parity)"""
        rate_g0 = np.mean(self.y_pred[self.group == 0])
        rate_g1 = np.mean(self.y_pred[self.group == 1])
        return float(abs(rate_g0 - rate_g1))

    def equalized_odds_difference(self) -> Dict[str, float]:
        """بررسی برابری نرخ مثبت واقعی (TPR) و نرخ مثبت کاذب (FPR)"""
        # True Positive Rate
        tpr_g0 = np.mean(self.y_pred[(self.group == 0) & (self.y_true == 1)])
        tpr_g1 = np.mean(self.y_pred[(self.group == 1) & (self.y_true == 1)])
        tpr_diff = abs(tpr_g0 - tpr_g1)

        # False Positive Rate
        fpr_g0 = np.mean(self.y_pred[(self.group == 0) & (self.y_true == 0)])
        fpr_g1 = np.mean(self.y_pred[(self.group == 1) & (self.y_true == 0)])
        fpr_diff = abs(fpr_g0 - fpr_g1)

        return {
            "tpr_disparity": float(tpr_diff),
            "fpr_disparity": float(fpr_diff)
        }
