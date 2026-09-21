import numpy as np
from typing import List, Dict

class ClinicalExplainer:
    """محاسبه سهم هر متغیر بالینی در تصمیم‌گیری مدل (Local & Global Feature Attribution)"""
    def __init__(self, predictor):
        self.predictor = predictor

    def explain_global_importance(self) -> List[Dict[str, float]]:
        """اهمیت سراسری ویژگی‌ها در تمام جمعیت بیماران"""
        importances = self.predictor.model.feature_importances_
        names = self.predictor.feature_names
        ranked_indices = np.argsort(importances)[::-1]
        
        return [{"feature": names[i], "importance": float(importances[i])} for i in ranked_indices]

    def explain_instance(self, patient_sample: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
        """تفسیر علت تصمیم‌گیری برای یک بیمار خاص نسبت به میانگین جمعیت"""
        base_pred = self.predictor.predict_proba(baseline.reshape(1, -1))[0]
        instance_pred = self.predictor.predict_proba(patient_sample.reshape(1, -1))[0]
        
        attributions = {}
        for idx, feat in enumerate(self.predictor.feature_names):
            perturbed = patient_sample.copy()
            perturbed[idx] = baseline[idx]
            perturbed_pred = self.predictor.predict_proba(perturbed.reshape(1, -1))[0]
            # تغییر در ریسک ناشی از این فاکتور خاص
            attributions[feat] = float(instance_pred - perturbed_pred)

        return attributions
