import numpy as np
from tabulate import tabulate
from src.model import ClinicalPredictor
from src.explainer import ClinicalExplainer
from src.fairness import FairnessAuditor

def generate_synthetic_clinical_data(n_samples: int = 1000):
    np.random.seed(42)
    # ویژگی‌ها: سن، فشار خون، کلسترول، گلوکز ناشتا، شاخص توده بدنی (BMI)
    age = np.random.normal(55, 12, n_samples)
    bp = np.random.normal(130, 18, n_samples)
    chol = np.random.normal(210, 35, n_samples)
    glucose = np.random.normal(105, 25, n_samples)
    bmi = np.random.normal(27, 4, n_samples)
    gender = np.random.binomial(1, 0.5, n_samples)  # 0: Female, 1: Male

    # ریسک واقعی بیماری
    risk_score = 0.03 * age + 0.02 * bp + 0.015 * chol + 0.025 * glucose + 0.04 * bmi - 9.5
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(int)

    X = np.column_stack([age, bp, chol, glucose, bmi])
    feature_names = ["Age", "BloodPressure", "Cholesterol", "Glucose", "BMI"]
    return X, y, gender, feature_names

def main():
    print("=== Trustworthy Clinical Decision Support: Audit Pipeline ===\n")
    X, y, gender, features = generate_synthetic_clinical_data(1000)

    # آموزش مدل
    predictor = ClinicalPredictor()
    predictor.fit(X, y, features)
    preds = predictor.predict(X)

    # ۱. ارزیابی تفسیرپذیری سراسری (Global Interpretability)
    explainer = ClinicalExplainer(predictor)
    importance_data = explainer.explain_global_importance()
    print("--- 1. Global Clinical Biomarker Attribution ---")
    table_imp = [[item["feature"], f"{item['importance']*100:.2f}%"] for item in importance_data]
    print(tabulate(table_imp, headers=["Clinical Marker", "Relative Attribution"], tablefmt="grid"))

    # ۲. ممیزی عدم سوگیری و عدالت (Fairness Audit)
    auditor = FairnessAuditor(y_true=y, y_pred=preds, sensitive_attribute=gender)
    dp_diff = auditor.demographic_parity_difference()
    eq_odds = auditor.equalized_odds_difference()

    print("\n--- 2. Algorithmic Fairness Audit (EU AI Act Compliance) ---")
    fairness_results = [
        ["Demographic Parity Disparity", f"{dp_diff:.4f}", "PASS (<0.10)" if dp_diff < 0.10 else "FLAGGED"],
        ["True Positive Rate Disparity", f"{eq_odds['tpr_disparity']:.4f}", "PASS (<0.08)" if eq_odds['tpr_disparity'] < 0.08 else "FLAGGED"],
        ["False Positive Rate Disparity", f"{eq_odds['fpr_disparity']:.4f}", "PASS (<0.08)" if eq_odds['fpr_disparity'] < 0.08 else "FLAGGED"]
    ]
    print(tabulate(fairness_results, headers=["Audit Metric", "Observed Value", "Compliance Status"], tablefmt="grid"))

if __name__ == "__main__":
    main()
