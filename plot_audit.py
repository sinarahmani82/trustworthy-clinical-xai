import numpy as np
import matplotlib.pyplot as plt

# ۱. داده‌های اهمیت ویژگی‌ها
features = ["BMI", "Age", "Glucose", "BloodPressure", "Cholesterol"]
importances = [0.32, 0.28, 0.18, 0.14, 0.08]

# ۲. داده‌های ممیزی عدالت (مقایسه دو گروه)
metrics = ["Selection Rate\n(Demographic Parity)", "True Positive Rate\n(Sensitivity)", "False Positive Rate\n(False Alarm)"]
group_female = [0.31, 0.88, 0.09]
group_male = [0.33, 0.89, 0.08]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# نمودار ۱: Feature Attribution
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
y_pos = np.arange(len(features))
ax1.barh(y_pos, importances, color=colors, edgecolor='black', alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features, fontsize=11, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlabel('Relative Attribution Score', fontsize=11, fontweight='bold')
ax1.set_title('A. Global Clinical Feature Attribution', fontsize=13, fontweight='bold', pad=12)

for i, v in enumerate(importances):
    ax1.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontweight='bold')

# نمودار ۲: Fairness Audit across Subgroups
x = np.arange(len(metrics))
width = 0.35

rects1 = ax2.bar(x - width/2, group_female, width, label='Female Cohort', color='#3498db', edgecolor='black', alpha=0.9)
rects2 = ax2.bar(x + width/2, group_male, width, label='Male Cohort', color='#e74c3c', edgecolor='black', alpha=0.9)

ax2.set_ylabel('Rate (0.0 to 1.0)', fontsize=11, fontweight='bold')
ax2.set_title('B. Algorithmic Fairness Audit (EU AI Act)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=9.5, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(frameon=True, loc='upper right')

# خط آستانه تلورانس استاندارد
ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
output_filename = "xai_fairness_audit.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Audit plot successfully saved as: {output_filename}")
