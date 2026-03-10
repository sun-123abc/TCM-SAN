# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

# ===============================
# 0. 基础配置
# ===============================
train_path = "train.xlsx"
test_path = "test.xlsx"
target_col = "证型"

# ===============================
# 1. 中文证型映射（固定 0–7）
# ===============================
class_names = [
    "气阴两虚证",
    "痰瘀互结证",
    "肺脾气虚证",
    "痰热阻肺证",
    "阴虚内热证",
    "痰湿蕴肺证",
    "脾虚痰湿证",
    "气滞血瘀证"
]
num_classes = len(class_names)

# ===============================
# 2. 读取数据
# ===============================
train_data = pd.read_excel(train_path)
test_data = pd.read_excel(test_path)

for df in [train_data, test_data]:
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.replace(r'[\u200b\u200c\u200d\uFEFF]', '', regex=True)
        .astype(int)
    )

X_train = train_data.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0)
y_train = train_data[target_col].astype(int)

X_test = test_data.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0)
y_test = test_data[target_col].astype(int)

# ===============================
# 3. 删除几乎全零特征
# ===============================
nonzero_cols = X_train.columns[X_train.sum(axis=0) > 1]
X_train = X_train[nonzero_cols]
X_test = X_test[nonzero_cols]

print(f"特征数：{len(nonzero_cols)}")

# ===============================
# 4. 特征标准化（SVM 必须）
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. SVM 模型
# ===============================
svm = SVC(
    C=10,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm.fit(X_train_scaled, y_train)

# ===============================
# 6. 预测
# ===============================
y_pred = svm.predict(X_test_scaled)
y_proba = svm.predict_proba(X_test_scaled)

# ===============================
# 7. 模型评估
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

y_test_bin = label_binarize(y_test, classes=range(num_classes))
auc_macro = roc_auc_score(
    y_test_bin, y_proba, average="macro", multi_class="ovr"
)

print("\n=== SVM 模型评估 ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC(macro): {auc_macro:.4f}")

# ===============================
# 8. 混淆矩阵（✅ 重点修复）
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("混淆矩阵-SVM", fontsize=18)
plt.xlabel("预测证型", fontsize=14)
plt.ylabel("真实证型", fontsize=14)
plt.xticks(range(num_classes), class_names, rotation=45, fontsize=12)
plt.yticks(range(num_classes), class_names, fontsize=12)

# 👉 核心：数值标注（你之前缺的）
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            fontsize=12,
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.colorbar(label="样本数")
plt.tight_layout()
plt.show()

# ===============================
# 9. ROC 曲线（多分类）
# ===============================
plt.figure(figsize=(8, 6))
for i, label in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测")
plt.title("ROC 曲线-SVM", fontsize=18)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(fontsize=9, loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== 运行结束：SVM ===")
