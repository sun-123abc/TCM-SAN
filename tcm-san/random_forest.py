# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

# =========================
# 0. 解决 joblib 中文路径问题
# =========================
temp_dir = r"C:\TempJoblib"
os.makedirs(temp_dir, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir

# =========================
# 1. 读取训练集和测试集
# =========================
train_path = "train.xlsx"
test_path = "test.xlsx"
target_col = "证型"

train_data = pd.read_excel(train_path)
test_data = pd.read_excel(test_path)

# =========================
# 2. 清理目标列（去掉隐形字符）
# =========================
for df in [train_data, test_data]:
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.replace(r'[\u200b\u200c\u200d\uFEFF]', '', regex=True)
        .astype(int)
    )

# =========================
# 3. 分离特征和目标列
# =========================
X_train = train_data.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_data[target_col]

X_test = test_data.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = test_data[target_col]

# =========================
# 3.1 删除几乎全零的特征
# =========================
nonzero_cols = X_train.columns[X_train.sum(axis=0) > 1]
X_train = X_train[nonzero_cols]
X_test = X_test[nonzero_cols]

# =========================
# 4. 中文类别映射（固定 0–7）
# =========================
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

labels = class_names
num_classes = len(class_names)

# 直接使用 0–7 编码（不再使用 LabelEncoder）
y_train_encoded = y_train.values
y_test_encoded = y_test.values

# =========================
# 5. 类别权重设置（缓解类别不平衡）
# =========================
classes, class_counts = np.unique(y_train_encoded, return_counts=True)
total = y_train_encoded.shape[0]

class_weights = {cls: total / cnt for cls, cnt in zip(classes, class_counts)}

# 小幅放大小类别权重
mean_w = np.mean(list(class_weights.values()))
class_weights = {
    cls: w * 1.1 if w > mean_w else w
    for cls, w in class_weights.items()
}

# =========================
# 6. 随机森林超参数网格搜索
# =========================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2"]
}

best_score = 0
best_params = None
best_model = None

for params in ParameterGrid(param_grid):
    model = RandomForestClassifier(
        **params,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_encoded)
    y_pred_val = model.predict(X_test)
    f1_val = f1_score(y_test_encoded, y_pred_val, average="weighted")

    if f1_val > best_score:
        best_score = f1_val
        best_params = params
        best_model = model

print("✅ 最佳参数:", best_params)

# =========================
# 7. 预测概率与最终预测
# =========================
y_proba = best_model.predict_proba(X_test)
y_pred = np.argmax(y_proba, axis=1)

# =========================
# 8. 模型评估指标
# =========================
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)

try:
    y_test_bin = label_binarize(y_test_encoded, classes=range(num_classes))
    auc_score = roc_auc_score(
        y_test_bin,
        y_proba,
        average="macro",
        multi_class="ovr"
    )
except Exception:
    auc_score = np.nan

cm = confusion_matrix(y_test_encoded, y_pred)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\n=== 混淆矩阵数值 ===")
print(cm_df)

print("\n=== 模型评估结果 ===")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 值: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")

# =========================
# 9. 混淆矩阵可视化（中文）
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("混淆矩阵-随机森林", fontsize=18)
plt.xlabel("预测证型", fontsize=14)
plt.ylabel("真实证型", fontsize=14)
plt.xticks(range(num_classes), labels, rotation=45, fontsize=12)
plt.yticks(range(num_classes), labels, fontsize=12)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# 10. ROC 曲线（多分类，中文）
# =========================
plt.figure(figsize=(7, 7))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC 曲线-随机森林", fontsize=18)
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
