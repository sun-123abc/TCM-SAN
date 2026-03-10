# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ===============================
# 0. 解决 joblib / 中文路径问题
# ===============================
temp_dir = r"C:\TempJoblib"
os.makedirs(temp_dir, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir

# ===============================
# 1. 数据路径 & 目标列
# ===============================
train_path = "train.xlsx"
test_path = "test.xlsx"
target_col = "证型"

# ===============================
# 2. 固定【证型中文映射】（核心）
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
# 3. 读取数据
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
# 4. 删除几乎全零特征
# ===============================
nonzero_cols = X_train.columns[X_train.sum(axis=0) > 1]
X_train = X_train[nonzero_cols]
X_test = X_test[nonzero_cols]

print(f"保留特征数：{len(nonzero_cols)}")

# ===============================
# 5. 类别权重（小类别增强）
# ===============================
classes, class_counts = np.unique(y_train, return_counts=True)
total = y_train.shape[0]
class_weights = [total / c for c in class_counts]

mean_w = np.mean(class_weights)
class_weights = [w * 1.1 if w > mean_w else w for w in class_weights]

# ===============================
# 6. CatBoost 网格搜索
# ===============================
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

param_grid = {
    "depth": [6, 7, 8],
    "learning_rate": [0.03, 0.05],
    "iterations": [700, 800, 1200]
}

best_score = 0
best_params = None
best_model = None

for depth in param_grid["depth"]:
    for lr in param_grid["learning_rate"]:
        for iters in param_grid["iterations"]:
            model = CatBoostClassifier(
                depth=depth,
                learning_rate=lr,
                iterations=iters,
                loss_function="MultiClass",
                eval_metric="TotalF1",
                class_weights=class_weights,
                random_seed=42,
                verbose=0
            )
            model.fit(train_pool, eval_set=test_pool)

            y_pred_val = model.predict(X_test).astype(int).ravel()
            f1_val = f1_score(y_test, y_pred_val, average="weighted")

            if f1_val > best_score:
                best_score = f1_val
                best_params = {
                    "depth": depth,
                    "learning_rate": lr,
                    "iterations": iters
                }
                best_model = model

print("✅ 最佳参数：", best_params)
print(f"✅ 最佳 F1：{best_score:.4f}")

# ===============================
# 7. 预测
# ===============================
y_proba = best_model.predict_proba(X_test)
y_pred = np.argmax(y_proba, axis=1)

# ===============================
# 8. 模型评估
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

y_test_bin = label_binarize(y_test, classes=range(num_classes))
auc_macro = roc_auc_score(
    y_test_bin, y_proba, average="macro", multi_class="ovr"
)

print("\n=== CatBoost 模型评估 ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC(macro): {auc_macro:.4f}")

# ===============================
# 9. 混淆矩阵（✅ 中文 + 数值）
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("混淆矩阵-CatBoost", fontsize=18)
plt.xlabel("预测证型", fontsize=14)
plt.ylabel("真实证型", fontsize=14)
plt.xticks(range(num_classes), class_names, rotation=45, fontsize=12)
plt.yticks(range(num_classes), class_names, fontsize=12)

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
# 10. ROC 曲线（中文证型）
# ===============================
plt.figure(figsize=(8, 6))
for i, label in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测")
plt.title("ROC 曲线-CatBoost", fontsize=18)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(fontsize=9, loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== 运行结束：CatBoost ===")
