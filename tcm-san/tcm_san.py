# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import random
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# =========================
# 设置中文字体（全局，解决乱码）
# =========================
font_path = r"C:\Windows\Fonts\simhei.ttf"
font_prop = FontProperties(fname=font_path)

rcParams['font.sans-serif'] = ['SimHei']   # 全局使用黑体
rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# =========================
# 0. 固定随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================
# 1. 数据路径
# =========================
train_path = "train.xlsx"
test_path = "test.xlsx"

# =========================
# 2. Dataset
# =========================
class TCMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# 3. Feature Attention
# =========================
class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        attn = torch.softmax(self.fc(x), dim=1)
        return x * attn

# =========================
# 4. SE Block
# =========================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(dim // reduction, 1), dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=0, keepdim=True))
        return x * w

# =========================
# 5. Group Dropout
# =========================
class GroupDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x, groups):
        if not self.training:
            return x
        x = x.clone()
        for g in groups:
            if len(g) == 0:
                continue
            g = [i for i in g if i < x.shape[1]]
            if torch.rand(1).item() < self.p:
                x[:, g] = 0.0
        return x

# =========================
# 6. MLP 主模型
# =========================
class TCM_SAN_MLP(nn.Module):
    def __init__(self, input_dim, num_classes, groups):
        super().__init__()
        self.groups = groups
        self.attn = FeatureAttention(input_dim)
        self.se = SEBlock(input_dim)
        self.group_dropout = GroupDropout(p=0.3)

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64, num_classes)

        # 残差映射
        self.res_fc2 = nn.Linear(256, 128)
        self.res_fc3 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.attn(x)
        x = self.se(x)
        x = self.group_dropout(x, self.groups)

        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)) + self.res_fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2) + self.res_fc3(h2))
        h3 = self.dropout(h3)

        logits = self.classifier(h3)
        return logits

# =========================
# 7. Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=3):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss

# =========================
# 8. 训练 / 测试函数
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)

# =========================
# 9. 可视化函数
# =========================
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("预测类别", fontproperties=font_prop, fontsize=16)
    plt.ylabel("真实类别", fontproperties=font_prop, fontsize=16)
    plt.title("混淆矩阵-tcm_san", fontproperties=font_prop, fontsize=18)
    # 旋转 X 轴标签
    plt.xticks(rotation=30, ha='right', fontsize=16)
    plt.yticks(rotation=0, fontsize=16)  # 保持 Y 轴标签水平
    plt.show()

def plot_roc_curve(y_true, y_prob, num_classes, class_names):
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("假阳性率", fontproperties=font_prop, fontsize=16)
    plt.ylabel("真阳性率", fontproperties=font_prop, fontsize=16)
    plt.title("ROC 曲线-tcm_san", fontproperties=font_prop, fontsize=18)
    plt.legend(loc="lower right", prop=font_prop,fontsize=12)
    plt.show()

# =========================
# 10. SHAP 分证型特征输出表格
# =========================
def shap_per_class_analysis_to_excel(model, X, y, feature_names, class_names, device, top_k=10, save_path="shap_feature_importance.xlsx"):
    import openpyxl
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    all_results = []

    for cls_idx, cls_name in enumerate(class_names):
        print(f"\n--- SHAP 分析: {cls_name} ---")
        X_cls = X_tensor[y == cls_idx]
        if len(X_cls) == 0:
            print("该类别无样本，跳过")
            continue
        background = X_tensor[:100] if len(X_tensor) > 100 else X_tensor
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_cls)  # list，每个类别shape=(n_samples, n_features)
        cls_shap = np.mean(np.abs(shap_values[cls_idx]), axis=0)
        sorted_idx = np.argsort(-cls_shap)
        top_features = [feature_names[i] for i in sorted_idx[:top_k]]
        top_values = cls_shap[sorted_idx[:top_k]]

        for i, (f, v) in enumerate(zip(top_features, top_values)):
            all_results.append({
                "证型": cls_name,
                "特征排名": i + 1,
                "特征名称": f,
                "SHAP 平均绝对值": v
            })

        print(f"{cls_name} 前 {len(top_features)} 个特征已保存到表格中")

    df_shap = pd.DataFrame(all_results)
    df_shap.to_excel(save_path, index=False)
    print(f"\n所有证型的 SHAP 前 {top_k} 特征已保存到 {save_path}")

# =========================
# 11. 主函数
# =========================
def main():
    # 数据读取
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    for df in [train_df, test_df]:
        if "编号" in df.columns:
            df.drop(columns=["编号"], inplace=True)

    label_col = "证型"
    X_train = train_df.drop(columns=[label_col]).values.astype(np.float32)
    y_train = LabelEncoder().fit_transform(train_df[label_col])
    X_test = test_df.drop(columns=[label_col]).values.astype(np.float32)
    y_test = LabelEncoder().fit_transform(test_df[label_col])

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE 后训练集大小: {X_train.shape}, 类别分布: {np.bincount(y_train)}")

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    input_dim = X_train.shape[1]

    # 四诊分组
    g1 = list(range(0, min(20, input_dim)))
    g2 = list(range(min(20, input_dim), min(30, input_dim)))
    g3 = list(range(min(30, input_dim), min(100, input_dim)))
    g4 = list(range(min(100, input_dim), input_dim))
    groups = [g1, g2, g3, g4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TCMDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TCMDataset(X_test, y_test), batch_size=16, shuffle=False)

    # 模型
    model = TCM_SAN_MLP(input_dim, num_classes, groups).to(device)
    criterion = FocalLoss(gamma=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5)

    # 训练
    epochs = 300
    best_macro = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        preds, labels, probs = evaluate(model, test_loader, device)
        macro = f1_score(labels, preds, average='macro')
        scheduler.step(macro)
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), "best_model.pth")
        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{epochs}] Loss: {loss:.4f} Test Macro-F1: {macro:.4f} LR: {lr:.6f}")

    # 测试指标
    model.load_state_dict(torch.load("best_model.pth"))
    preds, labels, probs = evaluate(model, test_loader, device)

    # 中文类别映射
    class_names = ["气阴两虚证", "痰瘀互结证", "肺脾气虚证", "痰热阻肺证",
                   "阴虚内热证", "痰湿蕴肺证", "脾虚痰湿证", "气滞血瘀证"]

    print("\n========== Test Results ==========")
    print("Accuracy:", accuracy_score(labels, preds))
    print("Macro Precision:", precision_score(labels, preds, average="macro"))
    print("Macro Recall:", recall_score(labels, preds, average="macro"))
    print("Macro F1:", f1_score(labels, preds, average="macro"))
    try:
        auc_score = roc_auc_score(label_binarize(labels, classes=np.arange(num_classes)), probs, average="macro")
        print("Macro AUC:", auc_score)
    except:
        print("AUC 计算失败（可能是单类别样本不足）")
    print(classification_report(labels, preds))

    # 可视化
    plot_confusion_matrix(labels, preds, classes=class_names)
    plot_roc_curve(labels, probs, num_classes, class_names)

    # SHAP 分证型分析保存表格
    feature_names = list(train_df.drop(columns=[label_col]).columns)
    shap_per_class_analysis_to_excel(model, X_test, labels, feature_names, class_names, device, top_k=10)

# =========================
# 12. 运行
# =========================
if __name__ == "__main__":
    main()
