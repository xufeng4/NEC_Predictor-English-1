import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 高分辨率
plt.rcParams['savefig.dpi'] = 300
# 自定义色阶：从极浅蓝（对应数值5）→ 中蓝（25）→ 深蓝（45）
cmap_custom = LinearSegmentedColormap.from_list(
    "custom_blue",
    ["#f0f8ff", "#87ceeb", "#003366"],  # 浅蓝→中蓝→深蓝
    N=256
)

# =====================================================
# 0️⃣ 数据读取 + 预处理（⚠️必须一致）
# =====================================================
save_dir = r"C:\Users\Feng\Desktop\研究生\医疗项目\新生儿坏死性小肠结肠炎\数据\传统方法\处理数据\最新结果表\英文版"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_excel(
    r"C:\Users\Feng\Desktop\研究生\医疗项目\新生儿坏死性小肠结肠炎\数据\传统方法\处理数据\机器学习输入\机器学习输入数据_三类数据_特征筛选(英文版).xlsx"
)

# -------- 特征矩阵 --------
feature_cols = [c for c in df.columns if c not in ["Hospitalization Number", "Outcome"]]

X = df[feature_cols]
y = df["Outcome"]

# =====================================================
# 🚫 不做任何特征筛选（核心区别）
# =====================================================
print(f"使用全部特征建模，特征数: {X.shape[1]}")

# =====================================================
# 1️⃣ Train/Test
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X.fillna(0), y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# 2️⃣ CatBoost模型
# =====================================================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    auto_class_weights="Balanced",
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# 💾 保存训练好的模型（用于网页端部署）
# =====================================================
model_path = r"C:\Users\Feng\Documents\Codex\2026-06-20\files-mentioned-by-the-user-5\outputs\NEC_Predictor\model\catboost_model.cbm"
model.save_model(model_path)
print(f"✅ 模型已保存至: {model_path}")

# ⚠️ 注意：此模型仅在训练集上训练。如需部署到生产环境，建议在完整数据集上重新训练最终模型。


# =====================================================
# 3️⃣ AUC + ROC
# # =====================================================
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"🚫 无特征筛选 AUC: {auc:.4f}")

# =====================================================
# 🧠 预测概率输出（仅概率）
# =====================================================
risk_df = pd.DataFrame({
    "住院号": df.loc[X_test.index, "Outcome"].values,
    "预测概率": y_prob,
    "真实结局": y_test.values
})

risk_df.to_csv(os.path.join(save_dir, "NEC预测概率.csv"), index=False)

print("\n预测概率已保存")

# ========= AUC =========
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_score:.3f}")

# 对角线
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend()

plt.grid(alpha=0.3)

plt.savefig(os.path.join(save_dir, "ROC标准曲线.png"))
plt.close()


# =====================================================
# 📊 4️⃣ 混淆矩阵
# =====================================================
# 👉 默认阈值0.5(需要注意）
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# 👉 数值打印
print("\n混淆矩阵:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# 👉 论文级可视化（完全匹配参考图）
plt.figure(figsize=(6, 5))
ax = plt.gca()

# 绘制热力图背景（匹配参考图色阶）
im = ax.imshow(cm, interpolation='nearest', cmap=cmap_custom, vmin=0, vmax=45)
# 添加右侧色条（匹配参考图刻度5-45）
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
cbar.set_ticks([5, 10, 15, 20, 25, 30, 35, 40, 45])
cbar.set_ticklabels(["5", "10", "15", "20", "25", "30", "35", "40", "45"])

# 设置坐标轴标签（完全匹配参考图）
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=['阴性', '阳性'],
    yticklabels=['阴性', '阳性'],
    title='CatBoost 混淆矩阵（Independent Test Set）',
    xlabel='预测标签',
    ylabel='真实标签'
)

# 旋转x轴标签，保持居中
plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

# 在每个格子中显示数值（居中、白色/黑色自适应）
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # 数值颜色：深色背景用白色，浅色背景用黑色
        text_color = "white" if cm[i, j] > 20 else "black"
        ax.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color=text_color,
            fontsize=14, fontweight='normal'
        )

# 调整布局，防止标签被裁剪
plt.tight_layout()

# 保存图片（高分辨率）
plt.savefig(os.path.join(save_dir, "CatBoost_混淆矩阵_独立测试集.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 混淆矩阵已保存至: {os.path.join(save_dir, 'CatBoost_混淆矩阵.png')}")
# =====================================================
# 📊 6️⃣ 决策曲线 DCA
# =====================================================
def decision_curve(y_true, y_prob, thresholds):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    n = len(y_true)
    net_benefit = []

    for pt in thresholds:

        # 防止极端值
        if pt >= 0.99:
            net_benefit.append(np.nan)
            continue

        pred = (y_prob >= pt).astype(int)

        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()

        nb = (tp / n) - (fp / n) * (pt / (1 - pt))

        net_benefit.append(nb)

    return np.array(net_benefit)

thresholds = np.linspace(0.01, 0.9, 100)

nb_model = decision_curve(y_test.values, y_prob, thresholds)

# 👉 treat all / none
prevalence = y_test.mean()
nb_all = []

for pt in thresholds:
    if pt >= 0.99:
        nb_all.append(np.nan)
    else:
        val = prevalence - (1 - prevalence) * (pt / (1 - pt))
        nb_all.append(val)

nb_all = np.array(nb_all)
nb_none = [0] * len(thresholds)

# 👉 画图
plt.figure(figsize=(6,5))

plt.plot(thresholds, nb_model, label="Model", linewidth=2)
plt.plot(thresholds, nb_all, linestyle="--", label="Treat All", linewidth=2)
plt.plot(thresholds, nb_none, linestyle="--", label="Treat None", linewidth=2)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")

plt.title("Decision Curve Analysis")

# ✅ 限制y轴（关键）
plt.ylim(
    min(np.nanmin(nb_model), np.nanmin(nb_all)) - 0.02,
    max(np.nanmax(nb_model), np.nanmax(nb_all)) + 0.02
)

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(os.path.join(save_dir, "决策曲线.png"))
plt.close()
# =====================================================
# 📉 7️⃣ 校准曲线
# =====================================================
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

brier = brier_score_loss(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Model")
plt.plot([0,1], [0,1], linestyle="--", label="Perfect")

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Probability")
plt.title(f"Calibration Curve (Brier={brier:.3f})")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(os.path.join(save_dir, "校准曲线.png"))
plt.close()
# =====================================================
# 4️⃣ SHAP分析
# =====================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 👉 CatBoost 二分类兼容
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# =====================================================
# ✅ Step 1：计算SHAP重要性（用于筛选Top特征）
# =====================================================
shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": shap_importance
}).sort_values(by="importance", ascending=False)

# 👉 Top10 特征
top10_feats = shap_df.head(10)["feature"].tolist()

# 👉 Top4 特征（用于依赖图）
top4_feats = shap_df.head(4)["feature"].tolist()

# =====================================================
# 📊 ① SHAP贡献条形图
# =====================================================
plt.figure(figsize=(10, 14))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False,
    max_display=10
)


plt.tight_layout()
plt.savefig(os.path.join(save_dir, "SHAP柱状图.png"), dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# 🐝 ② SHAP蜂群图（Top10）
# =====================================================
plt.figure()
shap.summary_plot(
    shap_values[:, [X_test.columns.get_loc(f) for f in top10_feats]],
    X_test[top10_feats],
    show=False
)

plt.savefig(os.path.join(save_dir, "SHAP蜂群图.png"))
plt.close()

# =====================================================
# 📈 ③ SHAP依赖图（Top4关键特征）
# =====================================================
for feat in top4_feats:
    plt.figure()
    shap.dependence_plot(
        feat,
        shap_values,
        X_test,
        interaction_index=None,
        show=False
    )

    plt.xlabel(f"{feat} value")
    plt.ylabel("SHAP value")

    # 👉 风格优化（接近你给的图）
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"SHAP_依赖图_{feat}.png"))
    plt.close()

# =====================================================
# 🌊 ④ SHAP瀑布图
# =====================================================
y_prob_test = model.predict_proba(X_test)[:, 1]
high_risk_idx = y_prob_test.argmax()
sample_idx = high_risk_idx

exp = shap.Explanation(
    values=shap_values[sample_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[sample_idx],
    feature_names=X_test.columns
)

plt.figure(figsize=(12, 7), dpi=300)
shap.plots.waterfall(exp, max_display=10, show=False)

# 自动去掉所有 y 轴标签里的数字和 = 号
ax = plt.gca()
labels = []
for label in ax.get_yticklabels():
    txt = label.get_text()
    # 去掉 "数字 = " 这部分
    if " = " in txt:
        txt = txt.split(" = ")[-1]
    labels.append(txt)
ax.set_yticklabels(labels)
# =====================================================

# 正确概率计算
base_logodds = exp.base_values
final_logodds = exp.base_values + exp.values.sum()
base_prob = 1 / (1 + np.exp(-base_logodds))
final_prob = 1 / (1 + np.exp(-final_logodds))

plt.tight_layout()
plt.subplots_adjust(left=0.3)

plt.savefig(os.path.join(save_dir, "SHAP瀑布图.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ 样本详情 (index={sample_idx})")
print(f"基线概率: {base_prob:.1%}")
print(f"最终预测概率: {final_prob:.1%}")
print(f"真实结局: {y_test.iloc[sample_idx]} (1=阳性)")
# =====================================================
# ✅ SHAP贡献值表
# =====================================================
direction = np.sign(shap_values).mean(axis=0)

shap_df_full = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": shap_importance,
    "direction": direction
}).sort_values(by="mean_abs_shap", ascending=False)

# 计算每个特征的贡献百分比
total_importance = shap_df_full["mean_abs_shap"].sum()
shap_df_full["contribution_percent"] = (shap_df_full["mean_abs_shap"] / total_importance * 100).round(2)
# ==================================================

shap_df_full.reset_index(drop=True, inplace=True)

# 保存【所有特征】表
shap_df_full.to_csv(os.path.join(save_dir, "SHAP贡献值.csv"),
                    index=False, encoding="utf-8-sig")

print("\n✅ 所有特征 SHAP 贡献值（含百分比）已保存！")
print(shap_df_full)
