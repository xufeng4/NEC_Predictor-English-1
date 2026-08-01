import os
import textwrap
import pandas as pd
import numpy as np
import shap
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from openpyxl import Workbook
from openpyxl.styles import PatternFill


plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False


def calc_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "Accuracy": acc,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv
    }

# =====================================================
# 0️⃣ 数据读取 + 预处理（🔥必须与第一部分完全一致）
# =====================================================
save_dir = r"C:\Users\Feng\Desktop\研究生\医疗项目\新生儿坏死性小肠结肠炎\数据\传统方法\处理数据\最新结果表\英文版"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_excel(
    r"C:\Users\Feng\Desktop\研究生\医疗项目\新生儿坏死性小肠结肠炎\数据\传统方法\处理数据\机器学习输入\机器学习输入数据_无图像特征(英文版).xlsx"
)

mapping_pairs = {
    'Gender': {'男': 0, '女': 1},
    '是否手术': {'否': 0, '是': 1},
    'Premature Birth ': {'否': 0, '是': 1},
    'Delivery Mode': {'剖腹产': 0, '顺产': 1},
    'IVF ': {'否': 0, '是': 1},
    'Twin Pregnancy': {'否': 0, '是': 1},
    'Blood Type': {'A': 0, 'B': 1, 'AB': 2, 'O': 3},
    'Blood Culture Result': {'阴性': 0, '革兰阳性球菌': 1, '革兰阴性球菌': 2}
}
for col, mp in mapping_pairs.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

ordinal_map = {'-': 0, '0-1': 1, '0-2': 2, '1-4': 3, '2-4': 4, '+': 5, '+++': 6}
ordinal_mapwhite = {'-': 0, '0-1': 1, '0-2': 2, '0-3': 3, '1-2': 4, '1-6': 5, '2-3': 6,
                    '4-6': 7, '6-8': 8, '+': 9, '++': 10, '+++': 11}
if 'Red Blood Cells' in df.columns:
    df['Red Blood Cells'] = df['Red Blood Cells'].map(ordinal_map)
if 'White/Pus Cells' in df.columns:
    df['White/Pus Cells'] = df['White/Pus Cells'].map(ordinal_mapwhite)

feature_cols = [c for c in df.columns if c not in ["Hospitalization Number", "Outcome"]]
X = df[feature_cols]
y = df["Outcome"]

# =====================================================
# 🚨 1️⃣ 先划分数据（核心！！！）
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# 🚨 2️⃣ 众数填补（只用训练集统计）
# =====================================================
def mode_impute(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    mode_dict = {}

    for col in train_df.columns:
        mode_val = train_df[col].mode(dropna=True)
        if len(mode_val) > 0:
            fill_val = mode_val[0]
        else:
            fill_val = 0

        mode_dict[col] = fill_val

        train_df[col] = train_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)

    return train_df, test_df, mode_dict

X_train, X_test, mode_dict = mode_impute(X_train, X_test)

# 👉 数值化
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# =====================================================
# ✅ Step 1：单因素筛选（🔥只用训练集）
# =====================================================
uni_results = []

for col in X_train.columns:
    try:
        if X_train[col].nunique() <= 1:
            continue

        xi = sm.add_constant(X_train[[col]])
        model = sm.Logit(y_train, xi).fit(disp=0)

        conf = model.conf_int().loc[col]

        uni_results.append({
            "feature": col,
            "coef": model.params[col],
            "p_value": model.pvalues[col],
            "OR": np.exp(model.params[col]),
            "OR_lower": np.exp(conf[0]),
            "OR_upper": np.exp(conf[1]),
            "status": "ok"
        })

    except Exception as e:
        uni_results.append({
            "feature": col,
            "coef": np.nan,
            "p_value": np.nan,
            "OR": np.nan,
            "OR_lower": np.nan,
            "OR_upper": np.nan,
            "status": "fail"
        })

uni_df = pd.DataFrame(uni_results).sort_values(by="p_value")

# 👉 保存（论文表）
uni_df.to_csv(os.path.join(save_dir, "Step1_单因素分析_无泄漏.csv"), index=False)

# 👉 筛选
uni_feats = uni_df[
    (uni_df["p_value"] < 0.1) & (uni_df["status"] == "ok")
]["feature"].tolist()

print(f"✅ 单因素筛选后: {len(uni_feats)}")

# =====================================================
# ✅ Step 2：Spearman筛选（🔥只用训练集）
# =====================================================
X_spear = X_train[uni_feats]

corr = X_spear.corr(method="spearman")
# 1. 绘制Spearman相关性热图（论文标准样式）
plt.figure(figsize=(16, 14), dpi=300)  # 足够大的画布，适配多特征
sns.heatmap(
    corr,
    annot=False,  # 不显示具体数值（避免拥挤），如需显示可改为True
    cmap="RdBu_r",  # 红蓝配色，红正蓝负，符合学术规范
    vmin=-1, vmax=1, center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Spearman Correlation Coefficient"}
)
plt.title("Spearman Correlation Heatmap (Training Set)", fontsize=16, pad=20)
# 放大颜色条（数值条形图及其文字）
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Spearman Correlation Coefficient", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Spearman相关性热图.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. 生成并保存完整Spearman相关系数表（论文可直接引用）
# 直接使用 corr 原生的宽格式（特征为索引+列）
corr_table_wide = corr.copy()

# 保存为 CSV（Excel 直接打开就是你要的宽表格格式）
corr_table_wide.to_csv(os.path.join(save_dir, "Spearman相关系数完整表.csv"))

print(f"\n✅ Spearman相关性热图已保存！")
print(f"✅ Spearman相关系数完整表（宽格式）已保存！")
print(corr_table_wide.head())

print(f"\n✅ Spearman相关性热图已保存！")

keep = []
corr_records = []

for col in corr.columns:
    drop_flag = False

    for k in keep:
        corr_val = corr.loc[col, k]

        if abs(corr_val) > 0.8:

            p1 = uni_df.loc[uni_df["feature"] == col, "p_value"].values[0]
            p2 = uni_df.loc[uni_df["feature"] == k, "p_value"].values[0]

            if p1 > p2:
                drop_flag = True
                corr_records.append({
                    "feature1": col,
                    "feature2": k,
                    "corr": corr_val,
                    "kept": k,
                    "removed": col
                })
                break
            else:
                keep.remove(k)
                corr_records.append({
                    "feature1": col,
                    "feature2": k,
                    "corr": corr_val,
                    "kept": col,
                    "removed": k
                })
                break

    if not drop_flag:
        keep.append(col)

# 👉 保存详细过程
pd.DataFrame(corr_records).to_csv(
    os.path.join(save_dir, "Step2_Spearman过程_无泄漏.csv"),
    index=False
)

# 👉 保存最终保留
pd.DataFrame({"kept_features": keep}).to_csv(
    os.path.join(save_dir, "Step2_Spearman结果_无泄漏.csv"),
    index=False
)

print(f"Spearman后: {len(keep)}")

# =====================================================
# ✅ Step 3：LASSO筛选（🔥只用训练集）
# =====================================================
lasso = LogisticRegressionCV(
    Cs=np.logspace(-2, 2, 50),
    penalty="l1",
    solver="saga",
    scoring="roc_auc",
    cv=5,
    class_weight="balanced",
    max_iter=10000,
    random_state=42
)

lasso.fit(X_train[keep], y_train)

coef = np.abs(lasso.coef_[0])

lasso_df = pd.DataFrame({
    "feature": keep,
    "coef": coef
}).sort_values(by="coef", ascending=False)

lasso_feats = lasso_df[lasso_df["coef"] > 1e-6]["feature"].tolist()

if len(lasso_feats) < 10:
    lasso_feats = lasso_df.head(10)["feature"].tolist()

lasso_df["selected"] = lasso_df["feature"].isin(lasso_feats)
lasso_df["rank"] = range(1, len(lasso_df)+1)

lasso_df.to_csv(os.path.join(save_dir, "Step3_LASSO_无泄漏.csv"), index=False)

print(f"LASSO后: {len(lasso_feats)}")

# ----------------------
# 2. 只对选中的特征画路径图（关键优化）
# ----------------------
# 只选LASSO选中的特征，避免线太多
X_lasso = StandardScaler().fit_transform(X_train[lasso_feats])

coefs = []
Cs = np.logspace(-2, 2, 50)

for C in Cs:
    lr = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000)
    lr.fit(X_lasso, y_train)
    coefs.append(lr.coef_[0])

coefs = np.array(coefs)

# ===== 优化后的系数路径图 =====
plt.figure(figsize=(8, 6))
# 给每条线加标签，方便看图
colors = plt.cm.tab10(np.linspace(0, 1, len(lasso_feats)))

for i in range(coefs.shape[1]):
    label = textwrap.fill(lasso_feats[i], width=22) if len(lasso_feats[i]) > 22 else lasso_feats[i]
    plt.plot(np.log10(Cs), coefs[:, i], color=colors[i], label=label)

plt.xlabel("log10(C)")
plt.ylabel("Coefficient")
plt.title("LASSO Coefficient Paths")
plt.xlim(-2, 3.5)
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(save_dir, "LASSO路径图.png"), dpi=300, bbox_inches='tight')
plt.close()
# =====================================================
# ✅ 最终特征表（🔥可解释版）
# =====================================================
final_df = pd.DataFrame({"feature": lasso_feats})

final_df = final_df.merge(
    uni_df[["feature", "p_value", "OR"]],
    on="feature", how="left"
)

final_df = final_df.merge(
    lasso_df[["feature", "coef"]],
    on="feature", how="left"
)

final_df.to_csv(os.path.join(save_dir, "Final_features_无泄漏.csv"), index=False)

print("\n🎯 最终特征:")
print(lasso_feats)

# =====================================================
# 🚨 Step 4：建模（仅使用筛选结果）
# =====================================================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    auto_class_weights="Balanced",
    verbose=0,
    random_state=42
)

model.fit(X_train[lasso_feats], y_train)

# =====================================================
# 💾 保存训练好的模型（用于网页端部署）
# =====================================================
model_path = r"C:\Users\Feng\Documents\Codex\2026-06-20\files-mentioned-by-the-user-5\outputs\NEC_Predictor\model\catboost_model.cbm"
model.save_model(model_path)
print(f"✅ 模型已保存至: {model_path}")

# ⚠️ 注意：此模型仅在训练集上训练（含特征筛选在训练集内完成）。
# 如需部署到生产环境，建议在完整数据集上重新训练最终模型。


# =====================================================
# 🚨 Step 5：测试集评估（真正泛化能力）
# =====================================================
y_prob = model.predict_proba(X_test[lasso_feats])[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"\n🎯 无泄漏AUC: {auc:.4f}")
