import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    recall_score
)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# =====================================================
# 保存路径
# =====================================================
save_dir = r"C:\Users\Feng\Desktop\NEC_5折结果"
os.makedirs(save_dir, exist_ok=True)

# =====================================================
# FeatureSelector
# =====================================================
class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        uni_p_threshold=0.1,
        spearman_thresh=0.8,
        min_features_to_select=5
    ):
        self.uni_p_threshold = uni_p_threshold
        self.spearman_thresh = spearman_thresh
        self.min_features_to_select = min_features_to_select

    def fit(self, X, y):

        X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()

        # =================================================
        # 单因素Logistic
        # =================================================
        pvals = {}

        for col in X.columns:

            try:
                xi = sm.add_constant(X[[col]].astype(float))
                model = sm.Logit(y, xi).fit(disp=0)
                pvals[col] = model.pvalues[col]

            except Exception:
                pvals[col] = 1.0

        uni_feats = [
            f for f, p in pvals.items()
            if p < self.uni_p_threshold
        ]

        if len(uni_feats) == 0:
            uni_feats = self.columns_

        # =================================================
        # Spearman去相关
        # =================================================
        subX = X[uni_feats].astype(float)

        corr = subX.corr(method="spearman").abs()

        keep = []

        for col in corr.columns:

            if all(corr.loc[col, k] < self.spearman_thresh for k in keep):
                keep.append(col)

        # =================================================
        # LASSO
        # =================================================
        self.lasso_ = LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 20),
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            cv=5,
            class_weight="balanced",
            max_iter=5000
        )

        self.lasso_.fit(subX[keep].fillna(0), y)

        coef = np.abs(self.lasso_.coef_[0])

        selected = [
            f for f, c in zip(keep, coef)
            if c > 1e-6
        ]

        if len(selected) < self.min_features_to_select:

            idx = np.argsort(-coef)[:self.min_features_to_select]
            selected = [keep[i] for i in idx]

        self.selected_features_ = selected

        return self

    def transform(self, X):

        X = pd.DataFrame(X, columns=self.columns_)

        return X[self.selected_features_]

# =====================================================
# 读取数据
# =====================================================
df = pd.read_excel(
    r"C:\Users\Feng\Desktop\研究生\医疗项目\新生儿坏死性小肠结肠炎\数据\传统方法\处理数据\机器学习输入\机器学习输入数据_无图像特征.xlsx"
)

# =====================================================
# 分类变量编码
# =====================================================
mapping_pairs = {

    '是否首诊': {'否': 0, '是': 1},
    '性别': {'男': 0, '女': 1},
    '是否手术': {'否': 0, '是': 1},
    '是否早产': {'否': 0, '是': 1},
    '入院是否NEC': {'否': 0, '是': 1},
    '生产方式': {'剖腹产': 0, '顺产': 1},
    '是否是试管婴儿': {'否': 0, '是': 1},
    '是否是双胎': {'否': 0, '是': 1},
    '血型': {'A': 0, 'B': 1, 'AB': 2, 'O': 3},
    '血培养': {'阴性': 0, '革兰阳性球菌': 1, '革兰阴性球菌': 2}
}

for col, mp in mapping_pairs.items():

    if col in df.columns:
        df[col] = df[col].map(mp)

# =====================================================
# 序数变量
# =====================================================
ordinal_map = {
    '-': 0,
    '0-1': 1,
    '0-2': 2,
    '1-4': 3,
    '2-4': 4,
    '+': 5,
    '+++': 6
}

ordinal_mapwhite = {
    '-': 0,
    '0-1': 1,
    '0-2': 2,
    '0-3': 3,
    '1-2': 4,
    '1-6': 5,
    '2-3': 6,
    '4-6': 7,
    '6-8': 8,
    '+': 9,
    '++': 10,
    '+++': 11
}

if '红细胞' in df.columns:
    df['红细胞'] = df['红细胞'].map(ordinal_map)

if '白/脓细胞' in df.columns:
    df['白/脓细胞'] = df['白/脓细胞'].map(ordinal_mapwhite)

# =====================================================
# 特征与标签
# =====================================================
feature_cols = [c for c in df.columns if c not in ["住院号", "结局"]]

X = df[feature_cols]
y = df["结局"]

# =====================================================
# 模型集合
# =====================================================
base_models = {

    "Logistic Regression":
        SkLogistic(
            class_weight="balanced",
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            subsample=0.8,
            scale_pos_weight=1.5,
            tree_method="hist",
            device="cpu",
            enable_categorical=False,
            random_state=42
        ),

    "LGBM":
        lgb.LGBMClassifier(
            class_weight="balanced",
            colsample_bytree=0.8,
            verbosity=-1,
            n_estimators=300,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=0,
            auto_class_weights="Balanced",
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "KNN":
        KNeighborsClassifier(n_neighbors=5)
}

# =====================================================
# 超参数
# =====================================================
param_grids = {

    "Logistic Regression": {
        "fs__uni_p_threshold": [0.05, 0.1],
        "fs__spearman_thresh": [0.8, 0.9],
        "model__C": [0.1, 1, 10]
    },

    "Random Forest": {
        "fs__uni_p_threshold": [0.05],
        "model__n_estimators": [200, 300],
        "model__max_depth": [4, 6]
    },

    "XGBoost": {
        "fs__uni_p_threshold": [0.05],
        "model__n_estimators": [300, 500],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.01, 0.03]
    },

    "LGBM": {
        "fs__uni_p_threshold": [0.05],
        "model__learning_rate": [0.01, 0.03]
    },

    "CatBoost": {
        "fs__uni_p_threshold": [0.05],
        "model__iterations": [300, 500],
        "model__learning_rate": [0.01, 0.03]
    },

    "SVM": {
        "fs__uni_p_threshold": [0.05],
        "model__C": [0.1, 1, 10]
    },

    "Naive Bayes": {
        "fs__uni_p_threshold": [0.05]
    },

    "KNN": {
        "fs__uni_p_threshold": [0.05],
        "model__n_neighbors": [3, 5, 7]
    }
}

# =====================================================
# 5折CV
# =====================================================
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

# =====================================================
# 保存各模型结果
# =====================================================
all_model_results = []

# =====================================================
# 循环所有模型
# =====================================================
for model_name, model in base_models.items():

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    fold_auc = []
    fold_sens = []
    fold_spec = []
    fold_acc = []

    best_params_all_folds = []

    # =================================================
    # Outer CV
    # =================================================
    for fold, (tr, te) in enumerate(outer_cv.split(X, y), 1):

        print(f"\nOuter Fold {fold}")

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # =================================================
        # pipeline
        # =================================================
        pipe = Pipeline([

            ("imputer",
             SimpleImputer(strategy="most_frequent")),

            ("scaler",
             StandardScaler()),

            ("fs",
             FeatureSelector(
                 uni_p_threshold=0.05,
                 spearman_thresh=0.75,
                 min_features_to_select=10
             )),

            ("model", model)
        ])

        # =================================================
        # GridSearch
        # =================================================
        grid = GridSearchCV(
            pipe,
            param_grids[model_name],
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_tr, y_tr)

        best_model = grid.best_estimator_

        best_params_all_folds.append(grid.best_params_)

        # =================================================
        # Outer test
        # =================================================
        y_prob = best_model.predict_proba(X_te)[:, 1]

        y_pred = (y_prob >= 0.5).astype(int)

        # =================================================
        # 指标
        # =================================================
        auc = roc_auc_score(y_te, y_prob)

        acc = accuracy_score(y_te, y_pred)

        sens = recall_score(y_te, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

        spec = tn / (tn + fp)

        # =================================================
        # 保存
        # =================================================
        fold_auc.append(auc)
        fold_acc.append(acc)
        fold_sens.append(sens)
        fold_spec.append(spec)

        print(f"AUC={auc:.3f}")
        print(f"Sensitivity={sens:.3f}")
        print(f"Specificity={spec:.3f}")
        print(f"Accuracy={acc:.3f}")

    # =====================================================
    # 最优超参数投票
    # =====================================================
    param_counter = defaultdict(list)

    for params in best_params_all_folds:

        for k, v in params.items():
            param_counter[k].append(v)

    final_best_params = {}

    for k, v_list in param_counter.items():

        most_common = Counter(v_list).most_common(1)[0][0]

        final_best_params[k] = most_common

    # =====================================================
    # 结果汇总
    # =====================================================
    result = {

        "模型": model_name,

        "AUC":
            f"{np.mean(fold_auc):.3f} "
            f"({np.percentile(fold_auc,2.5):.3f}–{np.percentile(fold_auc,97.5):.3f})",

        "灵敏度":
            f"{np.mean(fold_sens):.3f} "
            f"({np.percentile(fold_sens,2.5):.3f}–{np.percentile(fold_sens,97.5):.3f})",

        "特异性":
            f"{np.mean(fold_spec):.3f} "
            f"({np.percentile(fold_spec,2.5):.3f}–{np.percentile(fold_spec,97.5):.3f})",

        "准确性":
            f"{np.mean(fold_acc):.3f} "
            f"({np.percentile(fold_acc,2.5):.3f}–{np.percentile(fold_acc,97.5):.3f})",

        "最优超参数": str(final_best_params)
    }

    all_model_results.append(result)

# =====================================================
# 输出最终表格
# =====================================================
result_df = pd.DataFrame(all_model_results)

print("\n")
print("="*80)
print("最终模型性能比较")
print("="*80)

print(result_df)

# =====================================================
# 保存CSV
# =====================================================
csv_path = os.path.join(save_dir, "五折交叉验证模型比较.csv")

result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n结果已保存: {csv_path}")

# =====================================================
# 保存Excel
# =====================================================
excel_path = os.path.join(save_dir, "五折交叉验证模型比较.xlsx")

result_df.to_excel(
    excel_path,
    index=False
)

print(f"Excel已保存: {excel_path}")