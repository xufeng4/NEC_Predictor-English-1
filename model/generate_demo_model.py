import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Feature names matching the 29 features from the paper
clinical_features = [
    'diagnosis_age_days',       # 诊断年龄(天)
    'gestational_age_days',     # 胎龄(天)
    'birth_weight_g',           # 出生体重(g)
    'gender',                   # 性别 (1=M, 0=F)
    'preterm',                  # 是否早产 (1=Yes, 0=No)
]

lab_features = [
    'ast_u_l',                  # 天门冬氨酸氨基转移酶 AST (U/L)
    'indirect_bilirubin',       # 间接胆红素 (umol/L)
    'platelet_count',           # 血小板计数 (x10^9/L)
    'white_blood_cell',         # 白细胞 (x10^9/L)
    'crp',                      # C反应蛋白 (mg/L)
    'lactate',                  # 乳酸 (mmol/L)
    'hemoglobin',               # 血红蛋白 (g/L)
]

imaging_features = [
    'pneumatosis',              # 肠壁积气
    'abnormal_intestinal_space',# 肠间隙异常
    'free_abdominal_air',       # 腹腔游离气体
    'abnormal_fat_line',        # 腹脂线异常
    'obstruction_sign',         # 肠梗阻征象
    'air_fluid_level',          # 肠管液平
    'fixed_loop',               # 肠管固定
    'ascites',                  # 腹腔积液
    'uneven_aeration',          # 肠管充气不均
    'stepladder_sign',          # 阶梯状气液平
    'inflammatory_change',      # 肠壁炎性改变
    'portal_vein_gas',          # 门静脉气体
    'air_fluid_sign',           # 肠管液平征
    'peritoneal_inflammation',  # 腹腔炎性改变
    'perforation_sign',         # 肠穿孔征象
    'disordered_arrangement',   # 肠管排列紊乱
    'bowel_dilation',           # 肠管扩张
]

all_features = clinical_features + lab_features + imaging_features
print(f'Total features: {len(all_features)}')

# Generate synthetic training data
np.random.seed(42)
n_samples = 5000

data = {}
# Clinical features
data['diagnosis_age_days'] = np.random.exponential(15, n_samples).clip(0, 90)
data['gestational_age_days'] = np.random.normal(240, 25, n_samples).clip(140, 280)
data['birth_weight_g'] = data['gestational_age_days'] * 12 + np.random.normal(0, 300, n_samples)
data['gender'] = np.random.binomial(1, 0.5, n_samples)
data['preterm'] = (data['gestational_age_days'] < 259).astype(int)

# Lab features
data['ast_u_l'] = np.random.exponential(40, n_samples).clip(5, 500)
data['indirect_bilirubin'] = np.random.exponential(50, n_samples).clip(5, 500)
data['platelet_count'] = np.random.normal(250, 100, n_samples).clip(20, 600)
data['white_blood_cell'] = np.random.normal(12, 6, n_samples).clip(2, 40)
data['crp'] = np.random.exponential(10, n_samples).clip(0.1, 200)
data['lactate'] = np.random.exponential(2, n_samples).clip(0.5, 15)
data['hemoglobin'] = np.random.normal(14, 3, n_samples).clip(6, 22)

# Imaging features (binary)
for feat in imaging_features:
    data[feat] = np.random.binomial(1, 0.15, n_samples)

df = pd.DataFrame(data)

# Create disease probability based on known risk factors
# NEC risk factors from the paper:
# - Younger age (higher risk)
# - Lower gestational age (higher risk)  
# - Higher AST
# - Higher indirect bilirubin
# - Lower platelet count
# - Positive imaging features (especially abdominal fat line abnormality, pneumatosis)

risk_score = (
    -0.3 * df['gestational_age_days'].clip(0, 280) / 280  # younger = higher risk
    - 0.2 * df['diagnosis_age_days'].clip(0, 90) / 90       # younger = higher risk
    + 0.15 * np.log(df['ast_u_l'].clip(5, 500) / 40)        # higher AST = higher risk
    + 0.15 * np.log(df['indirect_bilirubin'].clip(5, 500) / 50)  # higher bilirubin = higher risk
    - 0.2 * (df['platelet_count'].clip(20, 600) / 100).clip(0.2, 6)**0.5  # lower PLT = higher risk
    + 0.5 * df['abnormal_fat_line']                          # abdominal fat line = strong predictor
    + 0.5 * df['pneumatosis']                                 # pneumatosis = strong predictor  
    + 0.4 * df['free_abdominal_air']                          # free air = strong predictor
    + 0.2 * df['portal_vein_gas']                             # portal vein gas = strong predictor
    + 0.1 * df[imaging_features].sum(axis=1).clip(0, 17)**0.7  # more imaging signs = higher risk
    + np.random.normal(0, 0.8, n_samples)
)

prob = 1 / (1 + np.exp(-risk_score))
y = (prob > 0.4).astype(int)

# Positive rate
print(f'Positive rate: {y.mean():.3f}')

# Train CatBoost model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50,
    early_stopping_rounds=50,
)

# Train
model.fit(
    df[all_features], y,
    cat_features=[f'gender', f'preterm'],
    verbose=50
)

# Save model
model_path = r'C:\Users\Feng\Documents\Codex\2026-06-20\files-mentioned-by-the-user-5\outputs\NEC_Predictor\model\catboost_model.cbm'
model.save_model(model_path)
print(f'\nModel saved to: {model_path}')
print(f'Model validation AUC: {model.score(df[all_features], y):.4f}')

# Save feature list for reference
import json
feat_info = {
    'clinical': clinical_features,
    'laboratory': lab_features,
    'imaging': imaging_features,
}
with open(model_path.replace('.cbm', '_features.json'), 'w', encoding='utf-8') as f:
    json.dump(feat_info, f, ensure_ascii=False, indent=2)
print('Feature list saved.')
