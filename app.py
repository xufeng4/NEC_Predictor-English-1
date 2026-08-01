import os, json, numpy as np
from flask import Flask, request, jsonify, render_template
from catboost import CatBoostClassifier, Pool

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)

model = CatBoostClassifier()
model.load_model(os.path.join(BASE_DIR, "model", "catboost_model.cbm"))

with open(os.path.join(BASE_DIR, "model", "catboost_model_features.json"), "r", encoding="utf-8-sig") as f:
    feat_info = json.load(f)

all_features = feat_info["clinical"] + feat_info["laboratory"] + feat_info["imaging"]
n_feat = len(all_features)

CLINICAL_LABELS = {
    "出生体重（g）": "Birth Weight (g)",
    "诊断年龄（天）": "Diagnostic Age (days)",
    "住院时长": "Length of Hospital Stay",
}
LAB_LABELS = {
    "腺苷脱氨酶": "Adenosine Deaminase",
    "丙氨酸氨基转移酶": "Alanine Aminotransferase (ALT)",
    "天门冬氨酸氨基转移酶": "Aspartate Aminotransferase (AST)",
    "肌酸激酶": "Creatine Kinase",
    "肌酐": "Creatinine",
    "间接胆红素": "Indirect Bilirubin",
    "总胆红素 tBil": "Total Bilirubin (tBil)",
    "平均血红蛋白浓度": "Mean Hemoglobin Concentration",
    "血小板计数": "Platelet Count (PLT)",
}
IMG_LABELS = {
    "肠壁积气": "Intestinal Wall Pneumatosis",
    "肠间隙异常": "Abnormal Intestinal Space",
    "腹腔游离气体": "Free Intraperitoneal Gas",
    "腹脂线异常": "Abnormal Abdominal Fat Line",
    "肠梗阻征象": "Intestinal Obstruction Signs",
    "肠管液平": "Intestinal Fluid Level",
    "肠管固定": "Intestinal Fixation",
    "腹腔积液": "Intraperitoneal Effusion",
    "肠管充气不均": "Uneven Intestinal Aeration",
    "阶梯状气液平": "Staircase-like Air-Fluid Level",
    "肠壁炎性改变": "Intestinal Wall Inflammatory Changes",
    "门静脉气体": "Portal Vein Gas",
    "肠管液平征": "Intestinal Fluid Level Sign",
    "腹腔炎性改变": "Intraperitoneal Inflammatory Changes",
    "肠穿孔征象": "Intestinal Perforation Signs",
    "肠管排列紊乱": "Disordered Intestinal Arrangement",
    "肠管扩张": "Intestinal Dilation",
}
LABELS = {"clinical": CLINICAL_LABELS, "laboratory": LAB_LABELS, "imaging": IMG_LABELS}
ALL_LABEL_DICTS = [CLINICAL_LABELS, LAB_LABELS, IMG_LABELS]
DEFAULTS = {"出生体重（g）": 1800, "诊断年龄（天）": 15, "住院时长": 30,
    "腺苷脱氨酶": 15, "丙氨酸氨基转移酶": 25, "天门冬氨酸氨基转移酶": 45,
    "肌酸激酶": 100, "肌酐": 50, "间接胆红素": 85,
    "总胆红素 tBil": 120, "平均血红蛋白浓度": 340, "血小板计数": 250}

@app.route("/")
def index():
    return render_template("index.html",
        clinical_features=feat_info["clinical"],
        laboratory_features=feat_info["laboratory"],
        imaging_features=feat_info["imaging"],
        labels=LABELS, defaults=DEFAULTS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        row = []
        for f in all_features:
            val = data.get(f, None)
            if val is None:
                row.append(float("nan"))
            else:
                row.append(float(val))
        X = np.array(row, dtype=float).reshape(1, -1)
        pool = Pool(X)

        prob = float(model.predict_proba(pool)[0, 1])
        pred = int(model.predict(pool)[0])

        shap_raw = model.get_feature_importance(pool, type="ShapValues")
        if len(shap_raw.shape) == 3:
            shap_contrib = shap_raw[0, :n_feat, 1]
        else:
            shap_contrib = shap_raw[0, :n_feat]

        top_idx = np.argsort(np.abs(shap_contrib))[::-1][:10]
        top_feats = []
        for idx in top_idx:
            fname = all_features[idx]
            clabel = fname
            for d in ALL_LABEL_DICTS:
                if fname in d:
                    clabel = d[fname]
                    break
            top_feats.append({
                "name": fname, "label": clabel,
                "value": float(shap_contrib[idx]),
                "abs_value": float(abs(shap_contrib[idx])),
            })

        if prob >= 0.7:
            rl, rc = "High Risk", "#dc3545"
        elif prob >= 0.4:
            rl, rc = "Moderate Risk", "#ff8c00"
        else:
            rl, rc = "Low Risk", "#28a745"

        return jsonify({
            "success": True, "probability": round(prob, 4),
            "prediction": pred,
            "prediction_label": "NEC Positive" if pred == 1 else "NEC Negative",
            "risk_level": rl, "risk_color": rc,
            "top_features": top_feats,
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "detail": traceback.format_exc()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
