"""
Phishing URL Detector - Eğitim Pipeline'ı (v2)

Önceki notebook'a göre eklenenler:
  1. Stratified K-Fold cross-validation (tek split yerine)
  2. Class imbalance handling (class_weight / scale_pos_weight)
  3. Hyperparameter tuning (RandomizedSearchCV)
  4. SHAP ile explainability (feature importance + örnek bazlı açıklama)
  5. Confusion matrix görselleştirme + FP/FN maliyet yorumu
  6. Ortak src/features.py kullanımı (train/serve skew riski yok)
  7. Metrikler reports/metrics.json'a kaydediliyor (README'yi buradan besleyeceğiz)

Kullanım (Kaggle ya da lokal, dataset indirilmiş halde):
    python train.py --data /path/to/urldata.csv --outdir models --reportdir reports
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES, extract_features_batch


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    assert "url" in df.columns, "CSV'de 'url' sütunu bulunamadı."
    assert "label" in df.columns, "CSV'de 'label' sütunu bulunamadı."
    return df


def build_feature_matrix(df: pd.DataFrame):
    X = extract_features_batch(df["url"])
    y = (df["label"].str.lower() == "malicious").astype(int)
    return X, y


def run_cross_validation(models: dict, X, y, cv_folds=5):
    print("\n" + "=" * 60)
    print(f"STRATIFIED {cv_folds}-FOLD CROSS VALIDATION (ROC-AUC)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
        print(f"{name:22s} AUC = {scores.mean():.4f} ± {scores.std():.4f}  (folds: {np.round(scores, 4).tolist()})")
    return cv_results


def tune_xgboost(X_train, y_train, scale_pos_weight, n_iter=20):
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (RandomizedSearchCV - XGBoost)")
    print("=" * 60)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    base = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"Tamamlandı ({time.time() - t0:.1f}s). En iyi CV AUC: {search.best_score_:.4f}")
    print("En iyi parametreler:", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_and_plot(model, X_test, y_test, reportdir: Path):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    report = classification_report(y_test, pred, target_names=["benign", "malicious"], output_dict=True)
    print("\n" + "=" * 60)
    print("FINAL MODEL - TEST SET SONUÇLARI")
    print("=" * 60)
    print(classification_report(y_test, pred, target_names=["benign", "malicious"]))
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n--- FP/FN Maliyet Yorumu ---")
    print(f"False Negative (kaçırılan phishing) : {fn}  -> en maliyetli hata, kullanıcı zarar görebilir")
    print(f"False Positive (yanlışlıkla bloklanan güvenli site): {fp}  -> kullanıcı deneyimini bozar ama daha az riskli")

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "malicious"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Final Model (Test Set)")
    fig.tight_layout()
    fig.savefig(reportdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {
        "roc_auc": float(auc),
        "classification_report": report,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_shap_analysis(model, X_test, reportdir: Path, sample_size=2000):
    try:
        import shap
    except ImportError:
        print("\n[UYARI] shap kurulu değil, `pip install shap` sonra tekrar dene. Explainability adımı atlandı.")
        return

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(reportdir / "shap_feature_importance.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(reportdir / "shap_summary_beeswarm.png", dpi=150)
    plt.close()
    print(f"SHAP grafikleri kaydedildi -> {reportdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="urldata.csv dosya yolu")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--reportdir", default="reports")
    parser.add_argument("--tune-iter", type=int, default=20, help="RandomizedSearchCV deneme sayısı")
    parser.add_argument("--skip-cv", action="store_true", help="Hızlı denemeler için CV adımını atla")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    reportdir = Path(args.reportdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reportdir.mkdir(parents=True, exist_ok=True)

    print("Veri yükleniyor...")
    df = load_data(args.data)
    print(f"Boyut: {df.shape}, sınıf dağılımı:\n{df['label'].value_counts()}")

    X, y = build_feature_matrix(df)
    class_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"\nFeature sayısı: {X.shape[1]}  |  scale_pos_weight (benign/malicious): {class_ratio:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = {"feature_names": FEATURE_NAMES, "class_ratio_benign_over_malicious": class_ratio}

    if not args.skip_cv:
        models_for_cv = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, random_state=42, n_jobs=-1, eval_metric="logloss",
                verbosity=0, scale_pos_weight=class_ratio,
            ),
        }
        metrics["cross_validation"] = run_cross_validation(models_for_cv, X_train, y_train)

    best_model, best_params, best_cv_auc = tune_xgboost(
        X_train, y_train, scale_pos_weight=class_ratio, n_iter=args.tune_iter
    )
    metrics["hyperparameter_tuning"] = {"best_params": best_params, "best_cv_auc": float(best_cv_auc)}

    best_model.fit(X_train, y_train)
    metrics["test_evaluation"] = evaluate_and_plot(best_model, X_test, y_test, reportdir)

    if not args.skip_shap:
        run_shap_analysis(best_model, X_test, reportdir)

    joblib.dump(best_model, outdir / "phishing_model.pkl")
    joblib.dump(FEATURE_NAMES, outdir / "feature_names.pkl")
    with open(reportdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Model kaydedildi -> {outdir / 'phishing_model.pkl'}")
    print(f"✅ Metrikler kaydedildi -> {reportdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
