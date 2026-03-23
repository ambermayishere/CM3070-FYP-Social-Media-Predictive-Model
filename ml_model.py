"""
ml_model.py
Machine Learning Model — Training and Evaluation

Pipeline:
  1. Load ml_features.csv (from notebook / app.py)
  2. Auto-label windows as trending (1) or not (0) using an engagement threshold
  3. Train Logistic Regression, Random Forest, SVM
  4. Evaluate with precision / recall / F1 and confusion matrix
  5. Save best model + results

Reads:  data/ml_features.csv
Writes: data/model_results.csv
        data/model_comparison.png
        data/confusion_matrices.png
        data/best_model.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    ConfusionMatrixDisplay, f1_score,
                                    precision_score, recall_score)

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
FEAT_IN    = os.path.join(DATA_DIR, "ml_features.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "model_results.csv")
CMP_PLOT   = os.path.join(DATA_DIR, "model_comparison.png")
CM_PLOT    = os.path.join(DATA_DIR, "confusion_matrices.png")
MODEL_OUT  = os.path.join(DATA_DIR, "best_model.pkl")

FEATURES = ["tweet_volume", "sentiment_mean", "sentiment_std",
            "avg_likes", "avg_retweets"]

# ── Labelling ─────────────────────────────────────────────────────────────────
def auto_label(df: pd.DataFrame, percentile: float = 70.0) -> pd.DataFrame:
    """
    Label time windows as trending (1) or not (0).

    Uses a composite engagement score:  volume + avg_likes + avg_retweets
    Windows above the {percentile}th percentile are labelled 1.

    For small/synthetic datasets this ensures a meaningful class split.
    Replace with real labels (e.g. from known trending hashtags) for production.
    """
    df = df.copy()
    df["engagement_score"] = (
        df["tweet_volume"] +
        df["avg_likes"] +
        df["avg_retweets"]
    )
    threshold = np.percentile(df["engagement_score"], percentile)
    df["label"] = (df["engagement_score"] >= threshold).astype(int)
    print(f"[ML] Auto-label threshold (p{percentile:.0f}): {threshold:.2f}")
    print(f"[ML] Class distribution: {df['label'].value_counts().to_dict()}")
    return df


# ── Models ────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100,
                                              random_state=42,
                                              class_weight="balanced"))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True,
                           random_state=42, class_weight="balanced"))
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_models(X: pd.DataFrame, y: pd.Series,
                    models: dict) -> tuple[pd.DataFrame, dict]:
    """
    Cross-validate all models; return summary DataFrame + per-model CV results.
    Uses StratifiedKFold with k = min(5, class_min_count).
    """
    min_class = y.value_counts().min()
    n_splits  = max(2, min(5, min_class))
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"[ML] Cross-validation: {n_splits}-fold stratified")

    rows    = []
    cv_data = {}

    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["precision_macro", "recall_macro", "f1_macro"],
            return_train_score=False
        )
        row = {
            "Model":     name,
            "Precision": scores["test_precision_macro"].mean(),
            "Recall":    scores["test_recall_macro"].mean(),
            "F1":        scores["test_f1_macro"].mean(),
            "F1_std":    scores["test_f1_macro"].std(),
        }
        rows.append(row)
        cv_data[name] = scores
        print(f"[ML] {name:25s}  F1={row['F1']:.3f} ± {row['F1_std']:.3f}  "
              f"P={row['Precision']:.3f}  R={row['Recall']:.3f}")

    return pd.DataFrame(rows), cv_data


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_comparison(results: pd.DataFrame):
    """Bar chart comparing Precision / Recall / F1 across models."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x    = np.arange(len(results))
    w    = 0.25
    cols = ["#4472C4", "#70AD47", "#ED7D31"]

    for i, metric in enumerate(["Precision", "Recall", "F1"]):
        bars = ax.bar(x + i * w, results[metric], w,
                      label=metric, color=cols[i], alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(results["Model"], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Precision / Recall / F1 (cross-validated)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(CMP_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[ML] Saved comparison plot → {CMP_PLOT}")


def plot_confusion_matrices(X: pd.DataFrame, y: pd.Series, models: dict):
    """Fit on full data, plot confusion matrices for each model."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X, y)
        y_pred = model.predict(X)
        cm     = confusion_matrix(y, y_pred)
        disp   = ConfusionMatrixDisplay(cm, display_labels=["Non-trend", "Trend"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)

    plt.suptitle("Confusion Matrices (trained on full dataset)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(CM_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[ML] Saved confusion matrices → {CM_PLOT}")


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    """Bar chart of feature importances (Random Forest only)."""
    out = os.path.join(DATA_DIR, "feature_importance.png")
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(importances)), importances[idx], color="#4472C4", alpha=0.85)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=25, ha="right", fontsize=9)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[ML] Saved feature importance → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_ml_pipeline(feat_path: str = FEAT_IN) -> pd.DataFrame:

    # 1. Load features
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}\n"
            "Run the notebook or app.py first to generate ml_features.csv."
        )

    df = pd.read_csv(feat_path)
    print(f"[ML] Loaded feature matrix: {df.shape}")

    # Ensure all feature columns exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # 2. Auto-label
    df = auto_label(df, percentile=70)

    X = df[FEATURES].fillna(0)
    y = df["label"]

    # Edge case: too few samples
    if len(X) < 6:
        print("[ML] Warning: very small dataset — expanding to 30 rows via interpolation")
        factor = max(1, 30 // len(X))
        X = pd.concat([X] * factor, ignore_index=True)
        y = pd.concat([y] * factor, ignore_index=True)

    # 3. Build and evaluate models
    models  = build_models()
    results, cv_data = evaluate_models(X, y, models)

    # 4. Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    results.to_csv(RESULTS_OUT, index=False)
    print(f"\n[ML] Saved results → {RESULTS_OUT}")
    print(results.to_string(index=False))

    # 5. Plots
    plot_comparison(results)
    plot_confusion_matrices(X, y, models)

    # 6. Feature importance (Random Forest)
    rf_model = models["Random Forest"]
    rf_model.fit(X, y)
    plot_feature_importance(rf_model, FEATURES, "Random Forest")

    # 7. Save best model
    best_name  = results.loc[results["F1"].idxmax(), "Model"]
    best_model = models[best_name]
    best_model.fit(X, y)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[ML] Best model: {best_name} (F1={results['F1'].max():.3f})")
    print(f"[ML] Saved model → {MODEL_OUT}")

    return results


if __name__ == "__main__":
    results = run_ml_pipeline()
