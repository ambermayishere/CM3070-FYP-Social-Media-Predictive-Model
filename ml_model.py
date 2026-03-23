"""
ml_model.py

trains and compares a few models using the features created earlier

since we do not have real trend labels, the script generates its own
based on engagement levels, then evaluates how well each model
can predict those labels

also saves the results, plots, and the best model for reuse
"""

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (confusion_matrix, ConfusionMatrixDisplay)
from sklearn.exceptions      import NotFittedError

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATA_DIR    = "data"
FEAT_IN     = os.path.join(DATA_DIR, "ml_features.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "model_results.csv")
CMP_PLOT    = os.path.join(DATA_DIR, "model_comparison.png")
CM_PLOT     = os.path.join(DATA_DIR, "confusion_matrices.png")
MODEL_OUT   = os.path.join(DATA_DIR, "best_model.pkl")

FEATURES = ["tweet_volume", "sentiment_mean", "sentiment_std",
            "avg_likes", "avg_retweets"]

# check that the dataset has all required features before training
# this avoids confusing errors later when selecting columns
def _validate_feature_matrix(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Feature matrix is missing {len(missing)} required column(s): "
            f"{missing}. Re-run app.py or the notebook to regenerate ml_features.csv."
        )
    if df.empty:
        raise ValueError("Feature matrix is empty — no rows to train on.")
    log.info("Feature matrix validated: %d rows, %d columns.", *df.shape)

# create labels based on engagement since no ground truth exists
# windows with higher activity are treated as "trending" so we can still train and evaluate models
def auto_label(df: pd.DataFrame, percentile: float = 70.0) -> pd.DataFrame:
    if not (0 < percentile < 100):
        raise ValueError(
            f"percentile must be between 0 and 100 exclusive, got {percentile}."
        )

    df = df.copy()
    df["engagement_score"] = (
        df["tweet_volume"] +
        df["avg_likes"]    +
        df["avg_retweets"]
    )
    threshold   = np.percentile(df["engagement_score"], percentile)
    df["label"] = (df["engagement_score"] >= threshold).astype(int)

    class_counts = df["label"].value_counts().to_dict()
    log.info("Auto-label threshold (p%.0f): %.2f", percentile, threshold)
    log.info("Class distribution: %s", class_counts)

    if len(class_counts) < 2:
        raise ValueError(
            "Labelling produced only one class. Try a different percentile "
            "or use a larger dataset so both classes are represented."
        )
    return df

# define models to compare
# each model is wrapped with scaling so everything is treated consistently during training and evaluation
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

# evaluate each model using cross-validation
# automatically adjusts number of folds if dataset is small so the evaluation still works
def evaluate_models(X: pd.DataFrame, y: pd.Series,
                    models: dict) -> tuple:
    min_class = int(y.value_counts().min())
    n_splits  = max(2, min(5, min_class))
    # choose number of folds based on smallest class size
    if n_splits < 5:
        log.warning(
            "Minority class has only %d samples — using %d-fold CV instead of 5. "
            "Collect more data for more stable estimates.",
            min_class, n_splits
        )

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    cv_data = {}
    # run cross-validation and collect performance metrics
    for name, model in models.items():
        try:
            scores = cross_validate(
                model, X, y, cv=cv,
                scoring=["precision_macro", "recall_macro", "f1_macro"],
                return_train_score=False
            )
        except Exception as exc:
            log.error("CV failed for %s: %s — skipping.", name, exc)
            continue

        row = {
            "Model":     name,
            "Precision": scores["test_precision_macro"].mean(),
            "Recall":    scores["test_recall_macro"].mean(),
            "F1":        scores["test_f1_macro"].mean(),
            "F1_std":    scores["test_f1_macro"].std(),
        }
        rows.append(row)
        cv_data[name] = scores
        log.info("%-25s  F1=%.3f ± %.3f  P=%.3f  R=%.3f",
                 name, row["F1"], row["F1_std"], row["Precision"], row["Recall"])

    if not rows:
        raise RuntimeError(
            "All models failed during cross-validation. "
            "Check that the feature matrix has no NaN columns."
        )

    return pd.DataFrame(rows), cv_data

# plot precision, recall and f1-scores for each model
# gives a quick visual comparison of performance
def plot_comparison(results: pd.DataFrame):
    """
    Grouped bar chart — one cluster per model, one bar per metric.
    Skips silently if the results table is empty.
    """
    if results.empty:
        log.warning("Results table is empty — skipping comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x    = np.arange(len(results))
    w    = 0.25
    cols = ["#4472C4", "#70AD47", "#ED7D31"]

    for i, metric in enumerate(["Precision", "Recall", "F1"]):
        bars = ax.bar(x + i * w, results[metric], w,
                      label=metric, color=cols[i], alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
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
    log.info("Comparison plot saved → %s", CMP_PLOT)

# show confusion matrix for each model
# trained on full dataset so both classes are visible
def plot_confusion_matrices(X: pd.DataFrame, y: pd.Series, models: dict):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        try:
            # train model on full data just for visualisation
            model.fit(X, y)
            y_pred = model.predict(X)
        except (NotFittedError, ValueError) as exc:
            log.error("Could not fit %s for confusion matrix: %s", name, exc)
            ax.set_title(f"{name}\n(fit failed)")
            ax.axis("off")
            continue

        cm   = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-trend", "Trend"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)

    plt.suptitle("Confusion Matrices (trained on full dataset)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(CM_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrices saved → %s", CM_PLOT)

# show which features matter most for the random forest model
# skips this step if the model does not support feature importance
def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    out = os.path.join(DATA_DIR, "feature_importance.png")
    clf = model.named_steps.get("clf")

    if clf is None or not hasattr(clf, "feature_importances_"):
        log.warning("%s does not expose feature_importances_ — skipping plot.", model_name)
        return

    importances = clf.feature_importances_
    if len(importances) != len(feature_names):
        log.warning("Importance array length mismatch — skipping feature importance plot.")
        return

    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(importances)), importances[idx], color="#4472C4", alpha=0.85)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx],
                       rotation=25, ha="right", fontsize=9)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Feature importance plot saved → %s", out)

# run the full machine learning workflow
# loads features, creates labels, trains models, compares results and saves the best model
def run_ml_pipeline(feat_path: str = FEAT_IN) -> pd.DataFrame:
    if not os.path.exists(feat_path):
        # make sure feature file exists before continuing
        raise FileNotFoundError(
            f"Feature file not found at '{feat_path}'. "
            "Run app.py or the Jupyter notebook first to generate ml_features.csv."
        )
    # load feature data generated from earlier steps
    df = pd.read_csv(feat_path)
    log.info("Loaded feature matrix: %s", str(df.shape))

    _validate_feature_matrix(df)

    # replace missing values so models do not break
    for col in FEATURES:
        if df[col].isna().any():
            n_missing = df[col].isna().sum()
            log.warning("Column '%s' has %d NaN value(s) — filling with 0.", col, n_missing)
            df[col] = df[col].fillna(0)

    df = auto_label(df, percentile=70)
    X  = df[FEATURES]
    y  = df["label"]

    # duplicate rows if dataset is too small for stable evaluation
    if len(X) < 6:
        log.warning(
            "Only %d rows available — duplicating to reach 30 for stable CV.", len(X)
        )
        factor = max(1, 30 // len(X))
        X = pd.concat([X] * factor, ignore_index=True)
        y = pd.concat([y] * factor, ignore_index=True)
    # prepare models for comparison
    models           = build_models()
    results, cv_data = evaluate_models(X, y, models)
    
    # save evaluation results for reporting
    os.makedirs(DATA_DIR, exist_ok=True)
    results.to_csv(RESULTS_OUT, index=False)
    log.info("Results saved → %s", RESULTS_OUT)
    log.info("\n%s", results.to_string(index=False))

    plot_comparison(results)
    plot_confusion_matrices(X, y, models)

    rf_model = models.get("Random Forest")
    if rf_model:
        rf_model.fit(X, y)
        plot_feature_importance(rf_model, FEATURES, "Random Forest")
    # pick the model with highest f1 score
    best_name = results.loc[results["F1"].idxmax(), "Model"]
    best_model = models[best_name]
    best_model.fit(X, y)
    # store best model so it can be reused later
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(best_model, f)
    log.info("Best model: %s (F1=%.3f)", best_name, results["F1"].max())
    log.info("Model saved → %s", MODEL_OUT)

    return results


if __name__ == "__main__":
    results = run_ml_pipeline()
