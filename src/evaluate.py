"""Metrics, cross-validation and per-category error analysis.

Accuracy alone is a poor summary for intrusion detection: a detector that
misses a whole attack family can still score well because the remaining
traffic is easy.  Recall on the attack class -- the fraction of intrusions
actually caught -- is the operationally important number, so every table here
reports it alongside precision and F1.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Everything recorded for one model on one split."""

    model_name: str
    split_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion: np.ndarray
    y_true: np.ndarray = field(repr=False, default=None)
    y_pred: np.ndarray = field(repr=False, default=None)
    y_score: np.ndarray | None = field(repr=False, default=None)
    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    def as_row(self) -> dict[str, Any]:
        tn, fp, fn, tp = self.confusion.ravel()
        return {
            "Model": self.model_name,
            "Split": self.split_name,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "ROC-AUC": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "False Alarm Rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
            "Miss Rate": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
            "Train (s)": round(self.train_seconds, 2),
            "Predict (s)": round(self.predict_seconds, 3),
        }


def _decision_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray | None:
    """Continuous attack scores for ROC, whichever interface the model offers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def train_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> tuple[BaseEstimator, float]:
    """Fit a model and return it together with the wall-clock training time."""
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    logger.info("Trained %s in %.2fs", type(model).__name__, elapsed)
    return model, elapsed


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    split_name: str,
    train_seconds: float = 0.0,
) -> EvaluationResult:
    """Score a fitted model on one split."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    predict_seconds = time.perf_counter() - start

    y_score = _decision_scores(model, X)
    y_true = np.asarray(y)

    try:
        auc = float(roc_auc_score(y_true, y_score)) if y_score is not None else None
    except ValueError:  # pragma: no cover - only when a split is single-class
        auc = None

    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        y_true=y_true,
        y_pred=np.asarray(y_pred),
        y_score=y_score,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
    )


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = config.CV_FOLDS,
    scoring: str = "f1",
) -> tuple[float, float]:
    """Mean and standard deviation of a stratified k-fold score."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def metrics_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Collapse a list of results into the comparison table used in the report."""
    return pd.DataFrame([r.as_row() for r in results])


def classification_text_report(result: EvaluationResult) -> str:
    """Per-class precision/recall/F1 block, saved verbatim to results/."""
    return classification_report(
        result.y_true, result.y_pred,
        target_names=config.TARGET_NAMES,
        digits=4,
        zero_division=0,
    )


def per_category_recall(
    result: EvaluationResult, categories: pd.Series
) -> pd.DataFrame:
    """Detection rate broken down by DoS / Probe / R2L / U2R.

    This is where an aggregate accuracy figure hides the interesting failure:
    the rare privilege-escalation families are far harder to catch than the
    high-volume flooding attacks.
    """
    frame = pd.DataFrame({
        "category": np.asarray(categories),
        "y_true": result.y_true,
        "y_pred": result.y_pred,
    })
    attacks = frame[frame["y_true"] == 1]

    rows = []
    for category, group in attacks.groupby("category"):
        rows.append({
            "Model": result.model_name,
            "Split": result.split_name,
            "Category": category,
            "Records": len(group),
            "Detected": int(group["y_pred"].sum()),
            "Detection Rate": round(group["y_pred"].mean(), 4),
        })

    normals = frame[frame["y_true"] == 0]
    if len(normals):
        rows.append({
            "Model": result.model_name,
            "Split": result.split_name,
            "Category": "Normal (correctly allowed)",
            "Records": len(normals),
            "Detected": int((normals["y_pred"] == 0).sum()),
            "Detection Rate": round((normals["y_pred"] == 0).mean(), 4),
        })

    return pd.DataFrame(rows).sort_values("Records", ascending=False).reset_index(drop=True)


def roc_points(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray] | None:
    """False-positive / true-positive pairs for the ROC overlay plot."""
    if result.y_score is None:
        return None
    fpr, tpr, _ = roc_curve(result.y_true, result.y_score)
    return fpr, tpr
