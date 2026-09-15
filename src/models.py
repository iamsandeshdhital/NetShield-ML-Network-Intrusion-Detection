"""The three classifiers compared in this study.

Scope note: the brief for this project restricts the comparison to classical
supervised models.  No deep-learning baseline is included -- on tabular flow
records of this size the tree ensembles are already at the practical ceiling,
and a neural model would add training cost and opacity without a corresponding
gain in detection rate.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import config

logger = logging.getLogger(__name__)


def build_logistic_regression() -> LogisticRegression:
    """Linear baseline.

    ``class_weight='balanced'`` keeps the decision boundary from drifting
    towards the slightly larger normal class, and the raised iteration cap is
    needed because the standardised byte-count features still have long tails.
    """
    # ``penalty`` is left at its default: scikit-learn 1.8 deprecated passing
    # it explicitly, and the default is the L2 objective this baseline wants.
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )


def build_decision_tree() -> DecisionTreeClassifier:
    """Single interpretable tree.

    Depth and leaf-size limits are deliberate: an unconstrained tree memorises
    the training split almost perfectly and transfers poorly to unseen attack
    families, which is precisely the failure mode this project measures.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    """Bagged ensemble; also the source of the feature-importance ranking."""
    return RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "Decision Tree": build_decision_tree,
    "Random Forest": build_random_forest,
}


def build_all_models() -> dict[str, BaseEstimator]:
    """Instantiate every model in the comparison, in reporting order."""
    models = {name: builder() for name, builder in MODEL_BUILDERS.items()}
    logger.info("Built %d models: %s", len(models), ", ".join(models))
    return models
