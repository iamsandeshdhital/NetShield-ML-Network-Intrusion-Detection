"""Three-stage feature selection for the NSL-KDD feature matrix.

Stage 1  Constant filter      - a feature that never varies carries no signal.
Stage 2  Correlation pruning  - of any near-duplicate pair keep one, so that
                                linear models are not destabilised and tree
                                importances are not split across twins.
Stage 3  Mutual-information ranking - keep the k features that share the most
                                information with the detection target.  MI is
                                used rather than an ANOVA F-test because it
                                also captures non-linear dependence, which
                                matters for the heavily skewed byte-count and
                                rate features in this dataset.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def find_constant_features(features: pd.DataFrame) -> list[str]:
    """Columns with a single distinct value across the whole training split."""
    return [col for col in features.columns if features[col].nunique(dropna=False) <= 1]


def find_correlated_features(
    features: pd.DataFrame, threshold: float = config.CORRELATION_THRESHOLD
) -> list[str]:
    """Second member of every feature pair whose |Pearson r| exceeds ``threshold``."""
    corr = features.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    return [col for col in upper.columns if (upper[col] >= threshold).any()]


def rank_by_mutual_information(
    features: pd.DataFrame,
    target: pd.Series,
    sample_size: int = 20_000,
    random_state: int = config.RANDOM_STATE,
) -> pd.Series:
    """Mutual information between each feature and the target, best first.

    MI with a k-nearest-neighbour estimator is O(n log n) per feature, so on a
    100k-row matrix it is estimated on a stratified subsample.  The ranking is
    stable well below the full sample size; only the absolute nats change.
    """
    if len(features) > sample_size:
        features, _, target, _ = train_test_split(
            features, target,
            train_size=sample_size,
            stratify=target,
            random_state=random_state,
        )
        logger.info("Estimating mutual information on a %d-row stratified sample",
                    sample_size)

    scores = mutual_info_classif(features, target, random_state=random_state)
    return pd.Series(scores, index=features.columns).sort_values(ascending=False)


@dataclass
class FeatureSelector:
    """Fit the three stages on training data, then subset any other split."""

    n_features: int = config.N_SELECTED_FEATURES
    correlation_threshold: float = config.CORRELATION_THRESHOLD
    selected_features_: list[str] = field(default_factory=list)
    dropped_constant_: list[str] = field(default_factory=list)
    dropped_correlated_: list[str] = field(default_factory=list)
    mi_scores_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "FeatureSelector":
        working = features

        self.dropped_constant_ = find_constant_features(working)
        working = working.drop(columns=self.dropped_constant_)
        logger.info("Stage 1 - dropped %d constant features", len(self.dropped_constant_))

        self.dropped_correlated_ = find_correlated_features(
            working, self.correlation_threshold
        )
        working = working.drop(columns=self.dropped_correlated_)
        logger.info("Stage 2 - dropped %d features correlated above |r| >= %.2f",
                    len(self.dropped_correlated_), self.correlation_threshold)

        self.mi_scores_ = rank_by_mutual_information(working, target)
        keep = min(self.n_features, len(self.mi_scores_))
        self.selected_features_ = list(self.mi_scores_.head(keep).index)
        logger.info("Stage 3 - kept the top %d of %d surviving features by mutual information",
                    keep, len(self.mi_scores_))

        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector.transform called before fit")
        return features[self.selected_features_]

    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(features, target).transform(features)

    def report(self) -> pd.DataFrame:
        """Tabular view of the ranking, for the results appendix."""
        return pd.DataFrame({
            "feature": self.mi_scores_.index,
            "mutual_information": self.mi_scores_.to_numpy().round(6),
            "selected": [f in self.selected_features_ for f in self.mi_scores_.index],
        })
