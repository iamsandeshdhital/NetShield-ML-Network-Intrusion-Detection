"""Cleaning, encoding and scaling for the NSL-KDD feature matrix.

The transformer is fitted on the training split only and then applied to every
other split.  Fitting on the full corpus would leak test-set statistics into
the scaler and the one-hot vocabulary and quietly inflate the reported scores.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_data(frame: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Remove metadata columns, exact duplicates and impossible values."""
    cleaned = frame.drop(columns=config.METADATA_COLUMNS, errors="ignore").copy()

    missing = int(cleaned.isna().sum().sum())
    if missing:
        logger.warning("Found %d missing cells; filling numeric with median", missing)
        numeric = cleaned.select_dtypes(include=np.number).columns
        cleaned[numeric] = cleaned[numeric].fillna(cleaned[numeric].median())
        cleaned = cleaned.dropna()

    if drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        removed = before - len(cleaned)
        if removed:
            logger.info("Dropped %d duplicate records (%.2f%%)",
                        removed, removed / before * 100)

    # ``su_attempted`` is documented as binary but the raw data contains a
    # stray value of 2; the dataset errata treat it as "attempted".
    if "su_attempted" in cleaned.columns:
        cleaned["su_attempted"] = cleaned["su_attempted"].replace(2, 1)

    return cleaned


def add_binary_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the 0/1 detection target from the fine-grained attack label."""
    out = frame.copy()
    out[config.TARGET_COLUMN] = (
        out[config.LABEL_COLUMN].str.lower() != "normal"
    ).astype(int)
    return out


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model inputs from the binary detection target."""
    drop_cols = [config.LABEL_COLUMN, config.TARGET_COLUMN, "attack_category"]
    features = frame.drop(columns=drop_cols, errors="ignore")
    target = frame[config.TARGET_COLUMN]
    return features, target


# --------------------------------------------------------------------------
# Encoding + scaling
# --------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """One-hot encodes the three nominal columns and standardises the rest."""

    categorical_columns: list[str] = field(
        default_factory=lambda: list(config.CATEGORICAL_COLUMNS)
    )
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names_: list[str] = field(default_factory=list)
    scaled_columns_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # -- internal ---------------------------------------------------------
    def _encode(self, features: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.categorical_columns if c in features.columns]
        return pd.get_dummies(features, columns=present, prefix=present, dtype=np.uint8)

    def _align(self, encoded: pd.DataFrame) -> pd.DataFrame:
        """Force an encoded frame onto the training column layout."""
        return encoded.reindex(columns=self.feature_names_, fill_value=0)

    # -- public API -------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "Preprocessor":
        encoded = self._encode(features)
        self.feature_names_ = list(encoded.columns)

        one_hot_prefixes = tuple(f"{c}_" for c in self.categorical_columns)
        self.scaled_columns_ = [
            col for col in encoded.columns
            if not col.startswith(one_hot_prefixes)
            and col not in config.BINARY_COLUMNS
        ]
        self.scaler.fit(encoded[self.scaled_columns_].astype(np.float64))
        self.fitted_ = True

        logger.info("Preprocessor fitted: %d features (%d standardised, %d one-hot/binary)",
                    len(self.feature_names_), len(self.scaled_columns_),
                    len(self.feature_names_) - len(self.scaled_columns_))
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Preprocessor.transform called before fit")

        encoded = self._align(self._encode(features)).astype(np.float64)
        encoded[self.scaled_columns_] = self.scaler.transform(
            encoded[self.scaled_columns_]
        )
        return encoded

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
