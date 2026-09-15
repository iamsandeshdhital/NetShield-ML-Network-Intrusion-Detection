"""Unit tests for the NetShield-ML components.

These run on small synthetic frames that follow the NSL-KDD schema, so the
suite is fast and does not need the raw dataset on disk.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, evaluate, models, preprocess
from src.data_loader import add_attack_category
from src.feature_selection import (
    FeatureSelector,
    find_constant_features,
    find_correlated_features,
)
from src.preprocess import Preprocessor

RNG = np.random.default_rng(0)


def make_frame(n_rows: int = 400) -> pd.DataFrame:
    """A synthetic frame with the real column names and plausible values."""
    frame = pd.DataFrame(
        RNG.integers(0, 100, size=(n_rows, len(config.COLUMN_NAMES))),
        columns=config.COLUMN_NAMES,
    )
    frame["protocol_type"] = RNG.choice(["tcp", "udp", "icmp"], n_rows)
    frame["service"] = RNG.choice(["http", "smtp", "ftp_data", "private"], n_rows)
    frame["flag"] = RNG.choice(["SF", "S0", "REJ"], n_rows)
    frame["label"] = RNG.choice(["normal", "neptune", "satan", "guess_passwd"], n_rows)
    frame["difficulty"] = RNG.integers(0, 21, n_rows)
    frame["num_outbound_cmds"] = 0          # constant in the real data too
    frame["su_attempted"] = RNG.choice([0, 1, 2], n_rows)   # real domain
    frame["land"] = RNG.integers(0, 2, n_rows)
    frame["logged_in"] = RNG.integers(0, 2, n_rows)
    frame["root_shell"] = RNG.integers(0, 2, n_rows)
    frame["is_host_login"] = 0
    frame["is_guest_login"] = RNG.integers(0, 2, n_rows)
    return add_attack_category(frame)


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def test_schema_has_expected_width():
    # 41 features + label + difficulty
    assert len(config.COLUMN_NAMES) == 43
    assert config.COLUMN_NAMES[-2:] == ["label", "difficulty"]


def test_every_taxonomy_value_is_known():
    assert set(config.ATTACK_CATEGORIES.values()) == {
        "DoS", "Probe", "R2L", "U2R", "Normal"
    }


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def test_clean_data_drops_metadata_column(frame):
    cleaned = preprocess.clean_data(frame)
    assert "difficulty" not in cleaned.columns


def test_clean_data_removes_duplicates():
    base = make_frame(50)
    doubled = pd.concat([base, base], ignore_index=True)
    cleaned = preprocess.clean_data(doubled)
    assert len(cleaned) == len(preprocess.clean_data(base))


def test_clean_data_normalises_su_attempted(frame):
    frame.loc[frame.index[:10], "su_attempted"] = 2
    cleaned = preprocess.clean_data(frame)
    assert set(cleaned["su_attempted"].unique()) <= {0, 1}


def test_binary_target_matches_label(frame):
    tagged = preprocess.add_binary_target(frame)
    normals = tagged[tagged["label"] == "normal"]
    attacks = tagged[tagged["label"] != "normal"]
    assert (normals[config.TARGET_COLUMN] == 0).all()
    assert (attacks[config.TARGET_COLUMN] == 1).all()


def test_split_features_target_removes_leaky_columns(frame):
    tagged = preprocess.add_binary_target(preprocess.clean_data(frame))
    X, y = preprocess.split_features_target(tagged)
    for leaked in ("label", "attack_category", config.TARGET_COLUMN):
        assert leaked not in X.columns
    assert len(X) == len(y)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def test_preprocessor_one_hot_encodes_nominal_columns(frame):
    tagged = preprocess.add_binary_target(preprocess.clean_data(frame))
    X, _ = preprocess.split_features_target(tagged)
    out = Preprocessor().fit_transform(X)

    for column in config.CATEGORICAL_COLUMNS:
        assert column not in out.columns
        assert any(c.startswith(f"{column}_") for c in out.columns)


def test_preprocessor_standardises_only_continuous_columns(frame):
    tagged = preprocess.add_binary_target(preprocess.clean_data(frame))
    X, _ = preprocess.split_features_target(tagged)
    pre = Preprocessor().fit(X)
    out = pre.transform(X)

    # A standardised column is centred; a one-hot column stays 0/1.
    assert abs(out["duration"].mean()) < 1e-9
    assert set(np.unique(out["protocol_type_tcp"])) <= {0.0, 1.0}
    assert "logged_in" not in pre.scaled_columns_


def test_preprocessor_aligns_unseen_categories():
    """A service present only at test time must not change the column layout."""
    train = make_frame(200)
    test = make_frame(80)
    test.loc[test.index[:5], "service"] = "telnet"   # unseen during fit

    train_t = preprocess.add_binary_target(preprocess.clean_data(train))
    test_t = preprocess.add_binary_target(preprocess.clean_data(test))
    X_train, _ = preprocess.split_features_target(train_t)
    X_test, _ = preprocess.split_features_target(test_t)

    pre = Preprocessor().fit(X_train)
    out = pre.transform(X_test)

    assert list(out.columns) == pre.feature_names_
    assert "service_telnet" not in out.columns


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(pd.DataFrame({"a": [1.0]}))


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
def test_find_constant_features():
    data = pd.DataFrame({"varies": [1, 2, 3], "constant": [7, 7, 7]})
    assert find_constant_features(data) == ["constant"]


def test_find_correlated_features_flags_a_duplicate():
    base = np.arange(50, dtype=float)
    data = pd.DataFrame({"a": base, "b": base * 2.0, "c": RNG.normal(size=50)})
    flagged = find_correlated_features(data, threshold=0.95)
    assert "b" in flagged and "a" not in flagged


def test_selector_keeps_requested_number_of_features(frame):
    tagged = preprocess.add_binary_target(preprocess.clean_data(frame))
    X, y = preprocess.split_features_target(tagged)
    encoded = Preprocessor().fit_transform(X)

    selector = FeatureSelector(n_features=8).fit(encoded, y)
    reduced = selector.transform(encoded)

    assert reduced.shape[1] == 8
    assert list(reduced.columns) == selector.selected_features_
    assert "num_outbound_cmds" in selector.dropped_constant_


def test_selector_report_covers_every_ranked_feature(frame):
    tagged = preprocess.add_binary_target(preprocess.clean_data(frame))
    X, y = preprocess.split_features_target(tagged)
    encoded = Preprocessor().fit_transform(X)

    selector = FeatureSelector(n_features=5).fit(encoded, y)
    report = selector.report()

    assert len(report) == len(selector.mi_scores_)
    assert int(report["selected"].sum()) == 5


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def test_exactly_three_classical_models_are_compared():
    built = models.build_all_models()
    assert list(built) == ["Logistic Regression", "Decision Tree", "Random Forest"]


def test_models_are_deterministic():
    for model in models.build_all_models().values():
        assert model.get_params()["random_state"] == config.RANDOM_STATE


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@pytest.fixture
def trained():
    """A tiny separable problem so the metrics are predictable."""
    X = pd.DataFrame({
        "f1": np.concatenate([RNG.normal(-3, 0.4, 120), RNG.normal(3, 0.4, 120)]),
        "f2": np.concatenate([RNG.normal(0, 0.4, 120), RNG.normal(1, 0.4, 120)]),
    })
    y = pd.Series([0] * 120 + [1] * 120)
    model = models.build_decision_tree()
    model, seconds = evaluate.train_model(model, X, y)
    return model, X, y, seconds


def test_evaluate_model_reports_sane_metrics(trained):
    model, X, y, seconds = trained
    result = evaluate.evaluate_model(model, X, y, "Decision Tree", "self", seconds)

    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == len(y)
    assert 0.9 <= result.accuracy <= 1.0
    assert result.roc_auc is not None and result.roc_auc > 0.9


def test_metrics_row_derives_error_rates(trained):
    model, X, y, seconds = trained
    result = evaluate.evaluate_model(model, X, y, "Decision Tree", "self", seconds)
    row = result.as_row()

    tn, fp, fn, tp = result.confusion.ravel()
    assert row["TN"] + row["FP"] + row["FN"] + row["TP"] == len(y)
    assert row["False Alarm Rate"] == pytest.approx(fp / (fp + tn), abs=1e-4)
    assert row["Miss Rate"] == pytest.approx(fn / (fn + tp), abs=1e-4)


def test_metrics_table_has_one_row_per_result(trained):
    model, X, y, seconds = trained
    results = [
        evaluate.evaluate_model(model, X, y, "Decision Tree", "a", seconds),
        evaluate.evaluate_model(model, X, y, "Decision Tree", "b", seconds),
    ]
    table = evaluate.metrics_table(results)
    assert len(table) == 2
    assert {"Accuracy", "Precision", "Recall", "F1"} <= set(table.columns)


def test_per_category_recall_counts_every_attack(trained):
    model, X, y, seconds = trained
    result = evaluate.evaluate_model(model, X, y, "Decision Tree", "self", seconds)
    categories = pd.Series(["Normal"] * 120 + ["DoS"] * 60 + ["Probe"] * 60)

    breakdown = evaluate.per_category_recall(result, categories)
    attack_rows = breakdown[~breakdown["Category"].str.startswith("Normal")]

    assert set(attack_rows["Category"]) == {"DoS", "Probe"}
    assert int(attack_rows["Records"].sum()) == 120


def test_cross_validation_returns_mean_and_spread(trained):
    model, X, y, _ = trained
    mean, std = evaluate.cross_validate_model(model, X, y, folds=3)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0
