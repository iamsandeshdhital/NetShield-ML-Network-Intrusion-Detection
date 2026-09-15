"""End-to-end NetShield-ML pipeline.

Order of operations, matching the project workflow:

    load -> clean -> binary target -> train/test split -> preprocess
         -> feature selection -> train three models -> evaluate
         -> feature importance -> artefacts

Two evaluation splits are reported.  The *held-out split* is a stratified 20%
slice of ``KDDTrain+`` and answers "how well does this detector work on the
kind of traffic it was trained on".  The *official KDDTest+ split* contains
attack families absent from training and answers the harder and more
operationally honest question, "how well does it work on attacks nobody has
shown it before".  Quoting only the first number is the standard way this
dataset gets oversold.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config, data_loader, evaluate, models, preprocess, visualize
from src.feature_selection import FeatureSelector
from src.preprocess import Preprocessor

logger = logging.getLogger(__name__)

HELD_OUT = "Held-out split (KDDTrain+ 20%)"
OFFICIAL = "Official KDDTest+ split"


def configure_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _banner(text: str) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("  %s", text)
    logger.info("=" * 78)


def run(skip_cv: bool = False, download: bool = True) -> dict[str, Any]:
    """Execute the whole study and write every artefact to ``results/``."""
    started = time.perf_counter()
    summary: dict[str, Any] = {}

    # ---------------------------------------------------------------- data
    _banner("1/7  Loading the NSL-KDD dataset")
    train_raw, test_raw = data_loader.load_dataset(download=download)
    summary["raw_train_records"] = int(len(train_raw))
    summary["raw_test_records"] = int(len(test_raw))
    summary["raw_columns"] = int(train_raw.shape[1] - 1)  # minus attack_category

    composition = pd.concat([
        data_loader.dataset_summary(train_raw, "KDDTrain+"),
        data_loader.dataset_summary(test_raw, "KDDTest+"),
    ], ignore_index=True)
    composition.to_csv(config.TABLES_DIR / "01_dataset_composition.csv", index=False)

    # ------------------------------------------------------------ cleaning
    _banner("2/7  Cleaning and target construction")
    train_clean = preprocess.add_binary_target(preprocess.clean_data(train_raw))
    test_clean = preprocess.add_binary_target(
        preprocess.clean_data(test_raw, drop_duplicates=False)
    )
    summary["clean_train_records"] = int(len(train_clean))
    summary["clean_test_records"] = int(len(test_clean))
    summary["train_attack_share"] = round(float(train_clean[config.TARGET_COLUMN].mean()), 4)
    summary["test_attack_share"] = round(float(test_clean[config.TARGET_COLUMN].mean()), 4)

    X_all, y_all = preprocess.split_features_target(train_clean)
    categories_all = train_clean["attack_category"]

    # -------------------------------------------------------------- splits
    _banner("3/7  Train / test split")
    X_train, X_holdout, y_train, y_holdout, cat_train, cat_holdout = train_test_split(
        X_all, y_all, categories_all,
        test_size=config.TEST_SIZE,
        stratify=y_all,
        random_state=config.RANDOM_STATE,
    )
    logger.info("Training rows: %d | held-out rows: %d", len(X_train), len(X_holdout))
    summary["train_rows"] = int(len(X_train))
    summary["holdout_rows"] = int(len(X_holdout))

    X_official, y_official = preprocess.split_features_target(test_clean)
    cat_official = test_clean["attack_category"]

    # -------------------------------------------------------- preprocessing
    _banner("4/7  Preprocessing")
    preprocessor = Preprocessor().fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_holdout_p = preprocessor.transform(X_holdout)
    X_official_p = preprocessor.transform(X_official)
    summary["encoded_features"] = int(X_train_p.shape[1])

    # ---------------------------------------------------- feature selection
    _banner("5/7  Feature selection")
    selector = FeatureSelector().fit(X_train_p, y_train)
    X_train_s = selector.transform(X_train_p)
    X_holdout_s = selector.transform(X_holdout_p)
    X_official_s = selector.transform(X_official_p)

    selector.report().to_csv(
        config.TABLES_DIR / "02_feature_selection.csv", index=False
    )
    summary["dropped_constant"] = len(selector.dropped_constant_)
    summary["dropped_correlated"] = len(selector.dropped_correlated_)
    summary["selected_features"] = len(selector.selected_features_)
    summary["constant_feature_names"] = selector.dropped_constant_
    summary["correlated_feature_names"] = selector.dropped_correlated_

    # -------------------------------------------------- training + scoring
    _banner("6/7  Training and evaluation")
    results: list[evaluate.EvaluationResult] = []
    cv_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, model in models.build_all_models().items():
        logger.info("--- %s ---", name)
        model, train_seconds = evaluate.train_model(model, X_train_s, y_train)
        fitted[name] = model

        results.append(evaluate.evaluate_model(
            model, X_holdout_s, y_holdout, name, HELD_OUT, train_seconds
        ))
        results.append(evaluate.evaluate_model(
            model, X_official_s, y_official, name, OFFICIAL, train_seconds
        ))

        if not skip_cv:
            mean_f1, std_f1 = evaluate.cross_validate_model(model, X_train_s, y_train)
            logger.info("%s | %d-fold CV F1 = %.4f +/- %.4f",
                        name, config.CV_FOLDS, mean_f1, std_f1)
            cv_rows.append({
                "Model": name,
                "CV folds": config.CV_FOLDS,
                "CV F1 mean": round(mean_f1, 4),
                "CV F1 std": round(std_f1, 4),
            })

        joblib.dump(model, config.MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    metrics = evaluate.metrics_table(results)
    metrics.to_csv(config.TABLES_DIR / "03_model_metrics.csv", index=False)

    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(
            config.TABLES_DIR / "04_cross_validation.csv", index=False
        )

    per_category = pd.concat(
        [
            evaluate.per_category_recall(
                r, cat_holdout if r.split_name == HELD_OUT else cat_official
            )
            for r in results
        ],
        ignore_index=True,
    )
    per_category.to_csv(config.TABLES_DIR / "05_per_category_recall.csv", index=False)

    report_lines = []
    for result in results:
        report_lines.append(f"{result.model_name} - {result.split_name}")
        report_lines.append("-" * 60)
        report_lines.append(evaluate.classification_text_report(result))
        report_lines.append("")
    (config.TABLES_DIR / "06_classification_reports.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    # -------------------------------------------------- feature importance
    forest = fitted["Random Forest"]
    importances = pd.Series(
        forest.feature_importances_, index=X_train_s.columns
    ).sort_values(ascending=False)
    importances.rename("importance").to_frame().to_csv(
        config.TABLES_DIR / "07_feature_importance.csv", index_label="feature"
    )
    summary["oob_score"] = round(float(getattr(forest, "oob_score_", float("nan"))), 4)
    summary["top_features"] = [
        {"feature": f, "importance": round(float(v), 6)}
        for f, v in importances.head(config.TOP_N_IMPORTANCES).items()
    ]

    # ------------------------------------------------------------- figures
    _banner("7/7  Figures")
    figures = [
        visualize.plot_class_distribution(train_raw, test_raw),
        visualize.plot_model_comparison(metrics, HELD_OUT, "02_model_comparison_holdout"),
        visualize.plot_model_comparison(metrics, OFFICIAL, "03_model_comparison_official"),
        visualize.plot_mutual_information(selector.mi_scores_),
        visualize.plot_feature_importance(importances),
        visualize.plot_confusion_matrices(results, HELD_OUT, "06_confusion_holdout"),
        visualize.plot_confusion_matrices(results, OFFICIAL, "07_confusion_official"),
        visualize.plot_roc_curves(results, HELD_OUT, "08_roc_holdout"),
        visualize.plot_roc_curves(results, OFFICIAL, "09_roc_official"),
        visualize.plot_category_detection(per_category, OFFICIAL, "10_category_detection_official"),
    ]
    summary["figures"] = [p.name for p in figures]

    # ------------------------------------------------------------ artefacts
    summary["metrics"] = metrics.to_dict(orient="records")
    summary["cross_validation"] = cv_rows
    summary["per_category"] = per_category.to_dict(orient="records")
    summary["mutual_information_top15"] = [
        {"feature": f, "mi": round(float(v), 6)}
        for f, v in selector.mi_scores_.head(15).items()
    ]
    summary["runtime_seconds"] = round(time.perf_counter() - started, 1)

    joblib.dump(
        {"preprocessor": preprocessor, "selector": selector},
        config.MODELS_DIR / "preprocessing.joblib",
    )
    (config.RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    _banner(f"Done in {summary['runtime_seconds']}s - artefacts in {config.RESULTS_DIR}")
    print()
    print(metrics.drop(columns=["Train (s)", "Predict (s)"]).to_string(index=False))
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NetShield-ML: compare classical classifiers for network "
                    "intrusion detection on NSL-KDD.",
    )
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip stratified k-fold cross-validation (much faster)")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of downloading missing raw data")
    parser.add_argument("--quiet", action="store_true", help="warnings only")
    args = parser.parse_args()

    configure_logging(verbose=not args.quiet)
    np.random.seed(config.RANDOM_STATE)
    run(skip_cv=args.skip_cv, download=not args.no_download)


if __name__ == "__main__":
    main()
