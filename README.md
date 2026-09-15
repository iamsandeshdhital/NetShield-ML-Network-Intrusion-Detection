# NetShield-ML — Network Intrusion Detection

Comparison of three classical supervised classifiers for detecting malicious
network traffic on the **NSL-KDD** benchmark, with an analysis of which flow
features actually drive the decision.

**Authors:** Anurag Jha, Sandesh Dhital

---

## What this project does

A network connection record is described by 41 features — duration, protocol,
service, byte counts, error rates, and per-host traffic statistics. The task is
to decide whether the record is normal traffic or an intrusion.

The project trains **Logistic Regression**, a **Decision Tree** and a **Random
Forest** on the same preprocessed feature matrix, scores them on identical
splits, and ranks the features by the Random Forest's impurity-decrease
importance. No deep learning is used: on tabular flow records of this size the
tree ensembles are already at the practical ceiling, and a neural model would
add training cost and opacity without a matching gain in detection rate.

## Pipeline

```
Network dataset (NSL-KDD)
        │
        ▼
Data cleaning ─────────────  drop metadata, duplicates, fix errata
        │
        ▼
Binary target ─────────────  normal = 0, any attack = 1
        │
        ▼
Train / test split ────────  stratified 80 / 20 on KDDTrain+
        │
        ▼
Preprocessing ─────────────  one-hot encode 3 nominal columns,
        │                    standardise continuous columns
        ▼
Feature selection ─────────  constant filter → correlation pruning
        │                    → mutual-information ranking
        ▼
   ┌────────────┬──────────────┬────────────────┐
   ▼            ▼              ▼
Logistic     Decision       Random
Regression   Tree           Forest
   └────────────┴──────────────┘
                │
                ▼
        Model evaluation ──  accuracy, precision, recall, F1,
                │            ROC-AUC, confusion matrix, 5-fold CV
                ▼
      Feature importance ──  Random Forest top-10 bar chart
```

## Two evaluation splits

Results are reported on **both** of these, and the difference between them is
the most informative thing in the study:

| Split | What it is | What it measures |
|---|---|---|
| Held-out split | stratified 20% of `KDDTrain+` | accuracy on traffic like the training data |
| Official `KDDTest+` | the dataset's own test file | generalisation to **attack families never seen in training** |

Papers that quote only the first number make NSL-KDD look solved. It is not.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                   # full study (downloads the data on first run)
python main.py --skip-cv         # faster: skips 5-fold cross-validation
```

The raw dataset is fetched automatically into `data/raw/` on the first run. To
work offline, place `KDDTrain+.txt` and `KDDTest+.txt` there yourself and pass
`--no-download`.

## Outputs

Everything is written under `results/`:

```
results/
├── figures/
│   ├── 01_class_distribution.png
│   ├── 02_model_comparison_holdout.png
│   ├── 03_model_comparison_official.png
│   ├── 04_mutual_information_top15.png
│   ├── 05_feature_importance_top10.png     ← the feature-importance chart
│   ├── 06_confusion_holdout.png
│   ├── 07_confusion_official.png
│   ├── 08_roc_holdout.png
│   ├── 09_roc_official.png
│   └── 10_category_detection_official.png
├── tables/
│   ├── 01_dataset_composition.csv
│   ├── 02_feature_selection.csv
│   ├── 03_model_metrics.csv                ← the headline comparison table
│   ├── 04_cross_validation.csv
│   ├── 05_per_category_recall.csv
│   ├── 06_classification_reports.txt
│   └── 07_feature_importance.csv
├── models/                                  (gitignored .joblib files)
└── run_summary.json                         every number in one file
```

## Repository layout

```
netshield-ml/
├── main.py                  CLI entry point
├── requirements.txt
├── src/
│   ├── config.py            paths, schema, taxonomy, hyper-parameters
│   ├── data_loader.py       download + load NSL-KDD
│   ├── preprocess.py        cleaning, target, encoding, scaling
│   ├── feature_selection.py three-stage selector
│   ├── models.py            the three classifiers
│   ├── evaluate.py          metrics, CV, per-family error analysis
│   ├── visualize.py         all figures
│   └── pipeline.py          orchestration
├── tests/                   pytest suite (no dataset required)
└── docs/                    full written report
```

## Tests

```bash
pytest
```

The suite runs on synthetic frames that follow the NSL-KDD schema, so it needs
no downloaded data and finishes in seconds.

## Documentation

The full write-up — dataset description, preprocessing rationale, results,
feature-importance discussion, limitations — is in
[`docs/NetShield-ML-Documentation.md`](docs/NetShield-ML-Documentation.md).

## Dataset

NSL-KDD, the cleaned redistribution of KDD Cup '99 produced by the Canadian
Institute for Cybersecurity at the University of New Brunswick. The duplicate
records that dominated the original corpus have been removed, which is what
makes accuracy on it meaningful.

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed
analysis of the KDD CUP 99 data set.* IEEE Symposium on Computational
Intelligence for Security and Defense Applications.
