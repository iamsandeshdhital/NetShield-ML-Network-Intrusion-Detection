"""Acquire and load the NSL-KDD network intrusion dataset.

NSL-KDD is the de-facto public benchmark for intrusion detection research.  It
is a cleaned redistribution of KDD Cup '99 in which the duplicate records that
biased the original corpus have been removed, which makes accuracy figures
meaningful rather than dominated by a handful of repeated flows.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def download_if_missing(destination: Path, urls: list[str]) -> Path:
    """Fetch ``destination`` from the first reachable mirror in ``urls``.

    Returns the path untouched when the file already exists, so the pipeline
    is safe to re-run offline once the data has been pulled down once.
    """
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Found existing %s (%.1f MB)",
                    destination.name, destination.stat().st_size / 1e6)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for url in urls:
        try:
            logger.info("Downloading %s from %s", destination.name, url)
            urllib.request.urlretrieve(url, destination)
            logger.info("Saved %s (%.1f MB)",
                        destination.name, destination.stat().st_size / 1e6)
            return destination
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover
            last_error = exc
            logger.warning("Mirror failed (%s): %s", url, exc)

    raise RuntimeError(
        f"Could not download {destination.name} from any mirror. "
        f"Place the file manually in {destination.parent}. Last error: {last_error}"
    )


def ensure_dataset() -> None:
    """Make sure both NSL-KDD split files are present on disk."""
    for filename, urls in config.DATA_SOURCES.items():
        download_if_missing(config.RAW_DIR / filename, urls)


def load_split(path: Path) -> pd.DataFrame:
    """Read one headerless NSL-KDD file into a named DataFrame."""
    frame = pd.read_csv(path, header=None, names=config.COLUMN_NAMES)
    logger.info("Loaded %s -> %d rows x %d columns",
                path.name, len(frame), frame.shape[1])
    return frame


def add_attack_category(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the four-way DoS/Probe/R2L/U2R taxonomy used for reporting."""
    frame = frame.copy()
    frame["attack_category"] = (
        frame[config.LABEL_COLUMN].map(config.ATTACK_CATEGORIES).fillna("Unknown")
    )
    unknown = frame.loc[frame["attack_category"] == "Unknown",
                        config.LABEL_COLUMN].unique()
    if len(unknown):
        logger.warning("Labels missing from the taxonomy: %s", sorted(unknown))
    return frame


def load_dataset(download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the NSL-KDD training and official test splits.

    The official ``KDDTest+`` split deliberately contains attack families that
    never appear in training, so it measures generalisation to novel attacks
    rather than in-distribution accuracy.  Both are returned; the pipeline
    reports on each separately.
    """
    if download:
        ensure_dataset()

    train = add_attack_category(load_split(config.TRAIN_FILE))
    test = add_attack_category(load_split(config.TEST_FILE))
    return train, test


def dataset_summary(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Build a small descriptive table used in the written report."""
    counts = frame["attack_category"].value_counts()
    summary = pd.DataFrame({
        "split": name,
        "category": counts.index,
        "records": counts.to_numpy(),
        "share_pct": (counts / len(frame) * 100).round(2).to_numpy(),
    })
    return summary.reset_index(drop=True)
