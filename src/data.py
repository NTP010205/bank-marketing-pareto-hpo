"""Load, validate, audit, and split the UCI Bank Marketing dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import resolve_project_path


EXPECTED_SHAPE = (45_211, 17)
EXPECTED_COLUMNS = (
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
)
TARGET_COLUMN = "y"
LEAKAGE_COLUMN = "duration"
VALID_LABELS = {"yes", "no"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the required bank-full.csv is absent."""


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    test_indices: np.ndarray


def missing_file_message(path: Path) -> str:
    return (
        f"Required dataset not found: {path}\n"
        "Download it from https://archive.ics.uci.edu/dataset/222/bank+marketing "
        "(DOI: 10.24432/C5K306), then extract bank+marketing.zip -> bank.zip -> "
        "bank-full.csv and place that exact file at data/raw/bank-full.csv. "
        "Do not substitute bank.csv or bank-additional-full.csv."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bank_data(path: str | Path, *, validate_shape: bool = True) -> pd.DataFrame:
    data_path = resolve_project_path(path)
    if not data_path.is_file():
        raise DatasetNotFoundError(missing_file_message(data_path))
    frame = pd.read_csv(data_path, sep=";", encoding="utf-8")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected schema. Expected columns in order: " + ", ".join(EXPECTED_COLUMNS)
        )
    if validate_shape and frame.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {frame.shape}; expected {EXPECTED_SHAPE}.")
    labels = set(frame[TARGET_COLUMN].dropna().unique())
    if labels != VALID_LABELS:
        raise ValueError(f"Target labels must be exactly {sorted(VALID_LABELS)}; got {sorted(labels)}.")
    return frame


def audit_dataframe(frame: pd.DataFrame, *, file_path: Path | None = None) -> dict[str, Any]:
    unknown_by_column = {
        column: int(frame[column].eq("unknown").sum())
        for column in frame.columns
        if frame[column].dtype == "object"
    }
    audit: dict[str, Any] = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "target_counts": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().items()},
        "target_proportions": {
            str(k): float(v) for k, v in frame[TARGET_COLUMN].value_counts(normalize=True).items()
        },
        "missing_total": int(frame.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "unknown_total": int(sum(unknown_by_column.values())),
        "unknown_by_column": unknown_by_column,
        "duplicate_rows": int(frame.duplicated().sum()),
        "excluded_from_features": [LEAKAGE_COLUMN],
    }
    if file_path is not None:
        audit["file"] = str(file_path)
        audit["sha256"] = sha256_file(file_path)
        audit["file_size_bytes"] = file_path.stat().st_size
    return audit


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-safe X and binary y; duration is removed before splitting."""
    X = frame.drop(columns=[TARGET_COLUMN, LEAKAGE_COLUMN]).copy()
    y = frame[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Target contains values outside {'yes', 'no'}.")
    return X, y.astype("int8")


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    positions = np.arange(len(X), dtype=np.int64)
    train_pos, test_pos = train_test_split(
        positions, test_size=test_size, random_state=random_state, stratify=y
    )
    return DataSplit(
        X_train=X.iloc[train_pos].copy(),
        X_test=X.iloc[test_pos].copy(),
        y_train=y.iloc[train_pos].copy(),
        y_test=y.iloc[test_pos].copy(),
        train_indices=train_pos,
        test_indices=test_pos,
    )


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit UCI Bank Marketing bank-full.csv.")
    parser.add_argument("--data", default="data/raw/bank-full.csv", help="Dataset path (project-relative by default).")
    parser.add_argument("--output", default="results/data_audit.json", help="Audit JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    data_path = resolve_project_path(args.data)
    output_path = resolve_project_path(args.output)
    try:
        frame = load_bank_data(data_path)
    except (DatasetNotFoundError, ValueError) as exc:
        print(exc)
        return 2
    audit = audit_dataframe(frame, file_path=data_path)
    write_json(audit, output_path)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Audit written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
