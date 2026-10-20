from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.week4_analysis import (
    PROJECT_ROOT,
    analysis_tables,
    frozen_train_eda_tables,
    load_frozen_training_frame,
    load_sources,
    validate_sources,
)


def test_week4_official_sources_are_consistent() -> None:
    report = validate_sources(load_sources())
    assert report["status"] == "PASS"
    assert report["sqlite_checks"]["random"]["trial_count"] == 40
    assert report["sqlite_checks"]["tpe"]["trial_count"] == 40
    assert report["official_pareto_hash_count"] == 13


def test_week4_eda_materializes_only_frozen_train_without_duration() -> None:
    frame = load_frozen_training_frame()
    summary, numeric, categorical = frozen_train_eda_tables(frame)
    assert len(frame) == 36_168
    assert "duration" not in frame.columns
    assert summary["scope"] == "frozen_training_rows_only"
    assert summary["target_counts"] == {"no": 31_937, "yes": 4_231}
    assert not numeric.empty
    assert not categorical.empty


def test_week4_tables_distinguish_best_and_selected() -> None:
    sources = load_sources()
    tables = analysis_tables(sources)
    key = tables["key_models"].set_index("model")
    assert key.loc["Best Tpe", "config_hash"] == "cd972e72ea0a"
    assert key.loc["Selected TPE", "config_hash"] == "4cdd6dd35f1d"
    assert key.loc["Best Tpe", "cv_ap_mean"] > key.loc["Selected TPE", "cv_ap_mean"]
    assert key.loc["Selected TPE", "cv_max_fold_size_mb"] < key.loc["Best Tpe", "cv_max_fold_size_mb"]
    assert key.loc["Selected TPE", "full_train_size_mb"] == 1.28934


def test_week4_dominance_and_near_best_counts() -> None:
    tables = analysis_tables(load_sources())
    assert int(tables["dominance"]["dominated"].sum()) == 67
    assert tables["dominance"]["config_hash"].nunique() == 70
    assert tables["near_best"]["config_hash"].nunique() == 18
    assert tables["near_best"].iloc[0]["config_hash"] == "4cdd6dd35f1d"


def test_week4_figures_are_png_and_at_least_300_dpi() -> None:
    figure_dir = PROJECT_ROOT / "figures/week4"
    paths = sorted(figure_dir.glob("*.png"))
    assert len(paths) == 7
    for path in paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert dpi[0] >= 299 and dpi[1] >= 299


def test_week4_notebooks_are_valid_json_with_safety_metadata() -> None:
    notebook_dir = PROJECT_ROOT / "notebooks"
    paths = sorted(notebook_dir.glob("0[1-5]_*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["metadata"]["week4_policy"] == {
            "read_only_analysis": True,
            "holdout_used": False,
            "training_performed": False,
            "hpo_performed": False,
        }
        assert any(cell["cell_type"] == "markdown" for cell in payload["cells"])
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])


def test_week4_notebook_execution_report_passes() -> None:
    path = PROJECT_ROOT / "results/analysis/week4/notebook_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["executed"] is True
    assert len(report["notebooks"]) == 5
    assert all(item["status"] == "PASS" for item in report["notebooks"])
    assert report["holdout_used"] is False

