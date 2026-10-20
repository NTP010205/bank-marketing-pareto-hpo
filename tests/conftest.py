from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS


@pytest.fixture
def synthetic_bank_frame() -> pd.DataFrame:
    """Synthetic data is test-only and must never be reported as a UCI result."""
    rng = np.random.default_rng(7)
    rows = 90
    data = {
        "age": rng.integers(18, 75, rows),
        "job": rng.choice(["admin.", "technician", "services"], rows),
        "marital": rng.choice(["single", "married", "divorced"], rows),
        "education": rng.choice(["primary", "secondary", "tertiary"], rows),
        "default": rng.choice(["no", "yes"], rows, p=[0.95, 0.05]),
        "balance": rng.integers(-500, 10_000, rows),
        "housing": rng.choice(["no", "yes"], rows),
        "loan": rng.choice(["no", "yes"], rows),
        "contact": rng.choice(["cellular", "telephone", "unknown"], rows),
        "day": rng.integers(1, 32, rows),
        "month": rng.choice(["jan", "feb", "mar", "apr"], rows),
        "duration": rng.integers(1, 1_500, rows),
        "campaign": rng.integers(1, 10, rows),
        "pdays": rng.choice([-1, 1, 30, 90], rows),
        "previous": rng.integers(0, 5, rows),
        "poutcome": rng.choice(["unknown", "failure", "success"], rows),
        "y": np.array(["yes"] * 18 + ["no"] * 72),
    }
    frame = pd.DataFrame(data)
    return frame.loc[:, EXPECTED_COLUMNS]
