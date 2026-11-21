import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallwatch.edge import fall_model as fm            # noqa: E402
from fallwatch.sim.fall_data import build_dataset      # noqa: E402


@pytest.fixture(scope="session")
def small_ds():
    return build_dataset(n_subjects=30, falls_per_subject=10, adl_per_subject=100, seed=3)


@pytest.fixture(scope="session")
def bundle(small_ds):
    return fm.train(small_ds, seed=3)
