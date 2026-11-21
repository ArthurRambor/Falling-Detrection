import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("sisfall", Path(__file__).resolve().parents[1] / "data" / "sisfall_loader.py")
sis = importlib.util.module_from_spec(spec); spec.loader.exec_module(sis)


def test_parse_and_window_synthetic_sisfall_file(tmp_path):
    n = 200 * 15                                           # 15 s at 200 Hz
    raw = np.zeros((n, 9)); raw[:, 2] = 256                # 1 g on z: 256 * 32/8192 = 1 g
    raw[1500, 2] = 900                                     # impact of about 3.5 g
    f = tmp_path / "F05_SE03_R01.txt"
    f.write_text("\n".join(",".join(str(int(v)) for v in r) + ";" for r in raw))
    acc = sis.read_file(f)
    assert abs(np.linalg.norm(acc[10]) - 1.0) < 1e-6
    w = sis.window_at_peak(acc)
    assert w.shape == (150, 3) and np.linalg.norm(w[50]).round(2) > 0.8
    ds = sis.load_dataset(tmp_path)
    assert ds["labels"].tolist() == [1] and ds["groups"].tolist() == ["elderly"]
