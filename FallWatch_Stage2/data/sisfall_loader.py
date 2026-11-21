"""
data/sisfall_loader.py  (optional, Arthur)
------------------------------------------
Runs the same feature set and subject-wise evaluation on the public SisFall
dataset (Sucerquia et al., 2017) instead of the simulator.

    1. Download SisFall and unzip it, e.g. to ~/SisFall_dataset/ (folders SA01..SE15).
    2. python data/sisfall_loader.py ~/SisFall_dataset

What the loader does
  * reads the ADXL345 columns (first three) of each .txt file; the sensor is
    +-16 g, 13 bit, so g = raw * 32 / 2**13
  * averages every 4 samples to go from 200 Hz to the 50 Hz of the prototype
  * takes the window from 1 s before to 2 s after the largest peak
  * labels F** files as falls, D** files as daily activities; SE** subjects are
    the elderly group, SA** the young adults

HONEST STATUS: not run on the real files during development (the dataset was not
available in the build environment). The parsing and windowing are tested on a
synthetic file written in the SisFall text format (tests/test_sisfall_loader.py).
Check one real file with --peek before trusting the numbers. SisFall's sensor is
worn at the waist, not the wrist, so scores are not comparable with the simulator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallwatch.config import FS, PEAK_IDX, POST_N, PRE_N, WINDOW_N     # noqa: E402

SRC_FS = 200
SCALE = 32.0 / 2 ** 13


def read_file(path) -> np.ndarray:
    """Return acceleration (n, 3) in g at 50 Hz."""
    rows = []
    for line in Path(path).read_text().replace(";", "").splitlines():
        parts = [p for p in line.strip().split(",") if p.strip() != ""]
        if len(parts) >= 3:
            rows.append([float(x) for x in parts[:3]])
    raw = np.array(rows) * SCALE
    k = SRC_FS // FS
    n = (len(raw) // k) * k
    return raw[:n].reshape(-1, k, 3).mean(axis=1)


def window_at_peak(acc: np.ndarray):
    mag = np.linalg.norm(acc, axis=1)
    p = int(np.argmax(mag))
    if p - PRE_N < 0 or p + POST_N > len(acc):
        pad = np.pad(acc, ((PRE_N, POST_N), (0, 0)), mode="edge")
        p += PRE_N
        acc = pad
    w = acc[p - PRE_N:p + POST_N]
    assert w.shape == (WINDOW_N, 3) and PEAK_IDX == PRE_N
    return w


def load_dataset(root) -> dict:
    windows, labels, sids, types, groups = [], [], [], [], []
    sid_map = {}
    for f in sorted(Path(root).rglob("*.txt")):
        parts = f.stem.split("_")                       # D05_SE03_R01
        if len(parts) != 3 or parts[0][0] not in "DF":
            continue
        act, subj = parts[0], parts[1]
        sid = sid_map.setdefault(subj, len(sid_map))
        windows.append(window_at_peak(read_file(f)))
        labels.append(int(act.startswith("F")))
        sids.append(sid); types.append(act); groups.append("elderly" if subj.startswith("SE") else "young")
    return {"windows": np.stack(windows), "labels": np.array(labels), "subject_ids": np.array(sids),
            "event_types": np.array(types), "groups": np.array(groups), "n_subjects": len(sid_map)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--peek", action="store_true", help="print one parsed file and exit")
    a = ap.parse_args()
    if a.peek:
        f = next(Path(a.root).rglob("F*_S*_R*.txt"))
        acc = read_file(f)
        print(f, acc.shape, "peak g:", np.linalg.norm(acc, axis=1).max().round(2))
        return
    from fallwatch.edge.fall_model import triggered_mask
    from fallwatch.evaluation.fall_eval import oof_probabilities, summarize
    ds = load_dataset(a.root)
    print(f"{len(ds['labels'])} recordings, {ds['labels'].sum()} falls, {ds['n_subjects']} subjects")
    from fallwatch.edge.features import FEATURE_NAMES
    p = oof_probabilities(ds, FEATURE_NAMES)
    m = summarize(p >= 0.5, ds, n_boot=300)
    print(f"subject-wise sensitivity {m['sensitivity']:.1%}  specificity {m['specificity']:.1%}")
    el = ds["groups"] == "elderly"
    for name, mask in (("elderly", el), ("young", ~el)):
        y, pr = ds["labels"][mask], (p >= 0.5)[mask]
        print(f"  {name}: sensitivity {(pr & (y == 1)).sum() / max((y == 1).sum(), 1):.1%}, "
              f"specificity {(~pr & (y == 0)).sum() / max((y == 0).sum(), 1):.1%}")


if __name__ == "__main__":
    main()
