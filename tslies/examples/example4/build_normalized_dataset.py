"""
Build a normalized dataset from real ACD weeks.

Every channel is normalized with a rolling mean and standard deviation over a centred window
(:func:`tslies.benchmark.rolling_normalize`), within each data segment: a gap in MET longer than
60 s starts a new segment. The residuals ``z = (y - mean) / std`` then behave almost like
independent N(0, 1) noise, which makes it possible to test the trigger statistics in a controlled
way, with anomalies of known strength added on top (see ``trigger_validation.ipynb``).

Output, in ``--out``:

- ``pk/wNNN.pk``: ``datetime``, ``MET``, the 15 channels, ``<channel>_mean`` and ``<channel>_std``,
  and the GOES X-ray fluxes, useful to recognise solar flares;
- ``normalized.json``: the parameters used.

Usage::

    python build_normalized_dataset.py --source <folder containing pk/wNNN.pk> --weeks 817 818
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from tslies.benchmark import rolling_normalize

FACES = ["top", "Xpos", "Xneg", "Ypos", "Yneg"]
BANDS = ["low", "middle", "high"]
Y_COLS = [f"{face}_{band}" for face in FACES for band in BANDS]
GAP_SECONDS = 60
SUPPORT_COLS = ["GOES_XRSA_HARD", "GOES_XRSB_SOFT"]


def normalize_week(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Keep time and channels of one week and add their rolling mean and standard deviation."""
    resets = (df["MET"].diff() > GAP_SECONDS).to_numpy()
    mean, std = rolling_normalize(df, Y_COLS, window, resets=resets)
    support = [col for col in SUPPORT_COLS if col in df.columns]
    return (df[["datetime", "MET"] + Y_COLS + support]
            .join(mean.add_suffix("_mean")).join(std.add_suffix("_std")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="folder containing pk/wNNN.pk")
    parser.add_argument("--weeks", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/normalized"))
    parser.add_argument("--window", type=int, default=600, help="rolling window, in samples (1 s each)")
    args = parser.parse_args()

    (args.out / "pk").mkdir(parents=True, exist_ok=True)
    for week in args.weeks:
        out = normalize_week(pd.read_pickle(args.source / "pk" / f"w{week}.pk"), args.window)
        out.to_pickle(args.out / "pk" / f"w{week}.pk")
        print(f"w{week}: {len(out)} samples")
    params = {"source": str(args.source), "weeks": args.weeks, "window": args.window,
              "gap_seconds": GAP_SECONDS, "y_cols": Y_COLS, "support_cols": SUPPORT_COLS}
    (args.out / "normalized.json").write_text(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
