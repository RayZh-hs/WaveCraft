#!/usr/bin/env python3
"""Phase VII: add bounded local surface detail to an assembled structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_detail_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-dir", type=Path, default=Path("datasets/phase6_generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase7_detail"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-version", type=int, default=3700)
    parser.add_argument("--variation-rate", type=float, default=0.08)
    parser.add_argument("--lantern-rate", type=float, default=0.03)
    args = parser.parse_args()

    report = run_detail_pass(
        args.phase6_dir,
        args.output_dir,
        args.seed,
        args.data_version,
        args.variation_rate,
        args.lantern_rate,
    )
    print(f"Wrote detailed house to {report['export']['output']}")


if __name__ == "__main__":
    main()
