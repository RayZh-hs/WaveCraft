#!/usr/bin/env python3
"""Phase VI: plan and assemble a rectangular medieval house."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_structure_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-dir", type=Path, default=Path("datasets/phase4_grammar"))
    parser.add_argument("--phase5-dir", type=Path, default=Path("datasets/phase5_modules"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase6_generated"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-version", type=int, default=3700)
    args = parser.parse_args()

    report = run_structure_generator(args.phase4_dir, args.phase5_dir, args.output_dir, args.seed, args.data_version)
    print(f"Wrote generated house to {report['export']['output']}")


if __name__ == "__main__":
    main()
