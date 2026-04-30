#!/usr/bin/env python3
"""Phase II: parse normalized arrays into architectural structure records."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_structure_parser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase2_structures"))
    parser.add_argument(
        "--originals-only",
        action="store_true",
        help="Parse only unaugmented Phase I arrays. By default augmentations are included as extra training views.",
    )
    args = parser.parse_args()

    summary = run_structure_parser(args.phase1_dir, args.output_dir, args.originals_only)
    print(f"Wrote {summary['record_count']} structure records to {summary['output']}")


if __name__ == "__main__":
    main()
