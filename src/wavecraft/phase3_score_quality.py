#!/usr/bin/env python3
"""Phase III: score parsed structures for grammar and module mining."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_quality_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2_structures"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase3_quality"))
    args = parser.parse_args()

    summary = run_quality_scorer(args.phase2_dir, args.output_dir)
    print(f"Wrote quality scores for {summary['record_count']} structures to {args.output_dir / 'quality_scores.json'}")


if __name__ == "__main__":
    main()
