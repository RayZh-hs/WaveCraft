#!/usr/bin/env python3
"""Phase IV: mine a transparent probabilistic building grammar."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_grammar_miner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2_structures"))
    parser.add_argument("--phase3-dir", type=Path, default=Path("datasets/phase3_quality"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase4_grammar"))
    args = parser.parse_args()

    grammar = run_grammar_miner(args.phase2_dir, args.phase3_dir, args.phase1_dir, args.output_dir)
    print(f"Wrote grammar from {grammar['training_record_count']} weighted structures to {args.output_dir / 'grammar.json'}")


if __name__ == "__main__":
    main()
