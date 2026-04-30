#!/usr/bin/env python3
"""Phase V: build a semantic module catalog for the MVP grammar."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavecraft.architecture import run_module_miner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase4-dir", type=Path, default=Path("datasets/phase4_grammar"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase5_modules"))
    args = parser.parse_args()

    catalog = run_module_miner(args.phase1_dir, args.phase4_dir, args.output_dir)
    print(f"Wrote {catalog['module_count']} semantic modules to {args.output_dir / 'module_catalog.json'}")


if __name__ == "__main__":
    main()
