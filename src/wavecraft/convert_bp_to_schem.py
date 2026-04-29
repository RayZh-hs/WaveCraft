#!/usr/bin/env python3
"""Convert raw .bp structure files into Sponge .schem files.

This intentionally wraps the external `schemconvert` CLI so the conversion step
that created datasets/schem is reproducible from Python.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def convert_file(input_path: Path, output_path: Path, schemconvert: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # schemconvert dispatches by extension. One source file is named .b even
    # though it contains BP data, so give the converter a temporary .bp path.
    if input_path.suffix == ".bp":
        converter_input = input_path
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="wavecraft_bp_")
        converter_input = Path(temp_dir.name) / f"{input_path.stem}.bp"
        shutil.copy2(input_path, converter_input)

    try:
        subprocess.run(
            [schemconvert, str(converter_input), str(output_path)],
            check=True,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/bp"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/schem"))
    parser.add_argument("--schemconvert", default="schemconvert")
    args = parser.parse_args()

    inputs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix in {".bp", ".b"}
    )
    if not inputs:
        raise SystemExit(f"No .bp/.b files found in {args.input_dir}")

    for input_path in inputs:
        output_path = args.output_dir / f"{input_path.stem}.schem"
        convert_file(input_path, output_path, args.schemconvert)
        print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
    main()

