"""Run the existing development-only model comparison workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import require_data

import development_analysis as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the locally obtained source CSV")
    args = parser.parse_args()
    analysis.RAW = require_data(args.data)
    analysis.main()


if __name__ == "__main__":
    main()
