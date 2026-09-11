#!/usr/bin/env python3

# PGK FILE DESCRIPTION
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

from __future__ import annotations

import argparse
import logging

from . import __version__
from ._judge import Judge
from ._util import annotate_csv

from typing import Optional

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    """Main function.

    Parameters
    ----------
    argv : list or str, or None, default = None
        Arguments passed to argparse.

    Returns
    -------
    int
        Exit status (zero is success).

    Examples
    --------
    >>>
    """

    parser = argparse.ArgumentParser(
        prog="qc_bench",
        description="Scores a list of translations.",
        epilog="(c) Micha Birklbauer, 2026",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="input file (str).",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="output file (str).",
        type=str,
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        judge = Judge()
        df, _result = annotate_csv(
            input_file=args.input, output_file=args.output, judge=judge
        )
        print(df)
        logger.info("Successfully scored and annotated all translations!")
    except Exception as _e:
        logger.exception("An error occurred while running the script!")
        return 1

    return 0
