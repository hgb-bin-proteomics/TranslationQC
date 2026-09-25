#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

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
    >>> from llm_judge import main
    >>> main(["-i", "data/test.csv", "-o", "data/test_annotated.csv", "--ollama"])
    """

    parser = argparse.ArgumentParser(
        prog="llm-judge",
        description="Quality estimation for a list of translations using LLMs.",
        epilog="(c) Micha Birklbauer, 2026",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="path/name of the translations '.csv' file containing the columns 'src', 'mt', 'src_lang', and 'mt_lang' (str).",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="path/name of the ouput file that should be written to disk (str).",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        required=False,
        help="path/name of the configuration file in TOML format (str).",
        type=str,
    )
    parser.add_argument(
        "--openai",
        dest="openai",
        action="store_true",
        default=False,
        help="use OpenAI model.",
    )
    parser.add_argument(
        "--anthropic",
        dest="anthropic",
        action="store_true",
        default=False,
        help="use Anthropic model.",
    )
    parser.add_argument(
        "--google",
        dest="google",
        action="store_true",
        default=False,
        help="use Google model.",
    )
    parser.add_argument(
        "--ollama",
        dest="ollama",
        action="store_true",
        default=False,
        help="use Ollama model.",
    )
    parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=None,
        required=False,
        help="which Ollama model to use, must be a valid Ollama model identifier (str).",
        type=str,
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        logger.info(f"Using OpenAI model: {args.openai}")
        logger.info(f"Using Anthropic model: {args.anthropic}")
        logger.info(f"Using Google model: {args.google}")
        logger.info(f"Using Ollama model: {args.ollama}")

        judge = Judge(
            openai=args.openai,
            anthropic=args.anthropic,
            google=args.google,
            ollama=str(args.ollama_model).strip()
            if (args.ollama_model is not None and args.ollama)
            else args.ollama,
            config=str(args.config).strip() if args.config is not None else None,
        )

        if args.ollama:
            logger.info(f"Selected Ollama model: {judge.config.ollama_model}")

        df, _result = annotate_csv(
            input_file=args.input, judge=judge, output_file=args.output
        )
        print(df)

        judge.close()
        logger.info("Successfully scored and annotated all translations!")
    except Exception as _e:
        logger.exception("An error occurred while running the script!")
        return 1

    return 0
