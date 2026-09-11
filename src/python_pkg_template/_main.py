#!/usr/bin/env python3

# PGK FILE DESCRIPTION
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

from __future__ import annotations

import argparse
import logging

from . import __version__
from ._util import character_factory
from ._util import battle

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
    >>> from python_pkg_template import main
    >>> main(["-f", "data/characters.csv"])
    INFO:main:Both characters have 130.0 hit points! The battle begins:
    INFO:main:Character Shadowheart has initiative!
    INFO:main:Character Shadowheart deals 311.13673321167755 damage!
    INFO:main:Character Shadowheart won!
    0
    """

    parser = argparse.ArgumentParser(
        prog="battle",
        description="Battles two characters.",
        epilog="(c) Micha Birklbauer, 2026",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        required=True,
        help="character file to read characters from (str).",
        type=str,
    )
    parser.add_argument(
        "-c1",
        "--character-1",
        dest="character_1",
        default=0,
        help="index of the first character to use (int).",
        type=int,
    )
    parser.add_argument(
        "-c2",
        "--character-2",
        dest="character_2",
        default=1,
        help="index of the second character to use (int).",
        type=int,
    )
    parser.add_argument(
        "-hp",
        "--hit-points",
        dest="health",
        default=130,
        help="health of all characters (int).",
        type=int,
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        characters = character_factory(args.file)
        character_1 = int(args.character_1)
        character_2 = int(args.character_2)
        health = float(args.health)
        logger.info(f"Both characters have {health} hit points! The battle begins:")
        if character_1 < 0 or character_1 >= len(characters):
            raise IndexError("Character 1 is not a valid index in the character file!")
        if character_2 < 0 or character_2 >= len(characters):
            raise IndexError("Character 1 is not a valid index in the character file!")
        _ = battle(characters[character_1], characters[character_2], health)
    except Exception as _e:
        logger.exception("An error occurred while running the script!")
        return 1

    return 0
