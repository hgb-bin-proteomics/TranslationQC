#!/usr/bin/env python3

# PGK FILE DESCRIPTION
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

r"""An example python package with a script that battles two characters.

.. code-block:: text
   :caption: Example Usage

   usage: battle [-h] -f FILE [-c1 CHARACTER_1] [-c2 CHARACTER_2] [-hp HEALTH] [--version]

   Battles two characters.

   options:
     -h, --help            show this help message and exit
     -f, --file FILE       character file to read characters from (str).
     -c1, --character-1 CHARACTER_1
                           index of the first character to use (int).
     -c2, --character-2 CHARACTER_2
                           index of the second character to use (int).
     -hp, --hit-points HEALTH
                           health of all characters (int).
     --version             show program's version number and exit

   (c) Micha Birklbauer, 2026

Examples
--------
>>> from python_pkg_template import character_factory, battle
>>> characters = character_factory("data/characters.csv")
>>> winner = battle(characters[0], characters[1], health=10000)
>>> winner.name
'Shadowheart'
"""

__all__ = ["main", "Character", "character_factory", "battle"]
__version__ = "0.1.0"
__author__ = "Your Name"

from ._main import main
from ._character import Character
from ._util import character_factory
from ._util import battle
