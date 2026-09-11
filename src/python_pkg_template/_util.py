#!/usr/bin/env python3

# PGK FILE DESCRIPTION
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

from __future__ import annotations

import random
import logging
import polars as pl

from ._character import Character

logger = logging.getLogger(__name__)


def character_factory(filename: str) -> list[Character]:
    r"""Creates a list of characters from a file.

    Parameters
    ----------
    filename : str
        The filename of the character ``csv`` file.

    Returns
    -------
    lisf of Character
        The parsed list of characters.

    Examples
    --------
    >>> from python_pkg_template import character_factory
    >>> characters = character_factory("data/characters.csv")
    >>> characters[0].name
    'Astarion'
    """
    df = pl.read_csv(filename)
    characters: list[Character] = list()
    for row in df.iter_rows(named=True):
        characters.append(
            Character(
                name=str(row["name"]),
                race=str(row["race"]) if "race" in row else None,  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]
                min_damage=float(row["min_damage"]),
                max_damage=float(row["max_damage"]),
            )
        )
    return characters


def battle(
    character_1: Character, character_2: Character, health: float = 100.0
) -> Character:
    r"""Makes two characters fight.

    Parameters
    ----------
    character_1 : Character
        One of the two characters that should battle.
    character_2 : Character
        One of the two characters that should battle.
    health : float, default = 100.0
        The amount of hit points both characters have.

    Returns
    -------
    Character
        The winner of the two characters.

    Examples
    --------
    >>> from python_pkg_template import character_factory, battle
    >>> characters = character_factory("data/characters.csv")
    >>> winner = battle(characters[0], characters[1], health=10000)
    >>> winner.name
    'Shadowheart'
    """
    health_1 = health
    health_2 = health
    initiative = random.random()  # noqa: S311
    if initiative < 0.5:
        logger.info(f"Character {character_1.name} has initiative!")
    else:
        logger.info(f"Character {character_2.name} has initiative!")
    while True:
        if initiative < 0.5:
            attack: float = character_1.attack()
            logger.info(f"Character {character_1.name} deals {attack} damage!")
            health_2 -= attack
            if health_2 <= 0:
                break
            attack: float = character_2.attack()
            logger.info(f"Character {character_2.name} deals {attack} damage!")
            health_1 -= attack
            if health_1 <= 0:
                break
        else:
            attack: float = character_2.attack()
            logger.info(f"Character {character_2.name} deals {attack} damage!")
            health_1 -= attack
            if health_1 <= 0:
                break
            attack: float = character_1.attack()
            logger.info(f"Character {character_1.name} deals {attack} damage!")
            health_2 -= attack
            if health_2 <= 0:
                break
    if health_1 <= 0:
        logger.info(f"Character {character_2.name} won!")
        return character_2
    logger.info(f"Character {character_1.name} won!")
    return character_1
