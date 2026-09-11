#!/usr/bin/env python3

# PKG NAME - TESTS
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com


def test1():
    from python_pkg_template import character_factory, battle

    characters = character_factory("data/characters.csv")
    winner = battle(characters[0], characters[1], health=10000)
    assert winner.name == "Shadowheart"
