#!/usr/bin/env python3

# PKG NAME - TESTS
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com


def test1():
    from python_pkg_template import character_factory

    characters = character_factory("data/characters.csv")
    assert characters[0].name == "Astarion"
