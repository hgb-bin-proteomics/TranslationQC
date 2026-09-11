#!/usr/bin/env python3

# PKG NAME - TESTS
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

import pytest


def test1():
    from python_pkg_template import Character

    character = Character(name="John Baldur")
    assert character["name"] == "John Baldur"


def test2():
    from python_pkg_template import Character

    character = Character(name="John Baldur")
    new_character = character.copy_with_update(update={"race": "Human"})
    assert new_character.race == "Human"


def test3():
    from python_pkg_template import Character

    character = Character(name="John Baldur")
    assert character.attack() == pytest.approx(0.0)
