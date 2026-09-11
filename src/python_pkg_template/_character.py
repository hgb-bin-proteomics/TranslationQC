#!/usr/bin/env python3

# PGK FILE DESCRIPTION
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com

from __future__ import annotations

import random
import logging
from pydantic import BaseModel, Field, ConfigDict, computed_field

from typing import Annotated, Optional, Literal, Any, override

logger = logging.getLogger(__name__)


class Character(BaseModel):
    r"""Core data structure representing a character.

    Bases Pydantic `BaseModel <https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/#pydantic.BaseModel>`_.

    Attributes Summary
    ------------------
    Here is a short summary about the class attributes, for more details
    on the specific Pydantic validation requirements please refer to the corresponding attributes
    themselves.

    Required
    ^^^^^^^^
    The following attributes are required:

    name : str
        The name of the character.

    Optional
    ^^^^^^^^
    The following attributes are optional:

    race : one of "Elf", "Half-Elf", "Human", or None, default = None
        The race of the character. Should be one of Elf, Half-Elf, or Human.
    min_damage : float, default = 0.0
        Minimum damage the character deals.
    max_damage : float, default = 0.0
        Maximum damage the character deals.

    Notes
    -----
    Minimum and maximum damage are automatically switched depending on which is
    greater.

    Examples
    --------
    >>> from python_pkg_template import Character
    >>> character = Character(name="John Baldur")
    """

    name: Annotated[str, Field(frozen=True, description="Name of the character.")]
    r"""
    Name of the character.
    """
    race: Annotated[
        Optional[Literal["Elf", "Half-Elf", "Human"]],
        Field(frozen=True, description="Race of the character."),
    ] = None
    r"""
    Race of the character. Should be one of Elf, Half-Elf, or Human.
    """
    min_damage: Annotated[
        float, Field(frozen=False, description="Minimum damage the character deals.")
    ] = 0.0
    r"""
    Minimum damage the character deals. Is automatically switched with max_damage
    if max_damage is smaller.
    """
    max_damage: Annotated[
        float, Field(frozen=False, description="Maximum damage the character deals.")
    ] = 0.0
    r"""
    Maximum damage the character deals. Is automatically switched with min_damage
    if min_damage is greater.
    """

    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """

    @computed_field(description="Average damage dealt by the character.")
    @property
    def avg_damage(self) -> float:
        r"""
        Average damage dealt by the character.
        """
        return (self.min_damage + self.max_damage) / 2.0

    @override
    def model_post_init(self, context: Any = None) -> None:
        r"""
        Performs extra validation and post init functions.

        Warnings
        --------
        This method should not be called manually!
        """
        if self.min_damage > self.max_damage:
            self.__dict__["min_damage"], self.__dict__["max_damage"] = (  # pyright: ignore[reportIndexIssue]
                self.max_damage,
                self.min_damage,
            )

    def __getitem__(self, key: str) -> Any:
        r"""
        Support for dict-like access.
        """
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(f"'{key}' is not a valid field!") from e

    def __contains__(self, key: str) -> bool:
        r"""
        Support for ``in`` operator.
        """
        return hasattr(self, key)

    def copy_with_update(self, update: dict[str, Any] = {}) -> Character:
        r"""Creates a deep copy of the class with optional attribute updates.

        Parameters
        ----------
        update : dict of str, any, default = empty dict
            Dictionary mapping attribute names (str) to their updated values.
            The default (empty dict) will create a deep copy with the original
            attribute values.

        Returns
        -------
        Character
            New character with optionally updated attributes.

        Examples
        --------
        >>> from python_pkg_template import Character
        >>> character = Character(name="John Baldur")
        >>> new_character = character.copy_with_update(update={"race": "Human"})
        """
        return Character(
            name=update["name"] if "name" in update else self.name,
            race=update["race"] if "race" in update else self.race,
            min_damage=update["min_damage"]
            if "min_damage" in update
            else self.min_damage,
            max_damage=update["max_damage"]
            if "max_damage" in update
            else self.max_damage,
        )

    def attack(self) -> float:
        r"""Get the attack damage of the next attack.

        Returns
        -------
        float
            The attack damage of the attack.

        Examples
        --------
        >>> from python_pkg_template import Character
        >>> character = Character(name="John Baldur")
        >>> character.attack()
        0.0
        """
        return self.min_damage + (self.max_damage - self.min_damage) * random.random()  # noqa: S311
