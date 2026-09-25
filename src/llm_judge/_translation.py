#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

from pydantic import BaseModel, Field

from typing import Literal


class TranslationError(BaseModel):
    error_type: Literal[
        "accuracy",
        "fluency",
        "style",
        "terminology",
        "non-translation",
        "other",
        "no-error",
    ] = Field(
        description=(
            "The type of translation error: "
            "One of 'accuracy', 'fluency', 'style', 'terminology', 'non-translation', 'other', or 'no-error'."
        )
    )
    error_subtype: Literal[
        "addition",
        "mistranslation",
        "omission",
        "untranslated text",
        "character encoding",
        "grammar",
        "inconsistency",
        "punctuation",
        "register",
        "spelling",
        "awkward",
        "inappropriate for context",
        "inconsistent use",
    ] = Field(
        description=(
            "The error subtype which can be for 'accuracy': 'addition', 'mistranslation', 'omission', 'untranslated text'; "
            "for 'fluency': 'character encoding', 'grammar', 'inconsistency', 'punctuation', 'register', 'spelling'; "
            "for 'style': 'awkward'; for 'terminology': 'inappropriate for context', 'inconsistent use'."
        )
    )
    error_severity: Literal["critical", "major", "minor"] = Field(
        description="Error severity which is one of 'critical', 'major', and 'minor'."
    )
    description: str = Field(
        description="A detailed description of the translation error and how to improve the translation."
    )
    location: str = Field(
        description="Where in the machine translation the error is located given as a substring."
    )


class QualityEstimation(BaseModel):
    source: str = Field(description="The source text.")
    machine_translation: str = Field(description="The machine translation text.")
    quality_estimation_value: float = Field(
        description=(
            "The quality estimation of the translation given as a value between 0 and 1 "
            "where 0 is complete gibberish and 1 would be a perfect translation."
        )
    )
    errors: list[TranslationError] = Field(
        description="List of all translation errors."
    )
