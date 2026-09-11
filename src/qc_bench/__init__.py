#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

r"""desc.

.. code-block:: text
   :caption: Example Usage



Examples
--------
>>>
"""

__all__ = [
    "main",
    "Judge",
    "JudgeResult",
    "JudgeModelResult",
    "TranslationError",
    "QualityEstimation",
    "annotate_csv",
]
__version__ = "0.1.0"
__author__ = "Your Name"

from ._main import main
from ._judge import Judge, JudgeResult, JudgeModelResult
from ._translation import TranslationError, QualityEstimation
from ._util import annotate_csv
