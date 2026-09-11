#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

r"""Performs quality estimation for a list of translations using LLMs from OpenAI, Anthropic, Google, and Ollama.

.. code-block:: text
   :caption: Example Usage

    usage: qc-bench [-h] -i INPUT -o OUTPUT [--openai] [--anthropic] [--google] [--ollama] [--version]

    Quality estimation for a list of translations using LLMs.

    options:
      -h, --help           show this help message and exit
      -i, --input INPUT    path/name of the translations '.csv' file containing the columns 'src', 'mt', 'src_lang', and 'mt_lang' (str).
      -o, --output OUTPUT  path/name of the ouput file that should be written to disk (str).
      --openai             use OpenAI model.
      --anthropic          use Anthropic model.
      --google             use Google model.
      --ollama             use Ollama model.
      --version            show program's version number and exit

    (c) Micha Birklbauer, 2026


Examples
--------
>>> from qc_bench import Judge
>>> judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
>>> jr = judge.score(
...     src="The mitochondria is the powerhouse of the cell.",
...     mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
...     src_lang="English",
...     mt_lang="German",
... )
>>> type(jr)
<class 'qc_bench._judge.JudgeResult'>
>>> jr.openai is None
True
>>> jr.anthropic is None
True
>>> jr.google is None
True
>>> jr.ollama is None
False
>>> type(jr.ollama)
<class 'qc_bench._judge.JudgeModelResult'>
>>> jr.ollama.model
'mistral:7b'
>>> jr.ollama.score
0.95
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
__author__ = "Micha Johannes Birklbauer"

from ._main import main
from ._judge import Judge, JudgeResult, JudgeModelResult
from ._translation import TranslationError, QualityEstimation
from ._util import annotate_csv
