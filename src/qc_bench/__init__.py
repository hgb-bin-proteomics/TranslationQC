#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

r"""Performs quality estimation for a list of translations using LLMs from OpenAI, Anthropic, Google, and Ollama.

.. code-block:: text
   :caption: Usage Options

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

.. code-block:: bash
   :caption: Example Usage

    qc-bench -i data/test.csv -o data/test_annotated.csv --ollama

.. code-block:: text
   :caption: Example Output

    INFO:qc_bench._main:Using OpenAI model: False
    INFO:qc_bench._main:Using Anthropic model: False
    INFO:qc_bench._main:Using Google model: False
    INFO:qc_bench._main:Using Ollama model: True
    INFO:qc_bench._main:Selected Ollama model: gemma4:e4b
    INFO:qc_bench._util:Reading file data\test.csv...
    INFO:qc_bench._util:Successfully read file data\test.csv!
    Annotating data\test.csv...:   0%|                                                                                     | 0/1 [00:00<?, ?it/s]
    INFO:httpx:HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
    INFO:qc_bench._judge:Successfully got a valid response after retry 0 for one query.
    Annotating data\test.csv...: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.20s/it]
    INFO:qc_bench._util:Finished annotation of data\test.csv!
    INFO:qc_bench._util:Writing files to disk...
    INFO:qc_bench._util:Successfully wrote file data\test_annotated.csv!
    INFO:qc_bench._util:Successfully wrote file data\test_annotated.csv.json!
    shape: (1, 8)
    ┌──────────────────────────────────┬──────────────────┬──────────┬─────────┬──────────────┬─────────────────┬──────────────┬─────────────────────────┐
    │ src                              ┆ mt               ┆ src_lang ┆ mt_lang ┆ score_openai ┆ score_anthropic ┆ score_google ┆ score_ollama_gemma4:e4b │
    │ ---                              ┆ ---              ┆ ---      ┆ ---     ┆ ---          ┆ ---             ┆ ---          ┆ ---                     │
    │ str                              ┆ str              ┆ str      ┆ str     ┆ f64          ┆ f64             ┆ f64          ┆ f64                     │
    ╞══════════════════════════════════╪══════════════════╪══════════╪═════════╪══════════════╪═════════════════╪══════════════╪═════════════════════════╡
    │ The lights are dimmable, but I…  ┆ Die Lichter sind ┆ English  ┆ German  ┆ NaN          ┆ NaN             ┆ NaN          ┆ 1.0                     │
    │                                  ┆ dimmbar, aber…   ┆          ┆         ┆              ┆                 ┆              ┆                         │
    └──────────────────────────────────┴──────────────────┴──────────┴─────────┴──────────────┴─────────────────┴──────────────┴─────────────────────────┘
    INFO:qc_bench._main:Successfully scored and annotated all translations!

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
