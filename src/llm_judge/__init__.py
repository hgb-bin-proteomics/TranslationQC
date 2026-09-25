#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

r"""Performs quality estimation for a list of translations using LLMs from OpenAI, Anthropic, Google, and Ollama.

.. code-block:: text
   :caption: Usage Options

    usage: llm-judge [-h] -i INPUT -o OUTPUT [-c CONFIG] [--openai] [--anthropic] [--google] [--ollama] [--ollama-model OLLAMA_MODEL] [--version]

    Quality estimation for a list of translations using LLMs.

    options:
      -h, --help            show this help message and exit
      -i, --input INPUT     path/name of the translations '.csv' file containing the columns 'src', 'mt', 'src_lang', and 'mt_lang' (str).
      -o, --output OUTPUT   path/name of the ouput file that should be written to disk (str).
      -c, --config CONFIG   path/name of the configuration file in TOML format (str).
      --openai              use OpenAI model.
      --anthropic           use Anthropic model.
      --google              use Google model.
      --ollama              use Ollama model.
      --ollama-model OLLAMA_MODEL
                            which Ollama model to use, must be a valid Ollama model identifier (str).
      --version             show program's version number and exit

    (c) Micha Birklbauer, 2026

.. code-block:: bash
   :caption: Example Usage

    llm-judge -i data/test.csv -o data/test_annotated.csv --ollama

.. code-block:: text
   :caption: Example Output

    INFO:llm_judge._main:Using OpenAI model: False
    INFO:llm_judge._main:Using Anthropic model: False
    INFO:llm_judge._main:Using Google model: False
    INFO:llm_judge._main:Using Ollama model: True
    INFO:llm_judge._judge:The following LLM-providers are enabled for this instance: Ollama!
    INFO:llm_judge._judge:Loaded the following configuration:
    -------------------- JudgeConfig --------------------
    OpenAI Model:               gpt-5.4
    OpenAI Thinking Level:      low
    Anthropic Model:            claude-opus-5
    Anthropic Thinking Level:   low
    Google Model:               gemini-3.1-pro-preview
    Google Thinking Level:      low
    Ollama Host:                http://localhost:11434
    Ollama Model:               gemma4:e4b
    Ollama Keep Alive Duration: -1
    Maximum Output Tokens:      2048
    Maximum Retries:            5
    Retry Waiting Time:         30.0
    Seeds:                      1337, 10081995, 18041970, 1071966, 3082023, 24042025
    -----------------------------------------------------
    INFO:llm_judge._main:Selected Ollama model: gemma4:e4b
    INFO:llm_judge._util:Reading file data/test.csv...
    INFO:llm_judge._util:Successfully read file data/test.csv!
    Annotating data/test.csv...:   0%|                                                                                     | 0/1 [00:00<?, ?it/s]
    INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    INFO:llm_judge._judge:Successfully got a valid response after retry 0 for one query.
    Annotating data/test.csv...: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:16<00:00, 16.74s/it]
    INFO:llm_judge._util:Finished annotation of data/test.csv!
    INFO:llm_judge._util:Writing files to disk...
    INFO:llm_judge._util:Successfully wrote file data/test_annotated.csv!
    INFO:llm_judge._util:Successfully wrote file data/test_annotated.csv.json!
    shape: (1, 8)
    ┌──────────────────────────────────┬──────────────────┬──────────┬─────────┬──────────────┬─────────────────┬──────────────┬─────────────────────────┐
    │ src                              ┆ mt               ┆ src_lang ┆ mt_lang ┆ score_openai ┆ score_anthropic ┆ score_google ┆ score_ollama_gemma4:e4b │
    │ ---                              ┆ ---              ┆ ---      ┆ ---     ┆ ---          ┆ ---             ┆ ---          ┆ ---                     │
    │ str                              ┆ str              ┆ str      ┆ str     ┆ f64          ┆ f64             ┆ f64          ┆ f64                     │
    ╞══════════════════════════════════╪══════════════════╪══════════╪═════════╪══════════════╪═════════════════╪══════════════╪═════════════════════════╡
    │ The lights are dimmable, but I…  ┆ Die Lichter sind ┆ English  ┆ German  ┆ NaN          ┆ NaN             ┆ NaN          ┆ 1.0                     │
    │                                  ┆ dimmbar, aber…   ┆          ┆         ┆              ┆                 ┆              ┆                         │
    └──────────────────────────────────┴──────────────────┴──────────┴─────────┴──────────────┴─────────────────┴──────────────┴─────────────────────────┘
    INFO:llm_judge._main:Successfully scored and annotated all translations!

Examples
--------
>>> from llm_judge import Judge
>>> judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
>>> jr = judge.score(
...     src="The mitochondria is the powerhouse of the cell.",
...     mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
...     src_lang="English",
...     mt_lang="German",
... )
>>> type(jr)
<class 'llm_judge._judge.JudgeResult'>
>>> jr.openai is None
True
>>> jr.anthropic is None
True
>>> jr.google is None
True
>>> jr.ollama is None
False
>>> type(jr.ollama)
<class 'llm_judge._judge.JudgeModelResult'>
>>> jr.ollama.model
'mistral:7b'
>>> jr.ollama.score
0.95

>>> from llm_judge import Judge
>>> judge = Judge(
...     openai=False,
...     anthropic=False,
...     google=False,
...     ollama=True,
...     config="config/judge_config.toml",
... )
>>> jr = judge.score(
...     src="The mitochondria is the powerhouse of the cell.",
...     mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
...     src_lang="English",
...     mt_lang="German",
... )
>>> type(jr)
<class 'llm_judge._judge.JudgeResult'>
>>> jr.openai is None
True
>>> jr.anthropic is None
True
>>> jr.google is None
True
>>> jr.ollama is None
False
>>> type(jr.ollama)
<class 'llm_judge._judge.JudgeModelResult'>
>>> jr.ollama.model
'gemma4:e4b'
>>> jr.ollama.score
1.0
"""

__all__ = [
    "main",
    "Judge",
    "JudgeResult",
    "JudgeModelResult",
    "JudgeConfig",
    "TranslationError",
    "QualityEstimation",
    "annotate_csv",
]
__version__ = "0.2.0"
__author__ = "Micha Johannes Birklbauer"

from ._main import main
from ._judge import Judge, JudgeResult, JudgeModelResult, JudgeConfig
from ._translation import TranslationError, QualityEstimation
from ._util import annotate_csv
