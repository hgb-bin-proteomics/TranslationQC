# LLM Judge

<img src="https://github.com/hgb-bin-proteomics/llm-judge/raw/master/docs/icons/icon_padded.png" class="dark-light" align="left" width="200px" style="padding: 5px 5px 5px 5px;"/>

**Rating natural language translations easily via proprietary and open-weights Large Language Models (LLMs).**

Sets up different LLM providers - including
[OpenAI](https://developers.openai.com/api/docs/models),
[Anthropic](https://platform.claude.com/docs/en/models/overview),
[Google](https://ai.google.dev/gemini-api/docs/models),
and [Ollama](https://ollama.com/search) for local LLMs - for quality estimation of natural language translations. Returns both error annotations
as well as a continuous score between `[0, 1]` (higher is better) which indicates the quality of the translation. Prompting is
based on the [MetricX-25 publication](https://doi.org/10.18653/v1/2025.wmt-1.70).

Can be used as a standalone tool or as a python
library with a convenient API, please refer to our documentation. Easily configurable via a convenient
[TOML](https://toml.io/)-based configuration file. Returns
[Pydantic](https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/#pydantic.BaseModel)
`BaseModel`-based class instances for easy
integration into production environments.

## Installation

- Add this to your [uv](https://docs.astral.sh/uv/) project with:
  ```bash
  uv add git+https://github.com/hgb-bin-proteomics/llm-judge.git
  ```
- Or install with pip:
  ```bash
  pip install git+https://github.com/hgb-bin-proteomics/llm-judge.git
  ```

## Usage with Python

- Checkout the example Jupyter notebook at `notebooks/llm-judge-usage.ipynb`.
- Checkout the python API documentation at https://hgb-bin-proteomics.github.io/llm-judge/

### Quick Start

- Import package:
  ```python
  from llm_judge import Judge
  ```
- Create a `Judge` instance with an LLM provider (here we use only [Ollama](https://ollama.com/)):
  ```python
  judge = Judge(
      openai=False,
      anthropic=False,
      google=False,
      ollama=True,
      config="config/judge_config.toml",
  )
  ```
- Rate one translation:
  ```python
  src = "The mitochondria is the powerhouse of the cell."
  mt = "Das Mitochondrium ist das Kraftwerk der Zelle."
  src_lang = "English"
  mt_lang = "German"
  jr = judge.score(src, mt, src_lang, mt_lang)
  ```
- Get the quality estimation score for the translation:
  ```python
  jr.ollama.score
  ```
- Close all connections and clients when you are done:
  ```python
  judge.close()
  ```

## Usage as a Standalone Tool

- You can use `llm-judge` as a standalone by installing via pip (see [Installation](#installation)).
- Alternatively, you can run `llm-judge` via [uvx](https://docs.astral.sh/uv/reference/cli/#uv-tool-run):
  ```bash
  uvx --from git+https://github.com/hgb-bin-proteomics/llm-judge.git llm-judge -h
  ```

### Usage Options

```text
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
```

### Example Usage

- natively:
  ```bash
  llm-judge -i data/test.csv -o data/test_annotated.csv --ollama
  ```
- with [uv](https://docs.astral.sh/uv/):
  ```bash
  uv run llm-judge -i data/test.csv -o data/test_annotated.csv --ollama
  ```

### Example Output

```text
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
INFO:llm_judge._judge:Successfully closed all connections and clients for this instance!
INFO:llm_judge._main:Successfully scored and annotated all translations!
```

## License

- [MIT](https://github.com/michabirklbauer/python-pkg_template/blob/master/LICENSE)

## Contact

- [micha.birklbauer@fh-hagenberg.at](mailto:micha.birklbauer@fh-hagenberg.at)
