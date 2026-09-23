# TranslationQC

<img src="https://github.com/hgb-bin-proteomics/TranslationQC/raw/master/docs/icons/icon_padded.png" class="dark-light" align="left" width="200px" style="padding: 5px 5px 5px 5px;"/>

Rating natural language translations easily via proprietary and open-weights LLMs. Sets up different LLM providers - including
OpenAI, Anthropic, Google, and Ollama for local LLMs - for quality estimation of translations. Returns both error annotations
as well as a continuous score between `[0, 1]` (higher is better) which indicates the quality of the translation. Prompting is
based on the [MetricX-25 publication](https://doi.org/10.18653/v1/2025.wmt-1.70). Can be used as a standalone tool or as a python
library with a convenient API, please refer to our documentation. Returns pydantic `BaseModel`-based class instances for easy
integration into production environments.

## Usage

### Usage with python

- Add this to your uv project with:
  ```bash
  uv add git+https://github.com/hgb-bin-proteomics/TranslationQC.git
  ```
- Or install with pip:
  ```bash
  pip install git+https://github.com/hgb-bin-proteomics/TranslationQC.git
  ```
- Checkout the example Jupyter notebook at `notebooks/qc-bench-usage.ipynb`.
- Checkout the python API documentation at https://hgb-bin-proteomics.github.io/TranslationQC/

### Usage Options

```text
usage: qc-bench [-h] -i INPUT -o OUTPUT [-c CONFIG] [--openai] [--anthropic] [--google] [--ollama] [--ollama-model OLLAMA_MODEL] [--version]

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
  qc-bench -i data/test.csv -o data/test_annotated.csv --ollama
  ```
- with uv:
  ```bash
  uv run qc-bench -i data/test.csv -o data/test_annotated.csv --ollama
  ```

### Example Output

```text
INFO:qc_bench._main:Using OpenAI model: False
INFO:qc_bench._main:Using Anthropic model: False
INFO:qc_bench._main:Using Google model: False
INFO:qc_bench._main:Using Ollama model: True
INFO:qc_bench._judge:The following LLM-providers are enabled for this instance: Ollama!
INFO:qc_bench._judge:Loaded the following configuration:
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
INFO:qc_bench._main:Selected Ollama model: gemma4:e4b
INFO:qc_bench._util:Reading file data/test.csv...
INFO:qc_bench._util:Successfully read file data/test.csv!
Annotating data/test.csv...:   0%|                                                                                             | 0/1 [00:00<?, ?it/s]INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
INFO:qc_bench._judge:Successfully got a valid response after retry 0 for one query.
Annotating data/test.csv...: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:20<00:00, 20.08s/it]
INFO:qc_bench._util:Finished annotation of data/test.csv!
INFO:qc_bench._util:Writing files to disk...
INFO:qc_bench._util:Successfully wrote file data/test_annotated.csv!
INFO:qc_bench._util:Successfully wrote file data/test_annotated.csv.json!
shape: (1, 8)
┌──────────────────────────────────┬──────────────────┬──────────┬─────────┬──────────────┬─────────────────┬──────────────┬─────────────────────────┐
│ src                              ┆ mt               ┆ src_lang ┆ mt_lang ┆ score_openai ┆ score_anthropic ┆ score_google ┆ score_ollama_gemma4:e4b │
│ ---                              ┆ ---              ┆ ---      ┆ ---     ┆ ---          ┆ ---             ┆ ---          ┆ ---                     │
│ str                              ┆ str              ┆ str      ┆ str     ┆ f64          ┆ f64             ┆ f64          ┆ f64                     │
╞══════════════════════════════════╪══════════════════╪══════════╪═════════╪══════════════╪═════════════════╪══════════════╪═════════════════════════╡
│ The lights are dimmable, but I…  ┆ Die Lichter sind ┆ English  ┆ German  ┆ NaN          ┆ NaN             ┆ NaN          ┆ 0.95                    │
│                                  ┆ dimmbar, aber…   ┆          ┆         ┆              ┆                 ┆              ┆                         │
└──────────────────────────────────┴──────────────────┴──────────┴─────────┴──────────────┴─────────────────┴──────────────┴─────────────────────────┘
INFO:qc_bench._main:Successfully scored and annotated all translations!
```

## License

- [MIT](https://github.com/michabirklbauer/python-pkg_template/blob/master/LICENSE)

## Contact

- [your.mail@mail.com](mailto:your.mail@mail.com)
