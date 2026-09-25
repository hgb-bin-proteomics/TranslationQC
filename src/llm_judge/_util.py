#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

from __future__ import annotations

import json
import logging
import polars as pl
from tqdm import tqdm

from ._judge import Judge, JudgeResult

from typing import Optional

logger = logging.getLogger(__name__)


def annotate_csv(
    input_file: str, judge: Judge, output_file: Optional[str] = None
) -> tuple[pl.DataFrame, list[JudgeResult]]:
    r"""Quality estimation for a list of translations in a '.csv' file using LLMs.

    Parameters
    ----------
    input_file : str
        Path/name of the translations '.csv' file containing the columns ``src``, ``mt``, ``src_lang``, and ``mt_lang``.
    judge : Judge
        The judge to be used for quality estimation.
    output_file : str, or None, default = None
        Path/name of the ouput file that should be written to disk. If ``None`` nothing is written to disk.

    Returns
    -------
    tuple of polars.DataFrame and list of JudgeResult
        Results as a ``polars.DataFrame`` and as list of ``JudgeResult``.

    Examples
    --------
    >>> from llm_judge import annotate_csv, Judge
    >>> judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    >>> pl_df, list_jr = annotate_csv("data/test.csv", judge=judge)
    Annotating data/test.csv...: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.96s/it]
    >>> pl_df
    shape: (1, 8)
    ┌──────────────────────────────────┬──────────────────┬──────────┬─────────┬──────────────┬─────────────────┬──────────────┬─────────────────────────┐
    │ src                              ┆ mt               ┆ src_lang ┆ mt_lang ┆ score_openai ┆ score_anthropic ┆ score_google ┆ score_ollama_mistral:7b │
    │ ---                              ┆ ---              ┆ ---      ┆ ---     ┆ ---          ┆ ---             ┆ ---          ┆ ---                     │
    │ str                              ┆ str              ┆ str      ┆ str     ┆ f64          ┆ f64             ┆ f64          ┆ f64                     │
    ╞══════════════════════════════════╪══════════════════╪══════════╪═════════╪══════════════╪═════════════════╪══════════════╪═════════════════════════╡
    │ The lights are dimmable, but I…  ┆ Die Lichter sind ┆ English  ┆ German  ┆ NaN          ┆ NaN             ┆ NaN          ┆ 0.95                    │
    │                                  ┆ dimmbar, aber…   ┆          ┆         ┆              ┆                 ┆              ┆                         │
    └──────────────────────────────────┴──────────────────┴──────────┴─────────┴──────────────┴─────────────────┴──────────────┴─────────────────────────┘
    >>> len(list_jr)
    1
    >>> type(list_jr[0])
    <class 'llm_judge._judge.JudgeResult'>
    >>> judge.close()
    """
    # data collection
    score_openai: list[float] = list()
    score_anthropic: list[float] = list()
    score_google: list[float] = list()
    score_ollama: list[float] = list()
    raw_data: list[JudgeResult] = list()
    json_data: list[str] = list()
    # file reading
    logger.info(f"Reading file {input_file}...")
    df: pl.DataFrame = pl.read_csv(input_file)
    logger.info(f"Successfully read file {input_file}!")
    # annotation
    for row in tqdm(
        df.iter_rows(named=True), total=df.shape[0], desc=f"Annotating {input_file}..."
    ):
        # get JudgeResult
        result = judge.score(
            src=str(row["src"]).strip(),
            mt=str(row["mt"]).strip(),
            src_lang=str(row["src_lang"]).strip(),
            mt_lang=str(row["mt_lang"]).strip(),
        )
        # save a json-able object
        json_data.append(result.model_dump(mode="json"))
        # save as JudgeResult
        raw_data.append(result)
        # save scores
        score_openai.append(
            result.openai.score if result.openai is not None else float("nan")
        )
        score_anthropic.append(
            result.anthropic.score if result.anthropic is not None else float("nan")
        )
        score_google.append(
            result.google.score if result.google is not None else float("nan")
        )
        score_ollama.append(
            result.ollama.score if result.ollama is not None else float("nan")
        )
    # expansion
    df = df.with_columns(
        pl.Series("score_openai", score_openai),
        pl.Series("score_anthropic", score_anthropic),
        pl.Series("score_google", score_google),
        pl.Series(f"score_ollama_{judge.config.ollama_model}", score_ollama),
    )
    logger.info(f"Finished annotation of {input_file}!")
    # saving
    if output_file is not None:
        logger.info("Writing files to disk...")
        df.write_csv(output_file)
        logger.info(f"Successfully wrote file {output_file}!")
        with open(f"{output_file}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f)
        logger.info(f"Successfully wrote file {output_file}.json!")
    return (df, raw_data)
