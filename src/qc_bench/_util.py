#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

from __future__ import annotations

import json
import random
import logging
import polars as pl
from tqdm import tqdm

from ._judge import Judge

logger = logging.getLogger(__name__)


def annotate_csv(input_file: str, output_file: str, judge: Judge) -> Tuple[pl.DataFrame, list[JudgeResult]]:
    r"""Creates a list of characters from a file.

    Parameters
    ----------
    input_file : str
        The filename of the input ``csv`` file.

    Returns
    -------

    Examples
    --------
    >>>
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
    for row in tqdm(df.iter_rows(named=True), total=df.shape[0], desc=f"Annotating {input_file}..."):
        result = judge.score(src=str(row["src"]).strip(), mt=str(row["mt"]).strip())
        json_data.append(result.model_dump_json())
        score_openai.append(result.openai.score)
        score_anthropic.append(result.anthropic.score)
        score_google.append(result.google.score)
        score_ollama.append(result.ollama.score)
    # expansion
    df = df.with_columns(
        pl.Series("score_openai", score_openai),
        pl.Series("score_anthropic", score_anthropic),
        pl.Series("score_google", score_google),
        pl.Series(f"score_ollama_{judge.ollama_model_name}", score_ollama),
    )
    logger.info(f"Finished annotation of {input_file}! Writing file to disk...")
    # saving
    df.write_csv(output_file)
    logger.info(f"Successfully wrote file {output_file}!")
    with open(f"{output_file}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    logger.info(f"Successfully wrote file {output_file}.json!")
    return (df, raw_data)
