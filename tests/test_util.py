#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import pytest


@pytest.mark.localonly
def test1():
    from llm_judge import annotate_csv, Judge

    judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    pl_df, list_jr = annotate_csv("data/test.csv", judge=judge)
    assert pl_df.shape == (1, 8)
    assert len(list_jr) == 1
    assert str(type(list_jr[0])) == "<class 'llm_judge._judge.JudgeResult'>"
    judge.close()
    assert judge.closed
