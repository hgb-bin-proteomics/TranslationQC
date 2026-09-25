#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import pytest


def test1():
    from llm_judge import JudgeConfig

    with pytest.raises(ValueError, match="Parameter 'seeds' must at least length"):
        _ = JudgeConfig(max_retry=5, seeds=[1, 2, 3])


def test2():
    from llm_judge import JudgeConfig

    jc = JudgeConfig.model_validate_toml("config/judge_config.toml")

    assert jc.openai_model == "gpt-6-astra"
    assert jc.openai_thinking_level == "medium"
    assert jc.anthropic_model == "claude-opus-5-5"
    assert jc.anthropic_thinking_level == "medium"
    assert jc.google_model == "gemini-3.8-flash"
    assert jc.google_thinking_level == "medium"
    assert jc.ollama_host == "http://localhost:11434"
    assert jc.ollama_model == "qwen3.8:27b"
    assert jc.ollama_keep_alive == "5m"
    assert jc.max_output_tokens == 4096
    assert jc.max_retry == 2
    assert jc.retry_wait_time == pytest.approx(10.0)
    assert jc.seeds == [41467, 26, 77407, 1]


def test3():
    from llm_judge import JudgeConfig

    jc = JudgeConfig.model_validate_toml("config/judge_config.toml")
    assert jc.openai_model == "gpt-6-astra"
