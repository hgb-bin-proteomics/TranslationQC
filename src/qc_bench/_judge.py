#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

from __future__ import annotations

import os
import time
import json
import logging
import tomllib
from openai import OpenAI
from anthropic import Anthropic
from google.genai import Client as Google
from google.genai import types as GoogleTypes  # noqa: N812
from ollama import Client as Ollama
from ollama import ChatResponse as OllamaChatResponse
from ollama import GenerateResponse as OllamaGenerateResponse
from ollama import ResponseError as OllamaResponseError
from ollama import pull as pull_ollama_model
from pydantic import BaseModel, Field, ConfigDict, computed_field

from typing import Optional, Annotated, Any, Literal, override

from ._translation import QualityEstimation
from ._constants import MAX_RETRY, MAX_OUTPUT_TOKENS, RETRY_WAIT_TIME, SEEDS
from ._constants import OPENAI_MODEL, OPENAI_THINKING_LEVEL
from ._constants import ANTHROPIC_MODEL, ANTHROPIC_THINKING_LEVEL
from ._constants import GOOGLE_MODEL, GOOGLE_THINKING_LEVEL
from ._constants import OLLAMA_HOST, OLLAMA_DEFAULT_MODEL, OLLAMA_KEEP_ALIVE

logger = logging.getLogger(__name__)


class JudgeModelResult(BaseModel):
    r"""Quality estimation result for one specific query and LLM."""

    model: Annotated[str, Field(frozen=True, description="The name of the LLM.")]
    r"""The name of the LLM."""
    prompt: Annotated[
        str, Field(frozen=True, description="The prompt used with the LLM.")
    ]
    r"""The prompt used with the LLM."""
    status: Annotated[
        Literal["ok", "error"],
        Field(frozen=True, description="Status of the response."),
    ]
    r"""Status of the response. Can be 'ok' or 'error'. Only 'ok' denote a successful response."""
    parameters: Annotated[
        Optional[dict[str, str]],
        Field(frozen=True, description="Additional parameters passed to the LLM."),
    ]
    r"""Additional parameters passed to the LLM."""
    response: Annotated[
        Optional[str], Field(frozen=True, description="The LLM response as raw text.")
    ]
    r"""The LLM response as raw text."""
    quality_estimation: Annotated[
        Optional[QualityEstimation],
        Field(frozen=True, description="The quality estimation returned by the LLM."),
    ]
    r"""The quality estimation returned by the LLM."""
    quality_estimation_dict: Annotated[
        Optional[dict[str, Any]],
        Field(
            frozen=True,
            description="The quality estimation returned by the LLM as json.",
        ),
    ]
    r"""The quality estimation returned by the LLM as json."""
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """

    @computed_field(description="Quality estimation score returned by the LLM.")
    @property
    def score(self) -> float:
        r"""The quality estimation score returned by the LLM (if any - maybe 'nan')."""
        if self.quality_estimation is not None:
            return self.quality_estimation.quality_estimation_value
        if self.quality_estimation_dict is not None:
            if "quality_estimation_value" in self.quality_estimation_dict:
                return float(self.quality_estimation_dict["quality_estimation_value"])
        return float("nan")


class JudgeResult(BaseModel):
    r"""Quality estimation results for one specific query and all LLMs."""

    openai: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the OpenAI model."),
    ]
    r"""Result of the OpenAI model. 'None' if the model was not setup."""
    anthropic: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Anthropic model."),
    ]
    r"""Result of the Anthropic model. 'None' if the model was not setup."""
    google: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Google model."),
    ]
    r"""Result of the Google model. 'None' if the model was not setup."""
    ollama: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Ollama model."),
    ]
    r"""Result of the Ollama model. 'None' if the model was not setup."""
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """


class JudgeConfig(BaseModel):
    r"""Configuration for all LLMs."""

    openai_model: Annotated[
        str,
        Field(frozen=True, description="The OpenAI model identifier."),
    ] = OPENAI_MODEL
    r"""The OpenAI model identifier. See
        `here <https://developers.openai.com/api/docs/models>`_.
    """
    openai_thinking_level: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        Field(frozen=True, description="The OpenAI thinking level of the model."),
    ] = OPENAI_THINKING_LEVEL
    r"""The OpenAI thinking level of the model. See
        `here <https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses>`_ and
        `here <https://developers.openai.com/api/reference/resources/$shared#(resource)%20%24shared%20%3E%20(model)%20reasoning_effort%20%3E%20(schema)>`_.
    """
    anthropic_model: Annotated[
        str,
        Field(frozen=True, description="The Anthropic model identifier."),
    ] = ANTHROPIC_MODEL
    r"""The Anthropic model identifier. See
        `here <https://platform.claude.com/docs/en/models/overview>`_.
    """
    anthropic_thinking_level: Annotated[
        Literal["low", "medium", "high", "xhigh", "max"],
        Field(frozen=True, description="The Anthropic thinking level of the model."),
    ] = ANTHROPIC_THINKING_LEVEL
    r"""The Anthropic thinking level of the model. See
        `here <https://platform.claude.com/docs/en/build-with-claude/effort>`_.
    """
    google_model: Annotated[
        str,
        Field(frozen=True, description="The Google model identifier."),
    ] = GOOGLE_MODEL
    r"""The Google model identifier. See
        `here <https://ai.google.dev/gemini-api/docs/models>`_.
    """
    google_thinking_level: Annotated[
        str,
        Field(frozen=True, description="The Google thinking level of the model."),
    ] = GOOGLE_THINKING_LEVEL
    r"""The Google thinking level of the model. See
        `here <https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#thinking_level>`_ and
        `here <https://ai.google.dev/gemini-api/docs/thinking#thinking-levels>`_.
    """
    ollama_host: Annotated[
        str,
        Field(frozen=True, description="The Ollama host address."),
    ] = OLLAMA_HOST
    r"""The Ollama host address (optionally including port)."""
    ollama_model: Annotated[
        str,
        Field(frozen=True, description="The Ollama model to use."),
    ] = OLLAMA_DEFAULT_MODEL
    r"""The Ollama model to use, given as a valid model identifier. See
        `here <https://ollama.com/search>`_.
    """
    ollama_keep_alive: Annotated[
        int | str,
        Field(frozen=True, description="The Ollama model in-memory duration."),
    ] = OLLAMA_KEEP_ALIVE
    r"""The Ollama model in-memory duration."""
    max_output_tokens: Annotated[
        int,
        Field(frozen=True, description="Maximum number of output tokens to generate."),
    ] = MAX_OUTPUT_TOKENS
    r"""Maximum number of output tokens to generate."""
    max_retry: Annotated[
        int,
        Field(frozen=True, description="The maximum number of request retries."),
    ] = MAX_RETRY
    r"""The maximum number of request retries for failed API calls."""
    retry_wait_time: Annotated[
        float,
        Field(
            frozen=True,
            description="Time in seconds to wait between failed API requests.",
        ),
    ] = RETRY_WAIT_TIME
    r"""Time in seconds to wait between failed API requests."""
    seeds: Annotated[
        list[int],
        Field(frozen=True, description="Random seeds to use for generation."),
    ] = SEEDS
    r"""Random seeds to use for generation.
        Length must be at least``max_retry + 1``.
    """
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """

    @override
    def model_post_init(self, context: Any = None) -> None:
        r"""
        Performs extra validation and post init functions.

        Warnings
        --------
        This method should not be called manually!
        """
        if len(self.seeds) < self.max_retry + 1:
            raise ValueError(
                f"Parameter 'seeds' must at least length {self.max_retry + 1}!"
            )

    @classmethod
    def model_validate_toml(cls, toml_path: str) -> JudgeConfig:
        parsed_toml = None
        with open(toml_path, "rb") as f:
            parsed_toml = tomllib.load(f)
            f.close()
        if parsed_toml is None:
            raise RuntimeError(
                f"Could not read {toml_path}. Is it in valid TOML format?"
            )
        openai_model = OPENAI_MODEL
        openai_thinking_level = OPENAI_THINKING_LEVEL
        anthropic_model = ANTHROPIC_MODEL
        anthropic_thinking_level = ANTHROPIC_THINKING_LEVEL
        google_model = GOOGLE_MODEL
        google_thinking_level = GOOGLE_THINKING_LEVEL
        ollama_host = OLLAMA_HOST
        ollama_model = OLLAMA_DEFAULT_MODEL
        ollama_keep_alive = OLLAMA_KEEP_ALIVE
        max_output_tokens = MAX_OUTPUT_TOKENS
        max_retry = MAX_RETRY
        retry_wait_time = RETRY_WAIT_TIME
        seeds = SEEDS
        if "OPENAI" in parsed_toml:
            if "openai_model" in parsed_toml["OPENAI"]:
                openai_model = parsed_toml["OPENAI"]["openai_model"]
            if "openai_thinking_level" in parsed_toml["OPENAI"]:
                openai_thinking_level = parsed_toml["OPENAI"]["openai_thinking_level"]
        if "ANTHROPIC" in parsed_toml:
            if "anthropic_model" in parsed_toml["ANTHROPIC"]:
                anthropic_model = parsed_toml["ANTHROPIC"]["anthropic_model"]
            if "anthropic_thinking_level" in parsed_toml["ANTHROPIC"]:
                anthropic_thinking_level = parsed_toml["ANTHROPIC"]["anthropic_thinking_level"]  # fmt: skip
        if "GOOGLE" in parsed_toml:
            if "google_model" in parsed_toml["GOOGLE"]:
                google_model = parsed_toml["GOOGLE"]["google_model"]
            if "google_thinking_level" in parsed_toml["GOOGLE"]:
                google_thinking_level = parsed_toml["GOOGLE"]["google_thinking_level"]
        if "OLLAMA" in parsed_toml:
            if "ollama_host" in parsed_toml["OLLAMA"]:
                ollama_host = parsed_toml["OLLAMA"]["ollama_host"]
            if "ollama_model" in parsed_toml["OLLAMA"]:
                ollama_model = parsed_toml["OLLAMA"]["ollama_model"]
            if "ollama_keep_alive" in parsed_toml["OLLAMA"]:
                ollama_keep_alive = parsed_toml["OLLAMA"]["ollama_keep_alive"]
        if "GENERAL" in parsed_toml:
            if "max_output_tokens" in parsed_toml["GENERAL"]:
                max_output_tokens = parsed_toml["GENERAL"]["max_output_tokens"]
            if "max_retry" in parsed_toml["GENERAL"]:
                max_retry = parsed_toml["GENERAL"]["max_retry"]
            if "retry_wait_time" in parsed_toml["GENERAL"]:
                retry_wait_time = parsed_toml["GENERAL"]["retry_wait_time"]
            if "seeds" in parsed_toml["GENERAL"]:
                seeds = parsed_toml["GENERAL"]["seeds"]
        return JudgeConfig(
            openai_model=openai_model,
            openai_thinking_level=openai_thinking_level,
            anthropic_model=anthropic_model,
            anthropic_thinking_level=anthropic_thinking_level,
            google_model=google_model,
            google_thinking_level=google_thinking_level,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            ollama_keep_alive=ollama_keep_alive,
            max_output_tokens=max_output_tokens,
            max_retry=max_retry,
            retry_wait_time=retry_wait_time,
            seeds=seeds,
        )

    @override
    def __str__(self) -> str:
        return (
            "-------------------- JudgeConfig --------------------\n"
            f"OpenAI Model:               {self.openai_model}\n"
            f"OpenAI Thinking Level:      {self.openai_thinking_level}\n"
            f"Anthropic Model:            {self.anthropic_model}\n"
            f"Anthropic Thinking Level:   {self.anthropic_thinking_level}\n"
            f"Google Model:               {self.google_model}\n"
            f"Google Thinking Level:      {self.google_thinking_level}\n"
            f"Ollama Host:                {self.ollama_host}\n"
            f"Ollama Model:               {self.ollama_model}\n"
            f"Ollama Keep Alive Duration: {self.ollama_keep_alive}\n"
            f"Maximum Output Tokens:      {self.max_output_tokens}\n"
            f"Maximum Retries:            {self.max_retry}\n"
            f"Retry Waiting Time:         {self.retry_wait_time}\n"
            f"Seeds:                      {', '.join([str(seed) for seed in self.seeds])}\n"
            "-----------------------------------------------------"
        )


class _OpenAIModel:
    @staticmethod
    def _get_openai_api_key() -> str:
        if "OPENAI_API_KEY" in os.environ:
            logger.info("Got OPENAI_API_KEY from environment.")
            return os.environ.get("OPENAI_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got OPENAI_API_KEY from env.json file.")
                return str(env["OPENAI_API_KEY"]).strip()
        logger.error(
            "Could not get an API key for OpenAI! Searched for 'OPENAI_API_KEY'."
        )
        raise RuntimeError(
            "Could not get an API key for OpenAI! Searched for 'OPENAI_API_KEY'."
        )
        return "err"

    @staticmethod
    def _get_system_instruction() -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """

    @staticmethod
    def _get_user_instruction(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return (
            f"{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"
        )

    @staticmethod
    def _get_openai_response(
        client: Optional[OpenAI],
        config: JudgeConfig,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        system_instruction: str = _OpenAIModel._get_system_instruction()
        user_instruction: str = _OpenAIModel._get_user_instruction(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        prompt: str = f"{system_instruction}\n{user_instruction}"
        response = None
        try:
            # https://developers.openai.com/api/docs/guides/structured-outputs/
            response = client.responses.parse(
                model=config.openai_model,
                input=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction},
                ],
                # # https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses
                reasoning={"effort": config.openai_thinking_level},
                text_format=QualityEstimation,
                max_output_tokens=config.max_output_tokens,
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from OpenAI API due to: {e}"
            )
            if retry < config.max_retry:
                time.sleep(config.retry_wait_time)
                return _OpenAIModel._get_openai_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting response at retry {retry} > MAX_RETRY ({config.max_retry}) from OpenAI API due to: {e}"
            )
            return JudgeModelResult(
                model=config.openai_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.openai_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < config.max_retry:
                return _OpenAIModel._get_openai_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.openai_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.openai_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.output_parsed is None:
            if retry < config.max_retry:
                return _OpenAIModel._get_openai_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.openai_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.openai_thinking_level},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            r = response.output_parsed.model_dump(mode="json")
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=config.openai_model,
                prompt=prompt,
                status="ok",
                parameters={"effort": config.openai_thinking_level},
                response=str(response),
                quality_estimation=response.output_parsed,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            if retry < config.max_retry:
                return _OpenAIModel._get_openai_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.openai_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.openai_thinking_level},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )
        logger.error(
            f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
        )
        return JudgeModelResult(
            model=config.openai_model,
            prompt=prompt,
            status="error",
            parameters={"effort": config.openai_thinking_level},
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _AnthropicModel:
    @staticmethod
    def _get_anthropic_api_key() -> str:
        if "ANTHROPIC_API_KEY" in os.environ:
            logger.info("Got ANTHROPIC_API_KEY from environment.")
            return os.environ.get("ANTHROPIC_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got ANTHROPIC_API_KEY from env.json file.")
                return str(env["ANTHROPIC_API_KEY"]).strip()
        logger.error(
            "Could not get an API key for Anthropic! Searched for 'ANTHROPIC_API_KEY'."
        )
        raise RuntimeError(
            "Could not get an API key for Anthropic! Searched for 'ANTHROPIC_API_KEY'."
        )
        return "err"

    @staticmethod
    def _get_system_instruction() -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """

    @staticmethod
    def _get_user_instruction(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return (
            f"{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"
        )

    @staticmethod
    def _get_anthropic_response(
        client: Optional[Anthropic],
        config: JudgeConfig,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        system_instruction: str = _AnthropicModel._get_system_instruction()
        user_instruction: str = _AnthropicModel._get_user_instruction(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        prompt: str = f"{system_instruction}\n{user_instruction}"
        response = None
        try:
            # https://platform.claude.com/docs/en/build-with-claude/structured-outputs#quick-start
            response = client.messages.parse(
                model=config.anthropic_model,
                # https://platform.claude.com/docs/en/build-with-claude/working-with-messages#system-role-in-messages
                system=system_instruction,
                messages=[
                    {"role": "user", "content": user_instruction},
                ],
                # https://platform.claude.com/docs/en/build-with-claude/effort
                output_config={"effort": config.anthropic_thinking_level},
                output_format=QualityEstimation,
                max_tokens=config.max_output_tokens,
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from Anthropic API due to: {e}"
            )
            if retry < config.max_retry:
                time.sleep(config.retry_wait_time)
                return _AnthropicModel._get_anthropic_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting response at retry {retry} > MAX_RETRY ({config.max_retry}) from Anthropic API due to: {e}"
            )
            return JudgeModelResult(
                model=config.anthropic_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.anthropic_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < config.max_retry:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.anthropic_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.anthropic_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.parsed_output is None:
            if retry < config.max_retry:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.anthropic_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.anthropic_thinking_level},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            r = response.parsed_output.model_dump(mode="json")
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=config.anthropic_model,
                prompt=prompt,
                status="ok",
                parameters={"effort": config.anthropic_thinking_level},
                response=str(response),
                quality_estimation=response.parsed_output,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            if retry < config.max_retry:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.anthropic_model,
                prompt=prompt,
                status="error",
                parameters={"effort": config.anthropic_thinking_level},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )
        logger.error(
            f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
        )
        return JudgeModelResult(
            model=config.anthropic_model,
            prompt=prompt,
            status="error",
            parameters={"effort": config.anthropic_thinking_level},
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _GoogleModel:
    @staticmethod
    def _get_gemini_api_key() -> str:
        if "GEMINI_API_KEY" in os.environ:
            logger.info("Got GEMINI_API_KEY from environment.")
            return os.environ.get("GEMINI_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got GEMINI_API_KEY from env.json file.")
                return str(env["GEMINI_API_KEY"]).strip()
        logger.error(
            "Could not get an API key for Google Gemini! Searched for 'GEMINI_API_KEY'."
        )
        raise RuntimeError(
            "Could not get an API key for Google Gemini! Searched for 'GEMINI_API_KEY'."
        )
        return "err"

    @staticmethod
    def _generate_prompt(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        base = """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """
        return f"{base}{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"

    @staticmethod
    def _get_gemini_response(
        client: Optional[Google],
        config: JudgeConfig,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        prompt: str = _GoogleModel._generate_prompt(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        response = None
        try:
            response = client.models.generate_content(
                model=config.google_model,
                contents=prompt,
                config=GoogleTypes.GenerateContentConfig(
                    # https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#thinking_level
                    # https://ai.google.dev/gemini-api/docs/thinking#thinking-levels
                    thinking_config=GoogleTypes.ThinkingConfig(
                        thinking_level=config.google_thinking_level
                    ),
                    # might be worth checking out: https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#structured_outputs_with_tools
                    response_mime_type="application/json",
                    response_json_schema=QualityEstimation.model_json_schema(),
                    max_output_tokens=config.max_output_tokens,
                    seed=config.seeds[retry],
                ),
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from Gemini API due to: {e}"
            )
            if retry < config.max_retry:
                time.sleep(config.retry_wait_time)
                return _GoogleModel._get_gemini_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.google_model,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": config.google_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < config.max_retry:
                time.sleep(config.retry_wait_time)
                return _GoogleModel._get_gemini_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.google_model,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": config.google_thinking_level},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.text is None:
            if retry < config.max_retry:
                time.sleep(config.retry_wait_time)
                return _GoogleModel._get_gemini_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.google_model,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": config.google_thinking_level},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            qe = QualityEstimation.model_validate_json(response.text)
            r = json.loads(response.text)
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=config.google_model,
                prompt=prompt,
                status="ok",
                parameters={"thinking_config": config.google_thinking_level},
                response=str(response),
                quality_estimation=qe,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            try:
                if retry < config.max_retry:
                    return _GoogleModel._get_gemini_response(
                        client,
                        config=config,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                r = json.loads(response.text)
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
                )
                return JudgeModelResult(
                    model=config.google_model,
                    prompt=prompt,
                    status="error",
                    parameters={"thinking_config": config.google_thinking_level},
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=r,
                )
            except Exception as _e:
                if retry < config.max_retry:
                    return _GoogleModel._get_gemini_response(
                        client,
                        config=config,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
                )
                return JudgeModelResult(
                    model=config.google_model,
                    prompt=prompt,
                    status="error",
                    parameters={"thinking_config": config.google_thinking_level},
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=None,
                )
        logger.error(
            f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
        )
        return JudgeModelResult(
            model=config.google_model,
            prompt=prompt,
            status="error",
            parameters={"thinking_config": config.google_thinking_level},
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _OllamaModel:
    @staticmethod
    def _get_system_instruction() -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """

    @staticmethod
    def _get_user_instruction(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return (
            f"{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"
        )

    @staticmethod
    def _generate_prompt(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        base = """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """
        return f"{base}{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"

    # this is using the chat API which is the recommended way for structured outputs
    # as of September 2026
    @staticmethod
    def _get_ollama_response(
        client: Optional[Ollama],
        config: JudgeConfig,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        response: OllamaChatResponse | None = None
        system_instruction: str = _OllamaModel._get_system_instruction()
        user_instruction: str = _OllamaModel._get_user_instruction(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        prompt: str = f"{system_instruction}\n{user_instruction}"
        try:
            response = client.chat(
                model=config.ollama_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction},
                ],
                format=QualityEstimation.model_json_schema(),
                keep_alive=config.ollama_keep_alive,
                options={
                    "num_predict": config.max_output_tokens,
                    "seed": config.seeds[retry],
                },
            )
        except OllamaResponseError as e:
            logger.error(f"Error in response: {e.error}")
            # if no model, retrive model and try again
            if e.status_code == 404:
                pull_ollama_model(config.ollama_model)
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            # use fallback to generate API
            return _OllamaModel._get_ollama_response_fallback(
                client,
                config=config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            )
        except Exception as e:
            logger.error(f"Error in response: {e}")

        if response is None:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            # use fallback to generate API
            return _OllamaModel._get_ollama_response_fallback(
                client,
                config=config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            )

        if response.message is None:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            # use fallback to generate API
            return _OllamaModel._get_ollama_response_fallback(
                client,
                config=config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            )

        if response.message.content is None:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            # use fallback to generate API
            return _OllamaModel._get_ollama_response_fallback(
                client,
                config=config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            )

        try:
            qe = QualityEstimation.model_validate_json(response.message.content)
            r = json.loads(response.message.content)
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=config.ollama_model,
                prompt=prompt,
                status="ok",
                parameters={
                    "num_predict": str(config.max_output_tokens),
                    "seed": str(config.seeds[retry]),
                },
                response=str(response),
                quality_estimation=qe,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            # use fallback to generate API
            return _OllamaModel._get_ollama_response_fallback(
                client,
                config=config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            )
        # use fallback to generate API
        return _OllamaModel._get_ollama_response_fallback(
            client,
            config=config,
            src=src,
            mt=mt,
            src_lang=src_lang,
            mt_lang=mt_lang,
        )

    @staticmethod
    def _get_ollama_response_fallback(
        client: Optional[Ollama],
        config: JudgeConfig,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        response: OllamaGenerateResponse | None = None
        prompt: str = _OllamaModel._generate_prompt(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        try:
            response = client.generate(
                model=config.ollama_model,
                prompt=prompt,
                format=QualityEstimation.model_json_schema(),
                keep_alive=config.ollama_keep_alive,
                options={
                    "num_predict": config.max_output_tokens,
                    "seed": config.seeds[retry],
                },
            )
        except OllamaResponseError as e:
            logger.error(f"Error in response: {e.error}")
            # if no model, retrive model and try again
            if e.status_code == 404:
                pull_ollama_model(config.ollama_model)
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response_fallback(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
        except Exception as e:
            logger.error(f"Error in response: {e}")

        if response is None:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response_fallback(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.ollama_model,
                prompt=prompt,
                status="error",
                parameters={
                    "num_predict": str(config.max_output_tokens),
                    "seed": str(config.seeds[retry]),
                },
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.response is None:
            if retry < config.max_retry:
                return _OllamaModel._get_ollama_response_fallback(
                    client,
                    config=config,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
            )
            return JudgeModelResult(
                model=config.ollama_model,
                prompt=prompt,
                status="error",
                parameters={
                    "num_predict": str(config.max_output_tokens),
                    "seed": str(config.seeds[retry]),
                },
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            qe = QualityEstimation.model_validate_json(response.response)
            r = json.loads(response.response)
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=config.ollama_model,
                prompt=prompt,
                status="ok",
                parameters={
                    "num_predict": str(config.max_output_tokens),
                    "seed": str(config.seeds[retry]),
                },
                response=str(response),
                quality_estimation=qe,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            try:
                if retry < config.max_retry:
                    return _OllamaModel._get_ollama_response_fallback(
                        client,
                        config=config,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                r = json.loads(response.response)
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
                )
                return JudgeModelResult(
                    model=config.ollama_model,
                    prompt=prompt,
                    status="error",
                    parameters={
                        "num_predict": str(config.max_output_tokens),
                        "seed": str(config.seeds[retry]),
                    },
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=r,
                )
            except Exception as _e:
                if retry < config.max_retry:
                    return _OllamaModel._get_ollama_response_fallback(
                        client,
                        config=config,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
                )
                return JudgeModelResult(
                    model=config.ollama_model,
                    prompt=prompt,
                    status="error",
                    parameters={
                        "num_predict": str(config.max_output_tokens),
                        "seed": str(config.seeds[retry]),
                    },
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=None,
                )
        logger.error(
            f"Failed getting a valid response at retry {retry} > MAX_RETRY ({config.max_retry})."
        )
        return JudgeModelResult(
            model=config.ollama_model,
            prompt=prompt,
            status="error",
            parameters={
                "num_predict": str(config.max_output_tokens),
                "seed": str(config.seeds[retry]),
            },
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class Judge:
    r"""Judge to rate translation quality.

    Judge to rate translation quality using LLMs from OpenAI, Anthropic, Google, and Ollama.

    Parameters
    ----------
    openai : str, or bool, or None, default = None
        - If a string is given, an ``OPENAI_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the OpenAI model will not be used as a judge.
    anthropic : str, or bool, or None, default = None
        - If a string is given, an ``ANTHROPIC_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the Anthropic model will not be used as a judge.
    google : str, or bool, or None, default = None
        - If a string is given, an ``GEMINI_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the Google model will not be used as a judge.
    ollama : str, or bool, or None, default = None
        - If a string is given, an Ollama model identifier (e.g. ``gemma4:e4b``) is expected.
        - If ``None`` or ``True`` the Ollama model from the config file will be used.
        - If ``False`` the Ollama model will not be used as a judge.
    config : JudgeConfig, str, or None, default = None
        - The configuration for the LLMs/Judge.
        - If a string is given, the path to a ``judge_config.toml`` file is expected.
        - If ``None`` the default configuration will be loaded.

    Raises
    ------
    RuntimeError
        If an Ollama model identifier was provided with parameter ``ollama`` but a configuration
        was also provided via ``config`` (potential clash in model identifiers).

    Examples
    --------
    >>> from qc_bench import Judge
    >>> judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    """

    def __init__(
        self,
        openai: Optional[str | bool] = None,
        anthropic: Optional[str | bool] = None,
        google: Optional[str | bool] = None,
        ollama: Optional[str | bool] = None,
        config: Optional[JudgeConfig | str] = None,
    ):
        # config
        if config is None:
            if isinstance(ollama, str):
                self.config = JudgeConfig(ollama_model=ollama)
            else:
                self.config = JudgeConfig()
        elif isinstance(config, str):
            self.config = JudgeConfig.model_validate_toml(config)
            if isinstance(ollama, str):
                logger.error(
                    "Parameter 'ollama' seems to be a model identifier but an "
                    "Ollama model is already given in the configuration! "
                    "Please only use one of the two options!"
                )
                raise RuntimeError(
                    "Parameter 'ollama' seems to be a model identifier but an "
                    "Ollama model is already given in the configuration! "
                    "Please only use one of the two options!"
                )
        else:
            self.config = config
            if isinstance(ollama, str):
                logger.error(
                    "Parameter 'ollama' seems to be a model identifier but an "
                    "Ollama model is already given in the configuration! "
                    "Please only use one of the two options!"
                )
                raise RuntimeError(
                    "Parameter 'ollama' seems to be a model identifier but an "
                    "Ollama model is already given in the configuration! "
                    "Please only use one of the two options!"
                )
        # openai
        if isinstance(openai, str):
            self.__openai = OpenAI(api_key=str(openai).strip())
        elif openai is None or openai:
            self.__openai = OpenAI(api_key=_OpenAIModel._get_openai_api_key())
        else:
            self.__openai = None
        # log openai
        if self.__openai is None:
            logger.info("OpenAI judge disabled for this instance!")
        else:
            logger.info("OpenAI judge enabled for this instance!")
        # anthropic
        if isinstance(anthropic, str):
            self.__anthropic = Anthropic(api_key=str(anthropic).strip())
        elif anthropic is None or anthropic:
            self.__anthropic = Anthropic(
                api_key=_AnthropicModel._get_anthropic_api_key()
            )
        else:
            self.__anthropic = None
        # log anthropic
        if self.__anthropic is None:
            logger.info("Anthropic judge disabled for this instance!")
        else:
            logger.info("Anthropic judge enabled for this instance!")
        # google
        if isinstance(google, str):
            self.__google = Google(api_key=str(google).strip())
        elif google is None or google:
            self.__google = Google(api_key=_GoogleModel._get_gemini_api_key())
        else:
            self.__google = None
        # log google
        if self.__google is None:
            logger.info("Google judge disabled for this instance!")
        else:
            logger.info("Google judge enabled for this instance!")
        # ollama
        if isinstance(ollama, str):
            self.__ollama = Ollama(host=self.config.ollama_host, headers={})
        elif ollama is None or ollama:
            self.__ollama = Ollama(host=self.config.ollama_host, headers={})
        else:
            self.__ollama = None
        # log ollama
        if self.__ollama is None:
            logger.info("Ollama judge disabled for this instance!")
        else:
            logger.info("Ollama judge enabled for this instance!")
        # log config
        logger.info(f"Loaded the following configuration:\n{self.config}")

    def score(self, src: str, mt: str, src_lang: str, mt_lang: str) -> JudgeResult:
        r"""Performs quality estimation using all setup LLMs for one translation.

        Parameters
        ----------
        src : str
            The source text.
        mt : str
            The machine translation.
        src_lang : str
            The language of the source text, e.g. ``"English"``.
        mt_lang : str
            The language of the machine translation, e.g. ``"German"``.

        Returns
        -------
        JudgeResult
            The results of all LLMs in a result container, see ``JudgeResult``.

        Examples
        --------
        >>> from qc_bench import Judge
        >>> judge = Judge(
        ...     openai=False, anthropic=False, google=False, ollama="mistral:7b"
        ... )
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

        >>> from qc_bench import Judge
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
        'gemma4:e4b'
        >>> jr.ollama.score
        1.0
        """
        return JudgeResult(
            openai=_OpenAIModel._get_openai_response(
                client=self.__openai,
                config=self.config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            ),
            anthropic=_AnthropicModel._get_anthropic_response(
                client=self.__anthropic,
                config=self.config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            ),
            google=_GoogleModel._get_gemini_response(
                client=self.__google,
                config=self.config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            ),
            ollama=_OllamaModel._get_ollama_response(
                client=self.__ollama,
                config=self.config,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            ),
        )
