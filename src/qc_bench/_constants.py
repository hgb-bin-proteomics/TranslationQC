#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

########## OPENAI ##########

# https://developers.openai.com/api/docs/models
OPENAI_MODEL = "gpt-5.4"
# https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses
# https://developers.openai.com/api/reference/resources/$shared#(resource)%20%24shared%20%3E%20(model)%20reasoning_effort%20%3E%20(schema)
OPENAI_THINKING_LEVEL = "low"

########## GOOGLE ##########

# https://ai.google.dev/gemini-api/docs/models
GOOGLE_MODEL = "gemini-3.1-pro-preview"
# https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#thinking_level
# https://ai.google.dev/gemini-api/docs/thinking#thinking-levels
GOOGLE_THINKING_LEVEL = "low"

########## OLLAMA ##########

# Where Ollama is hosted
OLLAMA_HOST = "http://localhost:11434"
# Ollama models: https://ollama.com/search
# the default Ollama model to use
OLLAMA_DEFAULT_MODEL = "gemma4:e4b"
# model in-memory duration
# see https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately
# -1 seems to be safer, see https://github.com/ollama/ollama/issues/7645
# KEEP_ALIVE = "5m"
OLLAMA_KEEP_ALIVE = -1

########## GENERAL ##########

# Max number generated output tokens, prevents infinite generation
MAX_OUTPUT_TOKENS = 2048
# Max number of request retries if API fails
MAX_RETRY = 5
# Time to wait between failing requests in seconds
RETRY_WAIT_TIME = 30.0
# Random seeds - has to be of lenght MAX_RETRY + 1
SEEDS = [1337, 10081995, 18041970, 1071966, 3082023, 24042025]
