#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

########## OPENAI ##########

# https://developers.openai.com/api/docs/models
OPENAI_MODEL = "gpt-5.4"
# https://developers.openai.com/api/reference/resources/$shared#(resource)%20%24shared%20%3E%20(model)%20reasoning_effort%20%3E%20(schema)
OPENAI_THINKING_LEVEL = "low"

########## GENERAL ##########

# Max number generated output tokens, prevents infinite generation
MAX_OUTPUT_TOKENS = 2048
# Max number of request retries if API fails
MAX_RETRY = 5
# Time to wait between failing requests in seconds
RETRY_WAIT_TIME = 30.0
