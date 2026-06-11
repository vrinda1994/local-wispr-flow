"""Client for the local Gemma server (LiteRT-LM, OpenAI-compatible)."""

import requests

from config import GEMMA_MODEL, SERVER_URL


def run_llm(messages: list[dict], max_tokens: int) -> str:
    """Send a chat message list to the local Gemma server, return the reply text."""
    payload = {
        "model": GEMMA_MODEL,
        "top_k": 1,            # greedy/deterministic. (temperature omitted on
        "max_tokens": max_tokens,  # purpose: temperature=0 crashes this GPU sampler.)
        "messages": messages,
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
