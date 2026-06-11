"""Spoken meta-command detection (pure functions, no state).

These run on the raw transcript *before* it reaches the LLM, so a whole-utterance
command is handled deterministically rather than dictated as text. A command must
be the entire utterance — a mid-sentence "scratch that" still flows to the model
as an inline self-correction.
"""

import re

# Whole-utterance phrases that mean "revert the last paste" (editor ⌘Z).
UNDO_PHRASES = {
    "undo", "undo that", "undo last", "scratch", "scratch that",
    "delete that", "delete that line", "delete line", "delete last line",
    "remove that", "remove that line",
}

# "dictate in Spanish", "translate to French", "switch to English", ...
_LANGUAGE_RE = re.compile(
    r"^(?:dictate|translate|write|speak|switch|talk)\s+(?:in|to|into)\s+([a-z]+)$"
)
_TRANSLATION_OFF = {"stop translating", "no translation", "translation off"}


def _clean(text: str) -> str:
    """Lowercase and strip punctuation for robust whole-utterance matching."""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()


def is_undo(text: str) -> bool:
    """True if the whole utterance is a revert/delete command."""
    return _clean(text) in UNDO_PHRASES


def parse_language_command(text: str):
    """Detect a prose-translation switch.

    Returns ('set', '<Language>') to start translating, ('off', None) to stop,
    or None if this utterance is not a language command.
    """
    t = _clean(text)
    if t in _TRANSLATION_OFF:
        return ("off", None)
    m = _LANGUAGE_RE.match(t)
    if m:
        lang = m.group(1)
        if lang == "english":          # English = source language → no translation
            return ("off", None)
        return ("set", lang.title())
    return None
