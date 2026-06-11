"""Editor side effects: paste at cursor, undo, and code-fence cleanup."""

import time

import pyperclip
from pynput import keyboard


def strip_code_fences(text: str) -> str:
    """Remove a leading ```lang fence and trailing ``` if the model added them."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]               # drop opening fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                   # drop closing fence
        t = "\n".join(lines)
    return t.strip("\n")


def paste(text: str):
    """Insert text at the cursor in the frontmost app (clipboard + ⌘V)."""
    previous = ""
    try:
        previous = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(0.05)
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.1)
    if previous:  # restore the user's old clipboard
        pyperclip.copy(previous)


def undo_in_editor():
    """Trigger the frontmost editor's own undo (⌘Z) — removes the last paste."""
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.cmd):
        kb.press("z")
        kb.release("z")
