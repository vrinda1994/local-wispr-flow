"""Console output helpers."""


def show_block(label: str, text: str):
    """Print a labelled, multi-line block with real line breaks (not repr)."""
    width = 52
    print(f"   ┌─ {label} " + "─" * max(2, width - len(label) - 5))
    for line in text.splitlines() or [""]:
        print(f"   │ {line}")
    print("   └" + "─" * (width - 1))
