"""Console output and prompts (rich), kept in one place so tests can swap them.

`ui.console` is the shared rich console. `ui.ask()` and friends read from an
injectable input function, and fall back to a default when stdin is not a
terminal (batch mode, pipes, tests) instead of blocking.
"""

from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)

# Replaceable for tests: a function that takes a prompt and returns a line.
_input_fn: Callable[[str], str] = input
_force_interactive: bool | None = None


def set_input(fn: Callable[[str], str] | None, interactive: bool | None = None) -> None:
    """Swap the input function (tests) and optionally force interactive on/off."""
    global _input_fn, _force_interactive
    _input_fn = fn or input
    _force_interactive = interactive


def is_interactive() -> bool:
    if _force_interactive is not None:
        return _force_interactive
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def ask(prompt: str, default: str = "") -> str:
    """One line of input; returns `default` when not interactive or on EOF."""
    if not is_interactive():
        return default
    try:
        answer = _input_fn(prompt).strip()
    except EOFError:
        return default
    return answer or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = ask(prompt + suffix, "y" if default else "n").lower()
    return answer.startswith("y")


def ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    """Ask for one of `choices` (key -> label). Returns the chosen key."""
    labels = " / ".join(f"[{key.upper()}] {label}" for key, label in choices.items())
    while True:
        answer = ask(f"{prompt} {labels}: ", default).lower()[:1]
        if answer in choices:
            return answer
        if not is_interactive():
            return default
        console.print(f"  Please answer one of: {', '.join(k.upper() for k in choices)}")


def _more_input_pending(timeout: float = 0.35) -> bool:
    """True when more pasted lines are already waiting on the terminal."""
    try:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(ready)
    except (ImportError, ValueError, OSError):
        return False


def read_multiline(prompt: str) -> str:
    """Read pasted text until a blank line (or EOF). Returns "" when not interactive.

    Job descriptions contain blank lines between paragraphs. On a real terminal
    a blank line only ends the paste when nothing else is waiting to be read,
    so a multi-paragraph paste arrives whole and a blank line typed afterwards
    finishes it.
    """
    if not is_interactive():
        return ""
    console.print(prompt)
    on_terminal = _input_fn is input
    lines: list[str] = []
    while True:
        try:
            if on_terminal:
                line = sys.stdin.readline()
                if line == "":             # Ctrl-D
                    break
            else:
                line = _input_fn("")
        except EOFError:
            break
        if not line.strip():
            if not lines:
                continue
            if on_terminal and _more_input_pending():
                lines.append("")           # blank line in the middle of a paste
                continue
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def info(message: str) -> None:
    console.print(message)


def step(message: str) -> None:
    console.print(f"[bold cyan]>[/bold cyan] {message}")


def success(message: str) -> None:
    console.print(f"[green]OK[/green]  {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]WARN[/yellow] {message}")


def error(message: str, hint: str | None = None) -> None:
    console.print(f"[bold red]ERROR[/bold red] {message}")
    if hint:
        console.print(f"[dim]{hint}[/dim]")


def panel(title: str, body: str, style: str = "cyan") -> None:
    console.print(Panel(body, title=title, border_style=style, expand=False))


def table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    grid = Table(title=title, show_lines=False, header_style="bold")
    for column in columns:
        grid.add_column(column, overflow="fold")
    for row in rows:
        grid.add_row(*[str(cell) for cell in row])
    console.print(grid)
