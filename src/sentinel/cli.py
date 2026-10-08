"""Command-line entry point: `uv run sentinel <command>`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {}


def command(name: str, help_: str):
    def register(fn: Callable[[argparse.Namespace], None]):
        COMMANDS[name] = (help_, fn)
        return fn

    return register


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="sentinel")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_, _) in COMMANDS.items():
        sub.add_parser(name, help=help_)
    args = parser.parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    main()
