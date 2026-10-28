"""Command-line entry point: `uv run shpe <command>`.

Each command group registers its subcommands in `_build_parser`. Commands are
thin wrappers; the logic lives in the library modules so notebooks and tests
call the same code.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shpe", description="SHPE Capital research tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
