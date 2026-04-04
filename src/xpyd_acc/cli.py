"""CLI entry point."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xpyd-acc",
        description="PD disaggregation accuracy diagnostic tool",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    sub.add_parser("compare-logprobs", help="Compare logprobs between two endpoints")
    sub.add_parser("check-kv", help="Check KV cache numerical accuracy")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    print(f"xpyd-acc {args.command} — not yet implemented")
