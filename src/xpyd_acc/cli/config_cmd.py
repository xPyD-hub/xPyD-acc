"""CLI handlers for init, config, profiles, completion."""

from __future__ import annotations

import argparse
import sys


def _run_init(args: argparse.Namespace) -> None:
    """Generate a starter config file."""
    from pathlib import Path

    from xpyd_acc.config_validate import generate_starter_config

    output = Path(args.output)
    try:
        path = generate_starter_config(output, force=args.force)
        print(f"Created config file: {path}")
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_config(args: argparse.Namespace) -> None:
    """Handle config subcommands."""
    if not hasattr(args, "config_command") or args.config_command is None:
        print("Usage: xpyd-acc config {validate}")
        return

    if args.config_command == "validate":
        from pathlib import Path

        from xpyd_acc.config_validate import validate_config

        path = Path(args.path)
        try:
            issues = validate_config(path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if not issues:
            print(f"✅ {path} is valid")
            sys.exit(0)
        else:
            has_errors = False
            for issue in issues:
                print(issue)
                if issue.startswith("error:"):
                    has_errors = True
            if has_errors:
                sys.exit(1)
            else:
                print(f"\n⚠️  {path} has warnings but no errors")
                sys.exit(0)


def _run_profiles(config: object) -> None:
    """List all available named profiles."""
    from xpyd_acc.profiles import list_profiles, parse_profiles

    user_profiles = parse_profiles(config.profiles_raw) if config is not None else None
    all_profiles = list_profiles(user_profiles)

    if not all_profiles:
        print("No profiles available.")
        return

    print("Available profiles:\n")
    for name in sorted(all_profiles):
        profile = all_profiles[name]
        settings = profile.to_dict()
        if settings:
            parts = [f"{k}={v}" for k, v in settings.items()]
            print(f"  {name}: {', '.join(parts)}")
        else:
            print(f"  {name}: (empty)")


def _run_completion(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Generate and output a shell completion script."""
    from xpyd_acc.completion import GENERATORS

    generator = GENERATORS[args.shell]
    script = generator(parser)

    if args.output:
        from pathlib import Path

        Path(args.output).write_text(script)
        print(f"Completion script written to {args.output}")
    else:
        print(script, end="")
