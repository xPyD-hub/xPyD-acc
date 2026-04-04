"""Shell completion script generation for xpyd-acc CLI."""

from __future__ import annotations

import argparse


def _get_subcommands_and_flags(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    """Extract subcommands and their flags from the argument parser.

    Returns a dict mapping subcommand name -> list of flag strings.
    Top-level flags are under the key ``""``.
    """
    result: dict[str, list[str]] = {}

    # Top-level flags
    top_flags: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        for opt in action.option_strings:
            top_flags.append(opt)
    result[""] = top_flags

    # Subcommands
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                flags: list[str] = []
                for sub_action in subparser._actions:
                    for opt in sub_action.option_strings:
                        flags.append(opt)
                result[name] = flags

    return result


def generate_bash(parser: argparse.ArgumentParser) -> str:
    """Generate a Bash completion script for the given parser."""
    info = _get_subcommands_and_flags(parser)
    subcommands = [k for k in info if k]

    lines = [
        "# Bash completion for xpyd-acc",
        "# eval \"$(xpyd-acc completion bash)\"",
        "",
        "_xpyd_acc_complete() {",
        "    local cur prev subcmds",
        "    COMPREPLY=()",
        "    cur=\"${COMP_WORDS[COMP_CWORD]}\"",
        "    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"",
        f"    subcmds=\"{' '.join(subcommands)}\"",
        "",
        "    if [[ $COMP_CWORD -eq 1 ]]; then",
        "        COMPREPLY=( $(compgen -W \"$subcmds\" -- \"$cur\") )",
        "        return 0",
        "    fi",
        "",
        "    local subcmd=\"${COMP_WORDS[1]}\"",
        "    case \"$subcmd\" in",
    ]

    for name in subcommands:
        flags_str = " ".join(info[name])
        lines.append(f"        {name})")
        lines.append(f"            COMPREPLY=( $(compgen -W \"{flags_str}\" -- \"$cur\") )")
        lines.append("            ;;")

    lines += [
        "    esac",
        "}",
        "",
        "complete -F _xpyd_acc_complete xpyd-acc",
    ]

    return "\n".join(lines) + "\n"


def generate_zsh(parser: argparse.ArgumentParser) -> str:
    """Generate a Zsh completion script for the given parser."""
    info = _get_subcommands_and_flags(parser)
    subcommands = [k for k in info if k]

    lines = [
        "#compdef xpyd-acc",
        "# Zsh completion for xpyd-acc",
        "# eval \"$(xpyd-acc completion zsh)\"",
        "",
        "_xpyd_acc() {",
        "    local -a subcmds",
        f"    subcmds=({' '.join(subcommands)})",
        "",
        "    _arguments -C \\",
        "        '1:command:->cmd' \\",
        "        '*::arg:->args'",
        "",
        "    case $state in",
        "        cmd)",
        "            _describe 'command' subcmds",
        "            ;;",
        "        args)",
        "            case $words[1] in",
    ]

    for name in subcommands:
        flag_entries = " ".join(f"'{f}'" for f in info[name])
        lines.append(f"                {name})")
        lines.append(f"                    _arguments {flag_entries}")
        lines.append("                    ;;")

    lines += [
        "            esac",
        "            ;;",
        "    esac",
        "}",
        "",
        "_xpyd_acc",
    ]

    return "\n".join(lines) + "\n"


def generate_fish(parser: argparse.ArgumentParser) -> str:
    """Generate a Fish completion script for the given parser."""
    info = _get_subcommands_and_flags(parser)
    subcommands = [k for k in info if k]

    lines = [
        "# Fish completion for xpyd-acc",
        "# xpyd-acc completion fish | source",
        "",
    ]

    # Subcommand completions
    for name in subcommands:
        lines.append(
            f"complete -c xpyd-acc -n '__fish_use_subcommand' "
            f"-a '{name}' -d '{name}'",
        )

    lines.append("")

    # Flag completions per subcommand
    for name in subcommands:
        for flag in info[name]:
            if flag.startswith("--"):
                short_flag = flag.lstrip("-")
                lines.append(
                    f"complete -c xpyd-acc -n '__fish_seen_subcommand_from {name}' "
                    f"-l '{short_flag}'",
                )
            elif flag.startswith("-") and len(flag) == 2:
                lines.append(
                    f"complete -c xpyd-acc -n '__fish_seen_subcommand_from {name}' "
                    f"-s '{flag[1]}'",
                )

    return "\n".join(lines) + "\n"


GENERATORS = {
    "bash": generate_bash,
    "zsh": generate_zsh,
    "fish": generate_fish,
}
