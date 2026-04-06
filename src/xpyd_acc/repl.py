"""Interactive REPL for exploratory comparison (M82)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ReplCommand:
    """Parsed REPL command."""

    name: str
    args: str = ""


@dataclass
class ReplEntry:
    """A single REPL interaction."""

    prompt: str
    baseline_output: str
    target_output: str
    match: bool
    timestamp: float = 0.0
    logprobs_baseline: list[dict[str, Any]] | None = None
    logprobs_target: list[dict[str, Any]] | None = None


@dataclass
class ReplSession:
    """Tracks state for an interactive REPL session."""

    baseline_url: str
    target_url: str
    model: str
    api_key: str | None = None
    show_logprobs: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    history: list[ReplEntry] = field(default_factory=list)

    def set_param(self, key: str, value: str) -> str:
        """Set a sampling parameter. Returns status message."""
        key = key.strip().lower()
        value = value.strip()
        if key == "temperature":
            self.temperature = float(value)
            return f"temperature = {self.temperature}"
        elif key == "top_p":
            self.top_p = float(value)
            return f"top_p = {self.top_p}"
        elif key == "seed":
            self.seed = int(value)
            return f"seed = {self.seed}"
        elif key == "max_tokens":
            self.max_tokens = int(value)
            return f"max_tokens = {self.max_tokens}"
        else:
            return f"Unknown parameter: {key}"

    def export_json(self) -> dict[str, Any]:
        """Export session history as JSON-serializable dict."""
        return {
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "model": self.model,
            "params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            },
            "entries": [
                {
                    "prompt": e.prompt,
                    "baseline_output": e.baseline_output,
                    "target_output": e.target_output,
                    "match": e.match,
                    "timestamp": e.timestamp,
                }
                for e in self.history
            ],
        }


def parse_command(line: str) -> ReplCommand | None:
    """Parse a REPL command (lines starting with ':')."""
    if not line.startswith(":"):
        return None
    parts = line[1:].split(None, 1)
    if not parts:
        return None
    return ReplCommand(name=parts[0].lower(), args=parts[1] if len(parts) > 1 else "")


def format_comparison(entry: ReplEntry) -> str:
    """Format a comparison result for terminal display."""
    status = "✅ MATCH" if entry.match else "❌ DIVERGE"
    lines = [
        f"\n{status}",
        "─── Baseline ───",
        entry.baseline_output or "(empty)",
        "─── Target ───",
        entry.target_output or "(empty)",
    ]
    return "\n".join(lines)


def format_diff(entry: ReplEntry) -> str:
    """Show token-level diff between baseline and target outputs."""
    b_tokens = entry.baseline_output.split()
    t_tokens = entry.target_output.split()
    max_len = max(len(b_tokens), len(t_tokens))
    if max_len == 0:
        return "(both outputs empty)"

    lines = ["Token diff (word-level):"]
    for i in range(max_len):
        b = b_tokens[i] if i < len(b_tokens) else "<END>"
        t = t_tokens[i] if i < len(t_tokens) else "<END>"
        marker = "  " if b == t else "→ "
        lines.append(f"  {marker}[{i}] {b!r:30s} | {t!r}")
    return "\n".join(lines)


def format_history(session: ReplSession) -> str:
    """Format session history for display."""
    if not session.history:
        return "(no history)"
    lines = ["# Session History", ""]
    for i, entry in enumerate(session.history, 1):
        status = "✅" if entry.match else "❌"
        prompt_preview = entry.prompt[:60] + ("..." if len(entry.prompt) > 60 else "")
        lines.append(f"  {i}. {status} {prompt_preview}")
    return "\n".join(lines)


async def send_prompt(
    url: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a prompt to an endpoint and return the output text."""
    import aiohttp

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    async with aiohttp.ClientSession() as cs:
        async with cs.post(
            url if url.endswith("/chat/completions") else f"{url.rstrip('/')}/v1/chat/completions",
            json=body,
            headers=headers,
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def run_repl_iteration(
    session: ReplSession,
    prompt: str,
) -> ReplEntry:
    """Send prompt to both endpoints and return the entry."""
    baseline_out, target_out = await asyncio.gather(
        send_prompt(
            session.baseline_url, session.model, prompt,
            api_key=session.api_key,
            temperature=session.temperature, top_p=session.top_p,
            seed=session.seed, max_tokens=session.max_tokens,
        ),
        send_prompt(
            session.target_url, session.model, prompt,
            api_key=session.api_key,
            temperature=session.temperature, top_p=session.top_p,
            seed=session.seed, max_tokens=session.max_tokens,
        ),
    )
    entry = ReplEntry(
        prompt=prompt,
        baseline_output=baseline_out,
        target_output=target_out,
        match=baseline_out == target_out,
        timestamp=time.time(),
    )
    session.history.append(entry)
    return entry


async def run_repl(
    baseline_url: str,
    target_url: str,
    model: str,
    api_key: str | None = None,
    print_fn: Callable[[str], None] | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> ReplSession:
    """Run the interactive REPL loop.

    Args:
        baseline_url: Baseline endpoint URL.
        target_url: Target endpoint URL.
        model: Model name.
        api_key: Optional API key.
        print_fn: Override print (for testing).
        input_fn: Override input (for testing).

    Returns:
        The session with all history.
    """
    _print = print_fn or print
    _input = input_fn or input

    session = ReplSession(
        baseline_url=baseline_url,
        target_url=target_url,
        model=model,
        api_key=api_key,
    )

    _print(f"xPyD-acc REPL — baseline: {baseline_url} | target: {target_url} | model: {model}")
    _print("Type a prompt to compare, or :help for commands. :quit to exit.\n")

    while True:
        try:
            line = _input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nBye!")
            break

        if not line:
            continue

        cmd = parse_command(line)
        if cmd is not None:
            if cmd.name in ("quit", "q", "exit"):
                _print("Bye!")
                break
            elif cmd.name == "help":
                _print(
                    ":logprobs     Toggle logprob display\n"
                    ":diff         Show token diff of last result\n"
                    ":history      Show session history\n"
                    ":set key=val  Set sampling parameter\n"
                    ":export path  Export session as JSON\n"
                    ":quit         Exit REPL"
                )
            elif cmd.name == "logprobs":
                session.show_logprobs = not session.show_logprobs
                _print(f"Logprobs display: {'on' if session.show_logprobs else 'off'}")
            elif cmd.name == "diff":
                if not session.history:
                    _print("(no results yet)")
                else:
                    _print(format_diff(session.history[-1]))
            elif cmd.name == "history":
                _print(format_history(session))
            elif cmd.name == "set":
                if "=" not in cmd.args:
                    _print("Usage: :set key=value")
                else:
                    k, v = cmd.args.split("=", 1)
                    _print(session.set_param(k, v))
            elif cmd.name == "export":
                path = cmd.args.strip()
                if not path:
                    _print("Usage: :export <path>")
                else:
                    with open(path, "w") as f:
                        json.dump(session.export_json(), f, indent=2)
                    _print(f"Session exported to {path}")
            else:
                _print(f"Unknown command: :{cmd.name}. Type :help for commands.")
        else:
            # It's a prompt — send to both endpoints
            try:
                entry = await run_repl_iteration(session, line)
                _print(format_comparison(entry))
            except Exception as exc:
                _print(f"Error: {exc}")

    return session
