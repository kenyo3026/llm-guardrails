"""CLI for CascadeGuard LLM Guardrails."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .main import Main
from .utils import resolve_config_path

console = Console()


def cmd_apply(args):
    """Execute guardrail apply command"""
    try:
        config_path = resolve_config_path(args.config)
        main = Main(config_path=config_path)

        pairs = []
        if args.prompt and args.output:
            pairs = [(args.prompt, args.output)]
        elif args.input_file:
            path = Path(args.input_file)
            if not path.exists():
                console.print(f"[bold red]Error:[/bold red] File not found: {args.input_file}")
                return 1
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    pairs.append((obj["prompt"], obj["output"]))
        else:
            console.print("[bold red]Error:[/bold red] Provide --prompt/--output or --input-file")
            return 1

        if not pairs:
            console.print("[bold red]Error:[/bold red] No pairs to process")
            return 1

        results = main.apply(
            pairs=pairs,
            guardrail_name=args.guardrail,
            winnow_down=args.winnow_down,
            return_as_dict=True,
        )

        for r in results:
            console.print(json.dumps(r, ensure_ascii=False))
        return 0

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1


def cmd_list(args):
    """List available guardrails"""
    try:
        config_path = resolve_config_path(args.config)
        main = Main(config_path=config_path)
        guardrails = main.list_guardrails()

        console.print("[bold cyan]Available Guardrails:[/bold cyan]")
        for g in guardrails:
            console.print(f"  - {g}")
        return 0

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CascadeGuard - LLM Guardrails with cascade preranker and fineranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "-g", "--guardrail",
        help="Guardrail name to use (default: first guardrail in config)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_apply = subparsers.add_parser("apply", help="Apply guardrail to prompt-output pairs")
    parser_apply.add_argument("--prompt", help="Prompt text (use with --output for single pair)")
    parser_apply.add_argument("--output", help="Output text (use with --prompt for single pair)")
    parser_apply.add_argument(
        "--input-file",
        help="Path to JSONL file with {'prompt': str, 'output': str} per line",
    )
    parser_apply.add_argument(
        "--winnow-invalid",
        action="store_true",
        help="Keep invalid (failed) items instead of valid (passed) ones",
    )

    subparsers.add_parser("list", help="List available guardrails")

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list(args)

    if args.command == "apply":
        args.winnow_down = not getattr(args, "winnow_invalid", False)
        return cmd_apply(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
