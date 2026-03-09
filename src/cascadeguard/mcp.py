"""
FastMCP-based MCP interface for CascadeGuard LLM Guardrails
"""

import argparse
from typing import Optional, List, Dict, Any

from fastmcp import FastMCP

from .main import Main
from .utils.path import resolve_config_path


def create_mcp_server(config_path: str = "configs/config.yaml") -> FastMCP:
    """
    Create and configure FastMCP server

    Args:
        config_path: Path to configuration file

    Returns:
        Configured FastMCP server instance
    """
    # Resolve config path
    config_path = resolve_config_path(config_path)

    # Initialize MCP server
    mcp = FastMCP("CascadeGuard")

    # Initialize Main instance
    main = Main(config_path=config_path)

    # Register tool: apply
    @mcp.tool()
    def apply(
        pairs: List[Dict[str, str]],
        guardrail: Optional[str] = None,
        winnow_down: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply guardrail to prompt-output pairs

        Args:
            pairs: List of dicts with 'prompt' and 'output' keys
            guardrail: Guardrail name (leave empty to use first guardrail in config)
            winnow_down: If true, keep valid items; if false, keep invalid items

        Returns:
            List of filtered results (RankData as dict)
        """
        # Convert to tuple format
        pair_tuples = [(p["prompt"], p["output"]) for p in pairs]

        results = main.apply(
            pairs=pair_tuples,
            guardrail_name=guardrail,
            winnow_down=winnow_down,
            return_as_dict=True,
        )
        return results

    # Register resource: list guardrails
    @mcp.resource("guardrail://guardrails")
    def list_guardrails() -> str:
        """
        List all available guardrails

        Returns:
            JSON string with list of guardrail names
        """
        import json
        guardrails = main.list_guardrails()
        return json.dumps({"guardrails": guardrails}, indent=2)

    return mcp


def main():
    """
    Main entry point for guardrail-mcp command
    """
    parser = argparse.ArgumentParser(
        description="Run CascadeGuard MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with stdio transport (for local tools like Claude Desktop)
  guardrail-mcp --transport stdio

  # Run with streamable-http transport (for remote deployment)
  guardrail-mcp --transport streamable-http --port 8002

  # Use custom config file
  guardrail-mcp -c configs/custom.yaml --transport stdio
"""
    )

    parser.add_argument(
        "-c", "--config",
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to for streamable-http (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to for streamable-http (default: 8000)"
    )

    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for streamable-http endpoint (default: /mcp)"
    )

    args = parser.parse_args()

    try:
        # Create MCP server
        mcp = create_mcp_server(config_path=args.config)

        # Run server with specified transport
        if args.transport == "stdio":
            # NOTE: In stdio mode, we MUST NOT print anything to stdout
            # as it will interfere with JSON-RPC communication
            # Disable banner in stdio mode to prevent JSON-RPC errors
            mcp.run(transport="stdio", show_banner=False)
        else:
            print(f"🚀 Starting MCP server with streamable-http transport...", flush=True)
            print(f"   Host: {args.host}", flush=True)
            print(f"   Port: {args.port}", flush=True)
            print(f"   Path: {args.path}", flush=True)
            mcp.run(
                transport="streamable-http",
                host=args.host,
                port=args.port,
                path=args.path
            )

    except FileNotFoundError as e:
        # Use stderr for error messages in stdio mode
        import sys
        print(f"❌ Error: {e}", file=sys.stderr, flush=True)
        return 1
    except Exception as e:
        import sys
        import traceback
        print(f"❌ Error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
