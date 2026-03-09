"""
FastAPI-based API interface for CascadeGuard LLM Guardrails
"""

import argparse
import asyncio

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from .main import Main
from .utils import resolve_config_path


class PairItem(BaseModel):
    """Single prompt-output pair"""
    prompt: str = Field(..., description="Prompt text")
    output: str = Field(..., description="Output text")


class ApplyRequest(BaseModel):
    """Request model for guardrail apply"""
    pairs: List[PairItem] = Field(..., description="List of prompt-output pairs to filter")
    guardrail: Optional[str] = Field(
        None,
        description="Guardrail name. Leave empty to use first guardrail in config.",
    )
    winnow_down: bool = Field(
        True,
        description="If true, keep valid items; if false, keep invalid items",
    )

    @field_validator("guardrail", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or (isinstance(v, str) and v.strip() == ""):
            return None
        return v


class ApplyResponse(BaseModel):
    """Response model for guardrail apply"""
    results: List[dict] = Field(..., description="Filtered results (RankData as dict)")


class GuardrailsResponse(BaseModel):
    """Response model for listing guardrails"""
    guardrails: List[str] = Field(..., description="List of available guardrail names")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")


def create_app(config_path: str = "configs/config.yaml") -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        config_path: Path to configuration file

    Returns:
        Configured FastAPI application
    """
    config_path = resolve_config_path(config_path)

    app = FastAPI(
        title="CascadeGuard API",
        description="API for LLM guardrails with cascade preranker and fineranker",
        version="0.1.0",
    )

    try:
        main = Main(config_path=config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CascadeGuard: {e}")

    @app.get("/", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(status="healthy", version="0.1.0")

    @app.get("/guardrails", response_model=GuardrailsResponse)
    async def list_guardrails():
        """
        List all available guardrails

        Returns:
            List of guardrail names
        """
        try:
            guardrails = main.list_guardrails()
            return GuardrailsResponse(guardrails=guardrails)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/list", response_model=GuardrailsResponse)
    async def list_guardrails_alias():
        """Alias for /guardrails"""
        return await list_guardrails()

    @app.post("/apply", response_model=ApplyResponse)
    async def apply_post(request: ApplyRequest):
        """
        Apply guardrail to prompt-output pairs

        Args:
            request: ApplyRequest with pairs and optional guardrail/winnow_down

        Returns:
            ApplyResponse with filtered results
        """
        try:
            pairs = [(p.prompt, p.output) for p in request.pairs]
            results = await asyncio.to_thread(
                main.apply,
                pairs=pairs,
                guardrail_name=request.guardrail,
                winnow_down=request.winnow_down,
                return_as_dict=True,
            )
            return ApplyResponse(results=results)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/apply", response_model=ApplyResponse)
    async def apply_get(
        prompt: str = Query(..., description="Prompt text"),
        output: str = Query(..., description="Output text"),
        guardrail: Optional[str] = Query(None, description="Guardrail name to use"),
        winnow_down: bool = Query(True, description="Keep valid (true) or invalid (false) items"),
    ):
        """
        Apply guardrail to single prompt-output pair (GET method)
        """
        if guardrail == "":
            guardrail = None
        request = ApplyRequest(
            pairs=[PairItem(prompt=prompt, output=output)],
            guardrail=guardrail,
            winnow_down=winnow_down,
        )
        return await apply_post(request)

    return app


app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: str = "configs/config.yaml",
    reload: bool = False,
):
    """Run the FastAPI server"""
    import uvicorn

    app_instance = create_app(config_path=config_path)
    uvicorn.run(app_instance, host=host, port=port, reload=reload)


def main():
    """Main entry point for guardrail-api command"""
    parser = argparse.ArgumentParser(description="Run CascadeGuard API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("-c", "--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
