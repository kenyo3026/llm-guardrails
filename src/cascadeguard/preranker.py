import importlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple, Union


@dataclass(frozen=True)
class ScannerType:
    INPUT  : Protocol = 'input'
    OUTPUT : Protocol = 'output'

class Preranker:

    def __init__(
        self,
        scanner: str,
        scanner_type: Union[ScannerType.INPUT, ScannerType.OUTPUT] = ScannerType.OUTPUT,
        scanner_kwargs: Dict[str, Any] = None,
    ):
        self.setup_scanner(
            scanner,
            scanner_type=scanner_type,
            scanner_kwargs=scanner_kwargs,
        )

    def setup_scanner(
        self,
        scanner: str,
        scanner_type: Union[ScannerType.INPUT, ScannerType.OUTPUT] = ScannerType.OUTPUT,
        scanner_kwargs: Dict[str, Any] = None,
    ):
        if scanner_kwargs is None:
            scanner_kwargs = {}
        self.scanner_kwargs = scanner_kwargs

        if model_kwargs := scanner_kwargs.get('model', {}):
            from llm_guard.model import Model
            scanner_kwargs = {**scanner_kwargs, 'model': Model(**model_kwargs)}

        if scanner_type == ScannerType.INPUT:
            module_name = 'llm_guard.input_scanners'
        elif scanner_type == ScannerType.OUTPUT:
            module_name = 'llm_guard.output_scanners'
        else:
            raise ValueError(f"Invalid scanner type: {scanner_type}")

        try:
            scanner_module = importlib.import_module(module_name)
            scanner_class = getattr(scanner_module, scanner)
        except (ImportError, AttributeError) as e:
            available = [
                name for name in dir(scanner_module)
                if not name.startswith('_') and name[0].isupper()
            ] if 'scanner_module' in locals() else []
            raise ValueError(
                f"Scanner '{scanner}' not found in {module_name}. "
                f"Available scanners: {', '.join(available)}"
            ) from e

        self.scanner = scanner_class(**scanner_kwargs)
