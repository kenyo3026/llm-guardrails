import pathlib
from typing import Union, List, Tuple, Optional, Dict, Any
from config_morpher import ConfigMorpher

from .guardrail import CascadeGuard, RankData


class Main:
    """Core main class for CascadeGuard LLM Guardrails"""

    def __init__(
        self,
        config_path: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]] = '../configs/config.yaml'
    ):
        """
        Initialize Main with configuration

        Args:
            config_path: Path(s) to config file(s). Can be a single path or list of paths.
                        If list, configs will be merged by config-morpher.
        """
        self.config_morpher = ConfigMorpher(config_path)

    def _get_guardrail_kwargs(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get guardrail configuration by name or use first guardrail

        Args:
            name: Guardrail name. If None, uses first guardrail.

        Returns:
            Dictionary of guardrail configuration
        """
        if name:
            # Use config_morpher path syntax (following drowcoder)
            kwargs = self.config_morpher.fetch(f'guardrails[name={name}]', None)
            if not kwargs:
                raise ValueError(f"Guardrail '{name}' not found in config")
        else:
            # Use first guardrail (following drowcoder pattern: models[0])
            kwargs = self.config_morpher.fetch('guardrails[0]', None)
            if not kwargs:
                raise ValueError("No guardrail configured")

        # Return a copy to avoid modifying the original config
        return dict(kwargs)

    def setup_guardrail(self, name: Optional[str] = None) -> CascadeGuard:
        """
        Setup guardrail instance

        Args:
            name: Guardrail name. If None, uses first guardrail.

        Returns:
            CascadeGuard instance
        """
        guardrail_kwargs = self._get_guardrail_kwargs(name)
        guardrail_kwargs.pop("name", None)  # Remove name field before passing to CascadeGuard
        return CascadeGuard(**guardrail_kwargs)

    def apply(
        self,
        pairs: List[Tuple[str, str]],
        guardrail_name: Optional[str] = None,
        winnow_down: bool = True,
        return_as_dict: bool = True,
    ) -> Union[List[RankData], List[Dict[str, Any]]]:
        """
        Execute guardrail filtering

        Args:
            pairs: List of (prompt, output) tuples to filter
            guardrail_name: Guardrail name to use. If None, uses first guardrail.
            winnow_down: If True, keep valid items; if False, keep invalid items
            return_as_dict: If True, return as list of dicts; if False, return as list of RankData

        Returns:
            Filtered results as list of RankData or dicts
        """
        guard = self.setup_guardrail(name=guardrail_name)
        return guard.apply(pairs, winnow_down=winnow_down, return_as_dict=return_as_dict)

    def list_guardrails(self) -> List[str]:
        """List all available guardrail names"""
        guardrails = self.config_morpher.fetch('guardrails', [])
        return [g.get('name', 'unnamed') for g in guardrails]
