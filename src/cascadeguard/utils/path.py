"""
Path resolution utilities for CascadeGuard

Resolves configuration file paths across different runtime environments (CLI, API).
"""

import pathlib
from typing import Union


def resolve_config_path(config_path: Union[str, pathlib.Path]) -> str:
    """
    Resolve configuration file path

    Tries multiple locations in order:
    1. Absolute path or path relative to current working directory
    2. Path relative to project root
    3. configs/config.yaml in project root
    4. ~/.cascadeguard/config.yaml in home directory

    Args:
        config_path: Path to configuration file

    Returns:
        Resolved absolute path to config file

    Raises:
        FileNotFoundError: If config file cannot be found in any location
    """
    path = pathlib.Path(config_path)

    if path.exists():
        return str(path.absolute())

    # Navigate from src/cascadeguard/utils/path.py -> project root
    project_root = pathlib.Path(__file__).parent.parent.parent.parent
    path = project_root / config_path
    if path.exists():
        return str(path.absolute())

    path = project_root / "configs" / "config.yaml"
    if path.exists():
        return str(path.absolute())

    path = pathlib.Path.home() / ".cascadeguard" / "config.yaml"
    if path.exists():
        return str(path.absolute())

    raise FileNotFoundError(
        f"Configuration file not found. Tried:\n"
        f"  - {config_path}\n"
        f"  - {project_root / config_path}\n"
        f"  - {project_root / 'configs' / 'config.yaml'}\n"
        f"  - {pathlib.Path.home() / '.cascadeguard' / 'config.yaml'}"
    )
