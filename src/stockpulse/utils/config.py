"""Configuration management for StockPulse."""

import os
from pathlib import Path
from typing import Any

import yaml

_config: dict[str, Any] | None = None


def find_config_file() -> Path:
    """Find the config file, checking multiple locations."""
    # Check common locations
    locations = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path(__file__).parent.parent.parent.parent / "config" / "config.yaml",
    ]

    for loc in locations:
        if loc.exists():
            return loc

    raise FileNotFoundError(
        f"Config file not found. Searched: {[str(l) for l in locations]}"
    )


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    global _config

    if config_path is None:
        config_path = find_config_file()

    config_path = Path(config_path)

    with open(config_path) as f:
        _config = yaml.safe_load(f)

    # Override with environment variables
    _apply_env_overrides(_config)

    return _config


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply environment variable overrides to config."""
    env_mappings = {
        "STOCKPULSE_EMAIL_SENDER": ("email", "sender"),
        "STOCKPULSE_EMAIL_RECIPIENT": ("email", "recipient"),
        "STOCKPULSE_EMAIL_PASSWORD": ("email", "password"),
        "STOCKPULSE_DB_PATH": ("database", "path"),
    }

    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value:
            _set_nested(config, path, value)


def _set_nested(d: dict, path: tuple[str, ...], value: Any) -> None:
    """Set a nested dictionary value."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def get_config() -> dict[str, Any]:
    """Get the loaded configuration, loading if necessary."""
    global _config
    if _config is None:
        load_config()
    return _config
