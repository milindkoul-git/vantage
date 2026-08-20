"""Configuration loading: defaults -> YAML file -> CLI overrides.

Later layers win, and every layer is optional, so the platform starts with no
config file at all while remaining fully configurable in deployment.

The loader is strict on purpose. An unknown or misspelled key is an error with
a suggestion attached, never a silently ignored setting - a typo'd
``targt_fps`` that quietly does nothing is exactly the kind of failure that
costs an afternoon.
"""

from __future__ import annotations

import dataclasses
import os
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from vantage.config.schema import VantageConfig
from vantage.core.errors import ConfigError

_ENV_VAR = "VANTAGE_CONFIG"


def default_config_path() -> Path:
    """Path to the bundled default configuration file."""
    return Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: list[str] | None = None,
    *,
    require_file: bool = False,
) -> VantageConfig:
    """Build a :class:`VantageConfig`.

    Args:
        path: YAML file to read. Falls back to ``$VANTAGE_CONFIG``, then to the
            bundled ``configs/default.yaml`` if it exists.
        overrides: ``dotted.key=value`` strings from the command line. Values are
            parsed as YAML scalars, so ``true``, ``30``, ``null`` and ``1.5``
            arrive as the right Python types.
        require_file: Raise if the resolved file does not exist, instead of
            falling back to built-in defaults.
    """
    data: dict[str, Any] = {}

    resolved = _resolve_path(path)
    if resolved is not None:
        if not resolved.is_file():
            if require_file or path is not None:
                raise ConfigError(f"configuration file not found: {resolved}")
        else:
            data = _read_yaml(resolved)
    elif require_file:
        raise ConfigError("no configuration file specified and no default found")

    for override in overrides or []:
        _apply_override(data, override)

    return _build(VantageConfig, data, path="")


def _resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser()
    bundled = default_config_path()
    return bundled if bundled.is_file() else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: could not be read: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must be 'dotted.key=value', got {override!r}")
    dotted, _, raw_value = override.partition("=")
    keys = [part for part in dotted.strip().split(".") if part]
    if not keys:
        raise ConfigError(f"override is missing a key: {override!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value  # a bare string that isn't valid YAML is still a string

    cursor = data
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively construct dataclass ``cls`` from mapping ``data``."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        key = unknown[0]
        raise ConfigError(
            f"unknown configuration key '{_join(path, key)}'"
            f"{_suggest(key, fields)}. Valid keys here: {sorted(fields)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, _join(path, name))

    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:  # pragma: no cover - schema guards first
        raise ConfigError(f"{path or 'config'}: {exc}") from exc


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(args[0], value, path)
        return value  # multi-type unions are passed through to the field validator

    if origin is list:
        # Without this, a bare string would satisfy the "sequence" duck-type and
        # then be iterated character by character downstream.
        if not isinstance(value, list):
            raise ConfigError(
                f"{path}: expected a list, got {value!r}. "
                "In YAML use '[person, car]' or a '- item' block."
            )
        # get_args returns a tuple; the name is reused from the union branch
        # above where it held a list. Reflection over annotations is dynamic
        # by nature and a checker cannot follow it.
        args = get_args(annotation)  # type: ignore[assignment]
        item_type = args[0] if args else str
        return [
            _coerce(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)
        ]

    if dataclasses.is_dataclass(annotation):
        # is_dataclass narrows to "instance or class"; here it is always the
        # class, because annotations are types.
        return _build(annotation, value, path)  # type: ignore[arg-type]

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _coerce_enum(annotation, value, path)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected true or false, got {value!r}")

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    return value


def _coerce_enum(enum_cls: type[Enum], value: Any, path: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.lower())
        except ValueError:
            pass
    valid = [member.value for member in enum_cls]
    raise ConfigError(f"{path}: expected one of {valid}, got {value!r}")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _suggest(key: str, fields: dict[str, Any]) -> str:
    """Offer the closest valid key, when one is close enough to be a typo."""
    import difflib

    close = difflib.get_close_matches(key, list(fields), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""
