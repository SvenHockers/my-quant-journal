#!/usr/bin/env python3
"""
Config validator for my-quant-journal.

Validates YAML config files, discovers available strategies dynamically,
and checks that provided strategy parameters match the selected strategy's
declared Backtrader params.

Usage:
  python scripts/validate_config.py validate --config path/to/config.yaml [--strict-params]
  python scripts/validate_config.py list-strategies
"""

import sys
import os
import argparse
import yaml
import inspect
import pkgutil
import importlib
from pathlib import Path
from typing import Dict, Any, Tuple, Set


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def normalize_identifier(value: str) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower()
    for ch in ("-", "_", " "):
        lowered = lowered.replace(ch, "")
    for token in ("strategy", "portfolio"):
        lowered = lowered.replace(token, "")
    return lowered


def camel_to_snake(name: str) -> str:
    out = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and not name[idx - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def discover_strategies() -> Dict[str, type]:
    from strategies.base_strategy import BaseStrategy  # import after sys.path updated
    import strategies as strategies_pkg

    identifier_to_class: Dict[str, type] = {}

    for mod_info in pkgutil.iter_modules(strategies_pkg.__path__):
        if mod_info.name.startswith("__"):
            continue
        try:
            module = importlib.import_module(f"strategies.{mod_info.name}")
        except Exception:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                class_name = obj.__name__
                snake = camel_to_snake(class_name)
                for suffix in ("_strategy", "strategy"):
                    if snake.endswith(suffix):
                        snake = snake[: -len(suffix)]
                        break

                candidates = set([mod_info.name, snake])
                if snake.endswith("_portfolio"):
                    candidates.add(snake[: -len("_portfolio")])

                for cand in candidates:
                    identifier_to_class[normalize_identifier(cand)] = obj

    return identifier_to_class


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def collect_allowed_params(strategy_cls: type) -> Set[str]:
    allowed: Set[str] = set()
    params_decl = getattr(strategy_cls, "params", ())
    if isinstance(params_decl, dict):
        allowed = set(params_decl.keys())
    else:
        try:
            for item in params_decl:
                if isinstance(item, tuple) and len(item) >= 1:
                    allowed.add(item[0])
        except Exception:
            pass
    return allowed


def validate_config(config_path: str, strict_params: bool = False) -> Tuple[bool, str]:
    try:
        cfg = load_config(config_path)
    except Exception as e:
        return False, f"Failed to load YAML: {e}"

    # Basic structure
    missing_keys = [k for k in ("backtesting", "data", "strategy") if k not in cfg]
    if missing_keys:
        return False, f"Missing required top-level keys: {missing_keys}"

    data = cfg.get("data", {})
    if not isinstance(data.get("symbols"), list) or not data.get("symbols"):
        return False, "data.symbols must be a non-empty list"

    # Strategy discovery and validation
    strategies = discover_strategies()
    raw_name = (cfg.get("strategy", {}) or {}).get("name")
    if not raw_name:
        return False, "strategy.name is required"
    norm_name = normalize_identifier(raw_name)
    strategy_cls = strategies.get(norm_name)
    if not strategy_cls:
        # Build a helpful list
        available = sorted(set(strategies.keys()))
        return False, (
            f"Unknown strategy '{raw_name}'. Available: "
            + ", ".join(available)
        )

    params = (cfg.get("strategy", {}) or {}).get("parameters", {}) or {}
    allowed = collect_allowed_params(strategy_cls)
    unknown = sorted(set(params.keys()) - allowed)

    # Prepare info string
    info_lines = [
        f"Detected strategy: {raw_name}",
        f"Class: {strategy_cls.__name__}",
        f"Allowed params: {sorted(allowed)}",
    ]

    if unknown:
        msg = f"Unknown parameters for strategy '{raw_name}': {unknown}"
        if strict_params:
            return False, "\n".join(info_lines + [msg])
        else:
            info_lines.append("Warning: " + msg)

    # Light check: date fields exist
    for key in ("start_date", "end_date"):
        if key not in data:
            info_lines.append(f"Warning: data.{key} not set")
    return True, "\n".join(info_lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate backtest config files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate a config file")
    p_val.add_argument("--config", required=True, help="Path to YAML config file")
    p_val.add_argument("--strict-params", action="store_true", help="Fail on unknown strategy parameters")

    p_list = sub.add_parser("list-strategies", help="List discovered strategies")

    args = parser.parse_args(argv)

    if args.cmd == "list-strategies":
        strategies = discover_strategies()
        if not strategies:
            print("No strategies discovered.")
            return 1
        # Render distinct class names and identifiers
        classes = {}
        for ident, cls in strategies.items():
            classes.setdefault(cls.__name__, set()).add(ident)
        for cls_name, idents in sorted(classes.items()):
            print(f"{cls_name}: {sorted(idents)}")
        return 0

    if args.cmd == "validate":
        ok, message = validate_config(args.config, strict_params=args.strict_params)
        print(message)
        return 0 if ok else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


