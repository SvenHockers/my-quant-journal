#!/usr/bin/env python3
"""
Streamlined backtest runner

This script loads configuration from YAML files and runs backtests
with minimal boilerplate code.
"""

import sys
import os
import yaml
import argparse
import inspect
import pkgutil
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from backtesting.engine import BacktestEngine


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _normalize_identifier(value: str) -> str:
    """Normalize strategy identifiers for comparison.

    Lowercase, remove separators, and drop common suffix tokens like 'strategy' or 'portfolio'.
    """
    if not isinstance(value, str):
        return ""
    lowered = value.lower()
    for ch in ["-", "_", " "]:
        lowered = lowered.replace(ch, "")
    # Remove common tokens anywhere in string to be lenient
    for token in ["strategy", "portfolio"]:
        lowered = lowered.replace(token, "")
    return lowered


def _camel_to_snake(name: str) -> str:
    out = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and not name[idx - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def discover_strategies():
    """Discover available strategies dynamically from the strategies package.

    Returns a dict mapping multiple identifiers to the class, and a canonical map of class->info.
    """
    from strategies.base_strategy import BaseStrategy  # local import after sys.path adjusted

    import strategies as strategies_pkg

    identifier_to_class = {}
    classes = []
    for mod_info in pkgutil.iter_modules(strategies_pkg.__path__):
        if mod_info.name.startswith("__"):
            continue
        try:
            module = importlib.import_module(f"strategies.{mod_info.name}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                classes.append((mod_info.name, obj))

    for module_name, cls in classes:
        # Build possible identifiers
        class_name = cls.__name__
        snake = _camel_to_snake(class_name)
        # Prefer to drop trailing '_strategy'
        for suffix in ["_strategy", "strategy"]:
            if snake.endswith(suffix):
                snake = snake[: -len(suffix)]
                break
        # Known aliases: module name itself is a good canonical id
        candidates = set()
        candidates.add(module_name)
        candidates.add(snake)
        # Also add variant without trailing '_portfolio'
        if snake.endswith("_portfolio"):
            candidates.add(snake[: -len("_portfolio")])

        # Normalized identifiers
        for cand in list(candidates):
            identifier_to_class[_normalize_identifier(cand)] = cls

    return identifier_to_class


def get_strategy_class(strategy_name: str):
    """Resolve a strategy class by a flexible name using discovery.

    Accepts module names (e.g., 'sma_crossover'), class-based slugs (e.g., 'sma_crossover_strategy'),
    and relaxed variants (e.g., 'regime_aware' mapping to 'regime_aware_portfolio').
    """
    identifiers = discover_strategies()
    key = _normalize_identifier(strategy_name)
    return identifiers.get(key)


def run_backtest(config_path):
    """Run backtest based on configuration file."""
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load market data
    symbols = config['data']['symbols']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
    data_dict = data_loader.get_multiple_symbols(symbols, start_date, end_date)

    # Optionally persist raw data for downstream analysis
    analysis_cfg = (config.get('analysis', {}) or {})
    if analysis_cfg.get('save_raw_data', False):
        raw_dir = Path(analysis_cfg.get('raw_data_dir') or analysis_cfg.get('output_dir') or 'analysis')
        # Default subdir if output_dir is used
        if raw_dir == Path(analysis_cfg.get('output_dir') or 'analysis'):
            raw_dir = raw_dir / 'raw_data'
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_format = (analysis_cfg.get('raw_data_format') or 'csv').lower()
        for sym, df in data_dict.items():
            try:
                if raw_format == 'parquet':
                    # Parquet requires pyarrow or fastparquet
                    df.to_parquet(raw_dir / f"{sym}.parquet")
                else:
                    df.to_csv(raw_dir / f"{sym}.csv")
            except Exception as e:
                print(f"Warning: failed to save raw data for {sym}: {e}")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_cash=config['backtesting']['initial_cash'],
        commission=config['backtesting']['commission']
    )
    
    # Add data feeds
    for symbol, data in data_dict.items():
        engine.add_data_feed(symbol, data)
    
    # Get strategy configuration
    strategy_name = config['strategy']['name']
    strategy_class = get_strategy_class(strategy_name)
    
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Get and sanitize strategy parameters against the selected strategy's declared params
    raw_params = config['strategy'].get('parameters', {}) or {}
    allowed_param_names = set()
    # Backtrader strategies declare params as a tuple of (name, default) pairs or a dict
    params_decl = getattr(strategy_class, 'params', ())
    try:
        if isinstance(params_decl, dict):
            allowed_param_names = set(params_decl.keys())
        else:
            # Expect iterable of pairs
            for item in params_decl:
                if isinstance(item, tuple) and len(item) >= 1:
                    allowed_param_names.add(item[0])
    except Exception:
        allowed_param_names = set()

    strategy_params = {k: v for k, v in raw_params.items() if k in allowed_param_names}
    unknown_params = sorted(set(raw_params.keys()) - allowed_param_names)
    if unknown_params:
        print(f"Warning: ignoring unknown parameters for '{strategy_name}': {unknown_params}")
    
    # Add strategy to engine
    engine.add_strategy(strategy_class, **strategy_params)
    
    # Run backtest
    print(f"Running {strategy_name} backtest...")
    results = engine.run()
    
    # Display results
    engine.display_results()
    
    # Save results if configured
    if config['backtesting'].get('save_results', True):
        output_dir = Path(config['backtesting']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        saved_path = engine.save_results(output_dir / f"{strategy_name}_results.json")

    # Plot classic Backtrader chart if configured
    if analysis_cfg.get('save_charts', True):
        chart_dpi = int(analysis_cfg.get('chart_dpi', 300))
        plot_style = analysis_cfg.get('plot_style', 'candle')
        symbols_str = ",".join(config['data']['symbols']) if config['data'].get('symbols') else ""
        result_key = f"{engine._strategy_class.__name__}_{symbols_str}"
        try:
            engine.plot_results(result_key, save_plot=True, dpi=chart_dpi, style=plot_style)
        except Exception as e:
            print(f"Warning: failed to generate chart: {e}")

    # Save analysis reports if configured
    if analysis_cfg.get('save_reports', True):
        analysis_out = Path(analysis_cfg.get('output_dir', 'analysis'))
        analysis_out.mkdir(parents=True, exist_ok=True)

        # Collect metrics from engine summary and strategy-specific metrics
        result_key = f"{engine._strategy_class.__name__}_{','.join(symbols)}"
        result_obj = engine.results.get(result_key, {})
        summary = engine._last_run_summary or {}
        strategy_obj = result_obj.get('strategy')
        try:
            strat_metrics = getattr(strategy_obj, 'get_strategy_metrics', None)
            if callable(strat_metrics):
                metrics = strat_metrics()
            else:
                metrics = getattr(strategy_obj, 'get_performance_summary', lambda: {})()
        except Exception:
            metrics = {}

        # Write metrics JSON
        try:
            import json
            with open(analysis_out / f"{strategy_name}_metrics.json", 'w') as f:
                json.dump({'summary': summary, 'metrics': metrics}, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: failed to save metrics JSON: {e}")

        # Write a simple Markdown report
        try:
            report_path = analysis_out / f"{strategy_name}_report.md"
            with open(report_path, 'w') as f:
                f.write(f"# Backtest Report: {strategy_name}\n\n")
                f.write(f"Symbols: {', '.join(symbols)}\n\n")
                f.write("## Summary\n")
                for k in [
                    'final_value', 'total_return', 'sharpe_ratio', 'max_drawdown', 'annual_return'
                ]:
                    if k in summary:
                        f.write(f"- {k}: {summary[k]}\n")
                if metrics:
                    f.write("\n## Strategy Metrics\n")
                    for mk, mv in metrics.items():
                        f.write(f"- {mk}: {mv}\n")
        except Exception as e:
            print(f"Warning: failed to save Markdown report: {e}")
    
    print("Backtest completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Run backtests using configuration files')
    parser.add_argument('config', help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        run_backtest(args.config)
    except Exception as e:
        print(f"Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
