#!/usr/bin/env python3
"""
Example: Run a grid search over SMA crossover parameters.

Usage:
  python -m utils.optimisers.example_grid_search
  # or, from repo root
  python src/utils/optimisers/example_grid_search.py

This script downloads AAPL daily data (cached), runs grid search
for SMACrossoverStrategy, prints the best params, and runs a final
backtest with those parameters.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path when executed directly
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtesting.engine import BacktestEngine
from strategies.sma_crossover import SMACrossoverStrategy
from utils.optimisers import GridSearchOptimiser


def main() -> int:
    engine = BacktestEngine(initial_cash=100_000, commission=0.001)

    # Define the parameter grid for the SMA strategy
    param_grid = {
        'fast_period': [5, 10, 20],
        'slow_period': [30, 50, 100],
        'printlog': [False],
    }

    optimiser = GridSearchOptimiser(
        engine,
        metric='sharpe_ratio',  # or a callable: lambda s: s.get('total_return', float('-inf'))
        mode='max',
        verbose=True,
    )

    result = optimiser.optimise(
        SMACrossoverStrategy,
        param_grid,
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2023-01-01',
    )

    best_params = result['best_params']
    best_score = result['best_score']
    print("\n=== Grid Search Complete ===")
    print("Best params:", best_params)
    print("Best score:", best_score)

    # Run a final backtest with the best parameters
    print("\nRunning final backtest with best parameters...")
    summary = engine.run_backtest(
        strategy_class=SMACrossoverStrategy,
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2023-01-01',
        strategy_params=best_params,
    )
    engine.display_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


