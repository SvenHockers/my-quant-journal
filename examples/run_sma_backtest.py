#!/usr/bin/env python3
"""
Example script demonstrating the SMA Crossover strategy backtest.
Shows how to use the complete backtesting framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.backtesting.engine import BacktestEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.analysis.performance import PerformanceAnalyzer


def main():
    """Run SMA crossover strategy backtest example."""
    
    print("=" * 60)
    print("SMA CROSSOVER STRATEGY BACKTEST EXAMPLE")
    print("=" * 60)
    
    # Initialize backtesting engine
    engine = BacktestEngine(
        initial_cash=100000.0,
        commission=0.001,
        output_dir="results"
    )
    
    # Strategy parameters
    strategy_params = {
        'fast_period': 10,
        'slow_period': 30,
        'printlog': True
    }
    
    # Run backtest
    print("\nRunning backtest...")
    results = engine.run_backtest(
        strategy_class=SMACrossoverStrategy,
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31',
        strategy_params=strategy_params
    )
    
    # Save results
    print("\nSaving results...")
    engine.save_results(list(engine.results.keys())[0])
    
    # Create performance analysis
    print("\nCreating performance analysis...")
    analyzer = PerformanceAnalyzer(output_dir="analysis")
    
    # Generate performance report
    report_path = analyzer.create_performance_report(results)
    print(f"Performance report created: {report_path}")
    
    # Generate performance charts
    figures = analyzer.plot_performance_charts(results)
    print(f"Generated {len(figures)} performance charts")
    
    # Parameter optimization example
    print("\nRunning parameter optimization...")
    param_ranges = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50]
    }
    
    optimization_results = engine.run_parameter_optimization(
        SMACrossoverStrategy,
        'AAPL',
        '2022-01-01',
        '2023-12-31',
        param_ranges
    )
    
    print("\nParameter optimization results:")
    print(optimization_results.head())
    
    # Save optimization results
    optimization_path = "results/optimization_results.csv"
    optimization_results.to_csv(optimization_path, index=False)
    print(f"\nOptimization results saved to: {optimization_path}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck the following directories for outputs:")
    print("- results/: Backtest results and optimization data")
    print("- analysis/: Performance reports and charts")
    print("- data/cache/: Cached market data")


if __name__ == "__main__":
    main()
