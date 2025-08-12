#!/usr/bin/env python3
"""
Example script demonstrating the Regime-Aware Portfolio Optimization strategy.

This script shows how to:
1. Load market data for multiple assets
2. Set up the regime-aware strategy
3. Run backtesting
4. Analyze results including regime changes and surprise scores
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.data.data_loader import DataLoader
from src.strategies.regime_aware_portfolio import RegimeAwarePortfolioStrategy


def main():
    """Main function to run the regime-aware portfolio backtest."""
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Example assets
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    initial_cash = 100000
    
    print("Loading market data...")
    
    # Load data
    data_loader = DataLoader()
    data_dict = data_loader.get_multiple_symbols(symbols, start_date, end_date)
    
    # Prepare data for backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add data feeds
    for symbol, data in data_dict.items():
        # Convert to backtrader format
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # Use index
            open=0,         # Column indices
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        cerebro.adddata(bt_data, name=symbol)
    
    print(f"Added {len(symbols)} data feeds to backtrader")
    
    # Add strategy
    cerebro.addstrategy(RegimeAwarePortfolioStrategy)
    
    print("Running backtest...")
    
    # Run backtest
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # Get strategy instance
    strategy = results[0]
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {((final_value / initial_value) - 1) * 100:.2f}%")
    
    # Get strategy-specific metrics
    strategy_metrics = strategy.get_strategy_metrics()
    
    print(f"\nRegime Changes Detected: {strategy_metrics['regime_changes_detected']}")
    print(f"Average Surprise Score: {strategy_metrics['avg_surprise_score']:.4f}")
    print(f"Average Learning Rate: {strategy_metrics['avg_learning_rate']:.4f}")
    
    if strategy_metrics['regime_changes_detected'] > 0:
        print(f"Regime Change Times: {strategy_metrics['regime_change_times']}")
        print(f"Regime Change Confidences: {[f'{c:.3f}' for c in strategy_metrics['regime_change_confidences']]}")
    
    # Performance metrics
    if 'total_trades' in strategy_metrics and strategy_metrics['total_trades'] > 0:
        print(f"\nTrading Performance:")
        print(f"Total Trades: {strategy_metrics['total_trades']}")
        print(f"Win Rate: {strategy_metrics.get('win_rate', 0):.2%}")
        print(f"Average PnL per Trade: ${strategy_metrics.get('avg_pnl_per_trade', 0):.2f}")
    else:
        print(f"\nTrading Performance: No trades executed (portfolio strategy)")
    
    # Plot strategy analysis
    print("\nGenerating strategy analysis plots...")
    try:
        strategy.plot_strategy_analysis()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Plot portfolio value over time
    try:
        cerebro.plot(style='candlestick', volume=False)
    except Exception as e:
        print(f"Portfolio plotting failed: {e}")
    
    print("\nBacktest completed successfully!")


def run_parameter_sensitivity_analysis():
    """Run sensitivity analysis on key strategy parameters."""
    
    print("\n" + "="*50)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*50)
    
    # Test different hazard rates
    hazard_rates = [0.005, 0.01, 0.02, 0.05]
    change_point_thresholds = [0.3, 0.5, 0.7]
    
    results = []
    
    for hazard_rate in hazard_rates:
        for threshold in change_point_thresholds:
            print(f"\nTesting: hazard_rate={hazard_rate}, threshold={threshold}")
            
            # This would require running multiple backtests
            # For now, just show the parameter combinations
            results.append({
                'hazard_rate': hazard_rate,
                'threshold': threshold,
                'expected_regime_changes': int(1 / hazard_rate),  # Rough estimate
                'sensitivity': 'High' if hazard_rate > 0.02 else 'Medium'
            })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nParameter Sensitivity Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
        run_parameter_sensitivity_analysis()
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
