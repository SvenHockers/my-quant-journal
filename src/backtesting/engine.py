"""
Backtesting engine for running trading strategy tests.
Provides standardized framework for strategy evaluation.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
import matplotlib.pyplot as plt
import os

from ..strategies.base_strategy import BaseStrategy
from ..data.data_loader import DataLoader


class BacktestEngine:
    """
    Standardized backtesting engine for trading strategies.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission: float = 0.001,
                 output_dir: str = "results"):
        """
        Initialize backtesting engine.
        
        Args:
            initial_cash: Starting capital
            commission: Commission rate per trade
            output_dir: Directory for results output
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.output_dir = output_dir
        self.data_loader = DataLoader()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.comparisons = {}
    
    def run_backtest(self,
                    strategy_class: Type[BaseStrategy],
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    strategy_params: Optional[Dict[str, Any]] = None,
                    data_interval: str = "1d") -> Dict[str, Any]:
        """
        Run a single backtest for a strategy.
        
        Args:
            strategy_class: Strategy class to test
            symbol: Stock symbol to test on
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_params: Parameters to pass to strategy
            data_interval: Data interval for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        print(f"Running backtest for {strategy_class.__name__} on {symbol}")
        
        # Get market data
        data = self.data_loader.get_data(symbol, start_date, end_date, data_interval)
        
        if not self.data_loader.validate_data(data):
            raise ValueError(f"Invalid data for {symbol}")
        
        # Create Backtrader engine
        cerebro = bt.Cerebro()
        
        # Add data feed
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Set initial cash and commission
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        results = cerebro.run()
        strategy = results[0]
        
        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Get analyzer results
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
        max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
        annual_return = strategy.analyzers.returns.get_analysis()['rnorm100']
        
        # Get trade analysis
        trade_analysis = strategy.analyzers.trades.get_analysis()
        
        # Compile results
        backtest_results = {
            'strategy_name': strategy_class.__name__,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'trade_analysis': trade_analysis,
            'strategy_performance': strategy.get_performance_summary(),
            'cerebro': cerebro,
            'strategy': strategy
        }
        
        # Store results
        key = f"{strategy_class.__name__}_{symbol}_{start_date}_{end_date}"
        self.results[key] = backtest_results
        
        print(f"Backtest completed. Final value: ${final_value:,.2f}, "
              f"Return: {total_return:.2%}")
        
        return backtest_results
    
    def run_parameter_optimization(self,
                                 strategy_class: Type[BaseStrategy],
                                 symbol: str,
                                 start_date: str,
                                 end_date: str,
                                 param_ranges: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy_class: Strategy class to optimize
            symbol: Stock symbol to test on
            start_date: Start date for backtest
            end_date: End date for backtest
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            DataFrame with optimization results
        """
        print(f"Running parameter optimization for {strategy_class.__name__}")
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        optimization_results = []
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                result = self.run_backtest(
                    strategy_class, symbol, start_date, end_date, params
                )
                
                # Extract key metrics
                optimization_results.append({
                    **params,
                    'final_value': result['final_value'],
                    'total_return': result['total_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown']
                })
                
            except Exception as e:
                print(f"Failed to test combination {params}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(optimization_results)
        
        # Sort by Sharpe ratio (best first)
        if not results_df.empty:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        return results_df
    
    def compare_strategies(self,
                          strategies: List[Type[BaseStrategy]],
                          symbol: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.
        
        Args:
            strategies: List of strategy classes to compare
            symbol: Stock symbol to test on
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Comparing {len(strategies)} strategies on {symbol}")
        
        comparison_results = []
        
        for strategy_class in strategies:
            try:
                result = self.run_backtest(
                    strategy_class, symbol, start_date, end_date
                )
                
                comparison_results.append({
                    'Strategy': strategy_class.__name__,
                    'Final Value': result['final_value'],
                    'Total Return': result['total_return'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown': result['max_drawdown'],
                    'Annual Return': result['annual_return']
                })
                
            except Exception as e:
                print(f"Failed to test {strategy_class.__name__}: {e}")
                continue
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Store comparison
        key = f"comparison_{symbol}_{start_date}_{end_date}"
        self.comparisons[key] = comparison_df
        
        return comparison_df
    
    def plot_results(self, result_key: str, save_plot: bool = True):
        """
        Plot backtest results.
        
        Args:
            result_key: Key for the results to plot
            save_plot: Whether to save the plot
        """
        if result_key not in self.results:
            raise ValueError(f"Result key {result_key} not found")
        
        result = self.results[result_key]
        cerebro = result['cerebro']
        
        # Plot
        fig = cerebro.plot(style='candlestick')[0][0]
        
        if save_plot:
            plot_filename = f"{result_key}_plot.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        return fig
    
    def save_results(self, result_key: str):
        """
        Save backtest results to file.
        
        Args:
            result_key: Key for the results to save
        """
        if result_key not in self.results:
            raise ValueError(f"Result key {result_key} not found")
        
        result = self.results[result_key]
        
        # Save summary to CSV
        summary_data = {
            'Metric': [
                'Strategy Name', 'Symbol', 'Start Date', 'End Date',
                'Initial Cash', 'Final Value', 'Total Return',
                'Sharpe Ratio', 'Max Drawdown', 'Annual Return'
            ],
            'Value': [
                result['strategy_name'], result['symbol'], result['start_date'],
                result['end_date'], result['initial_cash'], result['final_value'],
                result['total_return'], result['sharpe_ratio'],
                result['max_drawdown'], result['annual_return']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{result_key}_summary.csv"
        summary_path = os.path.join(self.output_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Results saved to {summary_path}")
        
        return summary_path
