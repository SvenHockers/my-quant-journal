"""
Backtesting engine for running trading strategy tests.
Provides standardized framework for strategy evaluation.
"""

import backtrader as bt
import pandas as pd
from typing import Dict, Any, List, Optional, Type
import os

from strategies.base_strategy import BaseStrategy
from data.data_loader import DataLoader


class BacktestEngine:
    """
    Standardized backtesting engine for trading strategies.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission: float = 0.0,
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

        # Maintain a persistent Cerebro instance for multi-asset workflows
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # Track data feeds and strategy configuration for run()
        self._symbols: List[str] = []
        self._strategy_class: Optional[Type[BaseStrategy]] = None
        self._strategy_params: Dict[str, Any] = {}
        self._last_run_summary: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------
    # New API used by examples/run_backtest.py
    # -------------------------------------------------------------
    def add_data_feed(self, symbol: str, data: pd.DataFrame) -> None:
        """Add a Pandas OHLCV DataFrame as a data feed to the engine.

        The DataFrame is expected to have columns: Open, High, Low, Close, Volume
        and a DatetimeIndex.
        """
        if data is None or data.empty:
            raise ValueError(f"Empty data provided for symbol {symbol}")

        # Basic validation using DataLoader's validator
        if not self.data_loader.validate_data(data):
            raise ValueError(f"Invalid data for {symbol}")

        data_feed = bt.feeds.PandasData(dataname=data)
        self.cerebro.adddata(data_feed, name=symbol)
        self._symbols.append(symbol)

    def add_strategy(self, strategy_class: Type[BaseStrategy], **strategy_params: Any) -> None:
        """Configure the strategy for the run() execution path."""
        self._strategy_class = strategy_class
        self._strategy_params = strategy_params or {}

    def run(self) -> Dict[str, Any]:
        """Execute a backtest using the data feeds and strategy previously added."""
        if self._strategy_class is None:
            raise ValueError("Strategy not configured. Call add_strategy() first.")

        # Add the strategy to Cerebro
        if self._strategy_params:
            self.cerebro.addstrategy(self._strategy_class, **self._strategy_params)
        else:
            self.cerebro.addstrategy(self._strategy_class)

        # Analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        results = self.cerebro.run()
        strategy = results[0]

        # Extract results
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # Analyzer results with safe access
        def safe_get(dct: Dict[str, Any], keys: List[str], default=None):
            current = dct
            try:
                for k in keys:
                    current = current[k]
                return current
            except Exception:
                return default

        sharpe_ratio = safe_get(strategy.analyzers.sharpe.get_analysis(), ['sharperatio'])
        max_drawdown = safe_get(strategy.analyzers.drawdown.get_analysis(), ['max', 'drawdown'])
        annual_return = safe_get(strategy.analyzers.returns.get_analysis(), ['rnorm100'])
        trade_analysis = strategy.analyzers.trades.get_analysis()

        # Compile summary consistent with existing results schema
        symbols_str = ",".join(self._symbols) if self._symbols else ""
        result_key = f"{self._strategy_class.__name__}_{symbols_str}"
        summary = {
            'strategy_name': self._strategy_class.__name__,
            'symbols': self._symbols,
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'trade_analysis': trade_analysis,
            'strategy_performance': getattr(strategy, 'get_performance_summary', lambda: {})(),
        }

        self.results[result_key] = {**summary, 'cerebro': self.cerebro, 'strategy': strategy}
        self._last_run_summary = summary
        return summary

    def display_results(self) -> None:
        """Print a concise summary of the last run."""
        if not self._last_run_summary:
            print("No results to display. Run the backtest first.")
            return

        s = self._last_run_summary
        print(
            f"Strategy: {s['strategy_name']} | Symbols: {', '.join(s['symbols'])}\n"
            f"Final Value: ${s['final_value']:,.2f} | Return: {s['total_return']:.2%}\n"
            f"Sharpe: {s['sharpe_ratio']} | Max DD: {s['max_drawdown']} | Annual Ret%: {s['annual_return']}"
        )

    def save_results(self, arg: Any):
        """Save results.

        Backward-compatible behavior:
        - If a filesystem path (str/Path) is passed, save the last run summary as JSON there.
        - Otherwise, treat the argument as a result key and save a CSV summary as before.
        """
        # Path-like: save JSON summary of last run
        try:
            from pathlib import Path
            import json
            if isinstance(arg, (str, os.PathLike, Path)) and (str(arg).endswith('.json') or '/' in str(arg)):
                if not self._last_run_summary:
                    raise ValueError("No results available to save. Run the backtest first.")
                path = Path(arg)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(self._last_run_summary, f, indent=2, default=str)
                print(f"Results saved to {path}")
                return str(path)
        except Exception as e:
            raise e

        # Fallback to legacy behavior treating arg as result_key and writing CSV
        result_key = str(arg)
        return self._save_results_legacy(result_key)

    # Internal helper for legacy save_results behavior
    def _save_results_legacy(self, result_key: str):
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
                result.get('strategy_name'), ','.join(result.get('symbols', [])), None,
                None, result.get('initial_cash'), result.get('final_value'),
                result.get('total_return'), result.get('sharpe_ratio'),
                result.get('max_drawdown'), result.get('annual_return')
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{result_key}_summary.csv"
        summary_path = os.path.join(self.output_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)

        print(f"Results saved to {summary_path}")
        return summary_path
    
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
    
    def plot_results(self, result_key: str, save_plot: bool = True, dpi: int = 300, style: str = 'candle'):
        """
        Plot backtest results.
        
        Args:
            result_key: Key for the results to plot
            save_plot: Whether to save the plot
            dpi: Resolution (dots per inch) for saved plot
        """
        if result_key not in self.results:
            raise ValueError(f"Result key {result_key} not found")
        
        result = self.results[result_key]
        cerebro = result['cerebro']
        
        # Plot. Backtrader returns a list of lists of figures.
        plots = cerebro.plot(style=style)
        if not isinstance(plots, (list, tuple)):
            plots = [[plots]]

        saved_paths = []
        for i, figset in enumerate(plots):
            # Some backtrader versions return a single list, normalize
            if not isinstance(figset, (list, tuple)):
                figset = [figset]
            for j, fig in enumerate(figset):
                if save_plot:
                    suffix = f"_{i}_{j}" if (i or j) else ""
                    plot_filename = f"{result_key}_plot{suffix}.png"
                    plot_path = os.path.join(self.output_dir, plot_filename)
                    try:
                        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                        saved_paths.append(plot_path)
                    except Exception as e:
                        print(f"Failed to save plot {plot_filename}: {e}")

        if saved_paths:
            print(f"Plot saved to {saved_paths[0]}" + (f" and {len(saved_paths)-1} more" if len(saved_paths) > 1 else ""))
        return plots
    
    # NOTE: legacy save_results moved into save_results() with compatibility layer
