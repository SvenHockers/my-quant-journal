"""
Notebook helper functions for easier experimentation and analysis.
Provides convenient utilities for Jupyter notebook workflows.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import warnings

# Suppress warnings for cleaner notebook output
warnings.filterwarnings('ignore')


def setup_notebook_environment(style: str = 'seaborn-v0_8', figsize: tuple = (12, 8)):
    """
    Set up the notebook environment with consistent plotting styles and configurations.
    
    Args:
        style: Matplotlib style to use
        figsize: Default figure size for plots
    """
    # Set plotting style
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    
    # Configure seaborn
    sns.set_palette("husl")
    sns.set_style("whitegrid")
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = 100
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    
    print("✅ Notebook environment configured successfully!")
    print(f"   - Plotting style: {style}")
    print(f"   - Default figure size: {figsize}")
    print(f"   - Pandas display options optimized")


def add_src_to_path():
    """
    Add the src directory to Python path for easy imports.
    """
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, 'src')
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ Added {src_path} to Python path")
    else:
        print(f"✅ {src_path} already in Python path")


def quick_data_preview(data: pd.DataFrame, symbol: str = "", max_rows: int = 10):
    """
    Quick preview of market data with key statistics.
    
    Args:
        data: DataFrame with market data
        symbol: Symbol name for display
        max_rows: Maximum rows to display
    """
    print(f"📊 Data Preview: {symbol}" if symbol else "📊 Data Preview")
    print("=" * 60)
    
    # Basic info
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Columns: {list(data.columns)}")
    
    # Sample data
    print(f"\nFirst {max_rows} rows:")
    print(data.head(max_rows))
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(data.describe())
    
    # Missing data check
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️  Missing data:")
        print(missing_data[missing_data > 0])
    else:
        print(f"\n✅ No missing data found")


def plot_price_volume(data: pd.DataFrame, symbol: str = "", figsize: tuple = (14, 8)):
    """
    Create a professional price and volume chart.
    
    Args:
        data: DataFrame with OHLCV data
        symbol: Symbol name for title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1.5, color='#1f77b4')
    ax1.set_title(f'{symbol} - Price and Volume Analysis' if symbol else 'Price and Volume Analysis', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Volume chart
    ax2.bar(data.index, data['Volume'], alpha=0.6, color='#2ca02c')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_strategies_side_by_side(results_dict: Dict[str, Dict[str, Any]], 
                                   metrics: List[str] = None):
    """
    Create a side-by-side comparison of multiple strategy results.
    
    Args:
        results_dict: Dictionary of strategy results
        metrics: List of metrics to compare
    """
    if metrics is None:
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annual_return']
    
    # Extract comparison data
    comparison_data = []
    for strategy_name, results in results_dict.items():
        row = {'Strategy': strategy_name}
        for metric in metrics:
            if metric in results:
                row[metric] = results[metric]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if metric in comparison_df.columns:
            values = comparison_df[metric]
            
            # Color coding for returns vs risk metrics
            if 'return' in metric.lower():
                colors = ['green' if x > 0 else 'red' for x in values]
            else:
                colors = ['red' if x < 0 else 'blue' for x in values]
            
            bars = ax.bar(comparison_df['Strategy'], values, color=colors, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, comparison_df


def create_experiment_summary(results_dict: Dict[str, Dict[str, Any]], 
                             experiment_name: str = "Strategy Experiment"):
    """
    Create a comprehensive summary of experiment results.
    
    Args:
        results_dict: Dictionary of strategy results
        experiment_name: Name of the experiment
    """
    print("=" * 80)
    print(f"🧪 EXPERIMENT SUMMARY: {experiment_name.upper()}")
    print("=" * 80)
    
    # Basic statistics
    total_strategies = len(results_dict)
    successful_strategies = len([r for r in results_dict.values() if 'final_value' in r])
    
    print(f"\n📈 EXPERIMENT OVERVIEW:")
    print(f"   Total Strategies Tested: {total_strategies}")
    print(f"   Successful Runs: {successful_strategies}")
    print(f"   Success Rate: {successful_strategies/total_strategies*100:.1f}%")
    
    if successful_strategies == 0:
        print("\n❌ No successful strategy runs found!")
        return
    
    # Performance analysis
    returns = [r.get('total_return', 0) for r in results_dict.values()]
    sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results_dict.values()]
    drawdowns = [r.get('max_drawdown', 0) for r in results_dict.values()]
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    print(f"   Returns Range: {min(returns):.2%} to {max(returns):.2%}")
    print(f"   Average Return: {np.mean(returns):.2%}")
    print(f"   Sharpe Ratio Range: {min(sharpe_ratios):.3f} to {max(sharpe_ratios):.3f}")
    print(f"   Average Sharpe: {np.mean(sharpe_ratios):.3f}")
    print(f"   Max Drawdown Range: {min(drawdowns):.2%} to {max(drawdowns):.2%}")
    
    # Best performers
    best_return = max(results_dict.items(), key=lambda x: x[1].get('total_return', -float('inf')))
    best_sharpe = max(results_dict.items(), key=lambda x: x[1].get('sharpe_ratio', -float('inf')))
    
    print(f"\n🏆 TOP PERFORMERS:")
    print(f"   Best Return: {best_return[0]} ({best_return[1].get('total_return', 0):.2%})")
    print(f"   Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].get('sharpe_ratio', 0):.3f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if np.mean(sharpe_ratios) > 1.0:
        print("   ✅ Overall strategy performance is good (Sharpe > 1.0)")
    elif np.mean(sharpe_ratios) > 0.5:
        print("   ⚠️  Strategy performance is moderate (Sharpe 0.5-1.0)")
    else:
        print("   ❌ Strategy performance needs improvement (Sharpe < 0.5)")
    
    if np.mean(drawdowns) < -0.1:
        print("   ⚠️  High drawdowns detected - consider risk management")
    
    print("\n" + "=" * 80)


def export_results_to_csv(results_dict: Dict[str, Dict[str, Any]], 
                          output_dir: str = "results",
                          filename_prefix: str = "experiment"):
    """
    Export experiment results to CSV files for further analysis.
    
    Args:
        results_dict: Dictionary of strategy results
        output_dir: Directory to save results
        filename_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export detailed results
    detailed_results = []
    for strategy_name, results in results_dict.items():
        row = {'Strategy': strategy_name}
        row.update(results)
        detailed_results.append(row)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(output_dir, f"{filename_prefix}_detailed_results.csv")
    detailed_df.to_csv(detailed_path, index=False)
    
    # Export summary
    summary_data = []
    for strategy_name, results in results_dict.items():
        summary_data.append({
            'Strategy': strategy_name,
            'Total Return': results.get('total_return', 0),
            'Sharpe Ratio': results.get('sharpe_ratio', 0),
            'Max Drawdown': results.get('max_drawdown', 0),
            'Annual Return': results.get('annual_return', 0),
            'Final Value': results.get('final_value', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✅ Results exported successfully!")
    print(f"   Detailed results: {detailed_path}")
    print(f"   Summary: {summary_path}")
    
    return detailed_path, summary_path


def quick_strategy_test(strategy_class, symbol: str, start_date: str, end_date: str, 
                       initial_cash: float = 100000, commission: float = 0.001):
    """
    Quick test of a strategy with minimal setup.
    
    Args:
        strategy_class: Strategy class to test
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        initial_cash: Initial capital
        commission: Commission rate
    
    Returns:
        Dictionary with test results
    """
    from backtesting.engine import BacktestEngine
    
    print(f"🚀 Quick testing {strategy_class.__name__} on {symbol}...")
    
    # Initialize engine
    engine = BacktestEngine(
        initial_cash=initial_cash,
        commission=commission,
        output_dir="results"
    )
    
    try:
        # Run backtest
        results = engine.run_backtest(
            strategy_class=strategy_class,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Display quick summary
        print(f"✅ Test completed successfully!")
        print(f"   Final Value: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


def render_pyfolio_tearsheet(engine, result_key: str,
                             live_start_date: Optional[str] = None,
                             round_trips: bool = True) -> None:
    """Render a PyFolio full tear sheet for a stored backtest result.

    This is a thin wrapper around the engine's `create_pyfolio_tearsheet`.
    """
    try:
        engine.create_pyfolio_tearsheet(
            result_key=result_key,
            live_start_date=live_start_date,
            round_trips=round_trips,
        )
    except Exception as e:
        print(f"❌ Unable to render PyFolio tear sheet: {e}")
