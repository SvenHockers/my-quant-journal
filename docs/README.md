# My Quant Journal - Documentation

## Overview

This repo contains a comprehensive framework for developing, testing, and analyzing algorithmic trading strategies. It provides a standardized approach to backtesting with reproducible results and comprehensive performance analysis.

## Architecture

The framework is organized into several key modules:

### 1. Strategies (`src/strategies/`)

- **Base Strategy**: Abstract base class providing common functionality
- **SMA Crossover**: Example implementation of a very simple MA crossover strategy
- **Custom Strategies**: Framework for implementing your own trading strategies

### 2. Data Management (`src/data/`)

- **Data Loader**: Handles fetching market data from Yahoo Finance
- **Caching**: Built-in data caching to avoid repeated downloads
- **Validation**: Data quality checks and validation

### 3. Backtesting (`src/backtesting/`)

- **Backtest Engine**: Core engine for running strategy tests
- **Parameter Optimization**: Automated parameter tuning
- **Strategy Comparison**: Side-by-side strategy evaluation

### 4. Analysis (`src/analysis/`)

- **Performance Analyzer**: Comprehensive performance metrics
- **Risk Analysis**: Risk metrics and drawdown analysis
- **Reporting**: Automated report generation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd my-quant-journal
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Your First Backtest

```python
from src.backtesting.engine import BacktestEngine
from src.strategies.sma_crossover import SMACrossoverStrategy

# Initialize engine
engine = BacktestEngine(initial_cash=100000)

# Run backtest
results = engine.run_backtest(
    strategy_class=SMACrossoverStrategy,
    symbol='AAPL',
    start_date='2022-01-01',
    end_date='2023-12-31'
)

# Save results
engine.save_results(list(engine.results.keys())[0])
```

### Example Script

Run the complete example:
```bash
python examples/run_sma_backtest.py
```

## Creating Custom Strategies

### 1. Inherit from BaseStrategy

```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def get_strategy_params(self):
        return {
            'param1': 10,
            'param2': 20
        }
    
    def next(self):
        # Your strategy logic here
        if self.position.size == 0:
            # No position - check for entry signal
            if self.entry_signal():
                self.buy()
        else:
            # Have position - check for exit signal
            if self.exit_signal():
                self.sell()
```

### 2. Required Methods

- **`get_strategy_params()`**: Return strategy parameters
- **`next()`**: Main strategy logic (called for each bar)

### 3. Available Methods

- **`self.buy()`**: Open long position
- **`self.sell()`**: Close position
- **`self.log()`**: Log messages
- **`self.position`**: Current position information

## Data Management

### Fetching Data

```python
from src.data.data_loader import DataLoader

loader = DataLoader()

# Single symbol
data = loader.get_data('AAPL', '2022-01-01', '2023-12-31')

# Multiple symbols
data_dict = loader.get_multiple_symbols(
    ['AAPL', 'GOOGL', 'MSFT'], 
    '2022-01-01', 
    '2023-12-31'
)
```

### Data Validation

```python
# Validate data quality
if loader.validate_data(data):
    print("Data is valid")
else:
    print("Data has issues")

# Get data summary
info = loader.get_data_info(data)
print(info)
```

## Advanced Features

### Parameter Optimization

```python
# Define parameter ranges
param_ranges = {
    'fast_period': [5, 10, 15, 20],
    'slow_period': [20, 30, 40, 50]
}

# Run optimization
optimization_results = engine.run_parameter_optimization(
    MyCustomStrategy,
    'AAPL',
    '2022-01-01',
    '2023-12-31',
    param_ranges
)
```

### Strategy Comparison

```python
# Compare multiple strategies
strategies = [SMACrossoverStrategy, MyCustomStrategy]
comparison = engine.compare_strategies(
    strategies,
    'AAPL',
    '2022-01-01',
    '2023-12-31'
)
```

### Performance Analysis

```python
from src.analysis.performance import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Generate performance report
report_path = analyzer.create_performance_report(results)

# Create performance charts
figures = analyzer.plot_performance_charts(results)
```

## Configuration

The framework uses a YAML configuration file (`config/config.yaml`) for default settings:

```yaml
backtesting:
  initial_cash: 100000.0
  commission: 0.001
  output_dir: "results"

data:
  default_interval: "1d"
  cache_enabled: true

strategies:
  sma_crossover:
    fast_period: 10
    slow_period: 30
```

## Output Structure

After running backtests, the framework creates:

```
results/           # Backtest results and summaries
analysis/          # Performance reports and charts
data/cache/        # Cached market data
logs/             # Application logs
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Best Practices

### 1. Strategy Development

- Start with simple strategies and add complexity gradually
- Use the base strategy class for consistent behavior
- Implement proper risk management
- Test with different market conditions

### 2. Data Management

- Always validate data before backtesting
- Use caching to avoid repeated downloads
- Handle missing data appropriately
- Consider data quality and gaps

### 3. Performance Analysis

- Look beyond just returns (Sharpe ratio, drawdown, etc.)
- Consider transaction costs and slippage
- Test parameter sensitivity
- Avoid overfitting to historical data

### 4. Reproducibility

- Use fixed random seeds where applicable
- Document all parameters and assumptions
- Save all results and configurations
- Version control your strategies

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory and have installed dependencies
2. **Data Issues**: Check internet connection and symbol validity
3. **Memory Issues**: Reduce data range or use smaller parameter grids for optimization
4. **Plotting Issues**: Ensure matplotlib backend is properly configured

### Getting Help

- Check the example scripts for usage patterns
- Review the test files for implementation details
- Ensure all dependencies are properly installed
- Check the logs for detailed error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
