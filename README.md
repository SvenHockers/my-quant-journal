# my-quant-journal

This is a personal research repository focused exclusively on the development and evaluation of algorithmic trading strategies.  
It serves as a structured research journal, documenting experiments and building standardized pipelines for strategy testing.

---

## Overview

This repository is dedicated to **systematic strategy research** from concept to performance evaluation using reproducible, data-driven methods.  
All experiments follow a standardized backtesting workflow built primarily with **Backtrader** and market data from **Yahoo Finance**.

The aim is to create a **consistent, reusable framework** for testing multiple strategies under comparable conditions,  
ensuring results are methodical, repeatable, and easy to analyze.

## Repository Structure

```
my-quant-journal/
├── src/                          # Source code
│   ├── strategies/               # Trading strategy implementations
│   │   ├── base_strategy.py     # Abstract base strategy class
│   │   └── sma_crossover.py     # Example SMA crossover strategy
│   ├── data/                    # Data management
│   │   └── data_loader.py       # Yahoo Finance data fetcher
│   ├── backtesting/             # Backtesting engine
│   │   └── engine.py            # Core backtesting framework
│   ├── analysis/                # Performance analysis
│   │   └── performance.py       # Performance metrics and reporting
│   └── utils/                   # Utility functions
│       └── notebook_helpers.py  # Jupyter notebook helpers
├── examples/                     # Example scripts and notebooks
│   ├── run_sma_backtest.py      # Complete backtest example
│   ├── quick_start.ipynb        # Quick start Jupyter notebook
│   └── notebook_examples.ipynb  # Comprehensive notebook examples
├── tests/                        # Unit tests
│   └── test_strategies.py       # Strategy testing
├── config/                       # Configuration files
│   └── config.yaml              # Default settings
├── docs/                         # Documentation
│   └── README.md                # Comprehensive guide
├── data/                         # Data storage
│   └── cache/                   # Cached market data
├── results/                      # Backtest results
├── analysis/                     # Analysis outputs
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── Makefile                     # Development commands
└── .gitignore                   # Git ignore rules
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd my-quant-journal

# Install dependencies
pip install -r requirements.txt

# Setup development environment
make setup-dev
```

### 2. Run Your First Backtest

```bash
# Run the complete example
make run-example
```

Or run the example directly:
```bash
python examples/run_sma_backtest.py
```

### 3. Jupyter Notebooks (Recommended for Research)

The framework is fully compatible with Jupyter notebooks for interactive research:

```bash
# Install Jupyter dependencies
make install-jupyter

# Start Jupyter notebook server
make jupyter

# Or start JupyterLab
make jupyter-lab
```

Then open the notebooks in `examples/`:
- **`quick_start.ipynb`** - Basic workflow and examples
- **`notebook_examples.ipynb`** - Comprehensive research examples

### 4. Create Custom Strategies

```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def get_strategy_params(self):
        return {'param1': 10}
    
    def next(self):
        # Your strategy logic here
        if self.entry_signal():
            self.buy()
        elif self.exit_signal():
            self.sell()
```

## Key Features

- **Standardized Framework**: Consistent interface for all strategies
- **Data Management**: Automated data fetching with caching
- **Comprehensive Analysis**: Performance metrics, risk analysis, and reporting
- **Parameter Optimization**: Automated strategy parameter tuning
- **Strategy Comparison**: Side-by-side strategy evaluation
- **Reproducible Results**: Consistent backtesting environment
- **Jupyter Integration**: Full notebook compatibility with helper functions

## Jupyter Notebook Workflow

The framework includes specialized notebook helpers for efficient research:

```python
# Setup notebook environment
from src.utils.notebook_helpers import setup_notebook_environment, quick_strategy_test

# Configure plotting and display
setup_notebook_environment()

# Quick strategy testing
results = quick_strategy_test(SMACrossoverStrategy, 'AAPL', '2023-01-01', '2023-12-31')

# Export results
from src.utils.notebook_helpers import export_results_to_csv
export_results_to_csv(results, 'my_experiment')
```

### Notebook Helper Functions

- **`setup_notebook_environment()`** - Configure plotting styles and pandas options
- **`quick_data_preview()`** - Quick market data analysis
- **`plot_price_volume()`** - Professional price/volume charts
- **`quick_strategy_test()`** - Fast strategy testing
- **`compare_strategies_side_by_side()`** - Strategy comparison visualizations
- **`create_experiment_summary()`** - Comprehensive experiment analysis
- **`export_results_to_csv()`** - Export results for further analysis

## Development

### Available Commands

```bash
make help           # Show all available commands
make install        # Install dependencies
make test           # Run test suite
make lint           # Run linting checks
make format         # Format code with black
make clean          # Clean generated files
make run-example    # Run the example backtest
make jupyter        # Start Jupyter notebook server
make jupyter-lab    # Start JupyterLab server
make setup-complete # Complete setup with Jupyter support
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Quick test run
make quick-test
```

## Documentation

- **User Guide**: See `docs/README.md` for comprehensive documentation
- **Examples**: Check `examples/` directory for usage patterns
- **Tests**: Review `tests/` for implementation details
- **Notebooks**: Interactive examples in Jupyter notebooks

## Dependencies

- **Backtrader**: Backtesting engine
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **YFinance**: Market data fetching
- **Jupyter**: Interactive notebook environment
- **Seaborn**: Statistical data visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

---

**Note**: This framework is designed for research and educational purposes. Always validate strategies thoroughly before any real trading applications.