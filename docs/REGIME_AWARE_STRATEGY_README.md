# Regime-Aware Portfolio Optimization Strategy

## Overview

This strategy implements the sophisticated regime-aware portfolio optimization framework described in the research document "Regime_Aware_Portfolio_Optimization_with_Surprise_Guided_Change_Point_Detection.pdf". The strategy combines several advanced techniques:

1. **Bayesian Online Change Point Detection (BOCPD)** - Detects structural breaks in asset return distributions
2. **Bayes Factor Surprise** - Quantifies model-data mismatch and adapts learning rates
3. **Recursive Moment Estimation** - Dynamically updates mean and covariance estimates
4. **Mean-Variance Optimization** - Solves portfolio allocation with turnover penalties

## Key Features

### 🎯 **Regime Detection**
- Automatically detects when market conditions change
- Uses probabilistic BOCPD framework for robust change point detection
- Adapts estimation windows based on detected regime changes

### 🧠 **Adaptive Learning**
- Learning rates automatically adjust based on statistical surprise
- Higher surprise = faster adaptation to new information
- Lower surprise = more stable, conservative updates

### 📊 **Portfolio Optimization**
- Mean-variance optimization with risk aversion control
- Turnover penalties to reduce transaction costs
- Constrained optimization (budget, weight limits, sector constraints)

### 📈 **Performance Analytics**
- Comprehensive regime change tracking
- Surprise score monitoring
- Learning rate analysis
- Standard trading performance metrics

## Mathematical Framework

### Change Point Detection
The strategy uses BOCPD to maintain a posterior distribution over run lengths:

```
p(r_t | r_{1:t}) ∝ Σ p(r_t | r_{t-1}) p(r_t | r_{t-r_t:t-1}) p(r_{t-1} | r_{1:t-1})
```

### Bayes Factor Surprise
Surprise is computed as the ratio of likelihoods:

```
BFS_t = p(r_t | π_0) / p(r_t | π_t)
```

Where:
- `π_t` = current model based on regime window
- `π_0` = uninformative alternative model

### Adaptive Learning Rate
Learning rate is modulated by surprise:

```
γ_t = (m × BFS_t) / (1 + m × BFS_t)
```

Where `m = p_c / (1 - p_c)` and `p_c` is the hazard rate.

### Portfolio Optimization
Weights are optimized by solving:

```
w_t = argmax_{w ∈ Ω_t} {w^T μ̂_t - λ w^T Σ̂_t w - κ ||w - w_{t-1}||_1}
```

Where:
- `λ` = risk aversion parameter
- `κ` = turnover penalty
- `Ω_t` = constraint set

## Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
- `numpy` - Numerical computations
- `scipy` - Optimization and statistics
- `pandas` - Data manipulation
- `backtrader` - Backtesting framework
- `matplotlib` - Plotting (optional)

## Usage

### Basic Usage

```python
from src.strategies.regime_aware_portfolio import RegimeAwarePortfolioStrategy
from src.data.data_loader import DataLoader

# Load data
data_loader = DataLoader()
data = data_loader.get_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'], '2022-01-01', '2023-12-31')

# Create strategy
strategy = RegimeAwarePortfolioStrategy()

# Run backtest
# ... (see example script for complete implementation)
```

### Running the Example

```bash
cd examples
python run_regime_aware_backtest.py
```

## Strategy Parameters

### Core Parameters
- **`hazard_rate`** (0.01): Prior probability of regime change per time step
- **`change_point_threshold`** (0.5): Confidence threshold for declaring change points
- **`max_window_length`** (252): Maximum estimation window (1 year for daily data)
- **`risk_aversion`** (2.0): Risk aversion parameter λ
- **`turnover_penalty`** (0.001): Penalty for portfolio turnover

### Portfolio Constraints
- **`min_weight`** (0.0): Minimum portfolio weight per asset
- **`max_weight`** (0.3): Maximum portfolio weight per asset
- **`rebalance_frequency`** (5): Days between rebalancing

### Advanced Parameters
- **`hazard_rate`**: Controls sensitivity to regime changes
  - Lower values (0.005): More stable, fewer regime changes
  - Higher values (0.05): More sensitive, frequent regime changes

- **`change_point_threshold`**: Controls confidence required for regime change
  - Lower values (0.3): More aggressive regime detection
  - Higher values (0.7): More conservative regime detection

## Performance Metrics

### Standard Metrics
- Total return, Sharpe ratio, maximum drawdown
- Win rate, average PnL per trade
- Number of trades and turnover

### Regime-Aware Metrics
- **Regime Changes Detected**: Number of structural breaks identified
- **Average Surprise Score**: Mean BFS across all observations
- **Average Learning Rate**: Mean adaptive learning rate
- **Regime Change Times**: When regime changes occurred
- **Regime Change Confidences**: Confidence levels of detected changes

## Example Results

```
==================================================
BACKTEST RESULTS
==================================================
Initial Portfolio Value: $100,000.00
Final Portfolio Value: $112,450.67
Total Return: 12.45%

Regime Changes Detected: 3
Average Surprise Score: 1.2345
Average Learning Rate: 0.0456

Regime Change Times: [45, 127, 189]
Regime Change Confidences: [0.723, 0.891, 0.654]

Trading Performance:
Total Trades: 156
Win Rate: 58.33%
Average PnL per Trade: $80.45
```

## Parameter Sensitivity

### Hazard Rate Sensitivity
- **Low (0.005)**: Stable, long-term regimes, conservative adaptation
- **Medium (0.01)**: Balanced approach, moderate regime detection
- **High (0.05)**: Aggressive, frequent regime changes, fast adaptation

### Change Point Threshold Sensitivity
- **Low (0.3)**: Frequent regime changes, higher false positives
- **Medium (0.5)**: Balanced detection accuracy
- **High (0.7)**: Conservative detection, fewer false positives

## Best Practices

### 1. **Data Quality**
- Use high-quality, clean market data
- Ensure consistent time series with no major gaps
- Consider using multiple timeframes for robustness

### 2. **Parameter Tuning**
- Start with default parameters
- Use cross-validation or walk-forward analysis
- Monitor regime change frequency and adjust hazard rate
- Balance turnover penalty with expected transaction costs

### 3. **Risk Management**
- Set appropriate weight limits based on asset characteristics
- Monitor regime change confidence levels
- Consider using regime-aware position sizing

### 4. **Performance Monitoring**
- Track regime change patterns over time
- Monitor surprise score distributions
- Analyze learning rate stability

## Troubleshooting

### Common Issues

1. **No Regime Changes Detected**
   - Increase hazard rate
   - Lower change point threshold
   - Check data quality and volatility

2. **Excessive Regime Changes**
   - Decrease hazard rate
   - Increase change point threshold
   - Check for data noise or outliers

3. **Poor Portfolio Performance**
   - Adjust risk aversion parameter
   - Review turnover penalty settings
   - Check constraint feasibility

4. **Optimization Failures**
   - Verify constraint parameters
   - Check covariance matrix conditioning
   - Review weight bounds

## Advanced Features

### Custom Constraints
You can extend the strategy to include:
- Sector allocation constraints
- Factor exposure limits
- ESG compliance requirements
- Liquidity constraints

### Alternative Models
The framework supports different distributional assumptions:
- Student-t distributions for fat tails
- Asymmetric distributions for skewness
- GARCH models for volatility clustering

### Multi-Asset Classes
Extend to include:
- Bonds and fixed income
- Commodities and currencies
- Alternative investments
- Cryptocurrencies

## Research Applications

This strategy is particularly useful for:
- **Academic Research**: Testing regime change theories
- **Risk Management**: Dynamic risk allocation
- **Quantitative Trading**: Systematic portfolio management
- **Asset Allocation**: Multi-asset class optimization

## Contributing

To contribute to this strategy:
1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

## License

This implementation is provided for educational and research purposes. Please cite the original research paper when using this strategy in academic work.

## References

- Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
- The original research document: "Regime_Aware_Portfolio_Optimization_with_Surprise_Guided_Change_Point_Detection.pdf"

---

For questions or support, please refer to the main project documentation or create an issue in the repository.
