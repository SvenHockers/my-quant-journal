"""
Regime-Aware Portfolio Optimization with Surprise-Guided Change Point Detection

This strategy implements the framework described in the research document:
- Bayesian Online Change Point Detection (BOCPD)
- Bayes Factor Surprise-based learning rate modulation
- Recursive moment estimation
- Dynamic portfolio optimization with turnover penalties
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from typing import Dict, Any


from .base_strategy import BaseStrategy


class RegimeAwarePortfolioStrategy(BaseStrategy):
    """
    Implements regime-aware portfolio optimization using BOCPD and Bayes Factor Surprise.
    
    This strategy dynamically adjusts portfolio weights based on detected regime changes
    and statistical surprise measures, optimizing for mean-variance objectives while
    penalizing excessive turnover.
    """
    
    def __init__(self):
        super().__init__()
        
        # Strategy parameters
        self.params = self.get_strategy_params()
        
        # Initialize state variables
        self.portfolio_weights = None
        self.estimated_mean = None
        self.estimated_covariance = None
        self.regime_start_time = 0
        self.run_length_posterior = None
        self.hazard_rate = self.params['hazard_rate']
        self.change_point_threshold = self.params['change_point_threshold']
        self.max_window_length = self.params['max_window_length']
        self.risk_aversion = self.params['risk_aversion']
        self.turnover_penalty = self.params['turnover_penalty']
        
        # Performance tracking
        self.regime_changes = []
        self.surprise_scores = []
        self.learning_rates = []
        
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            'hazard_rate': 0.01,           # Prior probability of regime change
            'change_point_threshold': 0.5,  # Threshold for declaring change point
            'max_window_length': 252,       # Maximum estimation window (1 year)
            'risk_aversion': 2.0,          # Risk aversion parameter λ
            'turnover_penalty': 0.001,     # Turnover penalty κ
            'min_weight': 0.0,             # Minimum portfolio weight
            'max_weight': 0.3,             # Maximum portfolio weight
            'rebalance_frequency': 5       # Rebalance every N days
        }
    
    def next(self):
        """Main strategy logic executed at each time step."""
        if len(self) < 2:  # Need at least 2 data points
            return
        
        # Get current returns
        current_returns = self._calculate_returns()
        
        # Update regime detection
        self._update_regime_detection(current_returns)
        
        # Update moment estimates
        self._update_moment_estimates(current_returns)
        
        # Rebalance portfolio if needed
        if len(self) % self.params['rebalance_frequency'] == 0:
            self._rebalance_portfolio()
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate returns for all assets."""
        returns = []
        for data in self.datas:
            if len(data) > 1:
                # Calculate log returns
                current_price = data.close[0]
                previous_price = data.close[-1]
                if previous_price > 0:
                    log_return = np.log(current_price / previous_price)
                else:
                    log_return = 0.0
                returns.append(log_return)
            else:
                returns.append(0.0)
        
        return np.array(returns)
    
    def _update_regime_detection(self, current_returns: np.ndarray):
        """Update regime detection using BOCPD."""
        if self.estimated_mean is None:
            # Initialize on first run
            self.estimated_mean = np.zeros_like(current_returns)
            self.estimated_covariance = np.eye(len(current_returns)) * 0.01
            self.run_length_posterior = np.array([1.0])  # Start with run length 0
            return
        
        # BOCPD update
        self._update_bocpd_posterior(current_returns)
        
        # Check for change point
        if self._detect_change_point():
            self.regime_start_time = len(self)
            self.regime_changes.append({
                'time': len(self),
                'confidence': self.run_length_posterior[0]
            })
    
    def _update_bocpd_posterior(self, current_returns: np.ndarray):
        """Update BOCPD posterior distribution."""
        # Prior over run length transitions
        hazard = self.hazard_rate
        max_run_length = len(self.run_length_posterior)
        
        # Initialize new posterior
        new_posterior = np.zeros(max_run_length + 1)
        
        # Update existing run lengths
        for r in range(max_run_length):
            if r == 0:
                # Change point occurred
                new_posterior[0] += hazard * self.run_length_posterior[r]
            else:
                # Run length continues
                new_posterior[r] += (1 - hazard) * self.run_length_posterior[r-1]
        
        # Add new run length
        new_posterior[max_run_length] = (1 - hazard) * self.run_length_posterior[-1]
        
        # Compute predictive likelihood for new observation
        likelihoods = np.zeros(max_run_length + 1)
        for r in range(max_run_length + 1):
            if r == 0:
                # New regime - use uninformative prior
                likelihoods[r] = self._compute_uninformative_likelihood(current_returns)
            else:
                # Use existing regime parameters
                likelihoods[r] = self._compute_regime_likelihood(current_returns, r)
        
        # Apply likelihood and normalize
        new_posterior *= likelihoods
        if np.sum(new_posterior) > 0:
            new_posterior /= np.sum(new_posterior)
        
        self.run_length_posterior = new_posterior
    
    def _compute_regime_likelihood(self, returns: np.ndarray, run_length: int) -> float:
        """Compute likelihood under current regime parameters."""
        try:
            # Use multivariate normal with current estimates
            rv = multivariate_normal(
                mean=self.estimated_mean,
                cov=self.estimated_covariance
            )
            return rv.pdf(returns)
        except:
            # Fallback to simple Gaussian
            return np.exp(-0.5 * np.sum(returns**2))
    
    def _compute_uninformative_likelihood(self, returns: np.ndarray) -> float:
        """Compute likelihood under uninformative prior."""
        # Simple Gaussian with large variance
        return np.exp(-0.5 * np.sum(returns**2) / 100.0)
    
    def _detect_change_point(self) -> bool:
        """Detect if a change point has occurred."""
        if len(self.run_length_posterior) > 0:
            return self.run_length_posterior[0] > self.change_point_threshold
        return False
    
    def _update_moment_estimates(self, current_returns: np.ndarray):
        """Update moment estimates using Bayes Factor Surprise."""
        if self.estimated_mean is None:
            return
        
        # Compute Bayes Factor Surprise
        bfs = self._compute_bayes_factor_surprise(current_returns)
        self.surprise_scores.append(bfs)
        
        # Compute adaptive learning rate
        m = self.hazard_rate / (1 - self.hazard_rate)
        learning_rate = (m * bfs) / (1 + m * bfs)
        self.learning_rates.append(learning_rate)
        
        # Update estimates
        self.estimated_mean = (
            (1 - learning_rate) * self.estimated_mean + 
            learning_rate * current_returns
        )
        
        # Update covariance
        diff = current_returns - self.estimated_mean
        self.estimated_covariance = (
            (1 - learning_rate) * self.estimated_covariance + 
            learning_rate * np.outer(diff, diff)
        )
    
    def _compute_bayes_factor_surprise(self, current_returns: np.ndarray) -> float:
        """Compute Bayes Factor Surprise."""
        try:
            # Current model likelihood
            current_likelihood = self._compute_regime_likelihood(
                current_returns, 
                len(self.run_length_posterior) - 1
            )
            
            # Uninformative model likelihood
            uninformative_likelihood = self._compute_uninformative_likelihood(current_returns)
            
            # Avoid division by zero
            if current_likelihood > 0:
                return uninformative_likelihood / current_likelihood
            else:
                return 1.0
        except:
            return 1.0
    
    def _rebalance_portfolio(self):
        """Rebalance portfolio using mean-variance optimization."""
        if self.estimated_mean is None or self.estimated_covariance is None:
            return
        
        # Get current portfolio weights
        current_weights = self._get_current_weights()
        
        # Optimize portfolio weights
        optimal_weights = self._optimize_portfolio(
            self.estimated_mean,
            self.estimated_covariance,
            current_weights
        )
        
        # Execute trades
        self._execute_trades(current_weights, optimal_weights)
        
        # Update portfolio weights
        self.portfolio_weights = optimal_weights
    
    def _get_current_weights(self) -> np.ndarray:
        """Get current portfolio weights."""
        if self.portfolio_weights is None:
            # Equal weight initialization
            n_assets = len(self.datas)
            return np.ones(n_assets) / n_assets
        
        return self.portfolio_weights
    
    def _optimize_portfolio(self, 
                           expected_returns: np.ndarray, 
                           covariance: np.ndarray, 
                           current_weights: np.ndarray) -> np.ndarray:
        """Solve mean-variance optimization problem."""
        n_assets = len(expected_returns)
        
        # Objective function: maximize return - risk - turnover penalty
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.dot(weights.T, np.dot(covariance, weights))
            turnover_penalty = self.turnover_penalty * np.sum(np.abs(weights - current_weights))
            
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk - turnover_penalty)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget constraint
        ]
        
        # Bounds
        bounds = [(self.params['min_weight'], self.params['max_weight'])] * n_assets
        
        # Initial guess
        x0 = current_weights
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints, bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
        except:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
    
    def _execute_trades(self, current_weights: np.ndarray, target_weights: np.ndarray):
        """Execute trades to achieve target weights."""
        portfolio_value = self.broker.getvalue()
        
        for i, data in enumerate(self.datas):
            current_weight = current_weights[i]
            target_weight = target_weights[i]
            
            if abs(target_weight - current_weight) > 0.01:  # 1% threshold
                target_value = target_weight * portfolio_value
                current_value = current_weight * portfolio_value
                
                if target_value > current_value:
                    # Buy
                    size = (target_value - current_value) / data.close[0]
                    self.buy(data=data, size=size)
                else:
                    # Sell
                    size = (current_value - target_value) / data.close[0]
                    self.sell(data=data, size=size)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific performance metrics."""
        metrics = super().get_performance_summary()
        
        # Add regime-aware metrics
        metrics.update({
            'regime_changes_detected': len(self.regime_changes),
            'avg_surprise_score': np.mean(self.surprise_scores) if self.surprise_scores else 0,
            'avg_learning_rate': np.mean(self.learning_rates) if self.learning_rates else 0,
            'regime_change_times': [rc['time'] for rc in self.regime_changes],
            'regime_change_confidences': [rc['confidence'] for rc in self.regime_changes]
        })
        
        return metrics
    
    def plot_strategy_analysis(self):
        """Plot strategy analysis including regime changes and surprise scores."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio weights over time
            if self.portfolio_weights is not None:
                axes[0, 0].plot(self.portfolio_weights)
                axes[0, 0].set_title('Portfolio Weights')
                axes[0, 0].set_xlabel('Asset Index')
                axes[0, 0].set_ylabel('Weight')
            
            # Surprise scores
            if self.surprise_scores:
                axes[0, 1].plot(self.surprise_scores)
                axes[0, 1].set_title('Bayes Factor Surprise')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Surprise Score')
            
            # Learning rates
            if self.learning_rates:
                axes[1, 0].plot(self.learning_rates)
                axes[1, 0].set_title('Adaptive Learning Rates')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Learning Rate')
            
            # Regime changes
            if self.regime_changes:
                change_times = [rc['time'] for rc in self.regime_changes]
                confidences = [rc['confidence'] for rc in self.regime_changes]
                axes[1, 1].scatter(change_times, confidences, alpha=0.7)
                axes[1, 1].set_title('Regime Changes')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Change Point Confidence')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
