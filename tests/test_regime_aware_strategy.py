"""
Tests for the Regime-Aware Portfolio Optimization Strategy.

This module tests the core functionality of the strategy including:
- BOCPD implementation
- Bayes Factor Surprise calculation
- Moment estimation updates
- Portfolio optimization
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import the strategy
from src.strategies.regime_aware_portfolio import RegimeAwarePortfolioStrategy


class TestRegimeAwarePortfolioStrategy(unittest.TestCase):
    """Test cases for the RegimeAwarePortfolioStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = RegimeAwarePortfolioStrategy()
        
        # Mock data feeds
        self.mock_data1 = Mock()
        self.mock_data1.close = [100, 101, 102, 103, 104]
        self.mock_data1.__len__ = lambda: len(self.mock_data1.close)
        
        self.mock_data2 = Mock()
        self.mock_data2.close = [50, 51, 52, 53, 54]
        self.mock_data2.__len__ = lambda: len(self.mock_data2.close)
        
        # Set up strategy with mock data
        self.strategy.datas = [self.mock_data1, self.mock_data2]
        self.strategy.broker = Mock()
        self.strategy.broker.getvalue.return_value = 100000
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy.params)
        self.assertIsNone(self.strategy.portfolio_weights)
        self.assertIsNone(self.strategy.estimated_mean)
        self.assertEqual(self.strategy.hazard_rate, 0.01)
        self.assertEqual(self.strategy.change_point_threshold, 0.5)
    
    def test_strategy_parameters(self):
        """Test strategy parameter configuration."""
        params = self.strategy.get_strategy_params()
        
        expected_params = {
            'hazard_rate', 'change_point_threshold', 'max_window_length',
            'risk_aversion', 'turnover_penalty', 'min_weight', 'max_weight',
            'rebalance_frequency'
        }
        
        for param in expected_params:
            self.assertIn(param, params)
    
    def test_calculate_returns(self):
        """Test return calculation."""
        # Mock the strategy to have length > 1
        self.strategy.__len__ = lambda: 2
        
        returns = self.strategy._calculate_returns()
        
        self.assertIsInstance(returns, np.ndarray)
        self.assertEqual(len(returns), 2)
        
        # Check that returns are calculated correctly
        expected_return1 = np.log(101 / 100)  # log(101/100)
        expected_return2 = np.log(51 / 50)    # log(51/50)
        
        np.testing.assert_almost_equal(returns[0], expected_return1, decimal=6)
        np.testing.assert_almost_equal(returns[1], expected_return2, decimal=6)
    
    def test_bocpd_posterior_update(self):
        """Test BOCPD posterior update."""
        # Initialize strategy state
        self.strategy.estimated_mean = np.array([0.0, 0.0])
        self.strategy.estimated_covariance = np.eye(2) * 0.01
        self.strategy.run_length_posterior = np.array([1.0])
        
        current_returns = np.array([0.01, 0.02])
        
        # Test posterior update
        self.strategy._update_bocpd_posterior(current_returns)
        
        # Check that posterior was updated
        self.assertIsNotNone(self.strategy.run_length_posterior)
        self.assertTrue(len(self.strategy.run_length_posterior) > 0)
        
        # Check that posterior sums to approximately 1
        np.testing.assert_almost_equal(
            np.sum(self.strategy.run_length_posterior), 1.0, decimal=6
        )
    
    def test_change_point_detection(self):
        """Test change point detection."""
        # Test with high confidence change point
        self.strategy.run_length_posterior = np.array([0.8, 0.2])  # High confidence in change
        
        self.assertTrue(self.strategy._detect_change_point())
        
        # Test with low confidence
        self.strategy.run_length_posterior = np.array([0.3, 0.7])  # Low confidence in change
        
        self.assertFalse(self.strategy._detect_change_point())
    
    def test_bayes_factor_surprise(self):
        """Test Bayes Factor Surprise calculation."""
        # Initialize strategy state
        self.strategy.estimated_mean = np.array([0.0, 0.0])
        self.strategy.estimated_covariance = np.eye(2) * 0.01
        self.strategy.run_length_posterior = np.array([0.5, 0.5])
        
        current_returns = np.array([0.01, 0.02])
        
        bfs = self.strategy._compute_bayes_factor_surprise(current_returns)
        
        self.assertIsInstance(bfs, float)
        self.assertGreater(bfs, 0)
    
    def test_moment_estimation_update(self):
        """Test moment estimation updates."""
        # Initialize strategy state
        self.strategy.estimated_mean = np.array([0.0, 0.0])
        self.strategy.estimated_covariance = np.eye(2) * 0.01
        self.strategy.run_length_posterior = np.array([0.5, 0.5])
        
        current_returns = np.array([0.01, 0.02])
        
        # Test moment update
        self.strategy._update_moment_estimates(current_returns)
        
        # Check that surprise scores and learning rates were recorded
        self.assertGreater(len(self.strategy.surprise_scores), 0)
        self.assertGreater(len(self.strategy.learning_rates), 0)
        
        # Check that estimates were updated
        self.assertIsNotNone(self.strategy.estimated_mean)
        self.assertIsNotNone(self.strategy.estimated_covariance)
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        expected_returns = np.array([0.05, 0.03])
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        current_weights = np.array([0.5, 0.5])
        
        optimal_weights = self.strategy._optimize_portfolio(
            expected_returns, covariance, current_weights
        )
        
        self.assertIsInstance(optimal_weights, np.ndarray)
        self.assertEqual(len(optimal_weights), 2)
        
        # Check constraints
        self.assertGreaterEqual(np.min(optimal_weights), self.strategy.params['min_weight'])
        self.assertLessEqual(np.max(optimal_weights), self.strategy.params['max_weight'])
        
        # Check budget constraint (approximately)
        np.testing.assert_almost_equal(np.sum(optimal_weights), 1.0, decimal=6)
    
    def test_portfolio_optimization_fallback(self):
        """Test portfolio optimization fallback to equal weights."""
        # Test with problematic inputs that might cause optimization failure
        expected_returns = np.array([0.05, 0.03])
        covariance = np.array([[0.0, 0.0], [0.0, 0.0]])  # Singular matrix
        current_weights = np.array([0.5, 0.5])
        
        optimal_weights = self.strategy._optimize_portfolio(
            expected_returns, covariance, current_weights
        )
        
        # Should fall back to equal weights
        expected_equal_weights = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(optimal_weights, expected_equal_weights)
    
    def test_get_current_weights(self):
        """Test current weights retrieval."""
        # Test initialization case
        weights = self.strategy._get_current_weights()
        expected_weights = np.array([0.5, 0.5])  # Equal weights for 2 assets
        
        np.testing.assert_array_almost_equal(weights, expected_weights)
        
        # Test with existing weights
        self.strategy.portfolio_weights = np.array([0.7, 0.3])
        weights = self.strategy._get_current_weights()
        
        np.testing.assert_array_almost_equal(weights, np.array([0.7, 0.3]))
    
    def test_strategy_metrics(self):
        """Test strategy metrics calculation."""
        # Add some mock data
        self.strategy.regime_changes = [
            {'time': 10, 'confidence': 0.8},
            {'time': 25, 'confidence': 0.9}
        ]
        self.strategy.surprise_scores = [1.2, 1.5, 0.8]
        self.strategy.learning_rates = [0.05, 0.03, 0.07]
        
        metrics = self.strategy.get_strategy_metrics()
        
        self.assertIn('regime_changes_detected', metrics)
        self.assertIn('avg_surprise_score', metrics)
        self.assertIn('avg_learning_rate', metrics)
        
        self.assertEqual(metrics['regime_changes_detected'], 2)
        self.assertAlmostEqual(metrics['avg_surprise_score'], 1.167, places=3)
        self.assertAlmostEqual(metrics['avg_learning_rate'], 0.05, places=3)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        self.strategy.datas = []
        returns = self.strategy._calculate_returns()
        self.assertEqual(len(returns), 0)
        
        # Test with single data point
        self.strategy.datas = [self.mock_data1]
        self.strategy.__len__ = lambda: 1
        returns = self.strategy._calculate_returns()
        self.assertEqual(returns[0], 0.0)  # Should return 0 for single point
        
        # Test with None estimates
        self.strategy.estimated_mean = None
        self.strategy.estimated_covariance = None
        
        # These should handle None gracefully
        self.strategy._update_moment_estimates(np.array([0.01, 0.02]))
        self.strategy._rebalance_portfolio()
    
    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        params = self.strategy.get_strategy_params()
        
        # Check parameter ranges
        self.assertGreater(params['hazard_rate'], 0)
        self.assertLess(params['hazard_rate'], 1)
        self.assertGreater(params['change_point_threshold'], 0)
        self.assertLess(params['change_point_threshold'], 1)
        self.assertGreater(params['risk_aversion'], 0)
        self.assertGreater(params['turnover_penalty'], 0)
        self.assertGreaterEqual(params['min_weight'], 0)
        self.assertLessEqual(params['max_weight'], 1)


if __name__ == '__main__':
    unittest.main()
