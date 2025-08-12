"""
Unit tests for trading strategies.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import strategies to test
from src.strategies.base_strategy import BaseStrategy
from src.strategies.sma_crossover import SMACrossoverStrategy


class TestBaseStrategy(unittest.TestCase):
    """Test base strategy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock strategy that inherits from BaseStrategy
        class MockStrategy(BaseStrategy):
            def get_strategy_params(self):
                return {'param1': 'value1'}
            
            def next(self):
                pass
        
        self.strategy = MockStrategy()
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy.params)
        self.assertEqual(self.strategy.params['param1'], 'value1')
        self.assertEqual(len(self.strategy.trades), 0)
        self.assertEqual(len(self.strategy.equity_curve), 0)
    
    def test_performance_summary_empty_trades(self):
        """Test performance summary with no trades."""
        summary = self.strategy.get_performance_summary()
        self.assertEqual(summary, {})
    
    def test_performance_summary_with_trades(self):
        """Test performance summary with trades."""
        # Add mock trades
        self.strategy.trades = [
            {'pnl': 100, 'entry_date': '2023-01-01', 'exit_date': '2023-01-02', 'size': 1},
            {'pnl': -50, 'entry_date': '2023-01-03', 'exit_date': '2023-01-04', 'size': 1},
            {'pnl': 75, 'entry_date': '2023-01-05', 'exit_date': '2023-01-06', 'size': 1}
        ]
        
        summary = self.strategy.get_performance_summary()
        
        self.assertEqual(summary['total_trades'], 3)
        self.assertEqual(summary['winning_trades'], 2)
        self.assertEqual(summary['losing_trades'], 1)
        self.assertEqual(summary['win_rate'], 2/3)
        self.assertEqual(summary['total_pnl'], 125)
        self.assertEqual(summary['avg_pnl_per_trade'], 125/3)


class TestSMACrossoverStrategy(unittest.TestCase):
    """Test SMA crossover strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SMACrossoverStrategy()
    
    def test_strategy_parameters(self):
        """Test strategy parameter defaults."""
        params = self.strategy.get_strategy_params()
        self.assertEqual(params['fast_period'], 10)
        self.assertEqual(params['slow_period'], 30)
        self.assertFalse(params['printlog'])
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        # Test that the strategy can be instantiated
        self.assertIsInstance(self.strategy, SMACrossoverStrategy)
        self.assertIsInstance(self.strategy, BaseStrategy)


class TestStrategyIntegration(unittest.TestCase):
    """Test strategy integration with mock data."""
    
    @patch('src.strategies.sma_crossover.bt.indicators.SMA')
    @patch('src.strategies.sma_crossover.bt.indicators.CrossOver')
    def test_sma_strategy_creation(self, mock_crossover, mock_sma):
        """Test SMA strategy creation with mocked indicators."""
        # Mock the indicator classes
        mock_sma.return_value = Mock()
        mock_crossover.return_value = Mock()
        
        # Create strategy
        strategy = SMACrossoverStrategy()
        
        # Verify indicators were created
        mock_sma.assert_called()
        mock_crossover.assert_called()


if __name__ == '__main__':
    unittest.main()
