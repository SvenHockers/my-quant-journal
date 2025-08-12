"""
Simple Moving Average Crossover Strategy
A basic example strategy for demonstration purposes.
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    
    Buy when fast SMA crosses above slow SMA
    Sell when fast SMA crosses below slow SMA
    """
    
    # Define strategy parameters using Backtrader's parameter system
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('printlog', False),
    )
    
    def __init__(self):
        super().__init__()
        
        # Add indicators
        self.fast_sma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_sma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        
        # Crossover signals
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
    
    def next(self):
        """Main strategy logic."""
        # Check if we have a position
        if not self.position:
            # No position - check for buy signal
            if self.crossover > 0:  # Fast SMA crosses above slow SMA
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                self.order = self.buy()
        else:
            # Have position - check for sell signal
            if self.crossover < 0:  # Fast SMA crosses below slow SMA
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()
    
    def log(self, txt, dt=None):
        """Log strategy messages if printlog is True."""
        if self.params.printlog:
            super().log(txt, dt)
