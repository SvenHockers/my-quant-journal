"""
Base strategy class for all trading strategies.
Provides common functionality and interface for strategy implementations.
"""

import backtrader as bt
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStrategy(bt.Strategy, ABC):
    """
    Abstract base class for all trading strategies.
    Implements common functionality and defines required interface.
    """
    
    def __init__(self):
        super().__init__()
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Strategy parameters
        self.params = self.get_strategy_params()
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
    
    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def next(self):
        """Main strategy logic. Must be implemented by subclasses."""
        pass
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        self.trades.append({
            'entry_date': trade.dtopen,
            'exit_date': trade.dtclose,
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'size': trade.size
        })
    
    def log(self, txt, dt=None):
        """Log strategy messages."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Return performance summary statistics."""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pnlcomm = sum(t['pnlcomm'] for t in self.trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_pnlcomm': total_pnlcomm,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
        }
