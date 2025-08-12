"""
Base Optimisers for Parameter Tuning

Defines a reusable abstract base that coordinates with the BacktestEngine
to evaluate strategies under different hyper-parameter configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Type, Union, cast


class BaseOptimiser(ABC):
    """Abstract base class for strategy hyper-parameter optimisers.

    This class provides common utilities to:
    - Run a single evaluation of a strategy with a given parameter set
    - Extract a numeric score from a backtest result
    - Validate that provided parameter names are supported by a strategy

    Subclasses should implement `optimise()` to explore the search space
    (e.g., grid search, random search, Bayesian optimisation) and return the
    best parameters and associated metrics.
    """

    def __init__(
        self,
        engine: Any,
        metric: Union[str, Callable[[Dict[str, Any]], float]] = "sharpe_ratio",
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        """Create a new optimiser.

        Args:
            engine: A `BacktestEngine` instance used to run evaluations.
            metric: Key in the result summary to score by (e.g., 'sharpe_ratio',
                'total_return'), or a callable mapping result summary -> float.
            mode: 'max' to maximize the metric, 'min' to minimize.
            verbose: Whether to print progress information.
        """
        # Duck-typed engine must provide run_backtest()
        if not hasattr(engine, "run_backtest") or not callable(getattr(engine, "run_backtest")):
            raise TypeError("engine must provide a callable 'run_backtest' method")
        if mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")

        self.engine = engine
        self.metric = metric
        self.mode = mode
        self.verbose = verbose

    # ------------- Public API -------------
    @abstractmethod
    def optimise(
        self,
        strategy_class: Type[Any],
        param_space: Dict[str, Iterable[Any]],
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Run the optimisation and return the best configuration.

        Expected return format (recommended):
            {
              'best_params': Dict[str, Any],
              'best_score': float,
              'best_summary': Dict[str, Any],
              'trials': List[Dict[str, Any]]  # optional per-evaluation summaries
            }
        """
        raise NotImplementedError

    # ------------- Utilities for subclasses -------------
    def evaluate_params(
        self,
        strategy_class: Type[Any],
        params: Dict[str, Any],
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run a single backtest with `params` and return (score, summary).

        The summary is the dictionary returned by `BacktestEngine.run_backtest()`
        without the heavy objects (cerebro/strategy) inlined; the optimiser only
        needs high-level metrics found in the summary.
        """
        self._validate_param_names(strategy_class, params)

        if self.verbose:
            print(f"Evaluating params: {params}")

        # Use the engine's one-shot backtest path so that each trial is isolated
        summary = self.engine.run_backtest(
            strategy_class=strategy_class,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_params=params,
        )

        score = self._score(summary)
        return score, summary

    def _score(self, summary: Dict[str, Any]) -> float:
        """Compute the numeric score from a backtest summary.

        If the requested metric is missing or None, returns a sentinel value
        that ranks worst given the specified `mode`.
        """
        value: Optional[float]
        if callable(self.metric):
            try:
                value = float(self.metric(summary))
            except Exception:
                value = None
        else:
            value = summary.get(str(self.metric))
            try:
                value = None if value is None else float(value)
            except Exception:
                value = None

        if value is None:
            # Worst-possible score depending on the optimisation direction
            return float("-inf") if self.mode == "max" else float("inf")
        return value if self.mode == "max" else -value

    def _validate_param_names(self, strategy_class: Type[Any], params: Dict[str, Any]) -> None:
        """Ensure provided parameter names are declared by the strategy.

        Backtrader strategies declare `params` as a tuple of (name, default)
        or as a dict. Unknown parameters are rejected to surface mistakes early.
        """
        allowed: Set[str] = set()
        params_decl: Any = getattr(strategy_class, "params", ())
        if isinstance(params_decl, dict):
            allowed = {str(k) for k in cast(Dict[Any, Any], params_decl).keys()}
        else:
            try:
                for item in cast(Iterable[Any], params_decl):
                    if isinstance(item, tuple) and item:
                        t = cast(Tuple[Any, ...], item)
                        name = str(t[0])
                        allowed.add(name)
            except Exception:
                allowed = set()

        unknown = sorted(set(params.keys()) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown strategy parameters for {strategy_class.__name__}: {unknown}. "
                f"Allowed: {sorted(allowed)}"
            )
