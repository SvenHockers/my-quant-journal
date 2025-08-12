"""
Grid Search Optimiser for Parameter Tuning.

Provides a simple exhaustive search over a discrete grid of parameters
for any Backtrader strategy implementing `params`.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Type

from .base_optimiser import BaseOptimiser


class GridSearchOptimiser(BaseOptimiser):
    """The classic grid search optimiser"""

    def _combinations(self, param_space: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
        names: List[str] = list(param_space.keys())
        values: List[Iterable[Any]] = [list(v) for v in param_space.values()]
        combos: List[Dict[str, Any]] = []
        for vals in product(*values):
            combos.append(dict(zip(names, vals)))
        return combos

    def optimise(
        self,
        strategy_class: Type[Any],
        param_space: Dict[str, Iterable[Any]],
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        trials: List[Dict[str, Any]] = []
        best_score: float = float("-inf") if self.mode == "max" else float("inf")
        best_params: Dict[str, Any] = {}
        best_summary: Dict[str, Any] = {}

        candidates = self._combinations(param_space)
        total = len(candidates)
        if self.verbose:
            print(f"GridSearch: evaluating {total} combinations...")

        for idx, params in enumerate(candidates, start=1):
            try:
                score, summary = self.evaluate_params(
                    strategy_class,
                    params,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Combination {idx}/{total} failed for params {params}: {e}")
                continue

            # Store trial record
            trials.append({
                **params,
                'score': score,
                'summary': {k: v for k, v in summary.items() if k not in {'cerebro', 'strategy'}},
            })

            is_better = score > best_score
            if is_better:
                best_score = score
                best_params = params
                best_summary = summary

            if self.verbose:
                direction = 'max' if self.mode == 'max' else 'min'
                print(f"[{idx}/{total}] score={score:.6f} ({direction} {self.metric}) | best={best_score:.6f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_summary': {k: v for k, v in best_summary.items() if k not in {'cerebro', 'strategy'}},
            'trials': trials,
        }
