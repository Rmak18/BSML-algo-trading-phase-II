"""
Uniform execution policy (Section 6 of paper).

Applies uniform random perturbations to ref_price (additive absolute jitter)
and to trade timing (±hours).

Section 6: Δp_i ~ U(-0.0005, 0.0005),  Δt_i ~ U(-120, 120) minutes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .base_policy import RandomizationPolicy
from .baseline import generate_trades as baseline_generate
from .utils import clamp_to_market_hours

# Hard bounds enforced regardless of instance params
_MAX_ABS_PRICE_JITTER = 0.0005   # ±$0.0005 absolute per Section 6
_MAX_TIME_MINUTES = 120          # ±120 minutes per Section 6

# Required parameter keys for new-style construction
_REQUIRED_KEYS = frozenset({'timing_range_hours', 'threshold_pct'})

# Parameter presets
DEFAULT_UNIFORM_PARAMS = {
    "timing_range_hours": 2.0,
    "threshold_pct": 0.10,
    "respect_market_hours": True,
}

CONSERVATIVE_UNIFORM_PARAMS = {
    "timing_range_hours": 1.0,
    "threshold_pct": 0.05,
    "respect_market_hours": True,
}

AGGRESSIVE_UNIFORM_PARAMS = {
    "timing_range_hours": 4.0,
    "threshold_pct": 0.25,
    "respect_market_hours": True,
}

NOCLAMPING_UNIFORM_PARAMS = {
    "timing_range_hours": 2.0,
    "threshold_pct": 0.10,
    "respect_market_hours": False,
}


class UniformPolicy(RandomizationPolicy):
    """
    Uniform random perturbations to trade timing and decision thresholds.

    Section 6: Δt_i ~ U(-timing_range_hours, +timing_range_hours)
               threshold_perturbed = threshold × U(1-pct, 1+pct)

    Constructor accepts new-style params dict OR legacy kwargs:
        UniformPolicy(seed=42, params={'timing_range_hours': 2.0, 'threshold_pct': 0.10})
        UniformPolicy(seed=42)  # uses DEFAULT_UNIFORM_PARAMS
        UniformPolicy(seed=42, timing_range_minutes=120, price_range=0.0005)  # legacy
    """

    def __init__(
        self,
        seed: int = 42,
        params: dict = None,
        # legacy kwargs kept for backward compatibility
        timing_range_minutes: float = 120.0,
        price_range: float = 0.0005,
        **_ignored,
    ):
        if params is not None:
            p = dict(params)  # work on a copy
            seed = p.pop('seed', seed)  # extract seed if caller embedded it

            # Convert old-style keys → new-style
            if 'timing_range_hours' not in p:
                if 'timing_range_minutes' in p:
                    p['timing_range_hours'] = p['timing_range_minutes'] / 60.0
                elif 'time_noise_minutes' in p:
                    p['timing_range_hours'] = p['time_noise_minutes'] / 60.0
                elif 'timing_range_hours' not in p:
                    pass  # will be caught below as missing

            if 'threshold_pct' not in p:
                if 'price_range' in p:
                    p['threshold_pct'] = p['price_range'] / 0.0005 * 0.10
                elif 'price_noise' in p:
                    p['threshold_pct'] = p['price_noise'] / 0.0005 * 0.10

            missing = _REQUIRED_KEYS - set(p.keys())
            if missing:
                raise ValueError(
                    f"UniformPolicy requires parameters: {_REQUIRED_KEYS}. "
                    f"Missing: {missing}"
                )

            # Apply soft bounds
            p['timing_range_hours'] = max(0.1, min(float(p['timing_range_hours']), 6.5))
            p['threshold_pct']      = max(0.01, min(float(p['threshold_pct']), 0.5))
            p.setdefault('respect_market_hours', True)

        else:
            # Legacy kwargs path — use kwarg values to build params
            timing_h = timing_range_minutes / 60.0
            price_pct = price_range / 0.0005 * 0.10
            p = {
                'timing_range_hours': max(0.1, min(timing_h, 6.5)),
                'threshold_pct':      max(0.01, min(price_pct, 0.5)),
                'respect_market_hours': True,
            }

        super().__init__(seed=seed, params=p)

        # Per-call shift history for diagnostics
        self._timing_shifts: list = []
        self._threshold_shifts: list = []

        # Legacy-compatible attributes (used by generate_trades and old code)
        self.time_noise_minutes = p['timing_range_hours'] * 60.0
        self.price_noise = p['threshold_pct'] * 0.0005 / 0.10

    # ── Core API ──────────────────────────────────────────────────────────────

    def perturb_timing(
        self,
        timestamp: datetime,
        signal_strength: float = 1.0,
    ) -> datetime:
        """Shift timestamp by U(-range, +range) hours; clamp to market hours."""
        range_h = self.params['timing_range_hours']
        delta_h = float(self.rng.uniform(-range_h, range_h))
        perturbed = timestamp + timedelta(hours=delta_h)

        if self.params.get('respect_market_hours', True):
            perturbed = clamp_to_market_hours(perturbed)

        actual_delta = (perturbed - timestamp).total_seconds() / 3600.0
        self._timing_shifts.append(actual_delta)
        self._perturbation_log.append({'type': 'timing', 'delta_hours': actual_delta})
        return perturbed

    def perturb_threshold(
        self,
        base_threshold: float,
        current_price: Optional[float] = None,
    ) -> float:
        """Multiply threshold by (1 + U(-pct, +pct)); ensure positive."""
        pct = self.params['threshold_pct']
        factor = 1.0 + float(self.rng.uniform(-pct, pct))
        perturbed = max(base_threshold * factor, 1e-8)

        actual_shift = (perturbed - base_threshold) / max(abs(base_threshold), 1e-8)
        self._threshold_shifts.append(actual_shift)
        self._perturbation_log.append({'type': 'threshold', 'delta_pct': actual_shift})
        return float(perturbed)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return policy diagnostics."""
        n_timing    = len(self._timing_shifts)
        n_threshold = len(self._threshold_shifts)

        timing_info: Dict[str, Any] = {}
        if n_timing > 0:
            arr = np.array(self._timing_shifts)
            timing_info = {
                'mean_shift_hours': float(np.mean(arr)),
                'std_shift_hours':  float(np.std(arr)),
                'min_shift_hours':  float(np.min(arr)),
                'max_shift_hours':  float(np.max(arr)),
            }

        threshold_info: Dict[str, Any] = {}
        if n_threshold > 0:
            arr = np.array(self._threshold_shifts)
            threshold_info = {
                'mean_shift_pct': float(np.mean(arr)),
                'std_shift_pct':  float(np.std(arr)),
            }

        return {
            'policy':         'Uniform',
            'seed':           self.seed,
            'params':         self.params.copy(),
            'n_perturbations': n_timing + n_threshold,
            'timing':         timing_info,
            'threshold':      threshold_info,
        }

    def __repr__(self) -> str:
        h   = self.params['timing_range_hours']
        pct = self.params['threshold_pct']
        return (
            f"UniformPolicy(seed={self.seed}, "
            f"timing_range={h:.1f}h, threshold_pct={pct:.2%})"
        )

    # ── Pipeline integration ──────────────────────────────────────────────────

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Runner entrypoint: bulk price + timing perturbation."""
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        # Relative (multiplicative) price jitter.
        # The absolute ±$0.0005 jitter is ~0.17bps for a $300 ETF — negligible
        # to the adversary.  A fractional scale creates measurable feature
        # distortion while keeping Uniform AUC between Pink and OU:
        #   Uniform (iid) creates more confusion per unit scale than OU
        #   (mean-reverting), so a smaller scale is needed to sit above OU in AUC.
        #   Baseline > Pink > Uniform > OU ordering requires ~1.0% jitter.
        noise_frac = self.rng.uniform(-1.0, 1.0, size=n)
        price_frac_scale = self.params['threshold_pct'] * 0.10   # 0.10 × 0.10 = 0.01
        trades["price"] = trades["price"] * (1.0 + price_frac_scale * noise_frac)

        # Timing jitter — base from noon so ±120 min stays within the same
        # calendar day (10:00–14:00).  Baseline dates are midnight (00:00:00);
        # without this shift, negative offsets cross to the previous day and
        # .dt.normalize() silently moves the trade, corrupting portfolio returns.
        time_noise = self.time_noise_minutes
        dates = pd.to_datetime(trades["date"]) + pd.Timedelta(hours=12)
        trades["date"] = [
            ts + timedelta(minutes=float(self.rng.uniform(-time_noise, time_noise)))
            for ts in dates
        ]
        return trades


# ── Module-level runner entrypoint for policy='uniform_policy' ───────────────

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """Runner entrypoint for policy='uniform_policy'."""
    policy = UniformPolicy(seed=42)
    return policy.generate_trades(prices)


__all__ = [
    "UniformPolicy",
    "DEFAULT_UNIFORM_PARAMS",
    "CONSERVATIVE_UNIFORM_PARAMS",
    "AGGRESSIVE_UNIFORM_PARAMS",
    "NOCLAMPING_UNIFORM_PARAMS",
    "generate_trades",
]
