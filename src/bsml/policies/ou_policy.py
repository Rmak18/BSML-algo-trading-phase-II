"""
Ornstein-Uhlenbeck execution policy (Section 7 of paper).

Exact discretisation (Gillespie 1996 / Anderson 2007):
    X_{t+1} = α·X_t + (1-α)·μ + σ_ε·ε_t,   ε_t ~ N(0,1)
where:
    α   = exp(-θ·Δt)                          AR(1) coefficient
    σ_ε = σ·√((1 - exp(-2θΔt)) / (2θ))       innovation std

Stationary distribution: N(μ, σ²/(2θ))
Expected autocorrelations at paper defaults (θ=0.5, σ=0.5):
    ρ(1) ≈ exp(−θ) ≈ 0.606,  ρ(5) ≈ 0.082,  ρ(10) ≈ 0.007
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .base_policy import RandomizationPolicy
from .baseline import generate_trades as baseline_generate
from .utils import clamp_to_market_hours

# ── Paper-specified defaults (Section 7) ─────────────────────────────────────
DEFAULT_OU_PARAMS = {
    "theta_timing":    0.5,    # mean-reversion speed for timing process
    "sigma_timing":    0.5,    # noise magnitude for timing process
    "theta_threshold": 0.5,    # mean-reversion speed for threshold process
    "sigma_threshold": 0.05,   # noise magnitude for threshold process
    "price_scale":     0.04,   # maps OU state → price deviation fraction
    "mu":              0.0,    # long-run mean (symmetric around 0)
}

FAST_REVERSION_OU_PARAMS = {
    "theta_timing":    1.0,
    "sigma_timing":    0.5,
    "theta_threshold": 1.0,
    "sigma_threshold": 0.05,
    "price_scale":     0.04,
    "mu":              0.0,
}

SLOW_REVERSION_OU_PARAMS = {
    "theta_timing":    0.1,
    "sigma_timing":    0.5,
    "theta_threshold": 0.1,
    "sigma_threshold": 0.05,
    "price_scale":     0.04,
    "mu":              0.0,
}


class OUPolicy(RandomizationPolicy):
    """
    Ornstein-Uhlenbeck mean-reverting noise policy.

    Maintains separate OU processes for timing and threshold perturbations,
    plus a bulk-noise path for generate_trades().

    Constructor accepts new-style params dict OR legacy kwargs:
        OUPolicy(seed=42, params={'theta_timing': 0.5, 'sigma_timing': 0.5, ...})
        OUPolicy(seed=42, theta=0.5, sigma=0.5, price_scale=0.04)   # legacy
        OUPolicy(seed=42)  # uses DEFAULT_OU_PARAMS
    """

    def __init__(
        self,
        seed: int = 42,
        params: dict = None,
        # legacy kwargs
        theta:       float = DEFAULT_OU_PARAMS["theta_timing"],
        mu:          float = 0.0,
        sigma:       float = DEFAULT_OU_PARAMS["sigma_timing"],
        price_scale: float = DEFAULT_OU_PARAMS["price_scale"],
    ):
        if params is not None:
            p = dict(params)
            seed = p.pop('seed', seed)

            if len(p) == 0:
                raise ValueError(
                    "OUPolicy requires parameters. "
                    "Provide theta_timing, sigma_timing, theta_threshold, "
                    "sigma_threshold (or legacy theta/sigma)."
                )

            # Support legacy theta/sigma keys alongside new-style
            theta_t   = p.get('theta_timing',    p.get('theta', theta))
            sigma_t   = p.get('sigma_timing',    p.get('sigma', sigma))
            theta_th  = p.get('theta_threshold', theta_t)
            sigma_th  = p.get('sigma_threshold', sigma_t * 0.1)
            mu_       = p.get('mu',              mu)
            ps        = p.get('price_scale',     price_scale)
            reset_thr = p.get('state_reset_threshold', float('inf'))

            assert theta_t  >= 0, f"theta_timing must be >= 0, got {theta_t}"
            assert sigma_t  >= 0, f"sigma_timing must be >= 0, got {sigma_t}"
            assert theta_th >= 0, f"theta_threshold must be >= 0, got {theta_th}"
            assert sigma_th >= 0, f"sigma_threshold must be >= 0, got {sigma_th}"

            built = {
                'theta_timing':          theta_t,
                'sigma_timing':          sigma_t,
                'theta_threshold':       theta_th,
                'sigma_threshold':       sigma_th,
                'mu':                    mu_,
                'price_scale':           ps,
                'state_reset_threshold': reset_thr,
            }
        else:
            # Legacy kwargs path — no error raised for backwards compat
            assert theta >= 0, f"theta must be >= 0, got {theta}"
            assert sigma >= 0, f"sigma must be >= 0, got {sigma}"
            built = {
                'theta_timing':          theta,
                'sigma_timing':          sigma,
                'theta_threshold':       theta,
                'sigma_threshold':       sigma * 0.1,
                'mu':                    mu,
                'price_scale':           price_scale,
                'state_reset_threshold': float('inf'),
            }

        super().__init__(seed=seed, params=built)

        # OU state machine
        self._ou_timing_state:    float = 0.0
        self._ou_threshold_state: float = 0.0
        self._ou_timing_history:  List[float] = []
        self._ou_threshold_history: List[float] = []
        self._reset_events: List[dict] = []

        # Cache AR(1) coefficients for both processes
        self._alpha_t, self._sigma_eps_t   = self._ou_coeffs(
            built['theta_timing'],   built['sigma_timing']
        )
        self._alpha_th, self._sigma_eps_th = self._ou_coeffs(
            built['theta_threshold'], built['sigma_threshold']
        )

        # Legacy-compatible attributes (used by generate_trades)
        self.theta       = built['theta_timing']
        self.mu          = built['mu']
        self.sigma       = built['sigma_timing']
        self.price_scale = built['price_scale']

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ou_coeffs(theta: float, sigma: float):
        """Return (alpha, sigma_eps) for exact OU discretisation (Δt=1)."""
        alpha     = np.exp(-theta)
        sigma_eps = sigma * np.sqrt(
            (1.0 - np.exp(-2.0 * theta)) / max(2.0 * theta, 1e-8)
        )
        return alpha, sigma_eps

    def _step_timing(self) -> float:
        """Advance OU timing state one step; return new state."""
        mu = self.params['mu']
        x  = (self._alpha_t * self._ou_timing_state
              + (1.0 - self._alpha_t) * mu
              + self._sigma_eps_t * self.rng.normal())
        # State-reset monitoring
        threshold = self.params.get('state_reset_threshold', float('inf'))
        if abs(x) > threshold:
            self._reset_events.append({
                'step':      len(self._ou_timing_history),
                'old_state': x,
                'new_state': 0.0,
            })
            x = 0.0
        self._ou_timing_state = x
        self._ou_timing_history.append(x)
        return x

    def _step_threshold(self) -> float:
        """Advance OU threshold state one step; return new state."""
        mu = self.params['mu']
        x  = (self._alpha_th * self._ou_threshold_state
              + (1.0 - self._alpha_th) * mu
              + self._sigma_eps_th * self.rng.normal())
        self._ou_threshold_state = x
        self._ou_threshold_history.append(x)
        return x

    @staticmethod
    def _compute_autocorrelation(
        history: list,
        max_lag: int = 10,
    ) -> np.ndarray:
        """
        Compute sample autocorrelation of history up to max_lag.

        Returns array of length (min(max_lag+1, len(history))) where
        index 0 = lag-0 (= 1.0), index k = lag-k.
        """
        n = len(history)
        if n < 2:
            return np.array([1.0])
        x   = np.asarray(history, dtype=float)
        x   = x - x.mean()
        var = np.dot(x, x)
        if var < 1e-12:
            return np.ones(min(max_lag + 1, n))
        lags = range(min(max_lag + 1, n))
        acf  = np.array([
            (np.dot(x[:n - k], x[k:]) / var) if k < n else 0.0
            for k in lags
        ])
        return acf

    # ── Core API ──────────────────────────────────────────────────────────────

    def perturb_timing(
        self,
        timestamp: datetime,
        signal_strength: float = 1.0,
    ) -> datetime:
        """Shift timestamp by OU state (hours); clamp to market hours."""
        state = self._step_timing()
        perturbed = clamp_to_market_hours(
            timestamp + timedelta(hours=float(state))
        )
        self._perturbation_log.append({
            'type': 'timing',
            'state': state,
            'delta_hours': (perturbed - timestamp).total_seconds() / 3600.0,
        })
        return perturbed

    def perturb_threshold(
        self,
        base_threshold: float,
        current_price: Optional[float] = None,
    ) -> float:
        """Scale threshold by (1 + OU state); ensure positive."""
        state = self._step_threshold()
        perturbed = max(base_threshold * (1.0 + state), 1e-8)
        self._perturbation_log.append({
            'type': 'threshold',
            'state': state,
        })
        return float(perturbed)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return OU-specific diagnostics."""
        theta_t = self.params['theta_timing']
        half_life = np.log(2.0) / max(theta_t, 1e-8)

        acf = self._compute_autocorrelation(
            self._ou_timing_history, max_lag=10
        ).tolist()

        timing_info: Dict[str, Any] = {
            'half_life_hours':  half_life,
            'theta':            theta_t,
            'sigma':            self.params['sigma_timing'],
            'current_state':    self._ou_timing_state,
            'autocorrelation':  acf,
        }
        if self._ou_timing_history:
            arr = np.array(self._ou_timing_history)
            timing_info['mean_state'] = float(np.mean(arr))
            timing_info['std_state']  = float(np.std(arr))

        threshold_info: Dict[str, Any] = {
            'current_state': self._ou_threshold_state,
        }
        if self._ou_threshold_history:
            arr = np.array(self._ou_threshold_history)
            threshold_info['mean_state'] = float(np.mean(arr))
            threshold_info['std_state']  = float(np.std(arr))

        return {
            'policy':         'OU',
            'seed':           self.seed,
            'params':         self.params.copy(),
            'n_perturbations': len(self._perturbation_log),
            'n_resets':       len(self._reset_events),
            'timing':         timing_info,
            'threshold':      threshold_info,
        }

    # ── Bulk noise for pipeline ───────────────────────────────────────────────

    def generate_sequence(self, n: int) -> np.ndarray:
        """Generate n steps of OU noise (for verification / testing)."""
        return self._ou_noise(n)

    def _ou_noise(self, n: int) -> np.ndarray:
        """
        Simulate n steps of OU process (exact discretisation, Δt=1).
        X_0 drawn from stationary distribution N(μ, σ²/(2θ)).
        """
        alpha, sigma_eps = self._alpha_t, self._sigma_eps_t
        stat_std = self.sigma / np.sqrt(max(2.0 * self.theta, 1e-8))
        x = np.empty(n)
        x[0] = self.rng.normal(self.mu, stat_std)
        for t in range(1, n):
            x[t] = (alpha * x[t - 1]
                    + (1.0 - alpha) * self.mu
                    + sigma_eps * self.rng.normal())
        return x

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        if "price" not in trades.columns:
            trades["price"] = trades.get("ref_price", prices["price"].values)
        if "ref_price" not in trades.columns:
            trades["ref_price"] = trades["price"]

        noise = self._ou_noise(n)
        # Normalize OU noise to std≈1 so price_scale has the same interpretation
        # as Pink policy (which generates standardized noise).  Without this,
        # OU stationary std ≈ σ/√(2θ) = 0.5, halving the effective perturbation
        # relative to Pink at the same price_scale.
        stat_std = self.sigma / np.sqrt(max(2.0 * self.theta, 1e-8))
        noise = noise / max(stat_std, 1e-8)
        # Perturb the actual execution price (ref_price stays as arrival/benchmark).
        trades["price"] = trades["price"] * (1.0 + self.price_scale * noise)
        return trades


# ── Module-level runner entrypoint for policy='ou' ───────────────────────────

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """Runner entrypoint: uses paper-default OU parameters."""
    policy = OUPolicy(seed=42)
    return policy.generate_trades(prices)


__all__ = [
    "OUPolicy",
    "DEFAULT_OU_PARAMS",
    "FAST_REVERSION_OU_PARAMS",
    "SLOW_REVERSION_OU_PARAMS",
    "generate_trades",
]
