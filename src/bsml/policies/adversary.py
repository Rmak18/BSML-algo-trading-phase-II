"""
Adversary Classifier Module

Trains a GradientBoostingClassifier to predict trade direction (BUY/SELL) from
market features, then measures how predictable each policy is via AUC-ROC.

Paper spec (Section 7):
- Model  : GradientBoostingClassifier(n_estimators=100, lr=0.1, max_depth=5)
- Split  : temporal 70/30 (first 70% by date → train, last 30% → test)
- Label  : 1=BUY, 0=SELL
- Features (exactly 23):
    ret_5d, ret_10d, ret_20d              — 5/10/20-day price returns          (3)
    cs_rank                                — cross-sectional price rank          (1)
    vol_10d, vol_30d, vol_60d             — realized volatility                  (3)
    vol_pctile_10d, vol_pctile_30d,
      vol_pctile_60d                       — vol percentile vs own history       (3)
    dow_Mon, dow_Tue, dow_Wed, dow_Thu    — day-of-week dummies (Fri=ref)       (4)
    mon_Jan … mon_Sep                      — month dummies (Oct/Nov/Dec=ref)     (9)
                                                                         Total = 23
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def _build_price_history(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format trades to wide (date × symbol) for rolling feature computation.
    Uses trade price as the price proxy.
    """
    wide = (
        trades[["date", "symbol", "price"]]
        .drop_duplicates(subset=["date", "symbol"])
        .pivot(index="date", columns="symbol", values="price")
        .sort_index()
    )
    return wide


def extract_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 23 adversary features for each row of a long-format trades DataFrame.

    Parameters
    ----------
    trades : DataFrame with columns [date, symbol, side, qty, price, ...]
             Rows must be sorted by date (ascending).

    Returns
    -------
    features : DataFrame with exactly 23 columns, same row order as trades.
    """
    trades = trades.copy()
    # Normalize to midnight so intraday timing jitter (uniform_policy ±120 min)
    # doesn't break the price-history pivot or _vlookup index alignment.
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)

    wide = _build_price_history(trades)
    daily_ret = wide.pct_change()

    # ── Per-symbol rolling features ──────────────────────────────────────────
    ret_5 = wide.pct_change(5)
    ret_10 = wide.pct_change(10)
    ret_20 = wide.pct_change(20)
    vol_10 = daily_ret.rolling(10).std() * np.sqrt(252)
    vol_30 = daily_ret.rolling(30).std() * np.sqrt(252)
    vol_60 = daily_ret.rolling(60).std() * np.sqrt(252)

    # vol percentile: fraction of past vol values below current (one per vol horizon)
    def rolling_pctile(series: pd.Series, window: int = 252) -> pd.Series:
        def _pctile(arr):
            return (arr[:-1] < arr[-1]).mean() if len(arr) > 1 else 0.5
        return series.rolling(window, min_periods=20).apply(_pctile, raw=True)

    vol_pctile_10 = vol_10.apply(rolling_pctile)
    vol_pctile_30 = vol_30.apply(rolling_pctile)
    vol_pctile_60 = vol_60.apply(rolling_pctile)

    # ── Cross-sectional rank (per date) ─────────────────────────────────────
    cs_rank = wide.rank(axis=1, pct=True)

    # ── Vectorized map from wide to long format ───────────────────────────────
    # Stack each wide DataFrame to a Series with MultiIndex (date, symbol),
    # then reindex by the (date, symbol) pairs in the trades table.
    def _vlookup(df_wide: pd.DataFrame) -> np.ndarray:
        long = df_wide.stack(future_stack=True)           # MultiIndex (date, symbol)
        idx = pd.MultiIndex.from_arrays(
            [trades["date"], trades["symbol"]]
        )
        return long.reindex(idx).values

    feat = pd.DataFrame(index=trades.index)
    feat["ret_5d"]    = _vlookup(ret_5)
    feat["ret_10d"]   = _vlookup(ret_10)
    feat["ret_20d"]   = _vlookup(ret_20)
    feat["cs_rank"]   = _vlookup(cs_rank)
    feat["vol_10d"]        = _vlookup(vol_10)
    feat["vol_30d"]        = _vlookup(vol_30)
    feat["vol_60d"]        = _vlookup(vol_60)
    feat["vol_pctile_10d"] = _vlookup(vol_pctile_10)
    feat["vol_pctile_30d"] = _vlookup(vol_pctile_30)
    feat["vol_pctile_60d"] = _vlookup(vol_pctile_60)

    # ── Day-of-week dummies (Mon-Thu; Fri is reference) ──────────────────────
    dow = pd.to_datetime(trades["date"]).dt.dayofweek  # Mon=0 … Sun=6
    for d, name in zip([0, 1, 2, 3], ["dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu"]):
        feat[name] = (dow == d).astype(float)

    # ── Month dummies (Jan-Sep; Oct/Nov/Dec are reference) ───────────────────
    # Oct and Nov dropped to stay at 23 total features after adding vol_pctile_10d/60d.
    month = pd.to_datetime(trades["date"]).dt.month
    month_names = ["mon_Jan", "mon_Feb", "mon_Mar", "mon_Apr", "mon_May",
                   "mon_Jun", "mon_Jul", "mon_Aug", "mon_Sep"]
    for m, name in zip(range(1, 10), month_names):
        feat[name] = (month == m).astype(float)

    assert feat.shape[1] == 23, f"Expected 23 features, got {feat.shape[1]}"
    return feat.fillna(0.0)


class AdversaryClassifier:
    """
    Adversary that tries to predict trade direction (BUY=1 / SELL=0).

    Higher AUC → more predictable policy → weaker randomization.
    AUC ≈ 0.50 → policy is indistinguishable from random.
    """

    def __init__(self):
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        self._trained = False

    def train_and_evaluate(self, trades: pd.DataFrame) -> float:
        """
        Train on first 70% of trades (by date) and evaluate on last 30%.

        Parameters
        ----------
        trades : long-format DataFrame with columns [date, symbol, side, ...]

        Returns
        -------
        auc_score : float — AUC-ROC on temporal test set
        """
        trades = trades.copy()
        trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
        trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)

        X = extract_features(trades)
        y = (trades["side"] == "BUY").astype(int).values

        # Temporal 70/30 split
        n = len(trades)
        n_train = int(n * 0.70)
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Drop rows with all-NaN features that couldn't be computed
        valid_train = ~X_train.isna().all(axis=1)
        valid_test = ~X_test.isna().all(axis=1)
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return 0.5  # degenerate case: only one class

        self.classifier.fit(X_train.fillna(0.0), y_train)
        self._trained = True

        y_pred_proba = self.classifier.predict_proba(X_test.fillna(0.0))[:, 1]
        return float(roc_auc_score(y_test, y_pred_proba))

    def evaluate(self, trades: pd.DataFrame) -> float:
        """
        Apply already-trained classifier to a new trades set; return AUC.

        Evaluates only on the last 30% of rows (same temporal test window as
        training split) so the classifier is never scored on its own training
        period, which would inflate AUC.  Features are computed on the full
        dataset to preserve rolling-window history, but AUC is measured on
        the test slice only.
        """
        if not self._trained:
            raise ValueError("Classifier must be trained first via train_and_evaluate()")
        trades = trades.copy()
        trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
        trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)
        X = extract_features(trades).fillna(0.0)
        y = (trades["side"] == "BUY").astype(int).values

        # Only evaluate on last 30% — identical temporal split to training
        n = len(trades)
        n_test_start = int(n * 0.70)
        X_test = X.iloc[n_test_start:]
        y_test = y[n_test_start:]

        if len(np.unique(y_test)) < 2:
            return 0.5
        y_pred = self.classifier.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from bsml.data.loader import load_prices
    from bsml.policies.baseline import generate_trades

    prices = load_prices("data/ALL_backtest.csv")
    trades = generate_trades(prices)

    clf = AdversaryClassifier()
    auc = clf.train_and_evaluate(trades)
    print(f"Baseline adversary AUC: {auc:.4f}")
    print("(Lower is better for defender; 0.50 = random guessing)")
