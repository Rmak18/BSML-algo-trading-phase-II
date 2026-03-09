"""
P6 - Standard Adversary v0.1 (Skeleton Implementation)

- Uses MOCK events (P3 will replace with real logs in v1.0).

    extract_features(trades)          -> X (features DataFrame)
    generate_labels(trades, delta_t) -> y (0/1 labels)
    train_adversary_classifier(...)  -> trained model
    compute_auc(...)                 -> AUC
    compute_pnr(...)                 -> PNR-like metric

- Trains on deterministic policy only.
- Evaluates on all policies and volatility regimes.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report



# CONFIG

DETERMINISTIC_POLICY_ID = "deterministic"

TRAIN_END = "2023-06-30"
VAL_END = "2023-09-30"

TOP_K_PNR = 0.2
ROLLING_WINDOWS = [10, 50]

# For timing labels: we'll approximate Δt as "next bar" in mock data.
DEFAULT_DELTA_STEPS = 1



# MOCK DATA GENERATION 

def generate_mock_events(
    n_events: int = 10000,
    n_symbols: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Synthetic events table approximating the real schema expected by P6/P7/P3.

    Columns:
        - timestamp, symbol, policy_id
        - mid_price, volume
        - action_side, action_size, is_market_order
        - realized_cost_lag, estimated_slippage_t_1
        - pnl
        - exec_flag  (1 if a trade execution happens at this bar, 0 otherwise)
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2023-01-01", periods=n_events // n_symbols, freq="H")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    policies = [DETERMINISTIC_POLICY_ID, "random_v1", "random_v2"]

    rows = []
    for t in dates:
        for sym in symbols:
            policy = rng.choice(policies, p=[0.4, 0.3, 0.3])

            # mid prices
            base_price = 100 + 0.01 * (t - dates[0]).total_seconds() / 3600.0
            noise = rng.normal(0, 0.5)
            mid_price = base_price + noise

            volume = float(max(1.0, rng.normal(1000, 200)))

            if policy == DETERMINISTIC_POLICY_ID:
                action_side = np.sign(noise)
                action_size = float(max(0.0, rng.normal(1.0, 0.2)))
            else:
                action_side = rng.choice([-1, 0, 1])
                action_size = float(max(0.0, rng.normal(0.8, 0.5)))

            is_market_order = int(rng.random() < 0.7)

            realized_cost_lag = float(rng.normal(0.01, 0.005))
            estimated_slippage_t_1 = float(rng.normal(0.02, 0.01))

            # PnL structure: deterministic has more structured behavior
            base_pnl = -0.01 * abs(noise) * action_side
            regime_bonus = 0.0
            if policy == DETERMINISTIC_POLICY_ID:
                regime_bonus = -0.02 if noise > 0.5 else 0.01
            pnl = float(base_pnl + regime_bonus + rng.normal(0, 0.01))

            # "exec_flag" = whether an order actually executed at this bar
            exec_flag = int(rng.random() < 0.4)  # ~40% of bars have executions

            rows.append({
                "timestamp": t,
                "symbol": sym,
                "policy_id": policy,
                "mid_price": mid_price,
                "volume": volume,
                "action_side": action_side,
                "action_size": action_size,
                "is_market_order": is_market_order,
                "realized_cost_lag": realized_cost_lag,
                "estimated_slippage_t_1": estimated_slippage_t_1,
                "pnl": pnl,
                "exec_flag": exec_flag,
            })

    df = pd.DataFrame(rows)
    return df



# FEATURE ENGINEERING (P6 → P7: extract_features)


def extract_features(trades: pd.DataFrame,
                     rolling_windows: List[int] = ROLLING_WINDOWS
                     ) -> pd.DataFrame:
    """
    P6 feature extraction function (P7 reuses this).

    - log_mid_return_1
    - rolling vol / volume stats per symbol
    - volatility regime flag
    - z-scored versions of core numeric features

    trades: DataFrame with at least:
        timestamp, symbol, mid_price, volume,
        realized_cost_lag, estimated_slippage_t_1
    """
    df = trades.sort_values(["symbol", "timestamp"]).copy()

    # 1-step log return
    df["log_mid_return_1"] = (
        df.groupby("symbol")["mid_price"]
          .apply(lambda x: np.log(x / x.shift(1)))
    )

    # rolling stats
    for w in rolling_windows:
        grp = df.groupby("symbol")

        df[f"roll_vol_price_{w}"] = (
            grp["mid_price"]
            .transform(lambda x: np.log(x).rolling(w, min_periods=max(3, w//5)).std())
        )
        df[f"roll_mean_volume_{w}"] = (
            grp["volume"]
            .transform(lambda x: x.rolling(w, min_periods=max(3, w//5)).mean())
        )
        df[f"roll_vol_volume_{w}"] = (
            grp["volume"]
            .transform(lambda x: x.rolling(w, min_periods=max(3, w//5)).std())
        )

    # volatility regime flag
    long_w = max(rolling_windows)
    vol_ref = df[f"roll_vol_price_{long_w}"]
    med = vol_ref.median()
    df["vol_regime"] = (vol_ref > med).astype(int)

    # z-score some numeric features
    z_cols = [
        "log_mid_return_1",
        "volume",
        "realized_cost_lag",
        "estimated_slippage_t_1",
    ] + [c for c in df.columns if c.startswith("roll_vol_") or c.startswith("roll_mean_volume_")]

    for col in z_cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    return df



# LABEL GENERATION (P6 → P7: generate_labels)


def generate_labels(trades: pd.DataFrame,
                    delta_steps: int = DEFAULT_DELTA_STEPS
                    ) -> np.ndarray:
    """
    P6 label construction used by P7 (timing / occurrence target).

    Concept: "trade in next Δt":
        label_t = 1 if there is an execution in (t, t+Δt] for the same symbol
                  0 otherwise.

    For mock data:
        - we approximate Δt by 'delta_steps' bars into the future.
        - we use the 'exec_flag' column to indicate executions.

    In real P6 v1.0:
        - this would look at actual event timestamps and windows of length Δt.
    """
    df = trades.sort_values(["symbol", "timestamp"]).copy()

    # Align next 'exec_flag' per symbol with current row via shift(-delta_steps)
    df["label"] = (
        df.groupby("symbol")["exec_flag"]
          .shift(-delta_steps)
          .fillna(0)
          .astype(int)
    )

    return df["label"].values



# TIME SPLITS


def make_time_splits(df: pd.DataFrame,
                     train_end: str = TRAIN_END,
                     val_end: str = VAL_END
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp")

    train = df[df["timestamp"] <= train_end]
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test = df[df["timestamp"] > val_end]

    return train, val, test



# METRICS: compute_pnr & compute_auc (P6 → P7)

def compute_pnr(df: pd.DataFrame,
                prob_col: str = "adv_prob",
                pnl_col: str = "pnl",
                top_k: float = TOP_K_PNR) -> float:
    """
    Simple PNR-like metric:

    - Take top_k fraction of rows with highest adversary probability.
    - Compute avg PnL in that group vs overall avg.
    - Return ratio.
    """
    if prob_col not in df.columns or pnl_col not in df.columns:
        return 0.0

    threshold = df[prob_col].quantile(1 - top_k)
    selected = df[df[prob_col] >= threshold]

    if selected.empty:
        return 0.0

    avg_sel = selected[pnl_col].mean()
    avg_all = df[pnl_col].mean()

    if avg_all == 0 or np.isnan(avg_all):
        return 0.0

    return avg_sel / abs(avg_all)


def compute_auc(classifier: Any,
                features: pd.DataFrame,
                labels: np.ndarray
                ) -> float:
    """
    P6 AUC computation (no bootstrap here in v0.1, but same interface).

    In v1.0:
        - This could be extended with block bootstrap as in the integration doc.
    """
    probs = classifier.predict_proba(features.values)[:, 1]
    auc = roc_auc_score(labels, probs)
    return float(auc)



# TRAINING: train_adversary_classifier (P6 → P7)


def train_adversary_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    model: str = "histgb",
    params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    P6 standard adversary training function.

    For v0.1:
        - We use HistGradientBoostingClassifier (sklearn).
    In v1.0:
        - This can be swapped for LightGBM/XGBoost with same interface.

    Args:
        features: DataFrame of numeric features
        labels: np.ndarray of 0/1
        model: str, kept for compatibility ('histgb' here)
        params: optional hyperparameters dict
    """
    if params is None:
        params = {}

    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.05)
    max_iter = params.get("max_iter", 300)
    min_samples_leaf = params.get("min_samples_leaf", 50)

    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
    )

    clf.fit(features.values, labels)
    return clf



# EVALUATION HELPERS (GLOBAL, PER POLICY, PER REGIME)


def _evaluate_split_global(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    name: str
) -> pd.DataFrame:
    X = df[feature_cols]
    y = df["label"].values

    probs = model.predict_proba(X.values)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y, probs)

    print(f"\n=== {name} - GLOBAL ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, preds, digits=3))

    out = df.copy()
    out["adv_prob"] = probs

    pnr = compute_pnr(out, "adv_prob", "pnl", top_k=TOP_K_PNR)
    print(f"{name} PNR (top {int(TOP_K_PNR*100)}% adv_prob): {pnr:.4f}")

    return out


def _evaluate_split_per_policy(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    name: str
) -> None:
    print(f"\n=== {name} - PER POLICY ===")
    for pid, g in df.groupby("policy_id"):
        if g["label"].nunique() < 2:
            print(f"- policy {pid}: skipped (only one label class)")
            continue

        X = g[feature_cols]
        y = g["label"].values
        probs = model.predict_proba(X.values)[:, 1]

        auc = roc_auc_score(y, probs)
        g = g.copy()
        g["adv_prob"] = probs
        pnr = compute_pnr(g, "adv_prob", "pnl", top_k=TOP_K_PNR)

        print(f"- policy {pid}: AUC={auc:.4f}, PNR={pnr:.4f}, n={len(g)}")


def _evaluate_split_by_regime(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    name: str
) -> None:
    print(f"\n=== {name} - PER VOLATILITY REGIME ===")
    for regime, g in df.groupby("vol_regime"):
        if g["label"].nunique() < 2:
            print(f"- regime {regime}: skipped (only one label class)")
            continue

        X = g[feature_cols]
        y = g["label"].values
        probs = model.predict_proba(X.values)[:, 1]

        auc = roc_auc_score(y, probs)
        print(f"- regime {regime}: AUC={auc:.4f}, n={len(g)}")



# MAIN P6 PIPELINE (v0.1, on mock data)


def run_standard_adversary_pipeline(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level P6 pipeline (mock v0.1):

    1. Extract features (P6 API).
    2. Generate timing labels (P6 API).
    3. Walk-forward split.
    4. Filter train/val to deterministic policy only.
    5. Train standard adversary.
    6. Evaluate globally, per policy, per vol regime.
    """
    print("Extracting features...")
    events_with_features = extract_features(events_df, ROLLING_WINDOWS)

    print("Generating timing labels...")
    labels = generate_labels(events_with_features, delta_steps=DEFAULT_DELTA_STEPS)
    events_with_features["label"] = labels

    # choose numeric feature columns
    exclude = {"timestamp", "symbol", "policy_id", "label", "pnl"}
    feature_cols = [
        c for c in events_with_features.columns
        if c not in exclude and np.issubdtype(events_with_features[c].dtype, np.number)
    ]

    print(f"Using {len(feature_cols)} feature columns.")

    print("Creating time-based splits...")
    train_df, val_df, test_df = make_time_splits(events_with_features, TRAIN_END, VAL_END)
    print(f"Split sizes (before policy filter): "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # deterministic-only training
    train_df = train_df[train_df["policy_id"] == DETERMINISTIC_POLICY_ID]
    val_df = val_df[val_df["policy_id"] == DETERMINISTIC_POLICY_ID]

    print(f"After filtering to deterministic policy ({DETERMINISTIC_POLICY_ID}): "
          f"train={len(train_df)}, val={len(val_df)}")

    print("Training standard adversary classifier...")
    X_train = train_df[feature_cols]
    y_train = train_df["label"].values

    clf = train_adversary_classifier(
        features=X_train,
        labels=y_train,
        model="histgb",
        params={"max_depth": 6, "learning_rate": 0.05, "max_iter": 300}
    )

    # evaluate splits
    train_out = _evaluate_split_global(clf, train_df, feature_cols,
                                       name="TRAIN (deterministic)")
    val_out = _evaluate_split_global(clf, val_df, feature_cols,
                                     name="VAL (deterministic)")
    test_out = _evaluate_split_global(clf, test_df, feature_cols,
                                      name="TEST (all policies)")

    _evaluate_split_per_policy(clf, test_out, feature_cols, name="TEST")
    _evaluate_split_by_regime(clf, test_out, feature_cols, name="TEST")

    results = {
        "model": clf,
        "feature_cols": feature_cols,
        "train_df": train_out,
        "val_df": val_out,
        "test_df": test_out,
    }
    return results



# MAIN (run on mock data)


if __name__ == "__main__":
    print("Generating mock events...")
    mock_events = generate_mock_events(n_events=5000, n_symbols=3, seed=42)

    print("Running P6 standard adversary v0.1 pipeline on mock data...")
    _ = run_standard_adversary_pipeline(mock_events)

    print("\n✓ P6 standard adversary v0.1 (mock) completed.")
