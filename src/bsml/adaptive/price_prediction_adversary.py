"""
Price Prediction Adversary - Regression-Based Exploitation Framework

Instead of just detecting patterns, these adversaries predict the EXACT PRICE
at which a randomized policy will trade, given the baseline price.

This is more realistic: real adversaries want to PROFIT, not just detect.

Key Metric: MAE% (Mean Absolute Error as % of price)
- If MAE% < 0.5% → Adversary can predict and exploit profitably
- If MAE% > 1.0% → Randomization is strong enough

Three Adversaries:
1. Baseline → Pink Noise Price Predictor
2. Baseline → OU Price Predictor  
3. Baseline → Uniform Price Predictor

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def extract_price_prediction_features(baseline_df: pd.DataFrame, policy_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features for predicting policy price from baseline price.
    
    The adversary knows:
    - Baseline price (the deterministic execution price)
    - Symbol (which asset)
    - Side (BUY or SELL)
    - Recent price history (momentum, volatility)
    - Time features (day of week, month)
    
    Goal: Predict policy's ref_price
    
    Args:
        baseline_df: Baseline trades
        policy_df: Policy trades (with randomized ref_price)
    
    Returns:
        Tuple of (features DataFrame, target array)
    """
    
    # Ensure alignment
    assert len(baseline_df) == len(policy_df), "Trade counts must match"
    assert (baseline_df['date'] == policy_df['date']).all(), "Dates must align"
    assert (baseline_df['symbol'] == policy_df['symbol']).all(), "Symbols must align"
    
    features = pd.DataFrame()
    
    # Core feature: baseline price (what adversary observes)
    features['baseline_price'] = baseline_df['price'].values
    
    # Categorical: symbol (one-hot encoded later)
    features['symbol'] = baseline_df['symbol'].values
    
    # Binary: side (BUY=1, SELL=0)
    features['side_binary'] = (baseline_df['side'] == 'BUY').astype(int).values
    
    # Time features
    dates = pd.to_datetime(baseline_df['date'])
    features['day_of_week'] = dates.dt.dayofweek.values
    features['month'] = dates.dt.month.values
    features['day_of_month'] = dates.dt.day.values
    
    # Price level features (normalized)
    features['price_level'] = baseline_df['price'].values / 100.0  # Normalize around 1
    features['log_price'] = np.log(baseline_df['price'].values + 1e-6)
    
    # Per-symbol features: momentum, volatility, relative position
    momentum_vals = []
    volatility_vals = []
    price_zscore_vals = []
    
    for symbol in baseline_df['symbol'].unique():
        symbol_mask = baseline_df['symbol'] == symbol
        symbol_prices = baseline_df.loc[symbol_mask, 'price'].values
        
        # Momentum: recent price change
        if len(symbol_prices) > 1:
            momentum = np.diff(symbol_prices, prepend=symbol_prices[0])
        else:
            momentum = np.zeros(len(symbol_prices))
        
        # Volatility: rolling std (window=20)
        if len(symbol_prices) >= 20:
            rolling_std = pd.Series(symbol_prices).rolling(window=20, min_periods=5).std().fillna(0).values
        else:
            rolling_std = pd.Series(symbol_prices).rolling(window=max(5, len(symbol_prices)), min_periods=1).std().fillna(0).values
        
        # Z-score: distance from mean
        symbol_mean = symbol_prices.mean()
        symbol_std = symbol_prices.std()
        if symbol_std > 0:
            z_score = (symbol_prices - symbol_mean) / symbol_std
        else:
            z_score = np.zeros(len(symbol_prices))
        
        momentum_vals.extend(momentum.tolist())
        volatility_vals.extend(rolling_std.tolist())
        price_zscore_vals.extend(z_score.tolist())
    
    features['momentum'] = momentum_vals
    features['volatility'] = volatility_vals
    features['price_zscore'] = price_zscore_vals
    
    # Interaction features (adversary looks for patterns)
    features['price_x_volatility'] = features['baseline_price'] * features['volatility']
    features['momentum_x_side'] = features['momentum'] * features['side_binary']
    
    # One-hot encode symbol
    features = pd.get_dummies(features, columns=['symbol'], prefix='symbol')
    
    # Target: policy's ref_price (what we want to predict)
    target = policy_df['ref_price'].values
    
    return features, target


class PricePredictionAdversary:
    """
    Base class for price prediction adversaries.
    
    Uses Random Forest Regressor with strong hyperparameters to ensure
    the adversary is as powerful as possible.
    """
    
    def __init__(self, policy_name: str, random_state: int = 42):
        """
        Initialize adversary.
        
        Args:
            policy_name: Name of policy being attacked
            random_state: Random seed
        """
        self.policy_name = policy_name
        self.random_state = random_state
        
        # Use strong model to maximize adversary power
        self.model = RandomForestRegressor(
            n_estimators=200,           # More trees for better predictions
            max_depth=20,               # Deep trees to capture complex patterns
            min_samples_split=5,        # Allow splits on small groups
            min_samples_leaf=2,         # Allow small leaves
            max_features='sqrt',        # Feature subsampling
            random_state=random_state,
            n_jobs=-1                   # Parallel processing
        )
        
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the adversary.
        
        Args:
            X_train: Training features
            y_train: Target prices
        
        Returns:
            Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'n_samples': len(X_train)
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, 
                 baseline_prices: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate adversary's prediction accuracy.
        
        Args:
            X_test: Test features
            y_test: True policy prices
            baseline_prices: Baseline prices (for percentage error calculation)
            verbose: Print detailed results
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Adversary must be trained before evaluation")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Absolute errors
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Percentage errors (KEY METRIC)
        absolute_errors = np.abs(y_test - y_pred)
        pct_errors = (absolute_errors / baseline_prices) * 100
        mae_pct = pct_errors.mean()
        median_pct_error = np.median(pct_errors)
        max_pct_error = pct_errors.max()
        
        # Exploitability check
        exploitable_threshold = 0.5  # 0.5% is typical transaction cost
        exploitable_fraction = (pct_errors < exploitable_threshold).mean()
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_pct': mae_pct,
            'median_pct_error': median_pct_error,
            'max_pct_error': max_pct_error,
            'exploitable_fraction': exploitable_fraction
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ADVERSARY: Predicting {self.policy_name} Prices")
            print(f"{'='*80}")
            print(f"\nAbsolute Errors:")
            print(f"  MAE:  ${mae:.4f}")
            print(f"  RMSE: ${rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            
            print(f"\nPercentage Errors (KEY METRIC):")
            print(f"  Mean:   {mae_pct:.4f}%")
            print(f"  Median: {median_pct_error:.4f}%")
            print(f"  Max:    {max_pct_error:.4f}%")
            
            print(f"\nExploitability Analysis:")
            print(f"  Trades predictable within 0.5%: {exploitable_fraction*100:.1f}%")
            
            if mae_pct < 0.5:
                print(f"\n  ⚠️  HIGHLY EXPLOITABLE - MAE < 0.5%")
                print(f"  → Adversary can profit after transaction costs")
            elif mae_pct < 1.0:
                print(f"\n  ⚠️  MODERATELY EXPLOITABLE - MAE < 1.0%")
                print(f"  → Adversary might profit in low-cost environments")
            else:
                print(f"\n  ✓ SAFE - MAE > 1.0%")
                print(f"  → Randomization is strong enough")
            
            # Feature importance
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Predictive Features:")
            print(feature_importance.head().to_string(index=False))
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict policy prices"""
        if not self.is_trained:
            raise ValueError("Adversary must be trained before prediction")
        return self.model.predict(X)


class BaselineToPinkPredictor(PricePredictionAdversary):
    """Adversary that predicts Pink Noise prices from baseline"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(policy_name="Pink Noise", random_state=random_state)


class BaselineToOUPredictor(PricePredictionAdversary):
    """Adversary that predicts OU prices from baseline"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(policy_name="OU", random_state=random_state)


class BaselineToUniformPredictor(PricePredictionAdversary):
    """Adversary that predicts Uniform prices from baseline"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(policy_name="Uniform", random_state=random_state)


def train_and_evaluate_price_predictor(
    adversary: PricePredictionAdversary,
    baseline_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train and evaluate a price prediction adversary.
    
    Args:
        adversary: Adversary instance
        baseline_df: Baseline trades
        policy_df: Policy trades
        test_size: Fraction for testing
        random_state: Random seed
        verbose: Print results
    
    Returns:
        Tuple of (train_metrics, test_metrics)
    """
    
    # Extract features
    if verbose:
        print(f"\nExtracting features for {adversary.policy_name}...")
    
    X, y = extract_price_prediction_features(baseline_df, policy_df)
    
    if verbose:
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples:  {X.shape[0]}")
    
    # Train/test split (temporal split would be better, but random for now)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Extract baseline prices for test set (for percentage calculation)
    baseline_prices_test = X_test['baseline_price'].values
    
    if verbose:
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    if verbose:
        print(f"\nTraining adversary...")
    
    train_metrics = adversary.train(X_train, y_train)
    
    if verbose:
        print(f"  Train MAE: ${train_metrics['train_mae']:.4f}")
        print(f"  Train R²:  {train_metrics['train_r2']:.4f}")
    
    # Evaluate
    test_metrics = adversary.evaluate(X_test, y_test, baseline_prices_test, verbose=verbose)
    
    return train_metrics, test_metrics


if __name__ == "__main__":
    """Test price prediction adversary"""
    
    print("="*80)
    print("PRICE PREDICTION ADVERSARY TEST")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    baseline_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n),
        'side': np.random.choice(['BUY', 'SELL'], n),
        'price': 100 + np.random.randn(n).cumsum() * 0.5,
        'ref_price': 100 + np.random.randn(n).cumsum() * 0.5
    })
    baseline_df['ref_price'] = baseline_df['price']  # Baseline: no randomization
    
    # Pink noise: add 1/f correlated noise
    pink_df = baseline_df.copy()
    pink_noise = np.random.randn(n).cumsum() * 0.3  # Brownian motion
    pink_df['ref_price'] = baseline_df['price'] + pink_noise
    
    # Test Pink adversary
    print("\n" + "="*80)
    print("Testing Pink Noise Price Predictor")
    print("="*80)
    
    pink_adversary = BaselineToPinkPredictor(random_state=42)
    train_metrics, test_metrics = train_and_evaluate_price_predictor(
        pink_adversary,
        baseline_df,
        pink_df,
        test_size=0.3,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("✓ Price prediction adversary tested successfully!")
    print("="*80)
    
    print(f"\nTest Results:")
    print(f"  MAE%: {test_metrics['mae_pct']:.4f}%")
    print(f"  Exploitable: {test_metrics['exploitable_fraction']*100:.1f}% of trades")
