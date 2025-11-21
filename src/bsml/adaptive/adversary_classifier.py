"""
P7 Adaptive Adversary Classifier - Production Grade

Advanced adversary with:
- 10-minute prediction window
- Cross-validation
- SMOTE resampling for imbalanced data
- Enhanced feature engineering
- Proper train/val/test splits

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class P7AdaptiveAdversary:
    """
    Production-grade adversary for 10-minute trade prediction.
    
    Predicts: "Will the next trade happen within 10 minutes?"
    
    Features:
    - SMOTE resampling for imbalanced labels
    - Cross-validation during training
    - Enhanced feature engineering
    - Ensemble of gradient boosting + random forest
    """
    
    def __init__(
        self, 
        window_threshold_minutes: float = 10,
        use_smote: bool = True,
        use_cv: bool = True,
        n_cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            window_threshold_minutes: Prediction window (10 min default)
            use_smote: Apply SMOTE for label balancing
            use_cv: Use cross-validation during training
            n_cv_folds: Number of CV folds
            random_state: Random seed for reproducibility
        """
        self.threshold_minutes = window_threshold_minutes
        self.use_smote = use_smote
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        
        self.model_gb = None  # Gradient boosting
        self.model_rf = None  # Random forest
        self.feature_cols = []
        self.cv_scores = []
        self.feature_importances = {}
        
    def extract_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rich features for intraday timing prediction.
        
        Enhanced features:
        - Multi-scale price momentum
        - Volatility at multiple horizons
        - Trade intensity metrics
        - Time-of-day patterns
        - Volume patterns
        - Price level indicators
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # =====================================================================
        # PRICE FEATURES (Multi-scale momentum)
        # =====================================================================
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_return_{lag}'] = (
                df.groupby('symbol')['mid_price'].pct_change(lag)
            )
        
        # Return acceleration
        df['return_accel'] = (
            df['price_return_1'] - df.groupby('symbol')['price_return_1'].shift(1)
        )
        
        # =====================================================================
        # VOLATILITY FEATURES (Multi-horizon)
        # =====================================================================
        for window in [3, 5, 10, 20]:
            # Price volatility
            df[f'price_vol_{window}'] = (
                df.groupby('symbol')['price_return_1']
                .transform(lambda x: x.rolling(window, min_periods=2).std())
            )
            
            # Volume volatility
            if 'volume' in df.columns:
                df[f'volume_vol_{window}'] = (
                    df.groupby('symbol')['volume']
                    .transform(lambda x: x.rolling(window, min_periods=2).std())
                )
        
        # Volatility of volatility (VoV)
        df['vol_of_vol'] = (
            df.groupby('symbol')['price_vol_5']
            .transform(lambda x: x.rolling(5, min_periods=2).std())
        )
        
        # =====================================================================
        # PRICE LEVEL INDICATORS (Normalized)
        # =====================================================================
        for window in [5, 10, 20, 50]:
            # Moving averages
            df[f'price_ma_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=2).mean())
            )
            
            # Distance from MA (z-score style)
            df[f'price_zscore_{window}'] = (
                (df['mid_price'] - df[f'price_ma_{window}']) / 
                (df[f'price_vol_{window}'] + 1e-8)
            )
        
        # High/Low tracking
        for window in [5, 10, 20]:
            df[f'price_high_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            df[f'price_low_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
            df[f'price_range_{window}'] = df[f'price_high_{window}'] - df[f'price_low_{window}']
        
        # =====================================================================
        # TIME FEATURES (Intraday patterns)
        # =====================================================================
        dt = pd.to_datetime(df['timestamp'])
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['day_of_week'] = dt.dt.dayofweek
        
        # Time-of-day indicators
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
        df['is_close'] = ((df['hour'] >= 15) & (df['hour'] < 16)).astype(int)
        
        # Day of week dummies
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Time since market open (approximate)
        df['minutes_since_open'] = (df['hour'] - 9) * 60 + df['minute']
        
        # =====================================================================
        # TRADE INTENSITY FEATURES
        # =====================================================================
        for window in [3, 5, 10, 20]:
            # Trade frequency
            df[f'trades_last_{window}'] = (
                df.groupby('symbol')['exec_flag']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            
            # Trade velocity (trades per unit time)
            df[f'trade_velocity_{window}'] = df[f'trades_last_{window}'] / window
        
        # Time since last trade (in minutes)
        df['minutes_since_last'] = (
            df.groupby('symbol')['timestamp'].diff().dt.total_seconds() / 60.0
        )
        df['minutes_since_last'] = df['minutes_since_last'].fillna(1440)  # 1 day default
        
        # =====================================================================
        # QUANTITY PATTERNS
        # =====================================================================
        if 'qty' in df.columns:
            df['qty_abs'] = df['qty'].abs()
            
            for window in [3, 5, 10]:
                df[f'qty_mean_{window}'] = (
                    df.groupby('symbol')['qty_abs']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                df[f'qty_std_{window}'] = (
                    df.groupby('symbol')['qty_abs']
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
            
            # Quantity change
            df['qty_change'] = df.groupby('symbol')['qty_abs'].diff()
        
        # =====================================================================
        # VOLUME PATTERNS
        # =====================================================================
        if 'volume' in df.columns:
            df['log_volume'] = np.log1p(df['volume'])
            
            for window in [3, 5, 10]:
                df[f'volume_ma_{window}'] = (
                    df.groupby('symbol')['volume']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                df[f'volume_ratio_{window}'] = (
                    df['volume'] / (df[f'volume_ma_{window}'] + 1)
                )
        
        return df
    
    def create_labels(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Create labels: Will next trade happen within 10 minutes?
        
        This is the CRITICAL prediction for intraday timing.
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # Calculate minutes until next trade
        df['minutes_to_next'] = (
            df.groupby('symbol')['timestamp']
            .shift(-1) - df['timestamp']
        ).dt.total_seconds() / 60.0
        
        # Binary label: 1 if next trade within threshold
        df['label'] = (df['minutes_to_next'] <= self.threshold_minutes).astype(int)
        
        # Last trade per symbol → label as 0
        df['label'] = df['label'].fillna(0).astype(int)
        
        return df['label']
    
    def _select_features(self, features_df: pd.DataFrame) -> list:
        """Select numeric feature columns, excluding metadata"""
        exclude = {
            'timestamp', 'symbol', 'policy_id', 'label', 'side', 
            'date', 'exec_flag', 'pnl', 'ref_price', 'minutes_to_next',
            'qty', 'volume', 'price', 'mid_price'
        }
        
        feature_cols = [
            c for c in features_df.columns
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        return feature_cols
    
    def train(self, trades_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train adversary with advanced techniques.
        
        Returns:
            Training metrics dict
        """
        if verbose:
            print(f"  [Adversary] Training on {len(trades_df)} samples...")
            print(f"  [Adversary] Prediction window: {self.threshold_minutes} minutes")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Select features
        self.feature_cols = self._select_features(features_df)
        
        if verbose:
            print(f"  [Adversary] Extracted {len(self.feature_cols)} features")
        
        # Prepare data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        # Check label distribution
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_rate = n_pos / len(y) * 100
        
        if verbose:
            print(f"  [Adversary] Labels → 0: {n_neg}, 1: {n_pos} ({pos_rate:.1f}% positive)")
        
        # Validation checks
        if n_pos < 5 or n_neg < 5:
            print(f"  [ERROR] Too few samples in one class!")
            return {'success': False, 'reason': 'insufficient_samples'}
        
        # Apply SMOTE if enabled and needed
        if self.use_smote and pos_rate < 40:
            if verbose:
                print(f"  [Adversary] Applying SMOTE resampling (imbalanced: {pos_rate:.1f}%)...")
            
            try:
                # Only resample to balance classes moderately
                target_ratio = 0.5  # Aim for 50% positive after resampling
                smote = SMOTE(
                    sampling_strategy=target_ratio,
                    random_state=self.random_state,
                    k_neighbors=min(5, n_pos - 1)
                )
                X, y = smote.fit_resample(X, y)
                
                if verbose:
                    n_pos_new = y.sum()
                    print(f"  [Adversary] After SMOTE → 0: {len(y) - n_pos_new}, "
                          f"1: {n_pos_new} ({n_pos_new/len(y)*100:.1f}%)")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] SMOTE failed: {e}. Continuing without resampling.")
        
        # =====================================================================
        # Train Gradient Boosting
        # =====================================================================
        if verbose:
            print(f"  [Adversary] Training Gradient Boosting...")
        
        self.model_gb = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.random_state
        )
        self.model_gb.fit(X, y)
        
        # =====================================================================
        # Train Random Forest (for ensemble)
        # =====================================================================
        if verbose:
            print(f"  [Adversary] Training Random Forest...")
        
        self.model_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_rf.fit(X, y)
        
        # =====================================================================
        # Cross-validation (on ORIGINAL unbalanced data)
        # =====================================================================
        cv_results = {}
        if self.use_cv:
            if verbose:
                print(f"  [Adversary] Running {self.n_cv_folds}-fold cross-validation...")
            
            # Re-prepare original data for CV
            X_cv = features_df[self.feature_cols].fillna(0).values
            y_cv = features_df['label'].values
            
            try:
                cv_scores = cross_val_score(
                    self.model_gb, 
                    X_cv, 
                    y_cv, 
                    cv=self.n_cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                self.cv_scores = cv_scores
                cv_results = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                if verbose:
                    print(f"  [Adversary] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] CV failed: {e}")
        
        # Feature importances
        if hasattr(self.model_gb, 'feature_importances_'):
            importances = self.model_gb.feature_importances_
            self.feature_importances = dict(zip(self.feature_cols, importances))
            
            if verbose:
                top_features = sorted(
                    self.feature_importances.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                print(f"  [Adversary] Top 5 features:")
                for feat, imp in top_features:
                    print(f"    - {feat}: {imp:.4f}")
        
        if verbose:
            print(f"  [Adversary] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'label_distribution': {'positive': int(y.sum()), 'negative': int(len(y) - y.sum())},
            **cv_results
        }
    
    def evaluate(
        self, 
        trades_df: pd.DataFrame, 
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate adversary on validation/test data.
        
        Returns:
            Dict with AUC, confusion matrix, and optionally predictions
        """
        if self.model_gb is None or self.model_rf is None:
            if verbose:
                print("  [ERROR] Models not trained!")
            return {'auc': 0.50, 'success': False}
        
        if verbose:
            print(f"  [Adversary] Evaluating on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Prepare data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        # Check label distribution
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        
        if verbose:
            print(f"  [Adversary] Val labels → 0: {n_neg}, 1: {n_pos} ({n_pos/len(y)*100:.1f}%)")
        
        if n_pos == 0 or n_neg == 0:
            if verbose:
                print(f"  [WARNING] All labels same class! Returning AUC=0.50")
            return {'auc': 0.50, 'success': False, 'reason': 'single_class'}
        
        # Ensemble prediction (average of both models)
        y_pred_gb = self.model_gb.predict_proba(X)[:, 1]
        y_pred_rf = self.model_rf.predict_proba(X)[:, 1]
        y_pred_proba = (y_pred_gb + y_pred_rf) / 2.0
        
        # Compute AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        # Confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        if verbose:
            print(f"  [Adversary] AUC = {auc:.4f}")
            print(f"  [Adversary] Confusion Matrix:")
            print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        result = {
            'auc': float(auc),
            'success': True,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y),
            'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
        }
        
        if return_predictions:
            result['predictions'] = y_pred_proba
            result['labels'] = y
        
        return result


def time_split_trades(
    trades_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split trades chronologically, preserving temporal order.
    
    Critical for time-series data to avoid lookahead bias.
    """
    df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
