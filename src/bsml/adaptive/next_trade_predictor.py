"""
P7 Next-Trade Predictor

Binary classifier: Will there be a trade tomorrow? (Yes/No)

Compare predictability between policies:
- High AUC = trades are predictable
- Low AUC = trades are unpredictable (better randomization)

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class NextTradePredictor:
    """
    Predicts whether a trade will occur tomorrow based on recent history.
    
    Goal: Measure how predictable a policy's trading pattern is.
    - High AUC (>0.70) = Very predictable (bad - adversary can anticipate)
    - Low AUC (~0.50-0.60) = Unpredictable (good - adversary can't anticipate)
    """
    
    def __init__(
        self,
        use_cv: bool = True,
        n_cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            use_cv: Use cross-validation for model validation
            n_cv_folds: Number of CV folds
            random_state: Random seed
        """
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        
        self.model = None
        self.feature_cols = None
        self.feature_importance_ = None
        self.cv_scores = []
    
    def _select_features(self, df: pd.DataFrame) -> list:
        """Select numeric feature columns, excluding metadata"""
        exclude = {
            'date', 'symbol', 'label', 'current_price'
        }
        
        # Use pandas select_dtypes which handles nullable types correctly
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns
        feature_cols = [c for c in numeric_cols if c not in exclude]
        
        return feature_cols
    
    def train(self, train_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train predictor to forecast next-day trades.
        
        Args:
            train_df: Training data with 'label' column (1=trade tomorrow, 0=no trade)
            verbose: Print training diagnostics
        
        Returns:
            Dict with training metrics
        """
        
        if verbose:
            print(f"  [Predictor] Training on {len(train_df)} samples...")
        
        self.feature_cols = self._select_features(train_df)
        
        if verbose:
            print(f"  [Predictor] Using {len(self.feature_cols)} features")
        
        X = train_df[self.feature_cols].fillna(0).values
        y = train_df['label'].values
        
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        
        if verbose:
            print(f"  [Predictor] Positive (trade tomorrow): {n_positive} ({n_positive/len(y)*100:.1f}%)")
            print(f"  [Predictor] Negative (no trade): {n_negative} ({n_negative/len(y)*100:.1f}%)")
        
        if n_positive < 10 or n_negative < 10:
            return {
                'success': False,
                'reason': 'too_few_samples_per_class',
                'n_samples': len(X),
                'n_features': len(self.feature_cols),
                'label_distribution': {'positive': int(n_positive), 'negative': int(n_negative)}
            }
        
        # Train Gradient Boosting classifier
        if verbose:
            print(f"  [Predictor] Training Gradient Boosting...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            max_features='sqrt',
            random_state=self.random_state,
            verbose=0
        )
        
        self.model.fit(X, y)
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if verbose:
            print(f"  [Predictor] Top 5 predictive features:")
            for idx, row in self.feature_importance_.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Cross-validation
        cv_results = {}
        if self.use_cv and len(X) >= 50:
            if verbose:
                print(f"  [Predictor] Running {self.n_cv_folds}-fold CV...")
            
            try:
                cv_scores = cross_val_score(
                    self.model, X, y,
                    cv=min(self.n_cv_folds, len(X) // 20),
                    scoring='roc_auc',
                    n_jobs=-1
                )
                self.cv_scores = cv_scores
                cv_results = {
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                }
                
                if verbose:
                    print(f"  [Predictor] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] CV failed: {e}")
        
        # Training set accuracy (for diagnostics)
        y_pred_train = self.model.predict(X)
        train_accuracy = accuracy_score(y, y_pred_train)
        
        if verbose:
            print(f"  [Predictor] Training accuracy: {train_accuracy:.4f}")
            print(f"  [Predictor] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'label_distribution': {'positive': int(n_positive), 'negative': int(n_negative)},
            'train_accuracy': float(train_accuracy),
            **cv_results
        }
    
    def evaluate(self, val_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Evaluate trade predictability.
        
        Lower AUC = less predictable (better randomization)
        Higher AUC = more predictable (worse - adversary can anticipate)
        
        Args:
            val_df: Validation data
            verbose: Print evaluation diagnostics
        
        Returns:
            Dict with evaluation metrics including AUC
        """
        
        if self.model is None:
            if verbose:
                print("  [ERROR] Model not trained!")
            return {
                'auc': 0.50,
                'success': False,
                'n_samples': 0
            }
        
        if verbose:
            print(f"  [Predictor] Evaluating predictability...")
        
        X = val_df[self.feature_cols].fillna(0).values
        y = val_df['label'].values
        
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        
        if verbose:
            print(f"  [Predictor] Val set → Positive: {n_positive}, Negative: {n_negative}")
        
        if n_positive == 0 or n_negative == 0:
            if verbose:
                print(f"  [WARNING] Single class in validation!")
            return {
                'auc': 0.50,
                'success': False,
                'n_samples': len(y)
            }
        
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        
        # Metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        if verbose:
            print(f"  [Predictor] Val AUC = {auc_score:.4f}")
            print(f"  [Predictor] Val Accuracy = {accuracy:.4f}")
            print(f"  [Predictor] Val F1 = {f1:.4f}")
            
            # Interpretation
            if auc_score > 0.70:
                print(f"  [Predictor] ⚠️  Highly PREDICTABLE - adversary can anticipate trades")
            elif auc_score < 0.60:
                print(f"  [Predictor] ✓ UNPREDICTABLE - good randomization!")
            else:
                print(f"  [Predictor] → Moderately predictable")
        
        return {
            'auc': float(auc_score),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'success': True,
            'n_samples': len(y)
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features for prediction.
        """
        if self.feature_importance_ is None:
            return pd.DataFrame()
        
        return self.feature_importance_.head(top_n)
