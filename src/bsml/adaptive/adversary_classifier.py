"""
P7 Adaptive Adversary Classifier V2 - Policy Distinguishability

Prediction task: "Which policy generated this trade - Baseline or Uniform?"
Label: 0 = Baseline, 1 = Uniform

Improvements over V1:
- Focused on distinguishability (not next-day prediction)
- Single strong model (Gradient Boosting)
- Proper feature importance tracking
- Inverted logic: Lower AUC = better randomization

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


class P7AdversaryV2:
    """
    Adversary classifier for policy distinguishability.
    
    Goal: Train a classifier to distinguish baseline vs uniform trades.
    - High AUC (>0.65) = Policies are distinguishable (bad - need more randomization)
    - Low AUC (~0.50-0.55) = Policies are indistinguishable (good - randomization works)
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
            'date', 'symbol', 'label', 'policy', 'policy_name',
            'price_trade', 'price_market', 'ref_price', 'side', 'qty',
            'date_only', 'side_numeric', 'direction_change'
        }
        
        feature_cols = [
            c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]
        
        return feature_cols
    
    def train(self, train_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train adversary to distinguish baseline vs uniform.
        
        Args:
            train_df: Training data with 'label' column (0=baseline, 1=uniform)
            verbose: Print training diagnostics
        
        Returns:
            Dict with training metrics
        """
        
        if verbose:
            print(f"  [Adversary V2] Training on {len(train_df)} samples...")
        
        self.feature_cols = self._select_features(train_df)
        
        if verbose:
            print(f"  [Adversary V2] Using {len(self.feature_cols)} features")
        
        X = train_df[self.feature_cols].fillna(0).values
        y = train_df['label'].values
        
        n_baseline = (y == 0).sum()
        n_uniform = (y == 1).sum()
        
        if verbose:
            print(f"  [Adversary V2] Baseline: {n_baseline}, Uniform: {n_uniform}")
        
        if n_baseline < 10 or n_uniform < 10:
            return {
                'success': False,
                'reason': 'too_few_samples_per_class',
                'n_samples': len(X),
                'n_features': len(self.feature_cols),
                'label_distribution': {'baseline': int(n_baseline), 'uniform': int(n_uniform)}
            }
        
        # Train Gradient Boosting classifier
        if verbose:
            print(f"  [Adversary V2] Training Gradient Boosting...")
        
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
            print(f"  [Adversary V2] Top 5 distinguishing features:")
            for idx, row in self.feature_importance_.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Cross-validation
        cv_results = {}
        if self.use_cv and len(X) >= 50:
            if verbose:
                print(f"  [Adversary V2] Running {self.n_cv_folds}-fold CV...")
            
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
                    print(f"  [Adversary V2] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] CV failed: {e}")
        
        # Training set accuracy (for diagnostics)
        y_pred_train = self.model.predict(X)
        train_accuracy = accuracy_score(y, y_pred_train)
        
        if verbose:
            print(f"  [Adversary V2] Training accuracy: {train_accuracy:.4f}")
            print(f"  [Adversary V2] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'label_distribution': {'baseline': int(n_baseline), 'uniform': int(n_uniform)},
            'train_accuracy': float(train_accuracy),
            **cv_results
        }
    
    def evaluate(self, val_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Evaluate policy distinguishability.
        
        Lower AUC = better randomization (policies are harder to distinguish)
        Higher AUC = worse randomization (policies are easy to distinguish)
        
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
                'n_samples': 0,
                'label_distribution': {'baseline': 0, 'uniform': 0}
            }
        
        if verbose:
            print(f"  [Adversary V2] Evaluating distinguishability...")
        
        X = val_df[self.feature_cols].fillna(0).values
        y = val_df['label'].values
        
        n_baseline = (y == 0).sum()
        n_uniform = (y == 1).sum()
        
        if verbose:
            print(f"  [Adversary V2] Val set → Baseline: {n_baseline}, Uniform: {n_uniform}")
        
        if n_baseline == 0 or n_uniform == 0:
            if verbose:
                print(f"  [WARNING] Single class in validation!")
            return {
                'auc': 0.50,
                'success': False,
                'n_samples': len(y),
                'label_distribution': {'baseline': int(n_baseline), 'uniform': int(n_uniform)}
            }
        
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        
        # Metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        if verbose:
            print(f"  [Adversary V2] Val AUC = {auc_score:.4f}")
            print(f"  [Adversary V2] Val Accuracy = {accuracy:.4f}")
            print(f"  [Adversary V2] Val F1 = {f1:.4f}")
            
            # Interpretation
            if auc_score > 0.65:
                print(f"  [Adversary V2] ⚠️  Policies are DISTINGUISHABLE - need more randomization")
            elif auc_score < 0.55:
                print(f"  [Adversary V2] ✓ Policies are INDISTINGUISHABLE - randomization working!")
            else:
                print(f"  [Adversary V2] → Borderline - acceptable randomization level")
        
        return {
            'auc': float(auc_score),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'success': True,
            'n_samples': len(y),
            'label_distribution': {'baseline': int(n_baseline), 'uniform': int(n_uniform)}
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features for distinguishing policies.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            return pd.DataFrame()
        
        return self.feature_importance_.head(top_n)
