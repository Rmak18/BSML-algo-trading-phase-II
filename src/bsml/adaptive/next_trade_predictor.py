"""
P7 Next-Trade Predictor

Binary classifier to predict if there will be a trade tomorrow.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score


class NextTradePredictor:
    """
    Predict if there will be a trade tomorrow based on recent trading patterns.
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.is_trained = False
    
    def train(self, train_data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train the next-trade predictor.
        
        Args:
            train_data: DataFrame with features and 'label' column
            verbose: Print training progress
            
        Returns:
            Dict with training metrics
        """
        
        # Separate features and labels
        feature_cols = [c for c in train_data.columns if c not in ['label', 'symbol', 'date']]
        X_train = train_data[feature_cols].copy()
        y_train = train_data['label'].copy()
        
        if verbose:
            print(f"  [Predictor] Training on {len(X_train)} samples...")
            print(f"  [Predictor] Using {len(feature_cols)} features")
            n_positive = (y_train == 1).sum()
            n_negative = (y_train == 0).sum()
            print(f"  [Predictor] Positive (trade tomorrow): {n_positive} ({n_positive/len(y_train)*100:.1f}%)")
            print(f"  [Predictor] Negative (no trade): {n_negative} ({n_negative/len(y_train)*100:.1f}%)")
        
        # Check if we have both classes
        if len(y_train.unique()) < 2:
            if verbose:
                print(f"  [ERROR] Only one class present in training data!")
                print(f"  [ERROR] Model not trained!")
            return {
                'success': False,
                'reason': 'single_class',
                'n_samples': len(X_train),
                'n_features': len(feature_cols)
            }
        
        # Fill NaNs
        X_train = X_train.fillna(0)
        
        # Train model
        if verbose:
            print(f"  [Predictor] Training Gradient Boosting...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols
        self.is_trained = True
        
        # Feature importance
        if verbose:
            importances = pd.Series(
                self.model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)
            
            print(f"  [Predictor] Top 5 predictive features:")
            for feat, imp in importances.head(5).items():
                print(f"    {feat}: {imp:.4f}")
        
        # Cross-validation
        if verbose:
            print(f"  [Predictor] Running 5-fold CV...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        
        if verbose:
            print(f"  [Predictor] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Training accuracy
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        
        if verbose:
            print(f"  [Predictor] Training accuracy: {train_acc:.4f}")
            print(f"  [Predictor] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X_train),
            'n_features': len(feature_cols),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'train_accuracy': float(train_acc),
            'feature_importance': importances.to_dict()
        }
    
    def evaluate(self, val_data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Evaluate predictor on validation data.
        
        Args:
            val_data: DataFrame with features and 'label' column
            verbose: Print evaluation results
            
        Returns:
            Dict with evaluation metrics
        """
        
        if not self.is_trained:
            if verbose:
                print(f"  [ERROR] Model not trained!")
            return {
                'success': False,
                'auc': 0.50,
                'reason': 'not_trained'
            }
        
        # Prepare data
        X_val = val_data[self.feature_cols].copy()
        y_val = val_data['label'].copy()
        
        X_val = X_val.fillna(0)
        
        if verbose:
            print(f"  [Predictor] Evaluating predictability...")
            n_positive = (y_val == 1).sum()
            n_negative = (y_val == 0).sum()
            print(f"  [Predictor] Val set → Positive: {n_positive}, Negative: {n_negative}")
        
        # Check if we have both classes
        if len(y_val.unique()) < 2:
            if verbose:
                print(f"  [WARNING] Single class in validation!")
            return {
                'success': True,
                'auc': 0.50,  # Random guessing
                'accuracy': float((y_val == y_val.mode()[0]).mean()),
                'reason': 'single_class_validation'
            }
        
        # Predict
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        
        # Metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if verbose:
            print(f"  [Predictor] Val AUC = {auc:.4f}")
            print(f"  [Predictor] Val Accuracy = {acc:.4f}")
            print(f"  [Predictor] Val F1 = {f1:.4f}")
            
            # Interpretation
            if auc > 0.75:
                print(f"  [Predictor] ⚠️  Highly PREDICTABLE - adversary can anticipate trades")
            elif auc > 0.60:
                print(f"  [Predictor] → Moderately predictable")
            else:
                print(f"  [Predictor] ✓ UNPREDICTABLE - good randomization!")
        
        return {
            'success': True,
            'auc': float(auc),
            'accuracy': float(acc),
            'f1': float(f1),
            'n_samples': len(X_val)
        }
