"""
Binary Classifiers for Adaptive Adversary Framework

Three adversarial classifiers:
1. Baseline vs Pink Noise Detector
2. Baseline vs OU Detector  
3. Baseline vs Uniform Detector

Each classifier predicts whether a trade came from Baseline or the specific policy.
High AUC (>0.7) indicates the policy is detectable and needs stronger randomization.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BaselineVsPinkClassifier:
    """
    Binary classifier to detect Pink Noise policy vs Baseline.
    
    Target: Detect 1/f spectral signature in price deviations.
    Key features: spectral_power_low_freq, deviation_autocorr_lag1
    """
    
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X_train: Feature matrix
            y_train: Labels (0=Baseline, 1=Pink)
        
        Returns:
            Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training AUC
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        return {
            'train_auc': train_auc,
            'n_samples': len(X_train)
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, verbose: bool = True) -> float:
        """
        Evaluate on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print detailed results
        
        Returns:
            Test AUC score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        pred_proba = self.model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, pred_proba)
        
        if verbose:
            predictions = self.model.predict(X_test)
            print(f"\nBaseline vs Pink Noise - Test AUC: {test_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions, 
                                       target_names=['Baseline', 'Pink Noise']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predictions))
            
            # Feature importance
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Important Features:")
            print(feature_importance.head())
        
        return test_auc
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]


class BaselineVsOUClassifier:
    """
    Binary classifier to detect OU policy vs Baseline.
    
    Target: Detect mean-reverting noise patterns.
    Key features: mean_reversion_signal, zero_crossing_frequency
    """
    
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """Train the classifier"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        return {
            'train_auc': train_auc,
            'n_samples': len(X_train)
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, verbose: bool = True) -> float:
        """Evaluate on test set"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        pred_proba = self.model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, pred_proba)
        
        if verbose:
            predictions = self.model.predict(X_test)
            print(f"\nBaseline vs OU - Test AUC: {test_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions,
                                       target_names=['Baseline', 'OU']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predictions))
            
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Important Features:")
            print(feature_importance.head())
        
        return test_auc
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]


class BaselineVsUniformClassifier:
    """
    Binary classifier to detect Uniform policy vs Baseline.
    
    Target: Detect time jitter AND uniform price noise.
    Key features: has_intraday_timestamp, deviation_uniformity_score
    """
    
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """Train the classifier"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        return {
            'train_auc': train_auc,
            'n_samples': len(X_train)
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, verbose: bool = True) -> float:
        """Evaluate on test set"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        pred_proba = self.model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, pred_proba)
        
        if verbose:
            predictions = self.model.predict(X_test)
            print(f"\nBaseline vs Uniform - Test AUC: {test_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions,
                                       target_names=['Baseline', 'Uniform']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predictions))
            
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Important Features:")
            print(feature_importance.head())
        
        return test_auc
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]


def train_and_evaluate_classifier(
    classifier,
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Train and evaluate a binary classifier with train/test split.
    
    Args:
        classifier: Classifier instance
        X: Features
        y: Labels
        test_size: Fraction of data for testing
        random_state: Random seed
        verbose: Print results
    
    Returns:
        Tuple of (train_auc, test_auc)
    """
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
    
    # Train
    train_metrics = classifier.train(X_train, y_train)
    train_auc = train_metrics['train_auc']
    
    if verbose:
        print(f"Training AUC: {train_auc:.4f}")
    
    # Evaluate
    test_auc = classifier.evaluate(X_test, y_test, verbose=verbose)
    
    return train_auc, test_auc


if __name__ == "__main__":
    """Test classifiers"""
    
    print("="*80)
    print("BINARY CLASSIFIERS TEST")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features for Pink vs Baseline
    X_pink = pd.DataFrame({
        'price_deviation_pct': np.random.randn(n_samples) * 2,
        'abs_deviation': np.abs(np.random.randn(n_samples) * 2),
        'deviation_autocorr_lag1': np.random.rand(n_samples) * 0.5 + 0.3,  # High autocorr
        'deviation_autocorr_lag2': np.random.rand(n_samples) * 0.3 + 0.2,
        'rolling_std_deviation': np.random.rand(n_samples) * 1.5,
        'spectral_power_low_freq': np.random.rand(n_samples) * 0.4 + 0.6  # High low-freq power
    })
    
    y_pink = np.random.randint(0, 2, n_samples)
    
    print("\n1. Testing Baseline vs Pink Classifier:")
    print("-" * 80)
    pink_classifier = BaselineVsPinkClassifier(random_state=42)
    train_auc, test_auc = train_and_evaluate_classifier(
        pink_classifier, X_pink, y_pink, verbose=True
    )
    
    # Features for OU vs Baseline
    X_ou = pd.DataFrame({
        'price_deviation_pct': np.random.randn(n_samples) * 2,
        'abs_deviation': np.abs(np.random.randn(n_samples) * 2),
        'mean_reversion_signal': np.random.choice([-1, 0, 1], n_samples),
        'deviation_change': np.random.randn(n_samples) * 1.5,
        'rolling_std_deviation': np.random.rand(n_samples) * 1.5,
        'zero_crossing_frequency': np.random.rand(n_samples) * 0.3 + 0.2
    })
    
    y_ou = np.random.randint(0, 2, n_samples)
    
    print("\n2. Testing Baseline vs OU Classifier:")
    print("-" * 80)
    ou_classifier = BaselineVsOUClassifier(random_state=42)
    train_auc, test_auc = train_and_evaluate_classifier(
        ou_classifier, X_ou, y_ou, verbose=True
    )
    
    # Features for Uniform vs Baseline
    X_uniform = pd.DataFrame({
        'price_deviation_pct': np.random.randn(n_samples) * 2,
        'abs_deviation': np.abs(np.random.randn(n_samples) * 2),
        'has_intraday_timestamp': np.random.randint(0, 2, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'minute_of_day': np.random.randint(0, 60, n_samples),
        'time_since_last_trade_hours': np.random.rand(n_samples) * 48,
        'deviation_uniformity_score': np.random.rand(n_samples)
    })
    
    y_uniform = np.random.randint(0, 2, n_samples)
    
    print("\n3. Testing Baseline vs Uniform Classifier:")
    print("-" * 80)
    uniform_classifier = BaselineVsUniformClassifier(random_state=42)
    train_auc, test_auc = train_and_evaluate_classifier(
        uniform_classifier, X_uniform, y_uniform, verbose=True
    )
    
    print("\n" + "="*80)
    print("✓ All classifiers tested successfully!")
    print("="*80)
