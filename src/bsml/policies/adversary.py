"""
Adversary Classifier Module
Trains machine learning classifier to predict trades and measures predictability
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class AdversaryClassifier:
    """
    Adversary that tries to predict trading decisions
    Uses XGBoost-style gradient boosting
    """
    
    def __init__(self, config):
        self.config = config
        self.classifier = None
        
    def extract_features(self, prices_df, signals_df):
        """
        Extract predictive features for adversary
        
        Features:
        - 5/10/20-day price momentum
        - 10/30/60-day realized volatility
        - Volume indicators (if available)
        - Time-of-day patterns
        - Recent signal changes
        """
        features_list = []
        
        for etf in self.config['universe']:
            # Price momentum features
            mom_5d = prices_df[etf].pct_change(5)
            mom_10d = prices_df[etf].pct_change(10)
            mom_20d = prices_df[etf].pct_change(20)
            
            # Volatility features
            returns = prices_df[etf].pct_change()
            vol_10d = returns.rolling(10).std() * np.sqrt(252)
            vol_30d = returns.rolling(30).std() * np.sqrt(252)
            vol_60d = returns.rolling(60).std() * np.sqrt(252)
            
            # Signal change indicators
            signal_change = signals_df[etf].diff()
            
            # Combine features for this ETF
            etf_features = pd.DataFrame({
                f'{etf}_mom5': mom_5d,
                f'{etf}_mom10': mom_10d,
                f'{etf}_mom20': mom_20d,
                f'{etf}_vol10': vol_10d,
                f'{etf}_vol30': vol_30d,
                f'{etf}_vol60': vol_60d,
                f'{etf}_signal_change': signal_change,
            })
            
            features_list.append(etf_features)
        
        # Combine all features
        features_df = pd.concat(features_list, axis=1)
        
        # Add time features
        features_df['day_of_week'] = pd.to_datetime(prices_df['date']).dt.dayofweek
        features_df['month'] = pd.to_datetime(prices_df['date']).dt.month
        
        return features_df
    
    def create_labels(self, signals_df):
        """
        Create binary labels for trade occurrence
        1 if trade occurs (signal changes), 0 otherwise
        """
        labels = {}
        
        for etf in self.config['universe']:
            # Trade occurs when signal changes
            trade_occurs = (signals_df[etf].diff() != 0).astype(int)
            labels[etf] = trade_occurs
        
        # Use first ETF as example (same logic applies to all)
        # In practice, could train separate classifiers per ETF
        return labels[self.config['universe'][0]]
    
    def train_and_evaluate(self, baseline_results):
        """
        Train adversary classifier on baseline strategy
        Returns AUC-ROC score
        """
        # Extract features and labels
        features_df = self.extract_features(
            pd.read_csv('prices.csv', parse_dates=['date']),
            baseline_results['signals']
        )
        labels = self.create_labels(baseline_results['signals'])
        
        # Remove NaN rows
        valid_idx = ~(features_df.isna().any(axis=1) | labels.isna())
        X = features_df[valid_idx].values
        y = labels[valid_idx].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train gradient boosting classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return auc_score
    
    def evaluate(self, policy_results):
        """
        Evaluate adversary performance on randomized policy
        
        Lower AUC = less predictable = better randomization
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained first")
        
        # Extract features
        features_df = self.extract_features(
            pd.read_csv('prices.csv', parse_dates=['date']),
            policy_results.get('signals', policy_results['signals'])
        )
        labels = self.create_labels(policy_results.get('signals', policy_results['signals']))
        
        # Remove NaN
        valid_idx = ~(features_df.isna().any(axis=1) | labels.isna())
        X = features_df[valid_idx].values
        y = labels[valid_idx].values
        
        # Predict
        y_pred_proba = self.classifier.predict_proba(X)[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y, y_pred_proba)
        
        return auc_score


class AdversaryMetrics:
    """
    Calculate additional adversary performance metrics
    """
    
    @staticmethod
    def calculate_precision_recall_auc(y_true, y_pred_proba):
        """Calculate Precision-Recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    
    @staticmethod
    def calculate_pnr(y_true, y_pred_proba, threshold=0.5):
        """
        Calculate Positive Net Reclassification (PNR)
        Measures net benefit of classifier predictions
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # True positives and false positives
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        
        # True negatives and false negatives  
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        # PNR = (TP - FP) / Total
        pnr = (tp - fp) / len(y_true)
        
        return pnr


if __name__ == '__main__':
    # Test adversary classifier
    import json
    from data_generator import generate_etf_prices
    from baseline_strategy import BaselineStrategy
    
    with open('../config.json', 'r') as f:
        config = json.load(f)[0]
    
    # Load or generate data
    import os
    if os.path.exists('../prices.csv'):
        prices_df = pd.read_csv('../prices.csv', parse_dates=['date'])
    else:
        prices_df = generate_etf_prices(config)
        prices_df.to_csv('../prices.csv', index=False)
    
    # Run baseline
    baseline = BaselineStrategy(config)
    baseline_results = baseline.run(prices_df)
    
    # Train and evaluate adversary
    adversary = AdversaryClassifier(config)
    auc_score = adversary.train_and_evaluate(baseline_results)
    
    print(f"Baseline Adversary AUC: {auc_score:.3f}")
    print(f"(Lower is better for defender, 0.50 = random guessing)")
