#!/usr/bin/env python
"""
Continuous ML Model Learner for Forex Spike Predictor

This script runs in the background and periodically updates the ML models
based on prediction performance and new data, enabling continuous learning
without disrupting the main prediction system.
"""

import os
import sys
import time
import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("continuous_learner.log")
    ]
)
logger = logging.getLogger("ContinuousLearner")

class ContinuousLearner:
    """Background process to update ML models based on performance."""
    
    def __init__(self, symbol='EUR_USD', timeframe='5m', update_interval_hours=12):
        """
        Initialize the continuous learner.
        
        Args:
            symbol: Currency pair to monitor
            timeframe: Data timeframe
            update_interval_hours: Hours between model updates
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.update_interval_hours = update_interval_hours
        
        # Base directories
        self.base_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..'
        ))
        
        # Data directories
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw', self.symbol, self.timeframe)
        self.analyzed_data_dir = os.path.join(self.data_dir, 'analyzed', self.symbol, self.timeframe, 'optimal')
        self.prediction_dir = os.path.join(self.data_dir, 'predictions', self.symbol, self.timeframe)
        self.performance_dir = os.path.join(self.data_dir, 'performance', self.symbol, self.timeframe)
        
        # Model directories - both standard model location and specialized
        self.specialized_model_dir = os.path.join(self.base_dir, 'models')
        self.root_model_dir = os.path.join(os.path.dirname(self.base_dir), 'models')
        
        # Ensure model directories exist
        os.makedirs(self.specialized_model_dir, exist_ok=True)
        os.makedirs(self.root_model_dir, exist_ok=True)
        
        # Training parameters
        self.min_training_samples = 500
        self.max_training_samples = 10000
        self.validation_split = 0.2
        
        logger.info(f"Initialized continuous learner for {symbol} {timeframe}")
        logger.info(f"Model directories: {self.specialized_model_dir}, {self.root_model_dir}")
        logger.info(f"Update interval: {update_interval_hours} hours")
    
    def find_latest_file(self, directory, pattern):
        """Find the latest file matching a pattern in a directory."""
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return None
            
        files = [f for f in os.listdir(directory) if pattern in f]
        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return None
            
        # Sort files by name (which includes date)
        files.sort(reverse=True)
        latest_file = os.path.join(directory, files[0])
        return latest_file
    
    def load_historical_data(self):
        """Load historical data for training."""
        try:
            # Load latest analyzed data
            latest_analyzed = self.find_latest_file(self.analyzed_data_dir, '_analyzed.csv')
            if latest_analyzed is None:
                logger.error("No analyzed data found")
                return None
                
            logger.info(f"Loading analyzed data from {latest_analyzed}")
            analyzed_df = pd.read_csv(latest_analyzed)
            
            # Convert timestamp to datetime
            if 'timestamp' in analyzed_df.columns:
                analyzed_df['timestamp'] = pd.to_datetime(analyzed_df['timestamp'])
            
            logger.info(f"Loaded {len(analyzed_df)} records from analyzed data")
            return analyzed_df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return None
    
    def load_prediction_performance(self):
        """Load prediction performance metrics for training feedback."""
        try:
            # Find latest performance file
            latest_perf = self.find_latest_file(self.performance_dir, 'performance.json')
            if latest_perf is None:
                logger.warning("No performance data found")
                return None
                
            # Load performance data
            with open(latest_perf, 'r') as f:
                performance = json.load(f)
                
            logger.info(f"Loaded performance data from {latest_perf}")
            return performance
            
        except Exception as e:
            logger.error(f"Error loading prediction performance: {str(e)}")
            return None
    
    def load_predictions(self):
        """Load recent predictions with outcomes for training feedback."""
        try:
            # Find latest predictions file
            latest_pred = self.find_latest_file(self.prediction_dir, 'predictions.json')
            if latest_pred is None:
                logger.warning("No prediction data found")
                return None
                
            # Load prediction data
            with open(latest_pred, 'r') as f:
                predictions = json.load(f)
                
            logger.info(f"Loaded predictions from {latest_pred}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return None
    
    def prepare_training_data(self, analyzed_df, predictions_data):
        """
        Prepare training data from analyzed data and prediction outcomes.
        
        Args:
            analyzed_df: DataFrame with analyzed data
            predictions_data: Dictionary with predictions and outcomes
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        try:
            if analyzed_df is None or predictions_data is None:
                logger.error("Missing data for training preparation")
                return None, None
                
            # Get predictions with outcomes
            if 'predictions' not in predictions_data:
                logger.warning("No predictions in data")
                return None, None
                
            valid_predictions = [p for p in predictions_data['predictions'] 
                               if 'outcome' in p and p['outcome'] in ['WIN', 'LOSS']]
                               
            if not valid_predictions:
                logger.warning("No predictions with outcomes")
                return None, None
                
            logger.info(f"Found {len(valid_predictions)} predictions with outcomes")
            
            # Prepare training data
            train_data = []
            
            for pred in valid_predictions:
                try:
                    # Get prediction timestamp
                    timestamp = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                    
                    # Find the data point in analyzed_df
                    # Look for the closest timestamp
                    closest_idx = analyzed_df['timestamp'].searchsorted(timestamp) - 1
                    if closest_idx < 0 or closest_idx >= len(analyzed_df):
                        continue
                        
                    # Get feature data for this prediction
                    features = analyzed_df.iloc[closest_idx].to_dict()
                    
                    # Add target label
                    # 1 = WIN (correct prediction), 0 = LOSS (incorrect prediction)
                    label = 1 if pred['outcome'] == 'WIN' else 0
                    
                    # Add to training data
                    features['target'] = label
                    train_data.append(features)
                    
                except Exception as e:
                    logger.warning(f"Error processing prediction: {str(e)}")
                    continue
            
            if not train_data:
                logger.warning("No valid training data after processing")
                return None, None
                
            # Convert to DataFrame
            train_df = pd.DataFrame(train_data)
            
            # Remove non-numeric columns
            train_df = train_df.select_dtypes(include=[np.number])
            
            # Fill NaN values
            train_df = train_df.fillna(0)
            
            # Separate features and target
            X = train_df.drop('target', axis=1)
            y = train_df['target']
            
            logger.info(f"Prepared {len(X)} samples for training")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def train_random_forest(self, X, y):
        """
        Train a Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Trained model
        """
        try:
            if X is None or y is None:
                return None
                
            logger.info("Training Random Forest model...")
            
            # Initialize model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
            
            # Train model
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            features = X.columns
            
            # Create feature importance dict
            feature_importances = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(features, importances)
            ]
            
            # Sort by importance
            feature_importances.sort(key=lambda x: x["importance"], reverse=True)
            
            # Create metadata
            metadata = {
                "model_type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced"
                },
                "feature_importances": feature_importances,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": self._evaluate_model(model, X, y)
            }
            
            logger.info("Random Forest model trained successfully")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return None, None
    
    def train_xgboost(self, X, y):
        """
        Train an XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Trained model
        """
        try:
            if X is None or y is None:
                return None
                
            logger.info("Training XGBoost model...")
            
            # Initialize model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Train model
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            features = X.columns
            
            # Create feature importance dict
            feature_importances = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(features, importances)
            ]
            
            # Sort by importance
            feature_importances.sort(key=lambda x: x["importance"], reverse=True)
            
            # Create metadata
            metadata = {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                },
                "feature_importances": feature_importances,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": self._evaluate_model(model, X, y)
            }
            
            logger.info("XGBoost model trained successfully")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            return None, None
    
    def _evaluate_model(self, model, X, y):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Get probabilities if available
            try:
                y_proba = model.predict_proba(X)[:, 1]
                roc_auc = roc_auc_score(y, y_proba)
            except:
                roc_auc = 0.5
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            cm = confusion_matrix(y, y_pred).tolist()
            
            # Create evaluation dict
            evaluation = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": cm,
                "roc_auc": float(roc_auc)
            }
            
            logger.info(f"Model evaluation: accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "roc_auc": 0.5,
                "error": str(e)
            }
    
    def save_models(self, rf_model, rf_metadata, xgb_model, xgb_metadata):
        """
        Save trained models to disk.
        
        Args:
            rf_model: Random Forest model
            rf_metadata: Random Forest metadata
            xgb_model: XGBoost model
            xgb_metadata: XGBoost metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if rf_model is not None and rf_metadata is not None:
                # Save Random Forest model
                rf_model_path = os.path.join(self.specialized_model_dir, f"random_forest_{timestamp}.joblib")
                joblib.dump(rf_model, rf_model_path)
                
                # Save metadata
                rf_meta_path = os.path.join(self.specialized_model_dir, f"random_forest_{timestamp}_metadata.json")
                with open(rf_meta_path, 'w') as f:
                    json.dump(rf_metadata, f, indent=2)
                
                # Create pickle version for predictor
                rf_pkl_path = os.path.join(self.root_model_dir, f"{self.symbol}_random_forest.pkl")
                with open(rf_pkl_path, 'wb') as f:
                    pickle.dump(rf_model, f)
                
                logger.info(f"Saved Random Forest model to {rf_model_path} and {rf_pkl_path}")
            
            if xgb_model is not None and xgb_metadata is not None:
                # Save XGBoost model
                xgb_model_path = os.path.join(self.specialized_model_dir, f"xgboost_{timestamp}.joblib")
                joblib.dump(xgb_model, xgb_model_path)
                
                # Save metadata
                xgb_meta_path = os.path.join(self.specialized_model_dir, f"xgboost_{timestamp}_metadata.json")
                with open(xgb_meta_path, 'w') as f:
                    json.dump(xgb_metadata, f, indent=2)
                
                # Create pickle version for predictor
                xgb_pkl_path = os.path.join(self.root_model_dir, f"{self.symbol}_xgboost.pkl")
                with open(xgb_pkl_path, 'wb') as f:
                    pickle.dump(xgb_model, f)
                
                logger.info(f"Saved XGBoost model to {xgb_model_path} and {xgb_pkl_path}")
            
            # Save training results
            results = {
                "timestamp": timestamp,
                "models_trained": [],
                "feature_count": 0
            }
            
            if rf_model is not None:
                results["models_trained"].append("random_forest")
            if xgb_model is not None:
                results["models_trained"].append("xgboost")
            
            if rf_metadata is not None:
                results["feature_count"] = len(rf_metadata["feature_importances"])
                results["top_features"] = rf_metadata["feature_importances"][:10]
            elif xgb_metadata is not None:
                results["feature_count"] = len(xgb_metadata["feature_importances"])
                results["top_features"] = xgb_metadata["feature_importances"][:10]
            
            results_path = os.path.join(self.specialized_model_dir, f"training_results_{timestamp}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved training results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def run_training_cycle(self):
        """Run a complete training cycle."""
        try:
            logger.info("Starting training cycle...")
            
            # Load data
            analyzed_df = self.load_historical_data()
            predictions_data = self.load_predictions()
            
            if analyzed_df is None or predictions_data is None:
                logger.error("Missing data for training cycle")
                return False
            
            # Prepare training data
            X, y = self.prepare_training_data(analyzed_df, predictions_data)
            
            if X is None or y is None or len(X) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(X) if X is not None else 0} samples")
                return False
            
            # Train models
            rf_model, rf_metadata = self.train_random_forest(X, y)
            xgb_model, xgb_metadata = self.train_xgboost(X, y)
            
            # Save models
            self.save_models(rf_model, rf_metadata, xgb_model, xgb_metadata)
            
            logger.info("Training cycle completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in training cycle: {str(e)}")
            return False
    
    def run_continuous_learning(self):
        """Run continuous learning process."""
        logger.info("Starting continuous learning process...")
        
        while True:
            try:
                # Run training cycle
                success = self.run_training_cycle()
                
                if success:
                    logger.info(f"Waiting {self.update_interval_hours} hours until next training cycle")
                else:
                    logger.info(f"Training cycle failed, retrying in 1 hour")
                    time.sleep(3600)  # Wait 1 hour before retrying
                    continue
                
                # Sleep until next update
                time.sleep(self.update_interval_hours * 3600)
                
            except KeyboardInterrupt:
                logger.info("Continuous learning process interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous learning process: {str(e)}")
                time.sleep(3600)  # Wait 1 hour before retrying

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous ML Model Learner')
    parser.add_argument('--symbol', default='EUR_USD', help='Currency pair symbol')
    parser.add_argument('--timeframe', default='5m', help='Data timeframe')
    parser.add_argument('--interval', type=int, default=12, help='Update interval in hours')
    
    args = parser.parse_args()
    
    learner = ContinuousLearner(
        symbol=args.symbol,
        timeframe=args.timeframe,
        update_interval_hours=args.interval
    )
    
    learner.run_continuous_learning()

if __name__ == "__main__":
    main()