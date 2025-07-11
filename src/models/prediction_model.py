# -*- coding: utf-8 -*-
"""
Bank Check Prediction Models

Fast machine learning algorithms for predicting:
1. Number of checks a client will issue
2. Maximum amount per check
"""

import json
import random
import math
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ..utils.data_utils import clean_numeric_data, calculate_metrics


class OptimizedLinearRegression:
    """Linear Regression with feature scaling."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, regularization: float = 0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.weights = []
        self.bias = 0.0
        self.training_history = []
    
    def fit(self, X: List[List[float]], y: List[float], verbose: bool = True) -> None:
        """Train linear regression with numerical stability."""
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        n_features = len(X[0])
        n_samples = len(X)
        
        # Normalize targets to prevent overflow
        y_mean = sum(y) / len(y)
        y_std = (sum((yi - y_mean) ** 2 for yi in y) / len(y)) ** 0.5
        y_std = max(y_std, 1e-8)  # Prevent division by zero
        y_norm = [(yi - y_mean) / y_std for yi in y]
        
        # Initialize weights with small values
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0  # Start with zero bias for normalized targets
        self.training_history = []
        
        # Store normalization parameters
        self.y_mean = y_mean
        self.y_std = y_std
        
        if verbose:
            print(f"[TERMINAL] Training Linear Regression: {n_samples} samples, {n_features} features")
        
        # Calculate feature means and stds for standardization
        feature_means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        feature_stds = []
        for j in range(n_features):
            variance = sum((X[i][j] - feature_means[j]) ** 2 for i in range(n_samples)) / n_samples
            feature_stds.append(max(variance ** 0.5, 1e-8))  # Prevent division by zero
        
        best_loss = float('inf')
        patience = 100
        no_improve = 0
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Forward pass with standardized features
            predictions = []
            for i, x in enumerate(X):
                pred = self.bias
                for j, feature in enumerate(x):
                    standardized_feature = (feature - feature_means[j]) / feature_stds[j]
                    pred += self.weights[j] * standardized_feature
                
                # Clip predictions to prevent overflow
                pred = max(-100, min(100, pred))
                predictions.append(pred)
            
            # Calculate loss with overflow protection
            try:
                squared_errors = []
                for i, pred in enumerate(predictions):
                    error = pred - y_norm[i]
                    # Clip error to prevent overflow
                    error = max(-50, min(50, error))
                    squared_errors.append(error ** 2)
                
                mse_loss = sum(squared_errors) / n_samples
                reg_loss = self.regularization * sum(w ** 2 for w in self.weights)
                total_loss = mse_loss + reg_loss
                
                # Check for overflow
                if total_loss > 1e10 or math.isnan(total_loss) or math.isinf(total_loss):
                    if verbose:
                        print(f"[TERMINAL] Numerical instability detected, reducing learning rate")
                    self.learning_rate *= 0.5
                    if self.learning_rate < 1e-8:
                        break
                    continue
                    
            except OverflowError:
                if verbose:
                    print(f"[TERMINAL] Overflow detected, reducing learning rate")
                self.learning_rate *= 0.5
                if self.learning_rate < 1e-8:
                    break
                continue
            
            self.training_history.append(total_loss)
            
            # Early stopping
            if total_loss < best_loss:
                best_loss = total_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > patience:
                    if verbose:
                        print(f"[TERMINAL] Early stopping at epoch {epoch}")
                    break
            
            # Gradient descent with clipping
            self._update_weights_standardized(X, y_norm, predictions, feature_means, feature_stds, n_samples)
            
            # Progress logging with ETA
            if verbose and epoch % 200 == 0:
                progress = (epoch / self.epochs) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (epoch + 1)) * (self.epochs - epoch) if epoch > 0 else 0
                print(f"[TERMINAL] {progress:5.1f}% | Epoch {epoch:4d} | Loss: {total_loss:.6f} | ETA: {eta:.1f}s")
        
        # Store standardization parameters for prediction
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        if verbose:
            total_time = time.time() - start_time
            print(f"[TERMINAL] Linear Regression completed in {total_time:.1f}s | Final loss: {best_loss:.6f}")
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for x in X:
            pred = self.bias
            for j, feature in enumerate(x):
                standardized_feature = (feature - self.feature_means[j]) / self.feature_stds[j]
                pred += self.weights[j] * standardized_feature
            
            # Denormalize prediction
            pred_denorm = pred * self.y_std + self.y_mean
            predictions.append(pred_denorm)
        return predictions
    
    def _update_weights_standardized(self, X, y_norm, predictions, feature_means, feature_stds, n_samples):
        # Update bias with clipping
        bias_grad = sum(predictions[i] - y_norm[i] for i in range(n_samples)) / n_samples
        bias_grad = max(-10, min(10, bias_grad))  # Clip gradient
        self.bias -= self.learning_rate * bias_grad
        
        # Update weights with standardized features and clipping
        for j in range(len(self.weights)):
            weight_grad = 0
            for i in range(n_samples):
                standardized_feature = (X[i][j] - feature_means[j]) / feature_stds[j]
                grad_contrib = (predictions[i] - y_norm[i]) * standardized_feature
                weight_grad += grad_contrib
            
            weight_grad /= n_samples
            weight_grad = max(-10, min(10, weight_grad))  # Clip gradient
            
            reg_grad = 2 * self.regularization * self.weights[j]
            total_grad = weight_grad + reg_grad
            
            # Clip weight update
            weight_update = self.learning_rate * total_grad
            weight_update = max(-1, min(1, weight_update))
            
            self.weights[j] -= weight_update
            
            # Clip weights to prevent explosion
            self.weights[j] = max(-100, min(100, self.weights[j]))


class FastGradientBoosting:
    """Fast Gradient Boosting optimized for speed."""
    
    def __init__(self, n_estimators: int = 50, max_depth: int = 3, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_prediction = 0.0
        self.training_history = []
    
    def fit(self, X: List[List[float]], y: List[float], verbose: bool = True) -> None:
        """Train fast gradient boosting."""
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        if verbose:
            print(f"[TERMINAL] Training Fast Gradient Boosting: {self.n_estimators} estimators, depth {self.max_depth}")
        
        # Initialize with mean
        self.initial_prediction = sum(y) / len(y)
        predictions = [self.initial_prediction] * n_samples
        
        self.trees = []
        self.training_history = []
        start_time = time.time()
        
        for i in range(self.n_estimators):
            # Calculate residuals
            residuals = [y[j] - predictions[j] for j in range(n_samples)]
            
            # Train decision stump
            tree = self._train_decision_stump(X, residuals, n_features)
            self.trees.append(tree)
            
            # Update predictions
            for j in range(n_samples):
                tree_pred = self._predict_tree(tree, X[j])
                predictions[j] += self.learning_rate * tree_pred
            
            # Calculate loss
            loss = sum((y[j] - predictions[j]) ** 2 for j in range(n_samples)) / n_samples
            self.training_history.append(loss)
            
            # Progress logging with ETA
            if verbose and i % 10 == 0:
                progress = (i / self.n_estimators) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (self.n_estimators - i) if i > 0 else 0
                print(f"[TERMINAL] {progress:5.1f}% | Estimator {i:3d} | Loss: {loss:.6f} | ETA: {eta:.1f}s")
        
        if verbose:
            total_time = time.time() - start_time
            print(f"[TERMINAL] Fast Gradient Boosting completed in {total_time:.1f}s | Final loss: {loss:.6f}")
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for x in X:
            pred = self.initial_prediction
            for tree in self.trees:
                pred += self.learning_rate * self._predict_tree(tree, x)
            predictions.append(pred)
        return predictions
    
    def _train_decision_stump(self, X, y, n_features):
        """Train a simple decision stump."""
        best_feature = 0
        best_threshold = 0.0
        best_loss = float('inf')
        
        for feature in range(n_features):
            # Get unique values for this feature
            values = sorted(set(X[i][feature] for i in range(len(X))))
            
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                # Split data
                left_indices = [j for j in range(len(X)) if X[j][feature] <= threshold]
                right_indices = [j for j in range(len(X)) if X[j][feature] > threshold]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate predictions
                left_pred = sum(y[j] for j in left_indices) / len(left_indices)
                right_pred = sum(y[j] for j in right_indices) / len(right_indices)
                
                # Calculate loss
                loss = sum((y[j] - left_pred) ** 2 for j in left_indices)
                loss += sum((y[j] - right_pred) ** 2 for j in right_indices)
                
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        
        # Create leaf predictions
        left_indices = [j for j in range(len(X)) if X[j][best_feature] <= best_threshold]
        right_indices = [j for j in range(len(X)) if X[j][best_feature] > best_threshold]
        
        left_pred = sum(y[j] for j in left_indices) / len(left_indices) if left_indices else 0
        right_pred = sum(y[j] for j in right_indices) / len(right_indices) if right_indices else 0
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_value': left_pred,
            'right_value': right_pred
        }
    
    def _predict_tree(self, tree, x):
        """Predict using a single tree."""
        if x[tree['feature']] <= tree['threshold']:
            return tree['left_value']
        else:
            return tree['right_value']


class OptimizedNeuralNetwork:
    """Simple and stable Neural Network."""
    
    def __init__(self, hidden_size: int = 16, learning_rate: float = 0.001, epochs: int = 200):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w1 = []  # Input to hidden weights
        self.b1 = []  # Hidden biases
        self.w2 = []  # Hidden to output weights
        self.b2 = 0.0  # Output bias
        self.training_history = []
    
    def fit(self, X: List[List[float]], y: List[float], verbose: bool = True) -> None:
        """Train simple neural network."""
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        if verbose:
            print(f"[TERMINAL] Training Neural Network: {n_features} -> {self.hidden_size} -> 1")
        
        # Normalize targets
        y_mean = sum(y) / len(y)
        y_std = (sum((yi - y_mean) ** 2 for yi in y) / len(y)) ** 0.5
        y_std = max(y_std, 1e-8)
        y_norm = [(yi - y_mean) / y_std for yi in y]
        
        self.y_mean = y_mean
        self.y_std = y_std
        
        # Standardize features
        feature_means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        feature_stds = []
        for j in range(n_features):
            variance = sum((X[i][j] - feature_means[j]) ** 2 for i in range(n_samples)) / n_samples
            feature_stds.append(max(variance ** 0.5, 1e-8))
        
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        # Standardize X
        X_std = []
        for i in range(n_samples):
            x_std = []
            for j in range(n_features):
                x_std.append((X[i][j] - feature_means[j]) / feature_stds[j])
            X_std.append(x_std)
        
        # Initialize weights with small random values
        self.w1 = [[random.uniform(-0.1, 0.1) for _ in range(self.hidden_size)] for _ in range(n_features)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [random.uniform(-0.1, 0.1) for _ in range(self.hidden_size)]
        self.b2 = 0.0
        
        self.training_history = []
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Forward pass
            total_loss = 0
            
            for sample_idx in range(n_samples):
                x = X_std[sample_idx]
                target = y_norm[sample_idx]
                
                # Hidden layer
                hidden = []
                for h in range(self.hidden_size):
                    z = self.b1[h]
                    for i in range(n_features):
                        z += x[i] * self.w1[i][h]
                    hidden.append(max(0, z))  # ReLU activation
                
                # Output layer
                output = self.b2
                for h in range(self.hidden_size):
                    output += hidden[h] * self.w2[h]
                
                # Loss
                error = output - target
                total_loss += error ** 2
                
                # Backward pass
                # Output layer gradients
                output_grad = error
                
                # Update output weights
                for h in range(self.hidden_size):
                    self.w2[h] -= self.learning_rate * output_grad * hidden[h]
                self.b2 -= self.learning_rate * output_grad
                
                # Hidden layer gradients
                for h in range(self.hidden_size):
                    if hidden[h] > 0:  # ReLU derivative
                        hidden_grad = output_grad * self.w2[h]
                        
                        # Update hidden weights
                        for i in range(n_features):
                            self.w1[i][h] -= self.learning_rate * hidden_grad * x[i]
                        self.b1[h] -= self.learning_rate * hidden_grad
            
            # Average loss
            avg_loss = total_loss / n_samples
            self.training_history.append(avg_loss)
            
            # Progress logging
            if verbose and epoch % 50 == 0:
                progress = (epoch / self.epochs) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (epoch + 1)) * (self.epochs - epoch) if epoch > 0 else 0
                print(f"[TERMINAL] {progress:5.1f}% | Epoch {epoch:4d} | Loss: {avg_loss:.6f} | ETA: {eta:.1f}s")
        
        if verbose:
            total_time = time.time() - start_time
            print(f"[TERMINAL] Neural Network completed in {total_time:.1f}s | Final loss: {avg_loss:.6f}")
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        
        for x in X:
            # Standardize input
            x_std = []
            for j in range(len(x)):
                x_std.append((x[j] - self.feature_means[j]) / self.feature_stds[j])
            
            # Forward pass
            hidden = []
            for h in range(self.hidden_size):
                z = self.b1[h]
                for i in range(len(x_std)):
                    z += x_std[i] * self.w1[i][h]
                hidden.append(max(0, z))
            
            output = self.b2
            for h in range(self.hidden_size):
                output += hidden[h] * self.w2[h]
            
            # Denormalize
            pred = output * self.y_std + self.y_mean
            predictions.append(pred)
        
        return predictions


class CheckPredictionModel:
    """Main prediction model class."""
    
    def __init__(self):
        self.model_type = 'gradient_boost'
        self.nbr_cheques_model = None
        self.montant_max_model = None
        self.is_trained = False
        self.metrics = {}
        self.feature_names = []
        
        print("[TERMINAL] CheckPredictionModel initialized")
    
    def set_model_type(self, model_type: str):
        """Set the ML algorithm type."""
        if model_type not in ['linear', 'gradient_boost', 'neural_network']:
            raise ValueError("Model type must be: linear, gradient_boost, or neural_network")
        
        self.model_type = model_type
        print(f"[TERMINAL] Model type set to: {model_type}")
    
    def fit(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the prediction models."""
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        print(f"[TERMINAL] ============ STARTING {self.model_type.upper()} TRAINING ============")
        
        # Create model instances
        print("[TERMINAL] Creating model instances...")
        
        if self.model_type == 'linear':
            self.nbr_cheques_model = OptimizedLinearRegression(epochs=1000)
            self.montant_max_model = OptimizedLinearRegression(epochs=1000)
        elif self.model_type == 'gradient_boost':
            self.nbr_cheques_model = FastGradientBoosting(n_estimators=100, max_depth=4)
            self.montant_max_model = FastGradientBoosting(n_estimators=100, max_depth=4)
        elif self.model_type == 'neural_network':
            self.nbr_cheques_model = OptimizedNeuralNetwork(hidden_size=16, epochs=200)
            self.montant_max_model = OptimizedNeuralNetwork(hidden_size=16, epochs=200)
        
        # Prepare features
        print("[TERMINAL] Preparing features for model training")
        X, y_nbr, y_montant = self._prepare_features(training_data)
        
        print(f"[TERMINAL] Prepared {len(X)} feature vectors with {len(X[0])} features")
        
        # Train number of checks model
        print("[TERMINAL] ============ TRAINING NUMBER OF CHECKS MODEL ============")
        start_time = time.time()
        self.nbr_cheques_model.fit(X, y_nbr, verbose=True)
        nbr_time = time.time() - start_time
        print(f"[TERMINAL] Number of checks model completed in {nbr_time:.1f}s")
        
        # Train maximum amount model
        print("[TERMINAL] ============ TRAINING MAXIMUM AMOUNT MODEL ============")
        start_time = time.time()
        self.montant_max_model.fit(X, y_montant, verbose=True)
        montant_time = time.time() - start_time
        print(f"[TERMINAL] Maximum amount model completed in {montant_time:.1f}s")
        
        # Evaluate models
        print("[TERMINAL] Evaluating model performance...")
        self._evaluate_models(X, y_nbr, y_montant)
        
        # Get R² scores
        nbr_r2 = self.metrics.get('nbr_cheques', {}).get('r2', 0)
        amount_r2 = self.metrics.get('montant_max', {}).get('r2', 0)
        
        print(f"[TERMINAL] ============ {self.model_type.upper()} RESULTS ============")
        print(f"[TERMINAL] Number of checks R²: {nbr_r2:.4f}")
        print(f"[TERMINAL] Maximum amount R²: {amount_r2:.4f}")
        
        total_time = nbr_time + montant_time
        print(f"[TERMINAL] ============ {self.model_type.upper()} TRAINING COMPLETED IN {total_time:.1f}s ============")
        
        self.is_trained = True
    
    def predict(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions for a client."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _, _ = self._prepare_features([client_data])
        
        # Make predictions
        nbr_pred = self.nbr_cheques_model.predict(X)[0]
        montant_pred = self.montant_max_model.predict(X)[0]
        
        # Ensure positive predictions
        nbr_pred = max(0, round(nbr_pred))
        montant_pred = max(0, montant_pred)
        
        return {
            'predicted_nbr_cheques': nbr_pred,
            'predicted_montant_max': montant_pred,
            'model_confidence': {
                'nbr_cheques_r2': self.metrics.get('nbr_cheques', {}).get('r2', 0),
                'montant_max_r2': self.metrics.get('montant_max', {}).get('r2', 0)
            }
        }
    
    def _prepare_features(self, data: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float], List[float]]:
        """Prepare features for training."""
        X = []
        y_nbr = []
        y_montant = []
        
        for record in data:
            features = [
                clean_numeric_data(record.get('Revenu_Estime', 0)),
                clean_numeric_data(record.get('Nbr_Cheques_2024', 0)),
                clean_numeric_data(record.get('Montant_Max_2024', 0)),
                clean_numeric_data(record.get('Ecart_Nbr_Cheques_2024_2025', 0)),
                clean_numeric_data(record.get('Ecart_Montant_Max_2024_2025', 0)),
                clean_numeric_data(record.get('A_Demande_Derogation', 0)),
                clean_numeric_data(record.get('Ratio_Cheques_Paiements', 0)),
                clean_numeric_data(record.get('Utilise_Mobile_Banking', 0)),
                clean_numeric_data(record.get('Nombre_Methodes_Paiement', 0)),
                clean_numeric_data(record.get('Montant_Moyen_Cheque', 0)),
                clean_numeric_data(record.get('Montant_Moyen_Alternative', 0)),
                1 if record.get('CLIENT_MARCHE') == 'PME' else 0,
                1 if record.get('CLIENT_MARCHE') == 'TPE' else 0,
                1 if record.get('CLIENT_MARCHE') == 'GEI' else 0,
                1 if record.get('CLIENT_MARCHE') == 'TRE' else 0,
            ]
            
            X.append(features)
            y_nbr.append(clean_numeric_data(record.get('Target_Nbr_Cheques_Futur', 0)))
            y_montant.append(clean_numeric_data(record.get('Target_Montant_Max_Futur', 0)))
        
        return X, y_nbr, y_montant
    
    def _evaluate_models(self, X, y_nbr, y_montant):
        """Evaluate model performance."""
        # Predictions
        nbr_pred = self.nbr_cheques_model.predict(X)
        montant_pred = self.montant_max_model.predict(X)
        
        # Calculate metrics
        nbr_metrics = calculate_metrics(y_nbr, nbr_pred)
        montant_metrics = calculate_metrics(y_montant, montant_pred)
        
        self.metrics = {
            'nbr_cheques': nbr_metrics,
            'montant_max': montant_metrics
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Serialize model weights and parameters
        nbr_model_data = None
        montant_model_data = None
        
        if self.nbr_cheques_model:
            if hasattr(self.nbr_cheques_model, 'weights'):
                # Linear model
                nbr_model_data = {
                    'weights': self.nbr_cheques_model.weights,
                    'bias': self.nbr_cheques_model.bias,
                    'y_mean': getattr(self.nbr_cheques_model, 'y_mean', 0),
                    'y_std': getattr(self.nbr_cheques_model, 'y_std', 1),
                    'feature_means': getattr(self.nbr_cheques_model, 'feature_means', []),
                    'feature_stds': getattr(self.nbr_cheques_model, 'feature_stds', [])
                }
            elif hasattr(self.nbr_cheques_model, 'trees'):
                # Gradient boosting model
                nbr_model_data = {
                    'trees': self.nbr_cheques_model.trees,
                    'initial_prediction': self.nbr_cheques_model.initial_prediction,
                    'learning_rate': self.nbr_cheques_model.learning_rate
                }
            elif hasattr(self.nbr_cheques_model, 'w1'):
                # Neural network model
                nbr_model_data = {
                    'w1': self.nbr_cheques_model.w1,
                    'b1': self.nbr_cheques_model.b1,
                    'w2': self.nbr_cheques_model.w2,
                    'b2': self.nbr_cheques_model.b2,
                    'hidden_size': self.nbr_cheques_model.hidden_size,
                    'y_mean': getattr(self.nbr_cheques_model, 'y_mean', 0),
                    'y_std': getattr(self.nbr_cheques_model, 'y_std', 1),
                    'feature_means': getattr(self.nbr_cheques_model, 'feature_means', []),
                    'feature_stds': getattr(self.nbr_cheques_model, 'feature_stds', [])
                }
        
        if self.montant_max_model:
            if hasattr(self.montant_max_model, 'weights'):
                # Linear model
                montant_model_data = {
                    'weights': self.montant_max_model.weights,
                    'bias': self.montant_max_model.bias,
                    'y_mean': getattr(self.montant_max_model, 'y_mean', 0),
                    'y_std': getattr(self.montant_max_model, 'y_std', 1),
                    'feature_means': getattr(self.montant_max_model, 'feature_means', []),
                    'feature_stds': getattr(self.montant_max_model, 'feature_stds', [])
                }
            elif hasattr(self.montant_max_model, 'trees'):
                # Gradient boosting model
                montant_model_data = {
                    'trees': self.montant_max_model.trees,
                    'initial_prediction': self.montant_max_model.initial_prediction,
                    'learning_rate': self.montant_max_model.learning_rate
                }
            elif hasattr(self.montant_max_model, 'w1'):
                # Neural network model
                montant_model_data = {
                    'w1': self.montant_max_model.w1,
                    'b1': self.montant_max_model.b1,
                    'w2': self.montant_max_model.w2,
                    'b2': self.montant_max_model.b2,
                    'hidden_size': self.montant_max_model.hidden_size,
                    'y_mean': getattr(self.montant_max_model, 'y_mean', 0),
                    'y_std': getattr(self.montant_max_model, 'y_std', 1),
                    'feature_means': getattr(self.montant_max_model, 'feature_means', []),
                    'feature_stds': getattr(self.montant_max_model, 'feature_stds', [])
                }
        
        model_data = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'nbr_model_data': nbr_model_data,
            'montant_model_data': montant_model_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.metrics = model_data['metrics']
        
        # Restore actual model objects if they exist
        if 'nbr_model_data' in model_data and model_data['nbr_model_data']:
            self.nbr_cheques_model = self._restore_model(model_data['nbr_model_data'], self.model_type)
        
        if 'montant_model_data' in model_data and model_data['montant_model_data']:
            self.montant_max_model = self._restore_model(model_data['montant_model_data'], self.model_type)
    
    def _restore_model(self, model_data: Dict[str, Any], model_type: str):
        """Restore a model instance from saved data."""
        if model_type == 'linear':
            model = OptimizedLinearRegression()
            model.weights = model_data.get('weights', [])
            model.bias = model_data.get('bias', 0.0)
            model.y_mean = model_data.get('y_mean', 0)
            model.y_std = model_data.get('y_std', 1)
            model.feature_means = model_data.get('feature_means', [])
            model.feature_stds = model_data.get('feature_stds', [])
            return model
        
        elif model_type == 'gradient_boost':
            model = FastGradientBoosting()
            model.trees = model_data.get('trees', [])
            model.initial_prediction = model_data.get('initial_prediction', 0.0)
            model.learning_rate = model_data.get('learning_rate', 0.1)
            return model
        
        elif model_type == 'neural_network':
            model = OptimizedNeuralNetwork()
            model.w1 = model_data.get('w1', [])
            model.b1 = model_data.get('b1', [])
            model.w2 = model_data.get('w2', [])
            model.b2 = model_data.get('b2', 0.0)
            model.hidden_size = model_data.get('hidden_size', 16)
            model.y_mean = model_data.get('y_mean', 0)
            model.y_std = model_data.get('y_std', 1)
            model.feature_means = model_data.get('feature_means', [])
            model.feature_stds = model_data.get('feature_stds', [])
            return model
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_trained or self.model_type != 'linear':
            return {}
        
        feature_names = [
            'Revenu_Estime', 'Nbr_Cheques_2024', 'Montant_Max_2024',
            'Ecart_Nbr_Cheques', 'Ecart_Montant_Max', 'A_Demande_Derogation',
            'Ratio_Cheques_Paiements', 'Utilise_Mobile_Banking',
            'Nombre_Methodes_Paiement', 'Montant_Moyen_Cheque',
            'Montant_Moyen_Alternative', 'CLIENT_MARCHE_PME',
            'CLIENT_MARCHE_TPE', 'CLIENT_MARCHE_GEI', 'CLIENT_MARCHE_TRE'
        ]
        
        if hasattr(self.nbr_cheques_model, 'weights'):
            weights = self.nbr_cheques_model.weights
            return {name: abs(weight) for name, weight in zip(feature_names, weights)}
        
        return {}