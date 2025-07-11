# -*- coding: utf-8 -*-
"""
Advanced Model Manager for Bank Check Prediction System
Supports multiple model storage, loading, and comparison
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .prediction_model import CheckPredictionModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Advanced model management with multiple model storage and selection."""
    
    def __init__(self, models_path: str = "data/models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry file
        self.registry_file = self.models_path / "model_registry.json"
        
        # Current active model
        self.active_model = None
        self.active_model_id = None
        
        # Load existing registry
        self.model_registry = self._load_registry()
        
        logger.info("ModelManager initialized")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        
        return {"models": {}, "active_model": None}
    
    def _save_registry(self) -> None:
        """Save the model registry."""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _generate_model_id(self, model_type: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{timestamp}"
    
    def save_model(self, model: CheckPredictionModel, model_name: Optional[str] = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained CheckPredictionModel
            model_name: Optional custom name for the model
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        if not model.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Generate model ID and filename
        model_id = self._generate_model_id(model.model_type)
        model_filename = f"{model_id}.json"
        model_filepath = self.models_path / model_filename
        
        # Generate display name
        if model_name is None:
            model_names = {
                'linear': 'Linear Regression',
                'gradient_boost': 'Gradient Boosting',
                'neural_network': 'Neural Network'
            }
            model_name = model_names.get(model.model_type, model.model_type)
        
        # Save model file
        model.save_model(str(model_filepath))
        
        # Update registry
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model.model_type,
            "filename": model_filename,
            "created_date": datetime.now().isoformat(),
            "metrics": model.metrics.copy() if model.metrics else {},
            "is_active": False
        }
        
        self.model_registry["models"][model_id] = model_info
        self._save_registry()
        
        logger.info(f"Model saved: {model_id} ({model_name})")
        return model_id
    
    def load_model(self, model_id: str) -> CheckPredictionModel:
        """
        Load a specific model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            CheckPredictionModel: Loaded model
        """
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        model_info = self.model_registry["models"][model_id]
        model_filepath = self.models_path / model_info["filename"]
        
        if not model_filepath.exists():
            raise FileNotFoundError(f"Model file not found: {model_filepath}")
        
        # Create and load model
        model = CheckPredictionModel()
        model.load_model(str(model_filepath))
        
        logger.info(f"Model loaded: {model_id} ({model_info['model_name']})")
        return model
    
    def set_active_model(self, model_id: str) -> None:
        """Set a model as the active/default model."""
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        # Update registry
        for mid, info in self.model_registry["models"].items():
            info["is_active"] = (mid == model_id)
        
        self.model_registry["active_model"] = model_id
        self._save_registry()
        
        # Load the active model
        self.active_model = self.load_model(model_id)
        self.active_model_id = model_id
        
        model_name = self.model_registry["models"][model_id]["model_name"]
        logger.info(f"Active model set: {model_id} ({model_name})")
    
    def get_active_model(self) -> Optional[CheckPredictionModel]:
        """Get the currently active model."""
        if self.active_model is None:
            active_id = self.model_registry.get("active_model")
            if active_id and active_id in self.model_registry["models"]:
                try:
                    self.active_model = self.load_model(active_id)
                    self.active_model_id = active_id
                except Exception as e:
                    logger.warning(f"Failed to load active model: {e}")
        
        return self.active_model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of all saved models with metadata."""
        models = []
        for model_id, info in self.model_registry["models"].items():
            model_data = info.copy()
            
            # Add performance summary
            if "metrics" in model_data and model_data["metrics"]:
                metrics = model_data["metrics"]
                nbr_r2 = metrics.get("nbr_cheques", {}).get("r2", 0)
                amount_r2 = metrics.get("montant_max", {}).get("r2", 0)
                model_data["performance_summary"] = {
                    "checks_accuracy": f"{nbr_r2:.1%}",
                    "amount_accuracy": f"{amount_r2:.1%}",
                    "overall_score": f"{(nbr_r2 + amount_r2) / 2:.1%}"
                }
            
            models.append(model_data)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_date"], reverse=True)
        return models
    
    def delete_model(self, model_id: str) -> None:
        """Delete a saved model."""
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        model_info = self.model_registry["models"][model_id]
        model_filepath = self.models_path / model_info["filename"]
        
        # Delete model file
        if model_filepath.exists():
            model_filepath.unlink()
        
        # Remove from registry
        del self.model_registry["models"][model_id]
        
        # Update active model if this was the active one
        if self.model_registry.get("active_model") == model_id:
            self.model_registry["active_model"] = None
            self.active_model = None
            self.active_model_id = None
        
        self._save_registry()
        
        model_name = model_info["model_name"]
        logger.info(f"Model deleted: {model_id} ({model_name})")
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparison of all models by type and performance."""
        models = self.list_models()
        
        comparison = {
            "by_type": {},
            "best_performers": {},
            "summary": {"total_models": len(models)}
        }
        
        # Group by type
        for model in models:
            model_type = model["model_type"]
            if model_type not in comparison["by_type"]:
                comparison["by_type"][model_type] = []
            comparison["by_type"][model_type].append(model)
        
        # Find best performers
        if models:
            # Best for checks
            best_checks = max(models, 
                key=lambda x: x.get("metrics", {}).get("nbr_cheques", {}).get("r2", 0))
            comparison["best_performers"]["checks"] = {
                "model_id": best_checks["model_id"],
                "model_name": best_checks["model_name"],
                "accuracy": best_checks.get("performance_summary", {}).get("checks_accuracy", "N/A")
            }
            
            # Best for amounts
            best_amount = max(models,
                key=lambda x: x.get("metrics", {}).get("montant_max", {}).get("r2", 0))
            comparison["best_performers"]["amounts"] = {
                "model_id": best_amount["model_id"],
                "model_name": best_amount["model_name"],
                "accuracy": best_amount.get("performance_summary", {}).get("amount_accuracy", "N/A")
            }
            
            # Best overall
            best_overall = max(models,
                key=lambda x: (x.get("metrics", {}).get("nbr_cheques", {}).get("r2", 0) +
                              x.get("metrics", {}).get("montant_max", {}).get("r2", 0)) / 2)
            comparison["best_performers"]["overall"] = {
                "model_id": best_overall["model_id"],
                "model_name": best_overall["model_name"],
                "accuracy": best_overall.get("performance_summary", {}).get("overall_score", "N/A")
            }
        
        return comparison
    
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        Keep only the most recent models of each type.
        
        Args:
            keep_count: Number of models to keep per type
            
        Returns:
            int: Number of models deleted
        """
        models = self.list_models()
        models_by_type = {}
        
        # Group by type
        for model in models:
            model_type = model["model_type"]
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model)
        
        deleted_count = 0
        
        # Keep only recent models per type
        for model_type, type_models in models_by_type.items():
            type_models.sort(key=lambda x: x["created_date"], reverse=True)
            
            # Delete old models (skip active model)
            for model in type_models[keep_count:]:
                if not model.get("is_active", False):
                    try:
                        self.delete_model(model["model_id"])
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete model {model['model_id']}: {e}")
        
        logger.info(f"Cleanup completed: {deleted_count} models deleted")
        return deleted_count
    
    def export_model_info(self, filepath: str) -> None:
        """Export model registry to file for backup."""
        registry_export = {
            "exported_date": datetime.now().isoformat(),
            "registry": self.model_registry
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(registry_export, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model registry exported to: {filepath}")