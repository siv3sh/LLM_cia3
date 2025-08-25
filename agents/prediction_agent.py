"""
Prediction Agent for machine learning model training and predictions
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from .base_agent import BaseAgent, AgentMessage
from core.config import Config


class PredictionAgent(BaseAgent):
    """
    Agent responsible for machine learning model training and predictions
    """
    
    def __init__(self, config: Config, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        
        # Model storage
        self.trained_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "model_class": RandomForestClassifier,
                "hyperparameters": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "gradient_boosting": {
                "model_class": GradientBoostingClassifier,
                "hyperparameters": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            },
            "logistic_regression": {
                "model_class": LogisticRegression,
                "hyperparameters": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            },
            "svm": {
                "model_class": SVC,
                "hyperparameters": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"]
                }
            }
        }
        
        # Setup prediction-specific message handlers
        self._setup_prediction_handlers()
        
        # Create models directory
        Path(self.config.model_backup_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_prediction_handlers(self):
        """Setup prediction-specific message handlers"""
        self.message_handlers.update({
            "train_model": self._handle_train_model,
            "make_predictions": self._handle_make_predictions,
            "evaluate_model": self._handle_evaluate_model,
            "get_model_info": self._handle_get_model_info,
            "update_model": self._handle_update_model,
            "ensemble_prediction": self._handle_ensemble_prediction,
        })
    
    async def _initialize_agent(self):
        """Initialize prediction agent specific components"""
        try:
            # Load existing models if available
            await self._load_existing_models()
            
            self.logger.info("Prediction agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction agent components: {e}")
            raise
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prediction-related tasks"""
        task_type = task_data.get("task_type")
        
        if task_type == "train_model":
            return await self._train_model(task_data)
        elif task_type == "make_predictions":
            return await self._make_predictions(task_data)
        elif task_type == "evaluate_model":
            return await self._evaluate_model(task_data)
        elif task_type == "get_model_info":
            return await self._get_model_info(task_data)
        elif task_type == "update_model":
            return await self._update_model(task_data)
        elif task_type == "ensemble_prediction":
            return await self._ensemble_prediction(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested prediction data"""
        data_type = request_data.get("data_type")
        
        if data_type == "trained_models":
            return list(self.trained_models.keys())
        elif data_type == "model_metadata":
            return self.model_metadata
        elif data_type == "model_performance":
            return self.model_performance
        elif data_type == "prediction_summary":
            return self._get_prediction_summary()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def _handle_train_model(self, message: AgentMessage):
        """Handle model training requests"""
        try:
            training_config = message.content
            result = await self._train_model(training_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="model_training_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="model_training_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_make_predictions(self, message: AgentMessage):
        """Handle prediction requests"""
        try:
            prediction_config = message.content
            result = await self._make_predictions(prediction_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="prediction_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="prediction_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_evaluate_model(self, message: AgentMessage):
        """Handle model evaluation requests"""
        try:
            evaluation_config = message.content
            result = await self._evaluate_model(evaluation_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="model_evaluation_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="model_evaluation_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_get_model_info(self, message: AgentMessage):
        """Handle model info requests"""
        try:
            info_config = message.content
            result = await self._get_model_info(info_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="model_info_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Model info retrieval failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="model_info_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_update_model(self, message: AgentMessage):
        """Handle model update requests"""
        try:
            update_config = message.content
            result = await self._update_model(update_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="model_update_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="model_update_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_ensemble_prediction(self, message: AgentMessage):
        """Handle ensemble prediction requests"""
        try:
            ensemble_config = message.content
            result = await self._ensemble_prediction(ensemble_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="ensemble_prediction_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="ensemble_prediction_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _train_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            self.state.current_task = "model_training"
            self.update_task_progress(0.1)
            
            # Get training data
            data = config.get("data")
            if data is None:
                raise ValueError("No training data provided")
            
            # Get model configuration
            model_type = config.get("model_type", "random_forest")
            if model_type not in self.model_configs:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model_config = self.model_configs[model_type]
            
            # Prepare data
            self.update_task_progress(0.2)
            X, y, feature_names = await self._prepare_training_data(data)
            
            # Split data
            self.update_task_progress(0.3)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train model
            self.update_task_progress(0.4)
            model, best_params = await self._train_model_with_hyperparameter_tuning(
                model_config, X_train, y_train
            )
            
            # Evaluate model
            self.update_task_progress(0.7)
            performance_metrics = await self._evaluate_model_performance(
                model, X_test, y_test
            )
            
            # Save model
            self.update_task_progress(0.9)
            model_id = await self._save_model(model, model_type, best_params, performance_metrics)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "performance_metrics": performance_metrics,
                "best_hyperparameters": best_params,
                "feature_names": feature_names,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        try:
            # Find target variable
            target_col = None
            for col in data.columns:
                if 'attrition' in col.lower() or 'left' in col.lower() or 'churn' in col.lower():
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError("No attrition target column found")
            
            # Prepare features
            feature_cols = [col for col in data.columns if col != target_col]
            X = data[feature_cols].select_dtypes(include=[np.number])
            
            if X.empty:
                raise ValueError("No numerical features available for training")
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = data[target_col].fillna(data[target_col].mode()[0] if len(data[target_col].mode()) > 0 else 0)
            
            # Convert target to binary if needed
            if y.dtype == 'object':
                y = y.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
            
            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values
            
            return X_array, y_array, list(X.columns)
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _train_model_with_hyperparameter_tuning(
        self, model_config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train model with hyperparameter tuning"""
        try:
            model_class = model_config["model_class"]
            hyperparameters = model_config["hyperparameters"]
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model_class())
            ])
            
            # Grid search for hyperparameter tuning
            param_grid = {
                f'classifier__{param}': values 
                for param, values in hyperparameters.items()
            }
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Extract actual hyperparameters (remove 'classifier__' prefix)
            clean_params = {
                param.replace('classifier__', ''): value 
                for param, value in best_params.items()
            }
            
            return best_model, clean_params
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            raise
    
    async def _evaluate_model_performance(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
            }
            
            if y_pred_proba is not None:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["classification_report"] = report
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _save_model(self, model: Any, model_type: str, hyperparameters: Dict[str, Any], 
                         performance: Dict[str, Any]) -> str:
        """Save trained model"""
        try:
            model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model file
            model_path = Path(self.config.model_backup_path) / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            
            # Store model metadata
            self.trained_models[model_id] = model
            self.model_metadata[model_id] = {
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "training_date": datetime.now().isoformat(),
                "model_path": str(model_path),
                "version": self.config.model_version
            }
            
            # Store performance metrics
            self.model_performance[model_id] = performance
            
            self.logger.info(f"Model {model_id} saved successfully")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    async def _make_predictions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using trained model"""
        try:
            self.state.current_task = "making_predictions"
            self.update_task_progress(0.1)
            
            # Get model ID
            model_id = config.get("model_id")
            if not model_id or model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            # Get prediction data
            prediction_data = config.get("data")
            if prediction_data is None:
                raise ValueError("No prediction data provided")
            
            # Load model
            self.update_task_progress(0.3)
            model = self.trained_models[model_id]
            
            # Prepare prediction data
            self.update_task_progress(0.5)
            X_pred = await self._prepare_prediction_data(prediction_data, model_id)
            
            # Make predictions
            self.update_task_progress(0.7)
            predictions = model.predict(X_pred)
            prediction_probas = model.predict_proba(X_pred)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Format results
            self.update_task_progress(0.9)
            results = {
                "predictions": predictions.tolist(),
                "prediction_probabilities": prediction_probas.tolist() if prediction_probas is not None else None,
                "model_id": model_id,
                "prediction_timestamp": datetime.now().isoformat()
            }
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "results": results,
                "total_predictions": len(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _prepare_prediction_data(self, data: pd.DataFrame, model_id: str) -> np.ndarray:
        """Prepare data for prediction"""
        try:
            # Get feature names from model metadata
            metadata = self.model_metadata[model_id]
            
            # For now, assume we need to select numerical columns
            # In a real implementation, you'd want to ensure feature alignment
            X = data.select_dtypes(include=[np.number])
            
            if X.empty:
                raise ValueError("No numerical features available for prediction")
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            return X.values
            
        except Exception as e:
            self.logger.error(f"Prediction data preparation failed: {e}")
            raise
    
    async def _evaluate_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a trained model"""
        try:
            model_id = config.get("model_id")
            if not model_id or model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            # Get evaluation data
            eval_data = config.get("data")
            if eval_data is None:
                raise ValueError("No evaluation data provided")
            
            # Load model
            model = self.trained_models[model_id]
            
            # Prepare evaluation data
            X_eval, y_eval, _ = await self._prepare_training_data(eval_data)
            
            # Evaluate performance
            performance = await self._evaluate_model_performance(model, X_eval, y_eval)
            
            # Update stored performance
            self.model_performance[model_id] = performance
            
            return {
                "status": "success",
                "model_id": model_id,
                "performance_metrics": performance
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _get_model_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a trained model"""
        try:
            model_id = config.get("model_id")
            if not model_id:
                return {
                    "available_models": list(self.trained_models.keys()),
                    "total_models": len(self.trained_models)
                }
            
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.model_metadata.get(model_id, {})
            performance = self.model_performance.get(model_id, {})
            
            return {
                "model_id": model_id,
                "metadata": metadata,
                "performance": performance,
                "model_size": self._get_model_size(model_id)
            }
            
        except Exception as e:
            self.logger.error(f"Model info retrieval failed: {e}")
            raise
    
    async def _update_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing model with new data"""
        try:
            model_id = config.get("model_id")
            if not model_id or model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            # Get new training data
            new_data = config.get("data")
            if new_data is None:
                raise ValueError("No new training data provided")
            
            # Retrain model with new data
            # This is a simplified version - in practice you might want incremental learning
            retrain_config = {
                "data": new_data,
                "model_type": self.model_metadata[model_id]["model_type"]
            }
            
            result = await self._train_model(retrain_config)
            
            return {
                "status": "success",
                "message": f"Model {model_id} updated successfully",
                "new_model_id": result["model_id"]
            }
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            raise
    
    async def _ensemble_prediction(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble predictions using multiple models"""
        try:
            # Get model IDs for ensemble
            model_ids = config.get("model_ids", list(self.trained_models.keys()))
            if not model_ids:
                raise ValueError("No models available for ensemble prediction")
            
            # Get prediction data
            prediction_data = config.get("data")
            if prediction_data is None:
                raise ValueError("No prediction data provided")
            
            # Make predictions with each model
            ensemble_predictions = []
            ensemble_probas = []
            
            for model_id in model_ids:
                if model_id in self.trained_models:
                    pred_config = {"model_id": model_id, "data": prediction_data}
                    pred_result = await self._make_predictions(pred_config)
                    
                    if pred_result["status"] == "success":
                        ensemble_predictions.append(pred_result["results"]["predictions"])
                        if pred_result["results"]["prediction_probabilities"]:
                            ensemble_probas.append(pred_result["results"]["prediction_probabilities"])
            
            if not ensemble_predictions:
                raise ValueError("No successful predictions from ensemble models")
            
            # Combine predictions (simple voting)
            ensemble_pred = np.array(ensemble_predictions)
            final_predictions = np.mean(ensemble_pred, axis=0) > 0.5
            
            # Combine probabilities if available
            final_probas = None
            if ensemble_probas:
                final_probas = np.mean(np.array(ensemble_probas), axis=0)
            
            return {
                "status": "success",
                "ensemble_predictions": final_predictions.tolist(),
                "ensemble_probabilities": final_probas.tolist() if final_probas is not None else None,
                "models_used": model_ids,
                "ensemble_method": "voting",
                "prediction_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing trained models from disk"""
        try:
            models_dir = Path(self.config.model_backup_path)
            if not models_dir.exists():
                return
            
            for model_file in models_dir.glob("*.joblib"):
                try:
                    model_id = model_file.stem
                    model = joblib.load(model_file)
                    
                    self.trained_models[model_id] = model
                    self.logger.info(f"Loaded existing model: {model_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing models: {e}")
    
    def _get_model_size(self, model_id: str) -> str:
        """Get model file size"""
        try:
            metadata = self.model_metadata.get(model_id, {})
            model_path = metadata.get("model_path")
            
            if model_path and Path(model_path).exists():
                size_bytes = Path(model_path).stat().st_size
                if size_bytes < 1024:
                    return f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024 * 1024):.1f} MB"
            
            return "Unknown"
            
        except Exception as e:
            self.logger.error(f"Failed to get model size: {e}")
            return "Unknown"
    
    def _get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction capabilities"""
        return {
            "total_models": len(self.trained_models),
            "model_types": list(set([meta.get("model_type", "unknown") for meta in self.model_metadata.values()])),
            "total_predictions_made": sum([len(meta.get("predictions", [])) for meta in self.model_metadata.values()]),
            "last_updated": datetime.utcnow().isoformat()
        }
