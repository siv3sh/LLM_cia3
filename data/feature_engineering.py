"""
Feature Engineering for creating and transforming features
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

from core.config import Config
from data.schemas import FeatureEngineeringConfig


class FeatureEngineer:
    """
    Handles feature engineering and transformation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature engineering components
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.polynomial_features: Optional[PolynomialFeatures] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.pca: Optional[PCA] = None
        
        # Feature metadata
        self.feature_metadata: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        
        # Engineering statistics
        self.engineering_stats: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the feature engineer"""
        try:
            # Initialize components
            self.scaler = StandardScaler()
            self.polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
            self.feature_selector = SelectKBest(score_func=f_classif, k=10)
            self.pca = PCA(n_components=0.95)  # Keep 95% variance
            
            self.logger.info("Feature engineer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature engineer: {e}")
            raise
    
    async def engineer_features(self, data: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
        """Perform comprehensive feature engineering"""
        try:
            self.logger.info("Starting feature engineering process")
            
            if data is None or data.empty:
                raise ValueError("No data provided for feature engineering")
            
            # Create a copy
            engineered_data = data.copy()
            
            # Store original shape
            original_shape = engineered_data.shape
            
            # Create temporal features
            if config.create_temporal_features:
                engineered_data = await self._create_temporal_features(engineered_data)
            
            # Create interaction features
            if config.create_interaction_features:
                engineered_data = await self._create_interaction_features(engineered_data)
            
            # Create polynomial features
            if config.create_polynomial_features:
                engineered_data = await self._create_polynomial_features(engineered_data, config.polynomial_degree)
            
            # Perform feature selection
            if config.max_features:
                engineered_data = await self._select_features(engineered_data, config.max_features, config.feature_selection_method)
            
            # Store engineering statistics
            self.engineering_stats["feature_engineering"] = {
                "original_shape": original_shape,
                "engineered_shape": engineered_data.shape,
                "features_added": engineered_data.shape[1] - original_shape[1],
                "config_used": config.dict(),
                "engineering_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Feature engineering completed: {engineered_data.shape}")
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
    
    async def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from existing data"""
        try:
            engineered_data = data.copy()
            
            # Create tenure-based features
            if 'years_at_company' in engineered_data.columns:
                # Tenure categories
                engineered_data['tenure_category'] = pd.cut(
                    engineered_data['years_at_company'],
                    bins=[0, 2, 5, 10, 20, 100],
                    labels=['New', 'Early', 'Mid', 'Senior', 'Veteran']
                )
                
                # Tenure risk (higher risk for new and very long tenure)
                engineered_data['tenure_risk'] = np.where(
                    (engineered_data['years_at_company'] <= 2) | (engineered_data['years_at_company'] >= 15),
                    1, 0
                )
            
            if 'years_in_current_role' in engineered_data.columns:
                # Role stagnation
                engineered_data['role_stagnation'] = np.where(
                    engineered_data['years_in_current_role'] >= 5, 1, 0
                )
                
                # Role progression
                engineered_data['role_progression'] = engineered_data['years_at_company'] - engineered_data['years_in_current_role']
            
            if 'years_since_last_promotion' in engineered_data.columns:
                # Promotion stagnation
                engineered_data['promotion_stagnation'] = np.where(
                    engineered_data['years_since_last_promotion'] >= 3, 1, 0
                )
            
            if 'training_times_last_year' in engineered_data.columns:
                # Training frequency
                engineered_data['training_frequency'] = pd.cut(
                    engineered_data['training_times_last_year'],
                    bins=[0, 1, 3, 6, 100],
                    labels=['None', 'Low', 'Medium', 'High']
                )
            
            # Store feature metadata
            self.feature_metadata["temporal_features"] = {
                "tenure_category": "Categorical feature based on years at company",
                "tenure_risk": "Binary risk indicator for new/veteran employees",
                "role_stagnation": "Binary indicator for role stagnation",
                "role_progression": "Difference between company and role tenure",
                "promotion_stagnation": "Binary indicator for promotion stagnation",
                "training_frequency": "Categorical feature for training frequency"
            }
            
            self.logger.info("Temporal features created successfully")
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Temporal feature creation failed: {e}")
            return data
    
    async def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        try:
            engineered_data = data.copy()
            
            # Age and tenure interaction
            if 'age' in engineered_data.columns and 'years_at_company' in engineered_data.columns:
                engineered_data['age_tenure_ratio'] = engineered_data['age'] / (engineered_data['years_at_company'] + 1)
                engineered_data['age_tenure_product'] = engineered_data['age'] * engineered_data['years_at_company']
            
            # Salary and performance interaction
            if 'monthly_income' in engineered_data.columns and 'performance_rating' in engineered_data.columns:
                engineered_data['salary_performance_ratio'] = engineered_data['monthly_income'] / engineered_data['performance_rating']
                engineered_data['salary_performance_alignment'] = np.where(
                    (engineered_data['monthly_income'] > engineered_data['monthly_income'].median()) & 
                    (engineered_data['performance_rating'] >= 4), 1, 0
                )
            
            # Education and role interaction
            if 'education' in engineered_data.columns and 'job_role' in engineered_data.columns:
                # Create education level mapping
                education_levels = {
                    'High School': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4
                }
                engineered_data['education_level'] = engineered_data['education'].map(education_levels).fillna(2)
                
                # Education-role mismatch
                engineered_data['education_role_mismatch'] = np.where(
                    (engineered_data['education_level'] >= 3) & 
                    (engineered_data['job_role'].str.contains('Associate|Assistant', case=False, na=False)), 1, 0
                )
            
            # Work-life balance indicators
            if 'overtime' in engineered_data.columns and 'business_travel' in engineered_data.columns:
                engineered_data['work_life_stress'] = np.where(
                    (engineered_data['overtime'] == 'Yes') & 
                    (engineered_data['business_travel'] == 'Travel_Frequently'), 2,
                    np.where(
                        (engineered_data['overtime'] == 'Yes') | 
                        (engineered_data['business_travel'] == 'Travel_Frequently'), 1, 0
                    )
                )
            
            # Store feature metadata
            self.feature_metadata["interaction_features"] = {
                "age_tenure_ratio": "Ratio of age to company tenure",
                "age_tenure_product": "Product of age and company tenure",
                "salary_performance_ratio": "Ratio of salary to performance rating",
                "salary_performance_alignment": "Binary indicator for salary-performance alignment",
                "education_level": "Numeric education level mapping",
                "education_role_mismatch": "Binary indicator for education-role mismatch",
                "work_life_stress": "Work-life balance stress indicator (0-2 scale)"
            }
            
            self.logger.info("Interaction features created successfully")
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Interaction feature creation failed: {e}")
            return data
    
    async def _create_polynomial_features(self, data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        try:
            engineered_data = data.copy()
            
            # Get numerical columns
            numerical_columns = engineered_data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_columns) == 0:
                self.logger.info("No numerical columns found for polynomial features")
                return engineered_data
            
            # Select important numerical features for polynomial creation
            important_features = ['age', 'years_at_company', 'monthly_income', 'performance_rating']
            selected_features = [col for col in important_features if col in numerical_columns]
            
            if len(selected_features) < 2:
                self.logger.info("Insufficient numerical features for polynomial creation")
                return engineered_data
            
            # Create polynomial features
            try:
                poly_data = engineered_data[selected_features].fillna(0)
                poly_features = self.polynomial_features.fit_transform(poly_data)
                
                # Create feature names
                feature_names = self.polynomial_features.get_feature_names_out(selected_features)
                
                # Add polynomial features to dataframe
                for i, feature_name in enumerate(feature_names):
                    if feature_name not in engineered_data.columns:
                        engineered_data[f'poly_{feature_name}'] = poly_features[:, i]
                
                # Store feature metadata
                self.feature_metadata["polynomial_features"] = {
                    "degree": degree,
                    "base_features": selected_features,
                    "total_polynomial_features": len(feature_names),
                    "feature_names": list(feature_names)
                }
                
                self.logger.info(f"Polynomial features created successfully (degree {degree})")
                
            except Exception as e:
                self.logger.warning(f"Polynomial feature creation failed: {e}")
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Polynomial feature creation failed: {e}")
            return data
    
    async def _select_features(self, data: pd.DataFrame, max_features: int, method: str = "correlation") -> pd.DataFrame:
        """Select the most important features"""
        try:
            engineered_data = data.copy()
            
            if method == "correlation":
                return await self._select_features_correlation(engineered_data, max_features)
            elif method == "mutual_info":
                return await self._select_features_mutual_info(engineered_data, max_features)
            elif method == "f_statistic":
                return await self._select_features_f_statistic(engineered_data, max_features)
            else:
                self.logger.warning(f"Unknown feature selection method: {method}")
                return engineered_data
                
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return data
    
    async def _select_features_correlation(self, data: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """Select features based on correlation with target"""
        try:
            if 'attrition' not in data.columns:
                self.logger.warning("Target column 'attrition' not found for correlation-based selection")
                return data
            
            # Calculate correlations with target
            correlations = data.corr()['attrition'].abs().sort_values(ascending=False)
            
            # Select top features
            top_features = correlations.head(max_features + 1).index.tolist()  # +1 to include target
            
            # Remove target from features
            if 'attrition' in top_features:
                top_features.remove('attrition')
            
            # Select only top features
            selected_data = data[top_features + ['attrition']]
            
            # Store feature importance
            self.feature_importance = correlations[top_features].to_dict()
            
            # Store selection metadata
            self.feature_metadata["feature_selection"] = {
                "method": "correlation",
                "max_features": max_features,
                "selected_features": top_features,
                "feature_importance": self.feature_importance
            }
            
            self.logger.info(f"Feature selection completed: {len(top_features)} features selected")
            
            return selected_data
            
        except Exception as e:
            self.logger.error(f"Correlation-based feature selection failed: {e}")
            return data
    
    async def _select_features_mutual_info(self, data: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """Select features based on mutual information with target"""
        try:
            if 'attrition' not in data.columns:
                self.logger.warning("Target column 'attrition' not found for mutual information selection")
                return data
            
            # Prepare data for mutual information
            X = data.drop(columns=['attrition']).select_dtypes(include=[np.number]).fillna(0)
            y = data['attrition']
            
            if X.shape[1] == 0:
                self.logger.warning("No numerical features found for mutual information selection")
                return data
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(X.columns, mi_scores))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            top_features = [feature for feature, score in sorted_features[:max_features]]
            
            # Select only top features
            selected_data = data[top_features + ['attrition']]
            
            # Store feature importance
            self.feature_importance = {feature: float(score) for feature, score in sorted_features[:max_features]}
            
            # Store selection metadata
            self.feature_metadata["feature_selection"] = {
                "method": "mutual_info",
                "max_features": max_features,
                "selected_features": top_features,
                "feature_importance": self.feature_importance
            }
            
            self.logger.info(f"Feature selection completed: {len(top_features)} features selected")
            
            return selected_data
            
        except Exception as e:
            self.logger.error(f"Mutual information feature selection failed: {e}")
            return data
    
    async def _select_features_f_statistic(self, data: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """Select features based on F-statistic with target"""
        try:
            if 'attrition' not in data.columns:
                self.logger.warning("Target column 'attrition' not found for F-statistic selection")
                return data
            
            # Prepare data for F-statistic
            X = data.drop(columns=['attrition']).select_dtypes(include=[np.number]).fillna(0)
            y = data['attrition']
            
            if X.shape[1] == 0:
                self.logger.warning("No numerical features found for F-statistic selection")
                return data
            
            # Calculate F-statistics
            f_scores, _ = f_classif(X, y)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(X.columns, f_scores))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            top_features = [feature for feature, score in sorted_features[:max_features]]
            
            # Select only top features
            selected_data = data[top_features + ['attrition']]
            
            # Store feature importance
            self.feature_importance = {feature: float(score) for feature, score in sorted_features[:max_features]}
            
            # Store selection metadata
            self.feature_metadata["feature_selection"] = {
                "method": "f_statistic",
                "max_features": max_features,
                "selected_features": top_features,
                "feature_importance": self.feature_importance
            }
            
            self.logger.info(f"Feature selection completed: {len(top_features)} features selected")
            
            return selected_data
            
        except Exception as e:
            self.logger.error(f"F-statistic feature selection failed: {e}")
            return data
    
    async def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        try:
            engineered_data = data.copy()
            
            # Get categorical columns
            categorical_columns = engineered_data.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_columns) == 0:
                self.logger.info("No categorical columns found for encoding")
                return engineered_data
            
            # Encode each categorical column
            for column in categorical_columns:
                if column == 'attrition':  # Skip target variable
                    continue
                
                try:
                    # Create label encoder
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(engineered_data[column].fillna('Unknown'))
                    
                    # Create new encoded column
                    engineered_data[f'{column}_encoded'] = encoded_values
                    
                    # Store encoder for later use
                    self.label_encoders[column] = le
                    
                    # Store encoding metadata
                    if "categorical_encoding" not in self.feature_metadata:
                        self.feature_metadata["categorical_encoding"] = {}
                    
                    self.feature_metadata["categorical_encoding"][column] = {
                        "encoder_type": "LabelEncoder",
                        "unique_values": le.classes_.tolist(),
                        "encoded_mapping": dict(zip(le.classes_, range(len(le.classes_))))
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to encode column {column}: {e}")
                    continue
            
            self.logger.info(f"Categorical encoding completed for {len(categorical_columns)} columns")
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Categorical encoding failed: {e}")
            return data
    
    async def scale_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        try:
            engineered_data = data.copy()
            
            # Get numerical columns (excluding target and encoded columns)
            numerical_columns = engineered_data.select_dtypes(include=[np.number]).columns
            target_columns = ['attrition']
            encoded_columns = [col for col in engineered_data.columns if col.endswith('_encoded')]
            
            # Columns to scale
            columns_to_scale = [col for col in numerical_columns if col not in target_columns + encoded_columns]
            
            if len(columns_to_scale) == 0:
                self.logger.info("No numerical columns found for scaling")
                return engineered_data
            
            # Scale features
            scaled_values = self.scaler.fit_transform(engineered_data[columns_to_scale])
            
            # Create scaled dataframe
            scaled_df = pd.DataFrame(scaled_values, columns=[f'{col}_scaled' for col in columns_to_scale])
            
            # Add scaled features to original data
            for col in scaled_df.columns:
                engineered_data[col] = scaled_df[col]
            
            # Store scaling metadata
            self.feature_metadata["feature_scaling"] = {
                "scaler_type": "StandardScaler",
                "scaled_columns": columns_to_scale,
                "scaled_column_names": list(scaled_df.columns),
                "scaler_params": {
                    "mean_": self.scaler.mean_.tolist(),
                    "scale_": self.scaler.scale_.tolist()
                }
            }
            
            self.logger.info(f"Feature scaling completed for {len(columns_to_scale)} columns")
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Feature scaling failed: {e}")
            return data
    
    async def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of engineered features"""
        try:
            return {
                "total_features_created": sum(len(features) for features in self.feature_metadata.values()),
                "feature_categories": list(self.feature_metadata.keys()),
                "feature_importance": self.feature_importance,
                "engineering_statistics": self.engineering_stats,
                "feature_metadata": self.feature_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get feature summary: {e}")
            return {"error": f"Feature summary retrieval failed: {str(e)}"}
    
    async def export_feature_metadata(self, path: str = None) -> str:
        """Export feature metadata to file"""
        try:
            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = f"./exports/feature_metadata_{timestamp}.json"
            
            # Create export directory
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata for export
            export_data = {
                "feature_metadata": self.feature_metadata,
                "feature_importance": self.feature_importance,
                "engineering_statistics": self.engineering_stats,
                "export_timestamp": datetime.utcnow().isoformat()
            }
            
            # Export to JSON
            import json
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Feature metadata exported to {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Feature metadata export failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the feature engineer"""
        try:
            # Clear components
            self.scaler = None
            self.label_encoders.clear()
            self.polynomial_features = None
            self.feature_selector = None
            self.pca = None
            
            # Clear metadata
            self.feature_metadata.clear()
            self.feature_importance.clear()
            self.engineering_stats.clear()
            
            self.logger.info("Feature engineer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Feature engineer shutdown failed: {e}")
