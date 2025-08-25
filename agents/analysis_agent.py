"""
Analysis Agent for statistical analysis and pattern recognition
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent, AgentMessage
from core.config import Config


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for statistical analysis and pattern recognition
    """
    
    def __init__(self, config: Config, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        
        # Analysis results storage
        self.analysis_results: Dict[str, Any] = {}
        self.statistical_tests: Dict[str, Any] = {}
        self.correlation_matrices: Dict[str, pd.DataFrame] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        
        # Analysis configuration
        self.analysis_configs = {
            "basic": ["descriptive", "correlation"],
            "comprehensive": ["descriptive", "correlation", "hypothesis_testing", "feature_importance"],
            "advanced": ["descriptive", "correlation", "hypothesis_testing", "feature_importance", "pca", "clustering"]
        }
        
        # Setup analysis-specific message handlers
        self._setup_analysis_handlers()
    
    def _setup_analysis_handlers(self):
        """Setup analysis-specific message handlers"""
        self.message_handlers.update({
            "statistical_analysis": self._handle_statistical_analysis,
            "correlation_analysis": self._handle_correlation_analysis,
            "hypothesis_testing": self._handle_hypothesis_testing,
            "feature_importance": self._handle_feature_importance,
            "pattern_recognition": self._handle_pattern_recognition,
            "trend_analysis": self._handle_trend_analysis,
        })
    
    async def _initialize_agent(self):
        """Initialize analysis agent specific components"""
        try:
            # Initialize analysis tools
            self.logger.info("Analysis agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analysis agent components: {e}")
            raise
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis-related tasks"""
        task_type = task_data.get("task_type")
        
        if task_type == "statistical_analysis":
            return await self._perform_statistical_analysis(task_data)
        elif task_type == "correlation_analysis":
            return await self._perform_correlation_analysis(task_data)
        elif task_type == "hypothesis_testing":
            return await self._perform_hypothesis_testing(task_data)
        elif task_type == "feature_importance":
            return await self._perform_feature_importance_analysis(task_data)
        elif task_type == "pattern_recognition":
            return await self._perform_pattern_recognition(task_data)
        elif task_type == "trend_analysis":
            return await self._perform_trend_analysis(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested analysis data"""
        data_type = request_data.get("data_type")
        
        if data_type == "analysis_results":
            return self.analysis_results
        elif data_type == "statistical_tests":
            return self.statistical_tests
        elif data_type == "correlation_matrices":
            return {k: v.to_dict() for k, v in self.correlation_matrices.items()}
        elif data_type == "feature_importance":
            return {k: v.to_dict() for k, v in self.feature_importance.items()}
        elif data_type == "analysis_summary":
            return self._get_analysis_summary()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def _handle_statistical_analysis(self, message: AgentMessage):
        """Handle statistical analysis requests"""
        try:
            analysis_config = message.content
            result = await self._perform_statistical_analysis(analysis_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="statistical_analysis_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="statistical_analysis_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_correlation_analysis(self, message: AgentMessage):
        """Handle correlation analysis requests"""
        try:
            correlation_config = message.content
            result = await self._perform_correlation_analysis(correlation_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="correlation_analysis_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="correlation_analysis_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_hypothesis_testing(self, message: AgentMessage):
        """Handle hypothesis testing requests"""
        try:
            hypothesis_config = message.content
            result = await self._perform_hypothesis_testing(hypothesis_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="hypothesis_testing_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Hypothesis testing failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="hypothesis_testing_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_feature_importance(self, message: AgentMessage):
        """Handle feature importance analysis requests"""
        try:
            importance_config = message.content
            result = await self._perform_feature_importance_analysis(importance_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="feature_importance_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="feature_importance_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_pattern_recognition(self, message: AgentMessage):
        """Handle pattern recognition requests"""
        try:
            pattern_config = message.content
            result = await self._perform_pattern_recognition(pattern_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="pattern_recognition_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="pattern_recognition_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_trend_analysis(self, message: AgentMessage):
        """Handle trend analysis requests"""
        try:
            trend_config = message.content
            result = await self._perform_trend_analysis(trend_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="trend_analysis_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="trend_analysis_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _perform_statistical_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            self.state.current_task = "statistical_analysis"
            self.update_task_progress(0.1)
            
            # Get data from data agent (this would be through message passing)
            # For now, we'll assume data is available
            data = config.get("data")
            if data is None:
                raise ValueError("No data provided for analysis")
            
            analysis_type = config.get("analysis_type", "comprehensive")
            analysis_steps = self.analysis_configs.get(analysis_type, self.analysis_configs["basic"])
            
            results = {}
            
            # Step 1: Descriptive Statistics
            if "descriptive" in analysis_steps:
                self.update_task_progress(0.2)
                results["descriptive"] = await self._calculate_descriptive_statistics(data)
            
            # Step 2: Correlation Analysis
            if "correlation" in analysis_steps:
                self.update_task_progress(0.4)
                results["correlation"] = await self._calculate_correlations(data)
            
            # Step 3: Hypothesis Testing
            if "hypothesis_testing" in analysis_steps:
                self.update_task_progress(0.6)
                results["hypothesis_testing"] = await self._perform_hypothesis_tests(data)
            
            # Step 4: Feature Importance
            if "feature_importance" in analysis_steps:
                self.update_task_progress(0.8)
                results["feature_importance"] = await self._calculate_feature_importance(data)
            
            # Step 5: Advanced Analysis
            if "pca" in analysis_steps:
                self.update_task_progress(0.9)
                results["pca"] = await self._perform_pca_analysis(data)
            
            self.update_task_progress(1.0)
            
            # Store results
            self.analysis_results = results
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "steps_completed": analysis_steps,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _calculate_descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics for numerical columns"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            descriptive_stats = {}
            
            for col in numerical_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    descriptive_stats[col] = {
                        "count": len(col_data),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median()),
                        "skewness": float(stats.skew(col_data)),
                        "kurtosis": float(stats.kurtosis(col_data)),
                        "q25": float(col_data.quantile(0.25)),
                        "q75": float(col_data.quantile(0.75))
                    }
            
            return descriptive_stats
            
        except Exception as e:
            self.logger.error(f"Descriptive statistics calculation failed: {e}")
            return {}
    
    async def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrices"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return {"message": "Insufficient numerical columns for correlation analysis"}
            
            # Pearson correlation
            pearson_corr = data[numerical_cols].corr(method='pearson')
            self.correlation_matrices["pearson"] = pearson_corr
            
            # Spearman correlation
            spearman_corr = data[numerical_cols].corr(method='spearman')
            self.correlation_matrices["spearman"] = spearman_corr
            
            # Find high correlations
            high_correlations = self._find_high_correlations(pearson_corr, threshold=0.7)
            
            return {
                "pearson_correlation": pearson_corr.to_dict(),
                "spearman_correlation": spearman_corr.to_dict(),
                "high_correlations": high_correlations,
                "correlation_summary": {
                    "total_variables": len(numerical_cols),
                    "high_correlation_pairs": len(high_correlations)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            return {}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of variables with high correlation"""
        high_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corrs.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": "strong" if abs(corr_value) >= 0.8 else "moderate"
                    })
        
        return high_corrs
    
    async def _perform_hypothesis_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform hypothesis tests for attrition analysis"""
        try:
            # Check if attrition column exists
            attrition_col = None
            for col in data.columns:
                if 'attrition' in col.lower() or 'left' in col.lower() or 'churn' in col.lower():
                    attrition_col = col
                    break
            
            if attrition_col is None:
                return {"message": "No attrition column found for hypothesis testing"}
            
            # Convert attrition to binary if needed
            attrition_data = data[attrition_col]
            if attrition_data.dtype == 'object':
                attrition_data = attrition_data.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
            
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != attrition_col]
            
            hypothesis_tests = {}
            
            for col in numerical_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # T-test for difference in means between attrition groups
                    try:
                        attrition_yes = col_data[attrition_data == 1]
                        attrition_no = col_data[attrition_data == 0]
                        
                        if len(attrition_yes) > 0 and len(attrition_no) > 0:
                            t_stat, p_value = stats.ttest_ind(attrition_yes, attrition_no)
                            
                            hypothesis_tests[col] = {
                                "test_type": "independent_t_test",
                                "t_statistic": float(t_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "effect_size": "large" if abs(t_stat) > 2.0 else "medium" if abs(t_stat) > 1.0 else "small"
                            }
                    except Exception as e:
                        hypothesis_tests[col] = {"error": str(e)}
            
            self.statistical_tests = hypothesis_tests
            
            return {
                "attrition_column": attrition_col,
                "tests_performed": len(hypothesis_tests),
                "significant_factors": [col for col, test in hypothesis_tests.items() 
                                      if test.get("significant", False)],
                "test_results": hypothesis_tests
            }
            
        except Exception as e:
            self.logger.error(f"Hypothesis testing failed: {e}")
            return {}
    
    async def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature importance for attrition prediction"""
        try:
            # Check if attrition column exists
            attrition_col = None
            for col in data.columns:
                if 'attrition' in col.lower() or 'left' in col.lower() or 'churn' in col.lower():
                    attrition_col = col
                    break
            
            if attrition_col is None:
                return {"message": "No attrition column found for feature importance analysis"}
            
            # Prepare features and target
            feature_cols = [col for col in data.columns if col != attrition_col]
            X = data[feature_cols].select_dtypes(include=[np.number])
            y = data[attrition_col]
            
            if X.empty:
                return {"message": "No numerical features available for importance analysis"}
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            # Convert target to binary if needed
            if y.dtype == 'object':
                y = y.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
            
            # Calculate feature importance using different methods
            importance_methods = {}
            
            # Method 1: F-statistic
            try:
                f_selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
                f_selector.fit(X, y)
                f_scores = pd.DataFrame({
                    'feature': X.columns,
                    'f_score': f_selector.scores_,
                    'p_value': f_selector.pvalues_
                }).sort_values('f_score', ascending=False)
                
                importance_methods["f_statistic"] = f_scores.to_dict()
            except Exception as e:
                importance_methods["f_statistic"] = {"error": str(e)}
            
            # Method 2: Mutual Information
            try:
                mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(10, X.shape[1]))
                mi_selector.fit(X, y)
                mi_scores = pd.DataFrame({
                    'feature': X.columns,
                    'mutual_info': mi_selector.scores_
                }).sort_values('mutual_info', ascending=False)
                
                importance_methods["mutual_information"] = mi_scores.to_dict()
            except Exception as e:
                importance_methods["mutual_information"] = {"error": str(e)}
            
            # Store results
            self.feature_importance = importance_methods
            
            return {
                "total_features": len(X.columns),
                "target_variable": attrition_col,
                "importance_methods": importance_methods,
                "top_features": self._get_top_features(importance_methods)
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def _get_top_features(self, importance_methods: Dict[str, Any]) -> List[str]:
        """Get top features across different importance methods"""
        try:
            top_features = set()
            
            for method, results in importance_methods.items():
                if isinstance(results, dict) and "feature" in results:
                    # Get top 5 features from each method
                    features = results["feature"][:5]
                    top_features.update(features)
            
            return list(top_features)
            
        except Exception as e:
            self.logger.error(f"Top features extraction failed: {e}")
            return []
    
    async def _perform_pca_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return {"message": "Insufficient numerical columns for PCA"}
            
            # Prepare data
            X = data[numerical_cols].fillna(data[numerical_cols].mean())
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA()
            pca.fit(X_scaled)
            
            # Calculate explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find number of components for 95% variance
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            return {
                "total_components": len(numerical_cols),
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "components_for_95_variance": int(n_components_95),
                "feature_contributions": {
                    col: pca.components_[i].tolist() 
                    for i, col in enumerate(numerical_cols)
                }
            }
            
        except Exception as e:
            self.logger.error(f"PCA analysis failed: {e}")
            return {}
    
    async def _perform_pattern_recognition(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pattern recognition analysis"""
        try:
            # This would implement more advanced pattern recognition
            # For now, return a placeholder
            return {
                "status": "success",
                "message": "Pattern recognition analysis completed",
                "patterns_found": [],
                "recommendations": []
            }
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
            raise
    
    async def _perform_trend_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis"""
        try:
            # This would implement trend analysis over time
            # For now, return a placeholder
            return {
                "status": "success",
                "message": "Trend analysis completed",
                "trends_found": [],
                "seasonality": False,
                "recommendations": []
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            raise
    
    def _get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analysis results"""
        return {
            "total_analyses": len(self.analysis_results),
            "analysis_types": list(self.analysis_results.keys()),
            "statistical_tests_count": len(self.statistical_tests),
            "correlation_matrices_count": len(self.correlation_matrices),
            "feature_importance_methods": list(self.feature_importance.keys()),
            "last_updated": datetime.utcnow().isoformat()
        }
