"""
Insight Agent for generating business insights and actionable recommendations
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentMessage
from core.config import Config


@dataclass
class BusinessInsight:
    """Represents a business insight"""
    insight_id: str
    insight_type: str  # risk, opportunity, trend, recommendation
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 10.0
    evidence: List[str]
    recommendations: List[str]
    created_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class RiskAssessment:
    """Represents a risk assessment"""
    risk_id: str
    risk_type: str
    description: str
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 10.0
    risk_score: float  # probability * impact
    mitigation_strategies: List[str]
    monitoring_metrics: List[str]
    created_at: datetime


class InsightAgent(BaseAgent):
    """
    Agent responsible for generating business insights and actionable recommendations
    """
    
    def __init__(self, config: Config, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        
        # Insight storage
        self.business_insights: List[BusinessInsight] = []
        self.risk_assessments: List[RiskAssessment] = []
        self.recommendations: List[Dict[str, Any]] = []
        
        # Insight generation configuration
        self.insight_configs = {
            "basic": ["risk_identification", "basic_recommendations"],
            "comprehensive": ["risk_identification", "trend_analysis", "detailed_recommendations", "impact_analysis"],
            "advanced": ["risk_identification", "trend_analysis", "detailed_recommendations", "impact_analysis", "predictive_insights", "cost_benefit_analysis"]
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        }
        
        # Setup insight-specific message handlers
        self._setup_insight_handlers()
    
    def _setup_insight_handlers(self):
        """Setup insight-specific message handlers"""
        self.message_handlers.update({
            "generate_insights": self._handle_generate_insights,
            "risk_assessment": self._handle_risk_assessment,
            "trend_analysis": self._handle_trend_analysis,
            "recommendations": self._handle_recommendations,
            "impact_analysis": self._handle_impact_analysis,
            "cost_benefit_analysis": self._handle_cost_benefit_analysis,
        })
    
    async def _initialize_agent(self):
        """Initialize insight agent specific components"""
        try:
            # Initialize insight generation tools
            self.logger.info("Insight agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize insight agent components: {e}")
            raise
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute insight-related tasks"""
        task_type = task_data.get("task_type")
        
        if task_type == "generate_insights":
            return await self._generate_comprehensive_insights(task_data)
        elif task_type == "risk_assessment":
            return await self._perform_risk_assessment(task_data)
        elif task_type == "trend_analysis":
            return await self._analyze_trends(task_data)
        elif task_type == "recommendations":
            return await self._generate_recommendations(task_data)
        elif task_type == "impact_analysis":
            return await self._analyze_impact(task_data)
        elif task_type == "cost_benefit_analysis":
            return await self._perform_cost_benefit_analysis(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested insight data"""
        data_type = request_data.get("data_type")
        
        if data_type == "business_insights":
            return [insight.__dict__ for insight in self.business_insights]
        elif data_type == "risk_assessments":
            return [risk.__dict__ for risk in self.risk_assessments]
        elif data_type == "recommendations":
            return self.recommendations
        elif data_type == "insight_summary":
            return self._get_insight_summary()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def _handle_generate_insights(self, message: AgentMessage):
        """Handle insight generation requests"""
        try:
            insight_config = message.content
            result = await self._generate_comprehensive_insights(insight_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="insight_generation_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="insight_generation_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_risk_assessment(self, message: AgentMessage):
        """Handle risk assessment requests"""
        try:
            risk_config = message.content
            result = await self._perform_risk_assessment(risk_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="risk_assessment_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="risk_assessment_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_trend_analysis(self, message: AgentMessage):
        """Handle trend analysis requests"""
        try:
            trend_config = message.content
            result = await self._analyze_trends(trend_config)
            
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
    
    async def _handle_recommendations(self, message: AgentMessage):
        """Handle recommendation generation requests"""
        try:
            rec_config = message.content
            result = await self._generate_recommendations(rec_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="recommendations_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="recommendations_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_impact_analysis(self, message: AgentMessage):
        """Handle impact analysis requests"""
        try:
            impact_config = message.content
            result = await self._analyze_impact(impact_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="impact_analysis_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Impact analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="impact_analysis_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_cost_benefit_analysis(self, message: AgentMessage):
        """Handle cost-benefit analysis requests"""
        try:
            cba_config = message.content
            result = await self._perform_cost_benefit_analysis(cba_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="cost_benefit_analysis_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Cost-benefit analysis failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="cost_benefit_analysis_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _generate_comprehensive_insights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        try:
            self.state.current_task = "generating_insights"
            self.update_task_progress(0.1)
            
            # Get analysis results
            analysis_results = config.get("analysis_results", {})
            if not analysis_results:
                raise ValueError("No analysis results provided for insight generation")
            
            insight_type = config.get("insight_type", "comprehensive")
            insight_steps = self.insight_configs.get(insight_type, self.insight_configs["basic"])
            
            insights = []
            
            # Step 1: Risk Identification
            if "risk_identification" in insight_steps:
                self.update_task_progress(0.2)
                risk_insights = await self._identify_risks(analysis_results)
                insights.extend(risk_insights)
            
            # Step 2: Trend Analysis
            if "trend_analysis" in insight_steps:
                self.update_task_progress(0.4)
                trend_insights = await self._identify_trends(analysis_results)
                insights.extend(trend_insights)
            
            # Step 3: Detailed Recommendations
            if "detailed_recommendations" in insight_steps:
                self.update_task_progress(0.6)
                recommendation_insights = await self._generate_detailed_recommendations(analysis_results)
                insights.extend(recommendation_insights)
            
            # Step 4: Impact Analysis
            if "impact_analysis" in insight_steps:
                self.update_task_progress(0.8)
                impact_insights = await self._analyze_business_impact(analysis_results)
                insights.extend(impact_insights)
            
            # Step 5: Advanced Insights
            if "predictive_insights" in insight_steps:
                self.update_task_progress(0.9)
                predictive_insights = await self._generate_predictive_insights(analysis_results)
                insights.extend(predictive_insights)
            
            self.update_task_progress(1.0)
            
            # Store insights
            self.business_insights.extend(insights)
            
            return {
                "status": "success",
                "insights_generated": len(insights),
                "insight_types": [insight.insight_type for insight in insights],
                "insights": [insight.__dict__ for insight in insights],
                "summary": await self._generate_insight_summary_text(insights)
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive insight generation failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _identify_risks(self, analysis_results: Dict[str, Any]) -> List[BusinessInsight]:
        """Identify potential risks from analysis results"""
        try:
            risks = []
            
            # Analyze statistical results
            if "hypothesis_testing" in analysis_results:
                hypothesis_results = analysis_results["hypothesis_testing"]
                significant_factors = hypothesis_results.get("significant_factors", [])
                
                for factor in significant_factors:
                    risk = BusinessInsight(
                        insight_id=f"risk_{factor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="risk",
                        title=f"High Attrition Risk: {factor}",
                        description=f"The factor '{factor}' shows statistically significant correlation with attrition, indicating a potential risk area.",
                        severity="high",
                        confidence=0.85,
                        impact_score=8.0,
                        evidence=[f"Statistical significance confirmed for {factor}"],
                        recommendations=[
                            f"Investigate root causes of {factor}",
                            f"Implement monitoring for {factor}",
                            f"Develop intervention strategies for {factor}"
                        ],
                        created_at=datetime.utcnow(),
                        tags=["risk", "attrition", "statistical"],
                        metadata={"factor": factor, "analysis_type": "hypothesis_testing"}
                    )
                    risks.append(risk)
            
            # Analyze correlation results
            if "correlation" in analysis_results:
                correlation_results = analysis_results["correlation"]
                high_correlations = correlation_results.get("high_correlations", [])
                
                for corr in high_correlations:
                    if corr.get("correlation", 0) > 0.8:
                        risk = BusinessInsight(
                            insight_id=f"corr_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            insight_type="risk",
                            title=f"Strong Correlation Risk: {corr['variable1']} and {corr['variable2']}",
                            description=f"Strong correlation between {corr['variable1']} and {corr['variable2']} may indicate redundancy or confounding factors.",
                            severity="medium",
                            confidence=0.75,
                            impact_score=6.0,
                            evidence=[f"Correlation coefficient: {corr['correlation']:.3f}"],
                            recommendations=[
                                "Investigate if both variables are necessary",
                                "Consider feature engineering to reduce redundancy",
                                "Monitor for multicollinearity effects"
                            ],
                            created_at=datetime.utcnow(),
                            tags=["risk", "correlation", "multicollinearity"],
                            metadata=corr
                        )
                        risks.append(risk)
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Risk identification failed: {e}")
            return []
    
    async def _identify_trends(self, analysis_results: Dict[str, Any]) -> List[BusinessInsight]:
        """Identify trends from analysis results"""
        try:
            trends = []
            
            # Analyze descriptive statistics
            if "descriptive" in analysis_results:
                descriptive_results = analysis_results["descriptive"]
                
                for variable, stats in descriptive_results.items():
                    # Check for skewed distributions
                    skewness = stats.get("skewness", 0)
                    if abs(skewness) > 1.0:
                        trend = BusinessInsight(
                            insight_id=f"trend_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            insight_type="trend",
                            title=f"Distribution Skewness: {variable}",
                            description=f"The variable '{variable}' shows a skewed distribution (skewness: {skewness:.3f}), which may indicate underlying patterns or data quality issues.",
                            severity="medium" if abs(skewness) < 2.0 else "high",
                            confidence=0.80,
                            impact_score=5.0,
                            evidence=[f"Skewness coefficient: {skewness:.3f}"],
                            recommendations=[
                                f"Investigate causes of skewness in {variable}",
                                "Consider data transformation if appropriate",
                                "Review data collection methods for {variable}"
                            ],
                            created_at=datetime.utcnow(),
                            tags=["trend", "distribution", "skewness"],
                            metadata={"variable": variable, "skewness": skewness}
                        )
                        trends.append(trend)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend identification failed: {e}")
            return []
    
    async def _generate_detailed_recommendations(self, analysis_results: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate detailed recommendations based on analysis results"""
        try:
            recommendations = []
            
            # Generate recommendations based on feature importance
            if "feature_importance" in analysis_results:
                importance_results = analysis_results["feature_importance"]
                top_features = importance_results.get("top_features", [])
                
                for i, feature in enumerate(top_features[:5]):  # Top 5 features
                    priority = "high" if i < 2 else "medium"
                    impact_score = 9.0 if i < 2 else 7.0
                    
                    rec = BusinessInsight(
                        insight_id=f"rec_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="recommendation",
                        title=f"Focus on {feature} for Attrition Prevention",
                        description=f"The feature '{feature}' is among the top predictors of attrition. Focused interventions on this factor could significantly reduce attrition rates.",
                        severity=priority,
                        confidence=0.90,
                        impact_score=impact_score,
                        evidence=[f"Feature importance ranking: {i+1}"],
                        recommendations=[
                            f"Develop targeted interventions for {feature}",
                            f"Monitor {feature} trends regularly",
                            f"Train managers on {feature} management",
                            f"Set KPIs related to {feature} improvement"
                        ],
                        created_at=datetime.utcnow(),
                        tags=["recommendation", "feature_importance", "intervention"],
                        metadata={"feature": feature, "ranking": i+1}
                    )
                    recommendations.append(rec)
            
            # Generate general recommendations
            general_rec = BusinessInsight(
                insight_id=f"general_rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="recommendation",
                title="Implement Comprehensive Attrition Prevention Strategy",
                description="Based on the analysis, implement a multi-faceted approach to reduce attrition rates.",
                severity="high",
                confidence=0.85,
                impact_score=9.0,
                evidence=["Statistical analysis completed", "Key factors identified"],
                recommendations=[
                    "Establish regular employee satisfaction surveys",
                    "Implement mentorship programs",
                    "Review compensation and benefits",
                    "Enhance career development opportunities",
                    "Improve work-life balance policies",
                    "Strengthen employee recognition programs"
                ],
                created_at=datetime.utcnow(),
                tags=["recommendation", "strategy", "comprehensive"],
                metadata={"recommendation_type": "general_strategy"}
            )
            recommendations.append(general_rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Detailed recommendation generation failed: {e}")
            return []
    
    async def _analyze_business_impact(self, analysis_results: Dict[str, Any]) -> List[BusinessInsight]:
        """Analyze business impact of attrition factors"""
        try:
            impact_insights = []
            
            # Analyze impact based on statistical significance
            if "hypothesis_testing" in analysis_results:
                hypothesis_results = analysis_results["hypothesis_testing"]
                significant_factors = hypothesis_results.get("significant_factors", [])
                
                for factor in significant_factors:
                    impact = BusinessInsight(
                        insight_id=f"impact_{factor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="impact",
                        title=f"Business Impact: {factor}",
                        description=f"The factor '{factor}' has significant impact on attrition, affecting business operations and costs.",
                        severity="high",
                        confidence=0.80,
                        impact_score=8.5,
                        evidence=[f"Statistically significant impact confirmed for {factor}"],
                        recommendations=[
                            f"Quantify financial impact of {factor}",
                            f"Develop ROI analysis for {factor} interventions",
                            f"Set improvement targets for {factor}",
                            f"Monitor business metrics related to {factor}"
                        ],
                        created_at=datetime.utcnow(),
                        tags=["impact", "business", "financial"],
                        metadata={"factor": factor, "impact_type": "statistical"}
                    )
                    impact_insights.append(impact)
            
            return impact_insights
            
        except Exception as e:
            self.logger.error(f"Business impact analysis failed: {e}")
            return []
    
    async def _generate_predictive_insights(self, analysis_results: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate predictive insights based on analysis results"""
        try:
            predictive_insights = []
            
            # Generate insights based on model performance
            if "prediction_phase" in analysis_results:
                prediction_results = analysis_results["prediction_phase"]
                
                if "training" in prediction_results:
                    training_results = prediction_results["training"]
                    model_performance = training_results.get("performance_metrics", {})
                    
                    # Analyze model accuracy
                    accuracy = model_performance.get("accuracy", 0)
                    if accuracy > 0.8:
                        insight = BusinessInsight(
                            insight_id=f"pred_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            insight_type="predictive",
                            title="High Prediction Accuracy Achieved",
                            description=f"The attrition prediction model achieved {accuracy:.1%} accuracy, enabling proactive intervention strategies.",
                            severity="low",
                            confidence=0.90,
                            impact_score=7.0,
                            evidence=[f"Model accuracy: {accuracy:.1%}"],
                            recommendations=[
                                "Implement proactive attrition prevention",
                                "Use model for early warning systems",
                                "Develop targeted intervention programs",
                                "Monitor prediction accuracy over time"
                            ],
                            created_at=datetime.utcnow(),
                            tags=["predictive", "accuracy", "proactive"],
                            metadata={"accuracy": accuracy, "model_type": "attrition_prediction"}
                        )
                        predictive_insights.append(insight)
            
            return predictive_insights
            
        except Exception as e:
            self.logger.error(f"Predictive insight generation failed: {e}")
            return []
    
    async def _perform_risk_assessment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            self.state.current_task = "risk_assessment"
            self.update_task_progress(0.1)
            
            # Get analysis results
            analysis_results = config.get("analysis_results", {})
            
            # Identify risks
            self.update_task_progress(0.3)
            risks = await self._identify_risks(analysis_results)
            
            # Assess risk levels
            self.update_task_progress(0.6)
            risk_assessments = []
            
            for risk in risks:
                # Calculate risk score based on severity and confidence
                severity_scores = {"low": 1, "medium": 3, "high": 6, "critical": 10}
                risk_score = severity_scores.get(risk.severity, 5) * risk.confidence
                
                assessment = RiskAssessment(
                    risk_id=risk.insight_id.replace("risk_", "risk_assessment_"),
                    risk_type=risk.insight_type,
                    description=risk.description,
                    probability=risk.confidence,
                    impact=risk.impact_score,
                    risk_score=risk_score,
                    mitigation_strategies=risk.recommendations,
                    monitoring_metrics=[f"Monitor {risk.title}"],
                    created_at=datetime.utcnow()
                )
                risk_assessments.append(assessment)
            
            # Store risk assessments
            self.risk_assessments.extend(risk_assessments)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "risks_identified": len(risks),
                "risk_assessments": [risk.__dict__ for risk in risk_assessments],
                "high_risk_count": len([r for r in risk_assessments if r.risk_score > 5]),
                "recommendations": await self._generate_risk_mitigation_recommendations(risk_assessments)
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _analyze_trends(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        try:
            self.state.current_task = "trend_analysis"
            self.update_task_progress(0.1)
            
            # Get analysis results
            analysis_results = config.get("analysis_results", {})
            
            # Identify trends
            self.update_task_progress(0.5)
            trends = await self._identify_trends(analysis_results)
            
            # Analyze trend patterns
            self.update_task_progress(0.8)
            trend_patterns = await self._identify_trend_patterns(trends)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "trends_identified": len(trends),
                "trend_patterns": trend_patterns,
                "trends": [trend.__dict__ for trend in trends],
                "recommendations": await self._generate_trend_based_recommendations(trends)
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _generate_recommendations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        try:
            self.state.current_task = "generating_recommendations"
            self.update_task_progress(0.1)
            
            # Get analysis results
            analysis_results = config.get("analysis_results", {})
            
            # Generate recommendations
            self.update_task_progress(0.5)
            recommendations = await self._generate_detailed_recommendations(analysis_results)
            
            # Prioritize recommendations
            self.update_task_progress(0.8)
            prioritized_recs = await self._prioritize_recommendations(recommendations)
            
            # Store recommendations
            self.recommendations = prioritized_recs
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "recommendations_generated": len(recommendations),
                "prioritized_recommendations": prioritized_recs,
                "implementation_plan": await self._generate_implementation_plan(prioritized_recs)
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _analyze_impact(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze business impact of insights"""
        try:
            self.state.current_task = "impact_analysis"
            self.update_task_progress(0.1)
            
            # Get analysis results
            analysis_results = config.get("analysis_results", {})
            
            # Analyze impact
            self.update_task_progress(0.5)
            impact_insights = await self._analyze_business_impact(analysis_results)
            
            # Quantify impact
            self.update_task_progress(0.8)
            impact_quantification = await self._quantify_impact(impact_insights)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "impact_insights": len(impact_insights),
                "impact_quantification": impact_quantification,
                "insights": [insight.__dict__ for insight in impact_insights],
                "recommendations": await self._generate_impact_based_recommendations(impact_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Impact analysis failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _perform_cost_benefit_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cost-benefit analysis for recommendations"""
        try:
            self.state.current_task = "cost_benefit_analysis"
            self.update_task_progress(0.1)
            
            # Get recommendations
            recommendations = config.get("recommendations", self.recommendations)
            
            # Perform CBA
            self.update_task_progress(0.5)
            cba_results = await self._calculate_cost_benefit(recommendations)
            
            # Generate ROI analysis
            self.update_task_progress(0.8)
            roi_analysis = await self._calculate_roi(cba_results)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "cba_results": cba_results,
                "roi_analysis": roi_analysis,
                "recommendations": await self._prioritize_by_roi(cba_results)
            }
            
        except Exception as e:
            self.logger.error(f"Cost-benefit analysis failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    # Helper methods for insight generation
    async def _identify_trend_patterns(self, trends: List[BusinessInsight]) -> Dict[str, Any]:
        """Identify patterns in trends"""
        try:
            patterns = {
                "distribution_issues": len([t for t in trends if "distribution" in t.title.lower()]),
                "data_quality_concerns": len([t for t in trends if "quality" in t.description.lower()]),
                "skewness_patterns": len([t for t in trends if "skew" in t.title.lower()])
            }
            return patterns
        except Exception as e:
            self.logger.error(f"Trend pattern identification failed: {e}")
            return {}
    
    async def _prioritize_recommendations(self, recommendations: List[BusinessInsight]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and feasibility"""
        try:
            prioritized = []
            for rec in recommendations:
                priority_score = rec.impact_score * rec.confidence
                prioritized.append({
                    "insight_id": rec.insight_id,
                    "title": rec.title,
                    "priority_score": priority_score,
                    "severity": rec.severity,
                    "recommendations": rec.recommendations,
                    "implementation_effort": self._estimate_implementation_effort(rec)
                })
            
            # Sort by priority score
            prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
            return prioritized
            
        except Exception as e:
            self.logger.error(f"Recommendation prioritization failed: {e}")
            return []
    
    async def _generate_implementation_plan(self, prioritized_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate implementation plan for recommendations"""
        try:
            phases = {
                "phase_1": [rec for rec in prioritized_recs[:3]],  # Top 3
                "phase_2": [rec for rec in prioritized_recs[3:6]],  # Next 3
                "phase_3": [rec for rec in prioritized_recs[6:]]   # Remaining
            }
            
            timeline = {
                "phase_1": "Immediate (0-3 months)",
                "phase_2": "Short-term (3-6 months)",
                "phase_3": "Long-term (6+ months)"
            }
            
            return {
                "phases": phases,
                "timeline": timeline,
                "total_recommendations": len(prioritized_recs)
            }
            
        except Exception as e:
            self.logger.error(f"Implementation plan generation failed: {e}")
            return {}
    
    async def _quantify_impact(self, impact_insights: List[BusinessInsight]) -> Dict[str, Any]:
        """Quantify the business impact of insights"""
        try:
            total_impact_score = sum([insight.impact_score for insight in impact_insights])
            avg_confidence = np.mean([insight.confidence for insight in impact_insights])
            
            return {
                "total_impact_score": total_impact_score,
                "average_confidence": avg_confidence,
                "high_impact_insights": len([i for i in impact_insights if i.impact_score > 7.0]),
                "estimated_attrition_reduction": f"{min(30, total_impact_score * 2):.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Impact quantification failed: {e}")
            return {}
    
    async def _calculate_cost_benefit(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate cost-benefit for recommendations"""
        try:
            cba_results = []
            for rec in recommendations:
                # Simplified cost-benefit calculation
                estimated_cost = self._estimate_implementation_cost(rec)
                estimated_benefit = rec.get("priority_score", 0) * 1000  # $1000 per priority point
                roi = (estimated_benefit - estimated_cost) / estimated_cost if estimated_cost > 0 else 0
                
                cba_results.append({
                    "insight_id": rec["insight_id"],
                    "title": rec["title"],
                    "estimated_cost": estimated_cost,
                    "estimated_benefit": estimated_benefit,
                    "roi": roi,
                    "payback_period": estimated_cost / (estimated_benefit / 12) if estimated_benefit > 0 else float('inf')
                })
            
            return cba_results
            
        except Exception as e:
            self.logger.error(f"Cost-benefit calculation failed: {e}")
            return []
    
    async def _calculate_roi(self, cba_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ROI analysis"""
        try:
            total_cost = sum([r["estimated_cost"] for r in cba_results])
            total_benefit = sum([r["estimated_benefit"] for r in cba_results])
            overall_roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0
            
            return {
                "total_cost": total_cost,
                "total_benefit": total_benefit,
                "overall_roi": overall_roi,
                "high_roi_recommendations": len([r for r in cba_results if r["roi"] > 2.0]),
                "break_even_point": total_cost / (total_benefit / 12) if total_benefit > 0 else float('inf')
            }
            
        except Exception as e:
            self.logger.error(f"ROI calculation failed: {e}")
            return {}
    
    async def _prioritize_by_roi(self, cba_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by ROI"""
        try:
            return sorted(cba_results, key=lambda x: x["roi"], reverse=True)
        except Exception as e:
            self.logger.error(f"ROI-based prioritization failed: {e}")
            return cba_results
    
    # Helper utility methods
    def _estimate_implementation_effort(self, insight: BusinessInsight) -> str:
        """Estimate implementation effort for a recommendation"""
        if insight.impact_score > 8.0:
            return "High"
        elif insight.impact_score > 6.0:
            return "Medium"
        else:
            return "Low"
    
    def _estimate_implementation_cost(self, recommendation: Dict[str, Any]) -> float:
        """Estimate implementation cost for a recommendation"""
        effort = recommendation.get("implementation_effort", "Medium")
        base_cost = {"Low": 5000, "Medium": 15000, "High": 50000}
        return base_cost.get(effort, 15000)
    
    async def _generate_risk_mitigation_recommendations(self, risk_assessments: List[RiskAssessment]) -> List[str]:
        """Generate risk mitigation recommendations"""
        try:
            recommendations = []
            high_risks = [r for r in risk_assessments if r.risk_score > 5]
            
            if high_risks:
                recommendations.append(f"Immediate attention required for {len(high_risks)} high-risk areas")
                recommendations.append("Implement risk monitoring dashboard")
                recommendations.append("Develop contingency plans for critical risks")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Risk mitigation recommendation generation failed: {e}")
            return []
    
    async def _generate_trend_based_recommendations(self, trends: List[BusinessInsight]) -> List[str]:
        """Generate recommendations based on trends"""
        try:
            recommendations = []
            
            if trends:
                recommendations.append("Monitor identified trends regularly")
                recommendations.append("Investigate root causes of distribution issues")
                recommendations.append("Implement data quality improvements")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Trend-based recommendation generation failed: {e}")
            return []
    
    async def _generate_impact_based_recommendations(self, impact_insights: List[BusinessInsight]) -> List[str]:
        """Generate recommendations based on impact analysis"""
        try:
            recommendations = []
            
            if impact_insights:
                recommendations.append("Prioritize high-impact factors for intervention")
                recommendations.append("Develop ROI analysis for recommended actions")
                recommendations.append("Set measurable improvement targets")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Impact-based recommendation generation failed: {e}")
            return []
    
    async def _generate_insight_summary_text(self, insights: List[BusinessInsight]) -> str:
        """Generate human-readable summary of insights"""
        try:
            if not insights:
                return "No insights generated."
            
            risk_count = len([i for i in insights if i.insight_type == "risk"])
            trend_count = len([i for i in insights if i.insight_type == "trend"])
            rec_count = len([i for i in insights if i.insight_type == "recommendation"])
            
            summary = f"Generated {len(insights)} insights: {risk_count} risks, {trend_count} trends, {rec_count} recommendations. "
            
            if risk_count > 0:
                high_risks = len([i for i in insights if i.insight_type == "risk" and i.severity in ["high", "critical"]])
                summary += f"High/critical risks identified: {high_risks}. "
            
            if rec_count > 0:
                summary += "Actionable recommendations provided for implementation."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Insight summary generation failed: {e}")
            return "Error generating insight summary."
    
    def _get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of all insights"""
        return {
            "total_insights": len(self.business_insights),
            "insight_types": list(set([insight.insight_type for insight in self.business_insights])),
            "risk_assessments_count": len(self.risk_assessments),
            "recommendations_count": len(self.recommendations),
            "high_priority_insights": len([i for i in self.business_insights if i.severity in ["high", "critical"]]),
            "last_updated": datetime.utcnow().isoformat()
        }
