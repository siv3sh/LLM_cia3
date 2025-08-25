"""
Coordinator Agent for orchestrating the multi-agent attrition analysis system
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentMessage
from .data_agent import DataAgent
from .analysis_agent import AnalysisAgent
from .prediction_agent import PredictionAgent
from .insight_agent import InsightAgent
from .chat_agent import ChatAgent
from core.config import Config
from core.workflow import WorkflowManager


@dataclass
class AnalysisJob:
    """Represents an attrition analysis job"""
    job_id: str
    company_name: str
    analysis_type: str  # basic, comprehensive, predictive
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class CoordinatorAgent(BaseAgent):
    """
    Main coordinator agent that orchestrates all other agents
    """
    
    def __init__(self, config: Config, workflow_manager: WorkflowManager, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        
        # Workflow management
        self.workflow_manager = workflow_manager
        
        # Subordinate agents
        self.data_agent: Optional[DataAgent] = None
        self.analysis_agent: Optional[AnalysisAgent] = None
        self.prediction_agent: Optional[PredictionAgent] = None
        self.insight_agent: Optional[InsightAgent] = None
        self.chat_agent: Optional[ChatAgent] = None
        
        # Job management
        self.active_jobs: Dict[str, AnalysisJob] = {}
        self.completed_jobs: Dict[str, AnalysisJob] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        
        # Agent communication
        self.agent_registry: Dict[str, BaseAgent] = {}
        
        # Setup coordinator-specific message handlers
        self._setup_coordinator_handlers()
    
    def _setup_coordinator_handlers(self):
        """Setup coordinator-specific message handlers"""
        self.message_handlers.update({
            "start_analysis": self._handle_start_analysis,
            "job_status": self._handle_job_status,
            "job_results": self._handle_job_results,
            "agent_registration": self._handle_agent_registration,
            "workflow_status": self._handle_workflow_status,
            "system_health": self._handle_system_health,
        })
    
    async def _initialize_agent(self):
        """Initialize coordinator agent and all subordinate agents"""
        try:
            self.logger.info("Initializing coordinator agent and subordinate agents...")
            
            # Initialize subordinate agents
            await self._initialize_subordinate_agents()
            
            # Register agents
            await self._register_agents()
            
            # Setup agent communication
            await self._setup_agent_communication()
            
            # Start background tasks
            asyncio.create_task(self._job_processor())
            asyncio.create_task(self._health_monitor())
            
            self.logger.info("Coordinator agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator agent: {e}")
            raise
    
    async def _initialize_subordinate_agents(self):
        """Initialize all subordinate agents"""
        try:
            # Initialize data agent
            self.data_agent = DataAgent(self.config)
            await self.data_agent.initialize()
            self.logger.info("Data agent initialized")
            
            # Initialize analysis agent
            self.analysis_agent = AnalysisAgent(self.config)
            await self.analysis_agent.initialize()
            self.logger.info("Analysis agent initialized")
            
            # Initialize prediction agent
            self.prediction_agent = PredictionAgent(self.config)
            await self.prediction_agent.initialize()
            self.logger.info("Prediction agent initialized")
            
            # Initialize insight agent
            self.insight_agent = InsightAgent(self.config)
            await self.insight_agent.initialize()
            self.logger.info("Insight agent initialized")

            # Initialize chat agent
            self.chat_agent = ChatAgent(self.config)
            await self.chat_agent.initialize()
            self.logger.info("Chat agent initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subordinate agents: {e}")
            raise
    
    async def _register_agents(self):
        """Register all agents in the agent registry"""
        try:
            agents = [
                ("data_agent", self.data_agent),
                ("analysis_agent", self.analysis_agent),
                ("prediction_agent", self.prediction_agent),
                ("insight_agent", self.insight_agent),
                ("chat_agent", self.chat_agent),
            ]
            
            for agent_name, agent in agents:
                if agent:
                    self.agent_registry[agent_name] = agent
                    await agent.subscribe_to_agent(self.agent_id)
                    await self.subscribe_to_agent(agent.agent_id)
                    self.logger.info(f"Registered agent: {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register agents: {e}")
            raise
    
    async def _setup_agent_communication(self):
        """Setup communication channels between agents"""
        try:
            # Setup message routing
            for agent_name, agent in self.agent_registry.items():
                # Subscribe to relevant message types
                if agent_name == "data_agent":
                    await self._setup_data_agent_communication(agent)
                elif agent_name == "analysis_agent":
                    await self._setup_analysis_agent_communication(agent)
                elif agent_name == "prediction_agent":
                    await self._setup_prediction_agent_communication(agent)
                elif agent_name == "insight_agent":
                    await self._setup_insight_agent_communication(agent)
                elif agent_name == "chat_agent":
                    await self._setup_chat_agent_communication(agent)
            
            self.logger.info("Agent communication setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup agent communication: {e}")
            raise
    
    async def _setup_data_agent_communication(self, agent: DataAgent):
        """Setup communication with data agent"""
        # Data agent can receive requests for data collection, preprocessing, etc.
        pass
    
    async def _setup_analysis_agent_communication(self, agent: AnalysisAgent):
        """Setup communication with analysis agent"""
        # Analysis agent can receive requests for statistical analysis
        pass
    
    async def _setup_prediction_agent_communication(self, agent: PredictionAgent):
        """Setup communication with prediction agent"""
        # Prediction agent can receive requests for model training and predictions
        pass
    
    async def _setup_insight_agent_communication(self, agent: InsightAgent):
        """Setup communication with insight agent"""
        # Insight agent can receive requests for business insights
        pass

    async def _setup_chat_agent_communication(self, agent: ChatAgent):
        """Setup communication with chat agent"""
        # Chat agent can receive requests for user interactions
        pass
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordinator tasks"""
        task_type = task_data.get("task_type")
        
        if task_type == "run_attrition_analysis":
            return await self._run_attrition_analysis(task_data)
        elif task_type == "get_job_status":
            return await self._get_job_status(task_data)
        elif task_type == "get_job_results":
            return await self._get_job_results(task_data)
        elif task_type == "cancel_job":
            return await self._cancel_job(task_data)
        elif task_type == "get_system_status":
            return await self._get_system_status()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested data"""
        data_type = request_data.get("data_type")
        
        if data_type == "active_jobs":
            return [job.__dict__ for job in self.active_jobs.values()]
        elif data_type == "completed_jobs":
            return [job.__dict__ for job in self.completed_jobs.values()]
        elif data_type == "agent_status":
            return {name: agent.get_performance_metrics() for name, agent in self.agent_registry.items()}
        elif data_type == "system_metrics":
            return await self._get_system_metrics()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def _handle_start_analysis(self, message: AgentMessage):
        """Handle start analysis requests"""
        try:
            analysis_config = message.content
            result = await self._run_attrition_analysis(analysis_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="analysis_started_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to start analysis: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="analysis_start_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_job_status(self, message: AgentMessage):
        """Handle job status requests"""
        try:
            job_id = message.content.get("job_id")
            if not job_id:
                raise ValueError("Job ID is required")
            
            status = await self._get_job_status({"job_id": job_id})
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=status,
                message_type="job_status_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="job_status_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_job_results(self, message: AgentMessage):
        """Handle job results requests"""
        try:
            job_id = message.content.get("job_id")
            if not job_id:
                raise ValueError("Job ID is required")
            
            results = await self._get_job_results({"job_id": job_id})
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=results,
                message_type="job_results_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to get job results: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="job_results_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_agent_registration(self, message: AgentMessage):
        """Handle agent registration requests"""
        try:
            agent_info = message.content
            agent_name = agent_info.get("agent_name")
            agent_id = agent_info.get("agent_id")
            
            if agent_name and agent_id:
                # This would typically register external agents
                self.logger.info(f"Agent registration request: {agent_name} ({agent_id})")
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"status": "registration_received"},
                message_type="agent_registration_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to handle agent registration: {e}")
    
    async def _handle_workflow_status(self, message: AgentMessage):
        """Handle workflow status requests"""
        try:
            workflow_status = await self.workflow_manager.get_status()
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=workflow_status,
                message_type="workflow_status_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="workflow_status_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_system_health(self, message: AgentMessage):
        """Handle system health requests"""
        try:
            health_status = await self._get_system_health()
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=health_status,
                message_type="system_health_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="system_health_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _run_attrition_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete attrition analysis"""
        try:
            # Create analysis job
            job = AnalysisJob(
                job_id=str(uuid.uuid4()),
                company_name=config.get("company_name", "Unknown Company"),
                analysis_type=config.get("analysis_type", "comprehensive"),
                status="pending",
                created_at=datetime.utcnow(),
                metadata=config
            )
            
            # Add job to queue
            await self.job_queue.put(job)
            self.active_jobs[job.job_id] = job
            
            self.logger.info(f"Created analysis job {job.job_id} for {job.company_name}")
            
            return {
                "status": "success",
                "job_id": job.job_id,
                "message": f"Analysis job created successfully. Job ID: {job.job_id}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis job: {e}")
            raise
    
    async def _job_processor(self):
        """Background task to process job queue"""
        while self.state.status == "running":
            try:
                if not self.job_queue.empty():
                    job = await self.job_queue.get()
                    await self._process_job(job)
                else:
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in job processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_job(self, job: AnalysisJob):
        """Process a single analysis job"""
        try:
            self.logger.info(f"Processing job {job.job_id}: {job.company_name}")
            
            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.progress = 0.0
            
            # Execute analysis workflow
            results = await self._execute_analysis_workflow(job)
            
            # Update job with results
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            job.results = results
            
            # Move job to completed
            self.completed_jobs[job.job_id] = job
            del self.active_jobs[job.job_id]
            
            self.logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Job {job.job_id} failed: {e}")
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            
            # Move job to completed (failed)
            self.completed_jobs[job.job_id] = job
            del self.active_jobs[job.job_id]
    
    async def _execute_analysis_workflow(self, job: AnalysisJob) -> Dict[str, Any]:
        """Execute the complete analysis workflow"""
        try:
            workflow_config = job.metadata
            results = {}
            
            # Phase 1: Data Collection and Preprocessing
            job.progress = 0.1
            self.logger.info(f"Job {job.job_id}: Starting data collection")
            
            data_result = await self._execute_data_phase(workflow_config)
            results["data_phase"] = data_result
            job.progress = 0.3
            
            # Phase 2: Statistical Analysis
            job.progress = 0.4
            self.logger.info(f"Job {job.job_id}: Starting statistical analysis")
            
            analysis_result = await self._execute_analysis_phase(workflow_config, data_result)
            results["analysis_phase"] = analysis_result
            job.progress = 0.6
            
            # Phase 3: Predictive Modeling
            job.progress = 0.7
            self.logger.info(f"Job {job.job_id}: Starting predictive modeling")
            
            prediction_result = await self._execute_prediction_phase(workflow_config, data_result, analysis_result)
            results["prediction_phase"] = prediction_result
            job.progress = 0.9
            
            # Phase 4: Business Insights
            job.progress = 0.95
            self.logger.info(f"Job {job.job_id}: Generating business insights")
            
            insight_result = await self._execute_insight_phase(workflow_config, results)
            results["insight_phase"] = insight_result
            job.progress = 1.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed for job {job.job_id}: {e}")
            raise
    
    async def _execute_data_phase(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection and preprocessing phase"""
        try:
            # Collect data
            collection_config = {
                "task_type": "collect_data",
                "sources": config.get("data_sources", ["csv"]),
                "csv_paths": config.get("csv_paths", []),
                "database_query": config.get("database_query"),
                "api_endpoints": config.get("api_endpoints", [])
            }
            
            collection_result = await self.data_agent._execute_task(collection_config)
            
            # Preprocess data
            preprocessing_config = {
                "task_type": "preprocess_data"
            }
            
            preprocessing_result = await self.data_agent._execute_task(preprocessing_config)
            
            # Engineer features
            feature_config = {
                "task_type": "engineer_features"
            }
            
            feature_result = await self.data_agent._execute_task(feature_config)
            
            return {
                "collection": collection_result,
                "preprocessing": preprocessing_result,
                "feature_engineering": feature_result
            }
            
        except Exception as e:
            self.logger.error(f"Data phase execution failed: {e}")
            raise
    
    async def _execute_analysis_phase(self, config: Dict[str, Any], data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis phase"""
        try:
            # Perform statistical analysis
            analysis_config = {
                "task_type": "statistical_analysis",
                "analysis_type": config.get("analysis_type", "comprehensive")
            }
            
            analysis_result = await self.analysis_agent._execute_task(analysis_config)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Analysis phase execution failed: {e}")
            raise
    
    async def _execute_prediction_phase(self, config: Dict[str, Any], data_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute predictive modeling phase"""
        try:
            # Train prediction model
            training_config = {
                "task_type": "train_model",
                "model_type": config.get("model_type", "random_forest"),
                "hyperparameters": config.get("hyperparameters", {})
            }
            
            training_result = await self.prediction_agent._execute_task(training_config)
            
            # Make predictions
            prediction_config = {
                "task_type": "make_predictions",
                "model_id": training_result.get("model_id")
            }
            
            prediction_result = await self.prediction_agent._execute_task(prediction_config)
            
            return {
                "training": training_result,
                "predictions": prediction_result
            }
            
        except Exception as e:
            self.logger.error(f"Prediction phase execution failed: {e}")
            raise
    
    async def _execute_insight_phase(self, config: Dict[str, Any], all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute business insights generation phase"""
        try:
            # Generate business insights
            insight_config = {
                "task_type": "generate_insights",
                "analysis_results": all_results,
                "insight_type": config.get("insight_type", "comprehensive")
            }
            
            insight_result = await self.insight_agent._execute_task(insight_config)
            
            return insight_result
            
        except Exception as e:
            self.logger.error(f"Insight phase execution failed: {e}")
            raise
    
    async def _get_job_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a specific job"""
        job_id = config.get("job_id")
        
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "company_name": job.company_name,
                "analysis_type": job.analysis_type,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "estimated_completion": None  # Could be calculated based on progress
            }
        elif job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "company_name": job.company_name,
                "analysis_type": job.analysis_type,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            }
        else:
            raise ValueError(f"Job {job_id} not found")
    
    async def _get_job_results(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get results of a completed job"""
        job_id = config.get("job_id")
        
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            if job.status == "completed":
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "results": job.results,
                    "metadata": job.metadata
                }
            else:
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "error_message": job.error_message
                }
        else:
            raise ValueError(f"Job {job_id} not found or not completed")
    
    async def _cancel_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a running job"""
        job_id = config.get("job_id")
        
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled successfully"
            }
        else:
            raise ValueError(f"Job {job_id} not found or not active")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "coordinator_status": self.state.status,
            "active_jobs_count": len(self.active_jobs),
            "completed_jobs_count": len(self.completed_jobs),
            "queue_size": self.job_queue.qsize(),
            "agents_status": {name: agent.state.status for name, agent in self.agent_registry.items()},
            "system_uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        return {
            "coordinator_metrics": self.get_performance_metrics(),
            "agent_metrics": {name: agent.get_performance_metrics() for name, agent in self.agent_registry.items()},
            "job_metrics": {
                "total_jobs": len(self.active_jobs) + len(self.completed_jobs),
                "success_rate": len([j for j in self.completed_jobs.values() if j.status == "completed"]) / max(len(self.completed_jobs), 1) * 100,
                "average_job_duration": self._calculate_average_job_duration()
            }
        }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            # Check agent health
            agent_health = {}
            for name, agent in self.agent_registry.items():
                try:
                    metrics = agent.get_performance_metrics()
                    agent_health[name] = {
                        "status": agent.state.status,
                        "uptime": metrics.get("uptime_seconds", 0),
                        "error_count": metrics.get("error_count", 0),
                        "success_count": metrics.get("success_count", 0)
                    }
                except Exception as e:
                    agent_health[name] = {"status": "error", "error": str(e)}
            
            # Overall health score
            health_score = self._calculate_health_score(agent_health)
            
            return {
                "overall_health": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy",
                "health_score": health_score,
                "agent_health": agent_health,
                "system_status": self._get_system_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {
                "overall_health": "error",
                "health_score": 0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _calculate_health_score(self, agent_health: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        try:
            if not agent_health:
                return 0.0
            
            total_score = 0.0
            agent_count = len(agent_health)
            
            for agent_name, health in agent_health.items():
                agent_score = 100.0
                
                # Deduct points for errors
                error_count = health.get("error_count", 0)
                agent_score -= min(error_count * 10, 50)  # Max 50 points deduction for errors
                
                # Deduct points for offline status
                if health.get("status") == "offline":
                    agent_score -= 30
                elif health.get("status") == "error":
                    agent_score -= 20
                
                total_score += max(0.0, agent_score)
            
            return total_score / agent_count
            
        except Exception as e:
            self.logger.error(f"Health score calculation failed: {e}")
            return 0.0
    
    def _calculate_average_job_duration(self) -> float:
        """Calculate average job duration in seconds"""
        try:
            completed_jobs = [j for j in self.completed_jobs.values() if j.started_at and j.completed_at]
            
            if not completed_jobs:
                return 0.0
            
            total_duration = sum([
                (j.completed_at - j.started_at).total_seconds() 
                for j in completed_jobs
            ])
            
            return total_duration / len(completed_jobs)
            
        except Exception as e:
            self.logger.error(f"Average job duration calculation failed: {e}")
            return 0.0
    
    async def _health_monitor(self):
        """Background task to monitor system health"""
        while self.state.status == "running":
            try:
                # Check system health every 30 seconds
                health_status = await self._get_system_health()
                
                if health_status["health_score"] < 50:
                    self.logger.warning(f"System health degraded: {health_status['health_score']}")
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup(self):
        """Cleanup resources before stopping"""
        try:
            # Cancel all active jobs
            for job in list(self.active_jobs.values()):
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                self.completed_jobs[job.job_id] = job
            
            self.active_jobs.clear()
            
            # Shutdown subordinate agents
            for agent in self.agent_registry.values():
                try:
                    await agent.shutdown()
                except Exception as e:
                    self.logger.error(f"Failed to shutdown agent {agent.agent_id}: {e}")
            
            self.logger.info("Coordinator agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Coordinator agent cleanup failed: {e}")
    
    async def run_attrition_analysis(self, company_data: str, analysis_type: str = "comprehensive") -> str:
        """Public method to run attrition analysis"""
        config = {
            "company_name": company_data.split("/")[-1].replace(".csv", ""),
            "analysis_type": analysis_type,
            "csv_paths": [company_data],
            "data_sources": ["csv"]
        }
        
        result = await self._run_attrition_analysis(config)
        return result["job_id"]
