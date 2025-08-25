"""
Workflow Manager for orchestrating the multi-agent attrition analysis system
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from core.config import Config


class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowPhase(Enum):
    """Workflow phase enumeration"""
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    INSIGHT_GENERATION = "insight_generation"
    REPORTING = "reporting"


@dataclass
class WorkflowStep:
    """Represents a workflow step"""
    step_id: str
    phase: WorkflowPhase
    name: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Represents a workflow execution"""
    execution_id: str
    workflow_name: str
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    steps: List[WorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class WorkflowManager:
    """
    Manages workflow execution and agent coordination
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Workflow definitions
        self.workflow_definitions: Dict[str, List[WorkflowStep]] = {}
        
        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.completed_executions: Dict[str, WorkflowExecution] = {}
        
        # Workflow templates
        self._setup_workflow_templates()
        
        # Execution queue
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info("Workflow manager initialized")
    
    def _setup_workflow_templates(self):
        """Setup predefined workflow templates"""
        # Basic workflow template
        basic_workflow = [
            WorkflowStep(
                step_id="data_collection",
                phase=WorkflowPhase.DATA_COLLECTION,
                name="Data Collection",
                description="Collect data from various sources",
                dependencies=[],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="data_preprocessing",
                phase=WorkflowPhase.DATA_PREPROCESSING,
                name="Data Preprocessing",
                description="Clean and preprocess collected data",
                dependencies=["data_collection"],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="statistical_analysis",
                phase=WorkflowPhase.STATISTICAL_ANALYSIS,
                name="Statistical Analysis",
                description="Perform statistical analysis on preprocessed data",
                dependencies=["data_preprocessing"],
                agent_id="analysis_agent"
            ),
            WorkflowStep(
                step_id="insight_generation",
                phase=WorkflowPhase.INSIGHT_GENERATION,
                name="Insight Generation",
                description="Generate business insights and recommendations",
                dependencies=["statistical_analysis"],
                agent_id="insight_agent"
            )
        ]
        
        # Comprehensive workflow template
        comprehensive_workflow = [
            WorkflowStep(
                step_id="data_collection",
                phase=WorkflowPhase.DATA_COLLECTION,
                name="Data Collection",
                description="Collect data from various sources",
                dependencies=[],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="data_preprocessing",
                phase=WorkflowPhase.DATA_PREPROCESSING,
                name="Data Preprocessing",
                description="Clean and preprocess collected data",
                dependencies=["data_collection"],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="feature_engineering",
                phase=WorkflowPhase.FEATURE_ENGINEERING,
                name="Feature Engineering",
                description="Create and engineer features for analysis",
                dependencies=["data_preprocessing"],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="statistical_analysis",
                phase=WorkflowPhase.STATISTICAL_ANALYSIS,
                name="Statistical Analysis",
                description="Perform comprehensive statistical analysis",
                dependencies=["feature_engineering"],
                agent_id="analysis_agent"
            ),
            WorkflowStep(
                step_id="model_training",
                phase=WorkflowPhase.MODEL_TRAINING,
                name="Model Training",
                description="Train machine learning models for prediction",
                dependencies=["statistical_analysis"],
                agent_id="prediction_agent"
            ),
            WorkflowStep(
                step_id="prediction",
                phase=WorkflowPhase.PREDICTION,
                name="Prediction",
                description="Make predictions using trained models",
                dependencies=["model_training"],
                agent_id="prediction_agent"
            ),
            WorkflowStep(
                step_id="insight_generation",
                phase=WorkflowPhase.INSIGHT_GENERATION,
                name="Insight Generation",
                description="Generate comprehensive business insights",
                dependencies=["prediction"],
                agent_id="insight_agent"
            ),
            WorkflowStep(
                step_id="reporting",
                phase=WorkflowPhase.REPORTING,
                name="Reporting",
                description="Generate comprehensive analysis report",
                dependencies=["insight_generation"],
                agent_id="insight_agent"
            )
        ]
        
        # Predictive workflow template
        predictive_workflow = [
            WorkflowStep(
                step_id="data_collection",
                phase=WorkflowPhase.DATA_COLLECTION,
                name="Data Collection",
                description="Collect historical and current data",
                dependencies=[],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="data_preprocessing",
                phase=WorkflowPhase.DATA_PREPROCESSING,
                name="Data Preprocessing",
                description="Clean and prepare data for modeling",
                dependencies=["data_collection"],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="feature_engineering",
                phase=WorkflowPhase.FEATURE_ENGINEERING,
                name="Feature Engineering",
                description="Create predictive features",
                dependencies=["data_preprocessing"],
                agent_id="data_agent"
            ),
            WorkflowStep(
                step_id="model_training",
                phase=WorkflowPhase.MODEL_TRAINING,
                name="Model Training",
                description="Train multiple prediction models",
                dependencies=["feature_engineering"],
                agent_id="prediction_agent"
            ),
            WorkflowStep(
                step_id="prediction",
                phase=WorkflowPhase.PREDICTION,
                name="Prediction",
                description="Generate attrition predictions",
                dependencies=["model_training"],
                agent_id="prediction_agent"
            ),
            WorkflowStep(
                step_id="insight_generation",
                phase=WorkflowPhase.INSIGHT_GENERATION,
                name="Predictive Insights",
                description="Generate insights from predictions",
                dependencies=["prediction"],
                agent_id="insight_agent"
            )
        ]
        
        self.workflow_definitions = {
            "basic": basic_workflow,
            "comprehensive": comprehensive_workflow,
            "predictive": predictive_workflow
        }
    
    async def initialize(self):
        """Initialize the workflow manager"""
        try:
            # Start background tasks
            self.background_tasks.append(
                asyncio.create_task(self._workflow_executor())
            )
            
            self.logger.info("Workflow manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow manager: {e}")
            raise
    
    async def start_workflow(self, workflow_type: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new workflow execution"""
        try:
            if workflow_type not in self.workflow_definitions:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Create workflow execution
            execution_id = str(uuid.uuid4())
            workflow_steps = self.workflow_definitions[workflow_type].copy()
            
            # Create execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_name=workflow_type,
                status=WorkflowStatus.IDLE,
                created_at=datetime.utcnow(),
                steps=workflow_steps,
                metadata=metadata or {}
            )
            
            # Add to active executions
            self.active_executions[execution_id] = execution
            
            # Add to execution queue
            await self.execution_queue.put(execution)
            
            self.logger.info(f"Started workflow {workflow_type} with execution ID: {execution_id}")
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow: {e}")
            raise
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow execution"""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
            elif execution_id in self.completed_executions:
                execution = self.completed_executions[execution_id]
            else:
                raise ValueError(f"Workflow execution {execution_id} not found")
            
            # Calculate progress
            total_steps = len(execution.steps)
            completed_steps = len([s for s in execution.steps if s.status == "completed"])
            progress = completed_steps / total_steps if total_steps > 0 else 0
            
            # Get current step
            current_step = None
            for step in execution.steps:
                if step.status == "running":
                    current_step = step
                    break
            
            return {
                "execution_id": execution_id,
                "workflow_name": execution.workflow_name,
                "status": execution.status.value,
                "progress": progress,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "current_step": current_step.name if current_step else None,
                "created_at": execution.created_at.isoformat(),
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "total_duration": execution.total_duration,
                "error_message": execution.error_message
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            raise
    
    async def get_all_workflow_statuses(self) -> Dict[str, Any]:
        """Get status of all workflow executions"""
        try:
            active_statuses = {}
            for execution_id, execution in self.active_executions.items():
                active_statuses[execution_id] = await self.get_workflow_status(execution_id)
            
            completed_statuses = {}
            for execution_id, execution in self.completed_executions.items():
                completed_statuses[execution_id] = await self.get_workflow_status(execution_id)
            
            return {
                "active_executions": active_statuses,
                "completed_executions": completed_statuses,
                "total_active": len(active_executions),
                "total_completed": len(completed_statuses)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get all workflow statuses: {e}")
            raise
    
    async def pause_workflow(self, execution_id: str) -> Dict[str, Any]:
        """Pause a workflow execution"""
        try:
            if execution_id not in self.active_executions:
                raise ValueError(f"Workflow execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.PAUSED
                
                # Pause current step
                for step in execution.steps:
                    if step.status == "running":
                        step.status = "paused"
                        step.end_time = datetime.utcnow()
                        if step.start_time:
                            step.duration = (step.end_time - step.start_time).total_seconds()
                        break
                
                self.logger.info(f"Paused workflow execution {execution_id}")
                
                return {
                    "status": "success",
                    "message": f"Workflow {execution_id} paused successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Workflow {execution_id} is not running"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to pause workflow: {e}")
            raise
    
    async def resume_workflow(self, execution_id: str) -> Dict[str, Any]:
        """Resume a paused workflow execution"""
        try:
            if execution_id not in self.active_executions:
                raise ValueError(f"Workflow execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                
                # Resume paused step
                for step in execution.steps:
                    if step.status == "paused":
                        step.status = "running"
                        step.start_time = datetime.utcnow()
                        break
                
                # Re-add to execution queue
                await self.execution_queue.put(execution)
                
                self.logger.info(f"Resumed workflow execution {execution_id}")
                
                return {
                    "status": "success",
                    "message": f"Workflow {execution_id} resumed successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Workflow {execution_id} is not paused"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to resume workflow: {e}")
            raise
    
    async def cancel_workflow(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a workflow execution"""
        try:
            if execution_id not in self.active_executions:
                raise ValueError(f"Workflow execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.FAILED
            execution.error_message = "Workflow cancelled by user"
            execution.completed_at = datetime.utcnow()
            
            if execution.started_at:
                execution.total_duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Move to completed executions
            self.completed_executions[execution_id] = execution
            del self.active_executions[execution_id]
            
            self.logger.info(f"Cancelled workflow execution {execution_id}")
            
            return {
                "status": "success",
                "message": f"Workflow {execution_id} cancelled successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow: {e}")
            raise
    
    async def _workflow_executor(self):
        """Background task to execute workflows"""
        while True:
            try:
                if not self.execution_queue.empty():
                    execution = await self.execution_queue.get()
                    await self._execute_workflow(execution)
                else:
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in workflow executor: {e}")
                await asyncio.sleep(5)
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute a workflow"""
        try:
            self.logger.info(f"Starting workflow execution {execution.execution_id}")
            
            # Update execution status
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            # Execute steps in dependency order
            for step in execution.steps:
                try:
                    # Check dependencies
                    if not await self._check_dependencies(execution, step):
                        continue
                    
                    # Execute step
                    await self._execute_step(execution, step)
                    
                except Exception as e:
                    self.logger.error(f"Step {step.name} failed: {e}")
                    step.status = "failed"
                    step.error_message = str(e)
                    step.end_time = datetime.utcnow()
                    
                    # Mark execution as failed
                    execution.status = WorkflowStatus.FAILED
                    execution.error_message = f"Step {step.name} failed: {e}"
                    execution.completed_at = datetime.utcnow()
                    
                    if execution.started_at:
                        execution.total_duration = (execution.completed_at - execution.started_at).total_seconds()
                    
                    # Move to completed executions
                    self.completed_executions[execution.execution_id] = execution
                    del self.active_executions[execution.execution_id]
                    
                    return
            
            # Mark execution as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
            if execution.started_at:
                execution.total_duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Move to completed executions
            self.completed_executions[execution.execution_id] = execution
            del self.active_executions[execution.execution_id]
            
            self.logger.info(f"Completed workflow execution {execution.execution_id}")
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            if execution.started_at:
                execution.total_duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Move to completed executions
            self.completed_executions[execution.execution_id] = execution
            del self.active_executions[execution.execution_id]
    
    async def _check_dependencies(self, execution: WorkflowExecution, step: WorkflowStep) -> bool:
        """Check if step dependencies are met"""
        try:
            for dep_id in step.dependencies:
                dep_step = next((s for s in execution.steps if s.step_id == dep_id), None)
                if not dep_step or dep_step.status != "completed":
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Dependency check failed for step {step.name}: {e}")
            return False
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Execute a single workflow step"""
        try:
            self.logger.info(f"Executing step: {step.name}")
            
            # Update step status
            step.status = "running"
            step.start_time = datetime.utcnow()
            
            # Simulate step execution (in real implementation, this would call agents)
            await asyncio.sleep(2)  # Simulate work
            
            # Mark step as completed
            step.status = "completed"
            step.end_time = datetime.utcnow()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "status": "success",
                "message": f"Step {step.name} completed successfully",
                "duration": step.duration
            }
            
            self.logger.info(f"Completed step: {step.name}")
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            step.status = "failed"
            step.error_message = str(e)
            step.end_time = datetime.utcnow()
            if step.start_time:
                step.duration = (step.end_time - step.start_time).total_seconds()
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get overall workflow manager status"""
        try:
            return {
                "status": "running",
                "active_executions": len(self.active_executions),
                "completed_executions": len(self.completed_executions),
                "queue_size": self.execution_queue.qsize(),
                "workflow_templates": list(self.workflow_definitions.keys()),
                "background_tasks": len(self.background_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self):
        """Shutdown the workflow manager"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.logger.info("Workflow manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Workflow manager shutdown failed: {e}")
