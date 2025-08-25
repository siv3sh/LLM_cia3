"""
FastAPI application for the Multi-Agent Attrition Analysis System
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from core.config import Config
from agents.coordinator_agent import CoordinatorAgent
from data.schemas import (
    JobRequest, JobStatus, SystemHealth, APIResponse, ErrorResponse,
    WorkflowConfig, AnalysisResult
)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Attrition Analysis System",
    description="A sophisticated multi-agent system for analyzing employee attrition using AI and machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
coordinator: Optional[CoordinatorAgent] = None
config: Optional[Config] = None


def create_app(app_config: Config, coordinator_agent: CoordinatorAgent) -> FastAPI:
    """Create and configure the FastAPI application"""
    global coordinator, config
    
    coordinator = coordinator_agent
    config = app_config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    return app


# Health check endpoint
@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Check system health"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        # Get system health from coordinator
        health_data = await coordinator._get_system_health()
        
        return SystemHealth(**health_data)
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# System status endpoint
@app.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """Get overall system status"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        status_data = await coordinator._get_system_status()
        return status_data
        
    except Exception as e:
        logging.error(f"Status retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status retrieval failed: {str(e)}"
        )


# Start analysis endpoint
@app.post("/analysis/start", response_model=APIResponse)
async def start_analysis(
    request: JobRequest,
    background_tasks: BackgroundTasks
):
    """Start a new attrition analysis job"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        # Start analysis in background
        background_tasks.add_task(
            coordinator._run_attrition_analysis,
            request.workflow_config.dict()
        )
        
        return APIResponse(
            status="success",
            message="Analysis job started successfully",
            data={"job_id": request.job_id},
            request_id=request.job_id
        )
        
    except Exception as e:
        logging.error(f"Failed to start analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}"
        )


# Get job status endpoint
@app.get("/analysis/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a specific analysis job"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        status_data = await coordinator._get_job_status({"job_id": job_id})
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatus(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


# Get job results endpoint
@app.get("/analysis/{job_id}/results", response_model=AnalysisResult)
async def get_job_results(job_id: str):
    """Get results of a completed analysis job"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        results_data = await coordinator._get_job_results({"job_id": job_id})
        
        if not results_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or not completed"
            )
        
        return AnalysisResult(**results_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get job results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job results: {str(e)}"
        )


# Cancel job endpoint
@app.post("/analysis/{job_id}/cancel", response_model=APIResponse)
async def cancel_job(job_id: str):
    """Cancel a running analysis job"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        cancel_result = await coordinator._cancel_job({"job_id": job_id})
        
        if not cancel_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=cancel_result.get("message", "Failed to cancel job")
            )
        
        return APIResponse(
            status="success",
            message="Job cancelled successfully",
            data={"job_id": job_id},
            request_id=job_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to cancel job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


# List active jobs endpoint
@app.get("/analysis/active", response_model=List[JobStatus])
async def list_active_jobs():
    """List all active analysis jobs"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        active_jobs = list(coordinator.active_jobs.values())
        
        # Convert to JobStatus models
        job_statuses = []
        for job in active_jobs:
            status_data = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "current_step": None,  # Placeholder
                "start_time": job.started_at,
                "estimated_completion": None,  # Placeholder
                "error_message": job.error_message,
                "last_updated": job.created_at  # Placeholder
            }
            job_statuses.append(JobStatus(**status_data))
        
        return job_statuses
        
    except Exception as e:
        logging.error(f"Failed to list active jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list active jobs: {str(e)}"
        )


# List completed jobs endpoint
@app.get("/analysis/completed", response_model=List[JobStatus])
async def list_completed_jobs():
    """List all completed analysis jobs"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        completed_jobs = list(coordinator.completed_jobs.values())
        
        # Convert to JobStatus models
        job_statuses = []
        for job in completed_jobs:
            status_data = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": 1.0,  # Completed
                "current_step": "completed",
                "start_time": job.started_at,
                "estimated_completion": job.completed_at,
                "error_message": job.error_message,
                "last_updated": job.completed_at or job.created_at
            }
            job_statuses.append(JobStatus(**status_data))
        
        return job_statuses
        
    except Exception as e:
        logging.error(f"Failed to list completed jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list completed jobs: {str(e)}"
        )


# Agent status endpoint
@app.get("/agents/status", response_model=Dict[str, Any])
async def get_agent_status():
    """Get status of all agents"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        agent_status = {}
        
        # Get status of each agent
        if coordinator.data_agent:
            agent_status["data_agent"] = coordinator.data_agent.state.__dict__
        
        if coordinator.analysis_agent:
            agent_status["analysis_agent"] = coordinator.analysis_agent.state.__dict__
        
        if coordinator.prediction_agent:
            agent_status["prediction_agent"] = coordinator.prediction_agent.state.__dict__
        
        if coordinator.insight_agent:
            agent_status["insight_agent"] = coordinator.insight_agent.state.__dict__
        
        agent_status["coordinator_agent"] = coordinator.state.__dict__
        
        return agent_status
        
    except Exception as e:
        logging.error(f"Failed to get agent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )


# Workflow status endpoint
@app.get("/workflow/status", response_model=Dict[str, Any])
async def get_workflow_status():
    """Get status of workflow manager"""
    try:
        if coordinator is None or coordinator.workflow_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow manager not initialized"
            )
        
        workflow_status = await coordinator.workflow_manager.get_status()
        return workflow_status
        
    except Exception as e:
        logging.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )


# System metrics endpoint
@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        if coordinator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Coordinator agent not initialized"
            )
        
        metrics = await coordinator._get_system_metrics()
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            error_code=f"HTTP_{exc.status_code}",
            error_message=exc.detail,
            details={"path": str(request.url)}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status="error",
            error_code="INTERNAL_ERROR",
            error_message="An internal server error occurred",
            details={"path": str(request.url), "error": str(exc)}
        ).dict()
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logging.info("Multi-Agent Attrition Analysis System API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logging.info("Multi-Agent Attrition Analysis System API shutting down")
    
    # Shutdown coordinator if available
    if coordinator:
        try:
            await coordinator.shutdown()
        except Exception as e:
            logging.error(f"Error during coordinator shutdown: {e}")


# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Multi-Agent Attrition Analysis System",
        "version": "1.0.0",
        "description": "A sophisticated multi-agent system for analyzing employee attrition",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "analysis": "/analysis",
            "agents": "/agents",
            "workflow": "/workflow",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
