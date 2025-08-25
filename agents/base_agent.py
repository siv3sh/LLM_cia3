"""
Base Agent class for the Multi-Agent Attrition Analysis System
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from core.config import Config


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    content: Any = None
    message_type: str = "data"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=high
    requires_response: bool = False
    response_to: Optional[str] = None


@dataclass
class AgentState:
    """Agent state information"""
    agent_id: str
    status: str = "idle"  # idle, busy, error, offline
    current_task: Optional[str] = None
    task_progress: float = 0.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    success_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    """
    
    def __init__(self, config: Config, agent_id: Optional[str] = None):
        self.config = config
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.agent_id}")
        
        # Agent state
        self.state = AgentState(agent_id=self.agent_id)
        self.memory: List[AgentMessage] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Communication
        self.subscribers: List[str] = []
        self.message_handlers: Dict[str, callable] = {}
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.task_history: List[Dict[str, Any]] = []
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        self.logger.info(f"Agent {self.agent_id} initialized")
    
    async def initialize(self):
        """Initialize the agent"""
        try:
            self.state.status = "initializing"
            await self._initialize_agent()
            self.state.status = "idle"
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
        except Exception as e:
            self.state.status = "error"
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            raise
    
    @abstractmethod
    async def _initialize_agent(self):
        """Agent-specific initialization logic"""
        pass
    
    async def start(self):
        """Start the agent's main loop"""
        try:
            self.state.status = "running"
            self.logger.info(f"Agent {self.agent_id} started")
            await self._run_agent_loop()
        except Exception as e:
            self.state.status = "error"
            self.logger.error(f"Agent {self.agent_id} failed: {e}")
            raise
    
    async def _run_agent_loop(self):
        """Main agent loop for processing messages"""
        while self.state.status == "running":
            try:
                # Process messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._process_message(message)
                
                # Run agent-specific tasks
                await self._run_agent_tasks()
                
                # Update state
                self.state.last_activity = datetime.utcnow()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                self.state.error_count += 1
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _run_agent_tasks(self):
        """Run agent-specific background tasks"""
        pass
    
    async def _process_message(self, message: AgentMessage):
        """Process incoming messages"""
        try:
            self.logger.debug(f"Processing message: {message.id} from {message.sender}")
            
            # Store message in memory
            self.memory.append(message)
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                await self._handle_unknown_message(message)
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")
            self.state.error_count += 1
    
    def _setup_message_handlers(self):
        """Setup default message handlers"""
        self.message_handlers = {
            "status_request": self._handle_status_request,
            "task_request": self._handle_task_request,
            "data_request": self._handle_data_request,
            "control": self._handle_control_message,
        }
    
    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request messages"""
        response = AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            content=self.state,
            message_type="status_response",
            response_to=message.id
        )
        await self.send_message(response)
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task request messages"""
        try:
            self.state.status = "busy"
            self.state.current_task = message.content.get("task_type", "unknown")
            
            # Execute task
            result = await self._execute_task(message.content)
            
            # Send response
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="task_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="task_error",
                response_to=message.id
            )
            await self.send_message(error_response)
        finally:
            self.state.status = "idle"
            self.state.current_task = None
    
    async def _handle_data_request(self, message: AgentMessage):
        """Handle data request messages"""
        try:
            data = await self._get_requested_data(message.content)
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=data,
                message_type="data_response",
                response_to=message.id
            )
            await self.send_message(response)
        except Exception as e:
            self.logger.error(f"Data request failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="data_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_control_message(self, message: AgentMessage):
        """Handle control messages (start, stop, restart, etc.)"""
        command = message.content.get("command")
        if command == "stop":
            await self.stop()
        elif command == "restart":
            await self.restart()
        elif command == "status":
            await self._handle_status_request(message)
    
    async def _handle_unknown_message(self, message: AgentMessage):
        """Handle messages with unknown types"""
        self.logger.warning(f"Unknown message type: {message.message_type}")
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested data (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _get_requested_data")
    
    async def send_message(self, message: AgentMessage):
        """Send a message to another agent"""
        try:
            # Add to recipient's queue if they're subscribed
            if message.recipient in self.subscribers:
                # This would typically go through a message broker
                # For now, we'll just log it
                self.logger.info(f"Sending message {message.id} to {message.recipient}")
            
            # Store in memory
            self.memory.append(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
    
    async def subscribe_to_agent(self, agent_id: str):
        """Subscribe to messages from another agent"""
        if agent_id not in self.subscribers:
            self.subscribers.append(agent_id)
            self.logger.info(f"Subscribed to agent {agent_id}")
    
    async def unsubscribe_from_agent(self, agent_id: str):
        """Unsubscribe from messages from another agent"""
        if agent_id in self.subscribers:
            self.subscribers.remove(agent_id)
            self.logger.info(f"Unsubscribed from agent {agent_id}")
    
    async def stop(self):
        """Stop the agent"""
        self.state.status = "stopping"
        self.logger.info(f"Agent {self.agent_id} stopping")
        
        # Cleanup
        await self._cleanup()
        
        self.state.status = "stopped"
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def restart(self):
        """Restart the agent"""
        self.logger.info(f"Agent {self.agent_id} restarting")
        await self.stop()
        await self.initialize()
        await self.start()
    
    async def _cleanup(self):
        """Cleanup resources before stopping"""
        pass
    
    async def shutdown(self):
        """Shutdown the agent completely"""
        await self.stop()
        self.logger.info(f"Agent {self.agent_id} shutdown complete")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "uptime_seconds": uptime,
            "status": self.state.status,
            "error_count": self.state.error_count,
            "success_count": self.state.success_count,
            "memory_size": len(self.memory),
            "queue_size": self.message_queue.qsize(),
            "subscriber_count": len(self.subscribers),
            "task_history_count": len(self.task_history)
        }
    
    def update_task_progress(self, progress: float):
        """Update current task progress"""
        self.state.task_progress = max(0.0, min(1.0, progress))
    
    def log_task_completion(self, task_type: str, duration: float, success: bool):
        """Log task completion for performance tracking"""
        task_record = {
            "task_type": task_type,
            "duration": duration,
            "success": success,
            "timestamp": datetime.utcnow(),
            "agent_id": self.agent_id
        }
        self.task_history.append(task_record)
        
        if success:
            self.state.success_count += 1
        else:
            self.state.error_count += 1
