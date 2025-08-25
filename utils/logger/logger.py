"""
Logging configuration for the Multi-Agent Attrition Analysis System
"""
import logging
import structlog
from pathlib import Path


def setup_logging():
    """Setup structured logging for the system"""
    try:
        # Create logs directory
        Path("./logs").mkdir(exist_ok=True)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/app.log'),
                logging.StreamHandler()
            ]
        )

        return True

    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return False


# Export the function for easy importing
__all__ = ['setup_logging']
