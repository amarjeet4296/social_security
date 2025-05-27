"""
Model Loader for the Enhanced Social Security Application System.
Provides centralized access to all models including the decision engine.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from enhanced_system.models.decision_engine import DecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Centralized loader for all models used in the system.
    Ensures models are loaded efficiently and only when needed.
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one model loader exists."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the model loader with optional database session."""
        # Only initialize once
        if self._initialized:
            return
            
        self.db_session = db_session
        self._initialized = True
        logger.info("Model Loader initialized")
    
    def get_decision_engine(self) -> DecisionEngine:
        """
        Get the decision engine instance.
        
        Returns:
            DecisionEngine: Instance of the decision engine
        """
        if "decision_engine" not in self._models:
            self._models["decision_engine"] = DecisionEngine(self.db_session)
            logger.info("Decision Engine loaded")
        
        return self._models["decision_engine"]
    
    def evaluate_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to evaluate an application using the decision engine.
        
        Args:
            application_data: Dictionary containing application information
            
        Returns:
            Dictionary with evaluation results
        """
        engine = self.get_decision_engine()
        return engine.evaluate_application(application_data)
