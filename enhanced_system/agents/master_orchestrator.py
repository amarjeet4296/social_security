"""
Master Orchestrator Agent - Coordinates the workflow between all specialized agents
Uses LangGraph for workflow orchestration with ReAct framework for reasoning
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json

from langchain.agents import AgentExecutor
from langchain.agents.react.base import ReActChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Import local modules
from utils.llm_factory import get_llm
from database.db_setup import Application, Document, Interaction, AuditLog, Recommendation, get_db
from database.chroma_manager import ChromaManager
from config.system_config import AGENT_CONFIG

# Import specialized agents
from agents.data_collector import DataCollector
from agents.validator import ValidatorAgent
from agents.assessor import AssessorAgent
from agents.counselor import CounselorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("master_orchestrator")

# Application state definition
class ApplicationState(Enum):
    SUBMISSION = "submission"
    DATA_COLLECTION = "data_collection"
    VALIDATION = "validation"
    ASSESSMENT = "assessment"
    COUNSELING = "counseling"
    COMPLETED = "completed"
    ERROR = "error"

class AppData(BaseModel):
    """Data structure for storing application information in the workflow graph."""
    app_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    emirates_id: Optional[str] = None
    income: Optional[float] = None
    family_size: Optional[int] = None
    address: Optional[str] = None
    employment_status: Optional[str] = None
    employer: Optional[str] = None
    job_title: Optional[str] = None
    employment_duration: Optional[int] = None
    monthly_expenses: Optional[float] = None
    assets_value: Optional[float] = None
    liabilities_value: Optional[float] = None
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    validation_status: str = "pending"
    assessment_status: str = "pending"
    risk_level: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    
class OrchestratorState(BaseModel):
    """State maintained by the orchestrator throughout the workflow."""
    application: AppData = Field(default_factory=AppData)
    current_state: ApplicationState = Field(default=ApplicationState.SUBMISSION)
    user_input: Optional[str] = None
    agent_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_count: int = Field(default=0)
    retry_count: int = Field(default=0)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)

class MasterOrchestratorAgent:
    """
    Master Orchestrator agent that coordinates the entire application workflow.
    Uses LangGraph for workflow management with ReAct reasoning framework.
    """
    
    def __init__(self):
        """Initialize the orchestrator agent with all necessary components."""
        logger.info("Initializing Master Orchestrator Agent")
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize database session
        self.db_session = next(get_db())
        
        # Initialize ChromaDB manager
        self.chroma_manager = ChromaManager()
        
        # Initialize specialized agents
        self.data_collector = DataCollector()
        self.validator = ValidatorAgent()
        self.assessor = AssessorAgent()
        
        # Initialize counselor only when needed to save resources
        try:
            self.counselor = CounselorAgent()
            logger.info("Counselor agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing counselor: {str(e)}")
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow_graph()
        
        # Create audit logger
        self.audit_logger = self._create_audit_logger()
        
        # Initialize performance metrics storage
        self.performance_metrics = {
            "total_applications": 0,
            "successful_applications": 0,
            "failed_applications": 0,
            "avg_processing_time": 0,
            "stage_metrics": {
                "submission": {"count": 0, "total_time": 0, "avg_time": 0, "success_rate": 100},
                "data_collection": {"count": 0, "total_time": 0, "avg_time": 0, "success_rate": 100},
                "validation": {"count": 0, "total_time": 0, "avg_time": 0, "success_rate": 100},
                "assessment": {"count": 0, "total_time": 0, "avg_time": 0, "success_rate": 100},
                "counseling": {"count": 0, "total_time": 0, "avg_time": 0, "success_rate": 100},
            },
            "error_counts": {},
            "bottlenecks": [],
            "last_updated": datetime.utcnow()
        }
        
        logger.info("Master Orchestrator Agent initialized successfully")
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the workflow graph using LangGraph with ReAct reasoning."""
        logger.info("Creating workflow graph")
        
        # Define the workflow graph with OrchestratorState
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes for each stage in the workflow
        workflow.add_node("submission", self._handle_submission)
        workflow.add_node("data_collection", self._handle_data_collection)
        workflow.add_node("validation", self._handle_validation)
        workflow.add_node("assessment", self._handle_assessment)
        workflow.add_node("counseling", self._handle_counseling)
        workflow.add_node("error_handler", self._handle_error)
        
        # Add the entry point from START to submission
        workflow.add_edge(lg.START, "submission")
        
        # Define the conditional edges
        workflow.add_conditional_edges(
            "submission",
            self._route_after_submission,
            {
                ApplicationState.DATA_COLLECTION: "data_collection",
                ApplicationState.ERROR: "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "data_collection",
            self._route_after_data_collection,
            {
                ApplicationState.VALIDATION: "validation",
                ApplicationState.ERROR: "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {
                ApplicationState.ASSESSMENT: "assessment",
                ApplicationState.DATA_COLLECTION: "data_collection",
                ApplicationState.ERROR: "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "assessment",
            self._route_after_assessment,
            {
                ApplicationState.COUNSELING: "counseling",
                ApplicationState.ERROR: "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "counseling",
            self._route_after_counseling,
            {
                ApplicationState.COMPLETED: END,
                ApplicationState.ERROR: "error_handler"
            }
        )
        
        # Error handler can route to any stage or end the workflow
        workflow.add_conditional_edges(
            "error_handler",
            self._route_after_error,
            {
                ApplicationState.SUBMISSION: "submission",
                ApplicationState.DATA_COLLECTION: "data_collection",
                ApplicationState.VALIDATION: "validation",
                ApplicationState.ASSESSMENT: "assessment",
                ApplicationState.COUNSELING: "counseling",
                ApplicationState.COMPLETED: END,
                ApplicationState.ERROR: END
            }
        )
        
        # Compile the workflow
        return workflow.compile()
    
    def _create_audit_logger(self):
        """Create an audit logger for tracking system actions."""
        def log_action(application_id: Optional[str], action: str, actor: str, details: Dict[str, Any] = None):
            """Log an action to the audit log."""
            try:
                audit_log = AuditLog(
                    application_id=application_id,
                    action=action,
                    actor=actor,
                    details=details or {},
                    timestamp=datetime.utcnow()
                )
                self.db_session.add(audit_log)
                self.db_session.commit()
                logger.info(f"Audit log created: {action} by {actor}")
            except Exception as e:
                logger.error(f"Error logging action: {str(e)}")
                self.db_session.rollback()
        
        return log_action
    
    def _log_audit_event(self, application_id: str, action: str, details: Dict[str, Any] = None) -> None:
        """Log an audit event to the database."""
        try:
            # Create audit log entry with metrics
            self._log_audit_event(
                application_id=None,
                action="performance_metrics_generated",
                details={
                    "metrics": self._generate_metrics(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info("Performance metrics stored to database")
        except Exception as e:
            logger.error(f"Error storing performance metrics: {str(e)}")
        
    async def process_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an application through the entire workflow.
        Returns the final state and recommendations.
        """
        logger.info(f"Processing application: {application_data.get('app_id', 'new application')}")
        
        # Create initial state
        initial_state = OrchestratorState(
            application=AppData(**application_data),
            current_state=ApplicationState.SUBMISSION,
            user_input=None,
            agent_output=None,
            error=None,
            history=[]
        )
        
        # Log the start of processing
        self.audit_logger(
            application_data.get("app_id"),
            "application_processing_started",
            "master_orchestrator",
            {"initial_state": application_data}
        )
        
        try:
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Log the completion of processing
            self.audit_logger(
                final_state.application.app_id,
                "application_processing_completed",
                "master_orchestrator",
                {"final_state": final_state.application.dict()}
            )
            
            return {
                "app_id": final_state.application.app_id,
                "status": final_state.current_state.value,
                "assessment_status": final_state.application.assessment_status,
                "validation_status": final_state.application.validation_status,
                "risk_level": final_state.application.risk_level,
                "recommendations": final_state.application.recommendations,
                "errors": final_state.application.errors
            }
        
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            
            # Log the error
            self.audit_logger(
                application_data.get("app_id"),
                "application_processing_error",
                "master_orchestrator",
                {"error": str(e)}
            )
            
            return {
                "app_id": application_data.get("app_id"),
                "status": "error",
                "error": str(e)
            }
    
    async def chat_with_application(self, app_id: str, message: str) -> Dict[str, Any]:
        """
        Allow interactive chat with an application.
        Uses the counselor agent to provide responses.
        """
        logger.info(f"Chat request for application {app_id}")
        
        try:
            # Get application from database
            application = self.db_session.query(Application).filter_by(filename=app_id).first()
            
            if not application:
                return {"status": "error", "message": "Application not found"}
            
            # Create interaction record
            interaction = Interaction(
                application_id=application.id,
                interaction_type="chat",
                content=message,
                agent="user",
                timestamp=datetime.utcnow()
            )
            self.db_session.add(interaction)
            self.db_session.commit()
            
            # Get application data as dictionary
            app_data = {
                "app_id": application.filename,
                "name": application.name,
                "income": application.income,
                "family_size": application.family_size,
                "address": application.address,
                "employment_status": application.employment_status,
                "assessment_status": application.assessment_status,
                "risk_level": application.risk_level
            }
            
            # Get recommendations
            recommendations = [
                {"category": rec.category, "priority": rec.priority, "description": rec.description}
                for rec in application.recommendations
            ]
            
            # Process with counselor agent
            response = await self.counselor.process_message(message, app_data, recommendations)
            
            # Create interaction record for agent response
            agent_interaction = Interaction(
                application_id=application.id,
                interaction_type="chat",
                content=response["message"],
                agent="counselor",
                timestamp=datetime.utcnow()
            )
            self.db_session.add(agent_interaction)
            self.db_session.commit()
            
            # Log the chat interaction
            self.audit_logger(
                app_id,
                "chat_interaction",
                "counselor_agent",
                {"user_message": message, "agent_response": response["message"]}
            )
            
            return {
                "status": "success",
                "message": response["message"],
                "recommendations": response.get("recommendations", [])
            }
        
        except Exception as e:
            logger.error(f"Error in chat interaction: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
    
    # Node handler methods for the workflow graph
    
    def _handle_submission(self, state: OrchestratorState) -> OrchestratorState:
        """Handle the initial submission of an application."""
        logger.info("Handling submission state")
        
        try:
            # Create a new application record in the database
            application = Application(
                filename=state.application.app_id,
                name=state.application.name,
                email=state.application.email,
                phone=state.application.phone,
                emirates_id=state.application.emirates_id,
                income=state.application.income or 0,
                family_size=state.application.family_size or 0,
                address=state.application.address or "",
                employment_status=state.application.employment_status,
                employer=state.application.employer,
                job_title=state.application.job_title,
                employment_duration=state.application.employment_duration,
                monthly_expenses=state.application.monthly_expenses,
                assets_value=state.application.assets_value,
                liabilities_value=state.application.liabilities_value,
                validation_status="pending",
                assessment_status="pending",
                risk_level="unknown"
            )
            
            self.db_session.add(application)
            self.db_session.commit()
            
            # Update state
            state.agent_output = {"message": "Application submitted successfully"}
            state.current_state = ApplicationState.DATA_COLLECTION
            state.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "state": "submission",
                "message": "Application submitted successfully"
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error in submission handling: {str(e)}")
            state.error = f"Submission error: {str(e)}"
            state.current_state = ApplicationState.ERROR
            return state
    
    def _handle_data_collection(self, state: OrchestratorState) -> OrchestratorState:
        """Handle the data collection process from documents."""
        logger.info("Handling data collection state")
        
        try:
            # Process each document in the application
            for doc in state.application.documents:
                # Extract data from document
                result = self.data_collector.process_document(
                    doc["file_bytes"],
                    doc["filename"],
                    doc["document_type"]
                )
                
                # Update application data with extracted information
                self._update_application_with_extracted_data(state.application, result)
                
                # Create document record in database
                self._create_document_record(state.application.app_id, doc, result)
            
            # Update state
            state.agent_output = {"message": "Data collection completed"}
            state.current_state = ApplicationState.VALIDATION
            state.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "state": "data_collection",
                "message": "Document data extraction completed"
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            state.error = f"Data collection error: {str(e)}"
            state.current_state = ApplicationState.ERROR
            return state
    
    def _handle_validation(self, state: OrchestratorState) -> OrchestratorState:
        """Handle the validation of application data."""
        logger.info("Handling validation state")
        
        try:
            # Convert application data to dictionary for validation
            app_data = state.application.dict()
            
            # Validate the application data
            is_valid, validation_errors = self.validator.validate(app_data)
            
            # Update validation status in database
            self._update_validation_status(
                state.application.app_id,
                "valid" if is_valid else "invalid",
                validation_errors
            )
            
            # Update application state
            state.application.validation_status = "valid" if is_valid else "invalid"
            if not is_valid:
                state.application.errors.extend(validation_errors)
            
            # Determine next state
            if is_valid:
                state.current_state = ApplicationState.ASSESSMENT
                state.agent_output = {"message": "Validation successful"}
            else:
                # If validation fails, go back to data collection or error based on severity
                if len(validation_errors) > 3:
                    state.current_state = ApplicationState.ERROR
                    state.error = "Multiple validation errors detected"
                else:
                    state.current_state = ApplicationState.DATA_COLLECTION
                    state.agent_output = {
                        "message": "Validation failed, returning to data collection",
                        "errors": validation_errors
                    }
            
            state.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "state": "validation",
                "message": "Validation completed",
                "is_valid": is_valid,
                "errors": validation_errors if not is_valid else []
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            state.error = f"Validation error: {str(e)}"
            state.current_state = ApplicationState.ERROR
            return state
    
    def _handle_assessment(self, state: OrchestratorState) -> OrchestratorState:
        """Handle the assessment of application eligibility."""
        logger.info("Handling assessment state")
        
        try:
            # Convert application data to dictionary for assessment
            app_data = state.application.dict()
            
            # Assess the application
            is_approved, reasons, assessment_details = self.assessor.assess_application(app_data)
            
            # Update assessment status in database
            self._update_assessment_status(
                state.application.app_id,
                "approved" if is_approved else "rejected",
                assessment_details["risk_level"],
                reasons
            )
            
            # Update application state
            state.application.assessment_status = "approved" if is_approved else "rejected"
            state.application.risk_level = assessment_details["risk_level"]
            
            # Create recommendations in database
            if is_approved:
                self._create_recommendations(state.application.app_id, assessment_details["recommendations"])
                state.application.recommendations = assessment_details["recommendations"]
            
            # Update state
            state.current_state = ApplicationState.COUNSELING
            state.agent_output = {
                "message": "Assessment completed",
                "is_approved": is_approved,
                "reasons": reasons,
                "details": assessment_details
            }
            
            state.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "state": "assessment",
                "message": "Assessment completed",
                "is_approved": is_approved,
                "risk_level": assessment_details["risk_level"]
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error in assessment: {str(e)}")
            state.error = f"Assessment error: {str(e)}"
            state.current_state = ApplicationState.ERROR
            return state
    
    def _handle_counseling(self, state: OrchestratorState) -> OrchestratorState:
        """Handle counseling, decision explanation, and recommendation generation."""
        logger.info("Handling counseling state")
        
        try:
            app_data = state.application.dict()
            
            # 1. Get decision explanation from CounselorAgent
            logger.info(f"Getting decision explanation for app: {state.application.app_id}")
            decision_explanation_result = self.counselor.explain_decision(app_data)
            
            # 2. Generate broader recommendations from CounselorAgent
            logger.info(f"Generating broader recommendations for app: {state.application.app_id}")
            recommendations_result = self.counselor.generate_recommendations(
                app_data,
                state.application.assessment_status, # Pass current assessment status
                state.application.risk_level       # Pass current risk level
            )
            
            # Combine recommendations for storage and state update
            final_broader_recommendations = recommendations_result.get("recommendations", [])
            
            # Update recommendations in the database
            if state.application.app_id and final_broader_recommendations:
                # Assuming _update_recommendations can handle the list of dicts format
                self._update_recommendations(
                    state.application.app_id,
                    final_broader_recommendations
                )
            
            # Update application state with the broader recommendations
            # Clear previous recommendations if any, then extend with new ones
            state.application.recommendations.clear()
            state.application.recommendations.extend(final_broader_recommendations)
            
            # Update overall state
            state.current_state = ApplicationState.COMPLETED
            state.agent_output = {
                "message": "Counseling and decision explanation completed.",
                "decision_status": decision_explanation_result.get("status"),
                "decision_explanation": decision_explanation_result.get("explanation"),
                "decision_next_steps": decision_explanation_result.get("next_steps"),
                "decision_timeline": decision_explanation_result.get("timeline"),
                "broader_recommendations": final_broader_recommendations,
                "counseling_summary": recommendations_result.get("summary")
            }
            
            state.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "state": "counseling",
                "message": "Counseling, decision explanation, and recommendation generation completed.",
                "assessment_status": state.application.assessment_status,
                "explanation_provided": bool(decision_explanation_result.get("explanation")),
                "recommendations_generated": len(final_broader_recommendations) > 0
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error in counseling: {str(e)}")
            state.error = f"Counseling error: {str(e)}"
            state.current_state = ApplicationState.ERROR
            return state
    
    def _handle_error(self, state: OrchestratorState) -> OrchestratorState:
        """Handle error conditions in the workflow."""
        logger.error(f"Error in workflow: {state.error}")
        
        # Categorize error by type
        if not state.error_type:
            state.error_type = self._categorize_error(state.error)
        
        # Increment error count
        state.error_count += 1
        
        # Update performance metrics
        self._track_performance(
            stage=state.current_state.value,
            success=False,
            error_type=state.error_type,
            state=state
        )
        
        # Log the error
        self._log_audit_event(
            application_id=state.application.app_id,
            action="error_occurred",
            details={
                "error": state.error,
                "error_type": state.error_type,
                "current_state": state.current_state.value,
                "retry_count": state.retry_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Determine if we should retry
        max_retries = self._get_max_retries(state.error_type)
        if state.retry_count < max_retries:
            # Implement exponential backoff
            backoff_time = 2 ** state.retry_count  # Exponential backoff
            logger.info(f"Retrying after {backoff_time} seconds (attempt {state.retry_count + 1}/{max_retries})")
            time.sleep(backoff_time)
            
            # Increment retry count
            state.retry_count += 1
            
            # Return to previous state for retry
            return state
        
        # Update application with error information
        if state.application and state.application.app_id:
            try:
                app = self.db_session.query(Application).filter_by(filename=state.application.app_id).first()
                if app:
                    app.status = "error"
                    app.updated_at = datetime.utcnow()
                    self.db_session.commit()
            except Exception as e:
                logger.error(f"Error updating application status: {str(e)}")
                self.db_session.rollback()
        
        # Return state with error status
        state.current_state = ApplicationState.ERROR
        return state
        
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error by type for targeted recovery strategies."""
        error_categories = {
            "document_processing": ["OCR", "image processing", "PDF extraction", "document"],
            "data_validation": ["validation", "invalid data", "missing field", "required field"],
            "assessment": ["assessment", "decision", "scoring", "eligibility"],
            "counseling": ["counseling", "recommendation", "advice"],
            "database": ["database", "SQL", "query", "connection"],
            "llm": ["LLM", "model", "token", "inference", "generation"],
        }
        
        for category, keywords in error_categories.items():
            if any(keyword.lower() in error_message.lower() for keyword in keywords):
                return category
        
        return "unknown"
    
    def _get_max_retries(self, error_type: str) -> int:
        """Get maximum retry count based on error type."""
        retry_map = {
            "document_processing": 3,
            "data_validation": 2,
            "assessment": 2,
            "counseling": 2,
            "database": 5,
            "llm": 3,
            "unknown": 1
        }
        
        return retry_map.get(error_type, 1)
    
    # Routing methods for the workflow graph
    
    def _route_after_submission(self, state: OrchestratorState) -> ApplicationState:
        """Determine next state after submission."""
        if state.error:
            return ApplicationState.ERROR
        return ApplicationState.DATA_COLLECTION
    
    def _route_after_data_collection(self, state: OrchestratorState) -> ApplicationState:
        """Determine next state after data collection."""
        if state.error:
            return ApplicationState.ERROR
        return ApplicationState.VALIDATION
    
    def _route_after_validation(self, state: OrchestratorState) -> ApplicationState:
        """Determine next state after validation."""
        if state.error:
            return ApplicationState.ERROR
        if state.application.validation_status == "valid":
            return ApplicationState.ASSESSMENT
        return ApplicationState.DATA_COLLECTION
    
    def _route_after_assessment(self, state: OrchestratorState) -> ApplicationState:
        """Determine next state after assessment."""
        if state.error:
            return ApplicationState.ERROR
        return ApplicationState.COUNSELING
    
    def _route_after_counseling(self, state: OrchestratorState) -> ApplicationState:
        """Determine next state after counseling."""
        if state.error:
            return ApplicationState.ERROR
        return ApplicationState.COMPLETED
    
    def _route_after_error(self, state: OrchestratorState) -> ApplicationState:
        """Determine what to do after an error."""
        # For now, just end the workflow
        return ApplicationState.ERROR
    
    # Helper methods for database operations
    
    def _update_application_with_extracted_data(self, application: AppData, extracted_data: Dict[str, Any]):
        """Update application data with information extracted from documents."""
        # Update each field if it exists in the extracted data
        for field, value in extracted_data.items():
            if hasattr(application, field) and value is not None:
                setattr(application, field, value)
    
    def _create_document_record(self, app_id: str, doc_info: Dict[str, Any], extracted_data: Dict[str, Any]):
        """Create a document record in the database."""
        # Get application ID from database
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        # Create document record
        document = Document(
            application_id=application.id,
            document_type=doc_info["document_type"],
            filename=doc_info["filename"],
            file_path=doc_info.get("file_path", ""),
            mime_type=doc_info.get("mime_type", "application/octet-stream"),
            extracted_data=extracted_data,
            uploaded_at=datetime.utcnow()
        )
        
        self.db_session.add(document)
        self.db_session.commit()
        
        logger.info(f"Document record created: {doc_info['filename']}")
    
    def _update_validation_status(self, app_id: str, status: str, errors: List[str] = None):
        """Update the validation status of an application in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        application.validation_status = status
        self.db_session.commit()
        
        # Log validation status
        self.audit_logger(
            app_id,
            "validation_status_updated",
            "validator_agent",
            {"status": status, "errors": errors or []}
        )
        
        logger.info(f"Validation status updated: {app_id} - {status}")
    
    def _update_assessment_status(self, app_id: str, status: str, risk_level: str, reasons: List[str] = None):
        """Update the assessment status of an application in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        application.assessment_status = status
        application.risk_level = risk_level
        self.db_session.commit()
        
        # Log assessment status
        self.audit_logger(
            app_id,
            "assessment_status_updated",
            "assessor_agent",
            {"status": status, "risk_level": risk_level, "reasons": reasons or []}
        )
        
        logger.info(f"Assessment status updated: {app_id} - {status}, risk level: {risk_level}")
    
    def _create_recommendations(self, app_id: str, recommendations: List[Dict[str, Any]]):
        """Create recommendation records in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        for rec in recommendations:
            recommendation = Recommendation(
                application_id=application.id,
                category=rec["category"],
                priority=rec["priority"],
                description=rec["description"],
                action_items=rec.get("action_items", []),
                created_at=datetime.utcnow()
            )
            self.db_session.add(recommendation)
        
        self.db_session.commit()
        
        # Log recommendations created
        self.audit_logger(
            app_id,
            "recommendations_created",
            "assessor_agent",
            {"count": len(recommendations)}
        )
        
        logger.info(f"Recommendations created for application: {app_id}")
    
    def _update_recommendations(self, app_id: str, recommendations: List[Dict[str, Any]]):
        """Update or create new recommendation records in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        for rec in recommendations:
            # Check if similar recommendation exists
            existing = self.db_session.query(Recommendation).filter_by(
                application_id=application.id,
                category=rec["category"]
            ).first()
            
            if existing:
                # Update existing recommendation
                existing.priority = rec["priority"]
                existing.description = rec["description"]
                existing.action_items = rec.get("action_items", existing.action_items)
            else:
                # Create new recommendation
                recommendation = Recommendation(
                    application_id=application.id,
                    category=rec["category"],
                    priority=rec["priority"],
                    description=rec["description"],
                    action_items=rec.get("action_items", []),
                    created_at=datetime.utcnow()
                )
                self.db_session.add(recommendation)
        
        self.db_session.commit()
        
        # Log recommendations updated
        self.audit_logger(
            app_id,
            "recommendations_updated",
            "counselor_agent",
            {"count": len(recommendations)}
        )
        
        logger.info(f"Recommendations updated for application: {app_id}")
    
    def _update_application_with_extracted_data(self, application: AppData, extracted_data: Dict[str, Any]):
        """Update application data with information extracted from documents."""
        # Update each field if it exists in the extracted data
        for field, value in extracted_data.items():
            if hasattr(application, field) and value is not None:
                setattr(application, field, value)
    
    def _create_document_record(self, app_id: str, doc_info: Dict[str, Any], extracted_data: Dict[str, Any]):
        """Create a document record in the database."""
        # Get application ID from database
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        # Create document record
        document = Document(
            application_id=application.id,
            document_type=doc_info["document_type"],
            filename=doc_info["filename"],
            file_path=doc_info.get("file_path", ""),
            mime_type=doc_info.get("mime_type", "application/octet-stream"),
            extracted_data=extracted_data,
            uploaded_at=datetime.utcnow()
        )
        
        self.db_session.add(document)
        self.db_session.commit()
        
        logger.info(f"Document record created: {doc_info['filename']}")
    
    def _update_validation_status(self, app_id: str, status: str, errors: List[str] = None):
        """Update the validation status of an application in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        application.validation_status = status
        self.db_session.commit()
        
        # Log validation status
        self.audit_logger(
            app_id,
            "validation_status_updated",
            "validator_agent",
            {"status": status, "errors": errors or []}
        )
        
        logger.info(f"Validation status updated: {app_id} - {status}")
    
    def _update_assessment_status(self, app_id: str, status: str, risk_level: str, reasons: List[str] = None):
        """Update the assessment status of an application in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        application.assessment_status = status
        application.risk_level = risk_level
        self.db_session.commit()
        
        # Log assessment status
        self.audit_logger(
            app_id,
            "assessment_status_updated",
            "assessor_agent",
            {"status": status, "risk_level": risk_level, "reasons": reasons or []}
        )
        
        logger.info(f"Assessment status updated: {app_id} - {status}, risk level: {risk_level}")
    
    def _create_recommendations(self, app_id: str, recommendations: List[Dict[str, Any]]):
        """Create recommendation records in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        for rec in recommendations:
            recommendation = Recommendation(
                application_id=application.id,
                category=rec["category"],
                priority=rec["priority"],
                description=rec["description"],
                action_items=rec.get("action_items", []),
                created_at=datetime.utcnow()
            )
            self.db_session.add(recommendation)
        
        self.db_session.commit()
        
        # Log recommendations created
        self.audit_logger(
            app_id,
            "recommendations_created",
            "assessor_agent",
            {"count": len(recommendations)}
        )
        
        logger.info(f"Recommendations created for application: {app_id}")
    
    def _update_recommendations(self, app_id: str, recommendations: List[Dict[str, Any]]):
        """Update or create new recommendation records in the database."""
        application = self.db_session.query(Application).filter_by(filename=app_id).first()
        
        if not application:
            logger.error(f"Application not found: {app_id}")
            return
        
        for rec in recommendations:
            # Check if similar recommendation exists
            existing = self.db_session.query(Recommendation).filter_by(
                application_id=application.id,
                category=rec["category"]
            ).first()
            
            if existing:
                # Update existing recommendation
                existing.priority = rec["priority"]
                existing.description = rec["description"]
                existing.action_items = rec.get("action_items", existing.action_items)
            else:
                # Create new recommendation
                recommendation = Recommendation(
                    application_id=application.id,
                    category=rec["category"],
                    priority=rec["priority"],
                    description=rec["description"],
                    action_items=rec.get("action_items", []),
                    created_at=datetime.utcnow()
                )
                self.db_session.add(recommendation)
        
        self.db_session.commit()
        
        # Log recommendations updated
        self.audit_logger(
            app_id,
            "recommendations_updated",
            "counselor_agent",
            {"count": len(recommendations)}
        )
        
        logger.info(f"Recommendations updated for application: {app_id}")
