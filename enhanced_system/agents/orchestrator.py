"""
Orchestrator Agent - The central agent that coordinates the workflow between specialized agents.
Uses LangGraph for workflow orchestration with the ReAct framework for reasoning.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json
import uuid

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.react.base import ReActChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Import utility for local LLM
from utils.llm_factory import get_llm
from database.db_setup import Application, Document, Interaction, AuditLog, get_db
from database.chroma_manager import ChromaManager

# Load environment variables
load_dotenv()

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
    history: List[Dict[str, Any]] = Field(default_factory=list)

class OrchestratorAgent:
    """
    Orchestrator agent that coordinates the entire application workflow.
    Uses LangGraph for workflow management and state transitions.
    """
    
    def __init__(self):
        """Initialize the orchestrator agent with all necessary components."""
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize database session
        self.db_session = next(get_db())
        
        # Initialize ChromaDB manager
        self.chroma_manager = ChromaManager()
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow_graph()
        
        # Create audit logger
        self.audit_logger = self._create_audit_logger()
        
    def _create_workflow_graph(self) -> StateGraph:
        """Create the workflow graph using LangGraph."""
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
            except Exception as e:
                print(f"Error logging action: {str(e)}")
                self.db_session.rollback()
        
        return log_action
    
    async def process_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an application through the entire workflow.
        
        Args:
            application_data: Initial application data
            
        Returns:
            Final application state and results
        """
        # Initialize state
        initial_state = OrchestratorState(
            application=AppData(**application_data),
            current_state=ApplicationState.SUBMISSION,
            user_input=json.dumps(application_data)
        )
        
        # Log application start
        self.audit_logger(
            application_data.get("app_id"),
            "application_started",
            "orchestrator",
            {"initial_data": application_data}
        )
        
        # Execute workflow
        try:
            final_state = await self.workflow.acall(initial_state)
            
            # Log application completion
            self.audit_logger(
                final_state.application.app_id,
                "application_completed",
                "orchestrator",
                {
                    "final_state": final_state.current_state.value,
                    "assessment_status": final_state.application.assessment_status,
                    "validation_status": final_state.application.validation_status
                }
            )
            
            return {
                "application_id": final_state.application.app_id,
                "status": final_state.current_state.value,
                "result": final_state.agent_output,
                "error": final_state.error,
                "application_data": final_state.application.dict()
            }
            
        except Exception as e:
            # Log error
            self.audit_logger(
                application_data.get("app_id"),
                "application_error",
                "orchestrator",
                {"error": str(e)}
            )
            
            return {
                "application_id": application_data.get("app_id"),
                "status": "error",
                "error": str(e),
                "application_data": application_data
            }
    
    async def handle_chat_message(self, application_id: str, message: str) -> Dict[str, Any]:
        """
        Handle a chat message from the user.
        
        Args:
            application_id: ID of the application
            message: User message
            
        Returns:
            Response from the appropriate agent
        """
        try:
            # Get application data from database
            application = self.db_session.query(Application).filter(
                Application.filename == application_id
            ).first()
            
            if not application:
                return {
                    "error": f"Application with ID {application_id} not found"
                }
            
            # Determine which agent should handle the message based on application state
            if application.assessment_status == "approved":
                # Use counselor agent for approved applications
                from agents.counselor import CounselorAgent
                counselor = CounselorAgent()
                response = await counselor.provide_guidance(application_id, message)
            elif application.assessment_status == "rejected":
                # Use counselor agent for rejected applications
                from agents.counselor import CounselorAgent
                counselor = CounselorAgent()
                response = await counselor.provide_guidance(application_id, message)
            elif application.validation_status == "invalid":
                # Use validator agent for invalid applications
                from agents.validator import ValidatorAgent
                validator = ValidatorAgent()
                response = await validator.explain_validation_errors(application_id, message)
            else:
                # Use a general response for other states
                response = {
                    "text": "Your application is still being processed. I'll help answer any questions you have about the process.",
                    "suggestions": [
                        "What documents do I need to provide?",
                        "How long will processing take?",
                        "What are the eligibility criteria?"
                    ]
                }
            
            # Log the interaction
            interaction = Interaction(
                application_id=application.id,
                interaction_type="chat",
                content=message,
                agent="orchestrator",
                timestamp=datetime.utcnow()
            )
            self.db_session.add(interaction)
            
            # Log the response
            interaction_response = Interaction(
                application_id=application.id,
                interaction_type="chat_response",
                content=response.get("text", ""),
                agent=response.get("agent", "orchestrator"),
                timestamp=datetime.utcnow()
            )
            self.db_session.add(interaction_response)
            self.db_session.commit()
            
            return response
            
        except Exception as e:
            self.db_session.rollback()
            return {
                "error": str(e),
                "text": "I'm sorry, I encountered an error while processing your message."
            }
    
    # Node handler methods
    def _handle_submission(self, state: OrchestratorState) -> OrchestratorState:
        """Handle application submission."""
        try:
            # Parse user input if it's a string
            if isinstance(state.user_input, str):
                try:
                    application_data = json.loads(state.user_input)
                    # Update application with parsed data
                    for key, value in application_data.items():
                        if hasattr(state.application, key):
                            setattr(state.application, key, value)
                except json.JSONDecodeError:
                    # Handle as a chat message if not valid JSON
                    if not state.application.chat_history:
                        state.application.chat_history = []
                    
                    state.application.chat_history.append({
                        "role": "user",
                        "content": state.user_input,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Create application record in database
            db_application = Application(
                application_id=state.application.app_id,
                name=state.application.name or "Unknown",
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
                assessment_status="pending"
            )
            
            self.db_session.add(db_application)
            self.db_session.commit()
            
            # Update state
            state.current_state = ApplicationState.DATA_COLLECTION
            state.agent_output = {
                "message": "Application submitted successfully",
                "application_id": state.application.app_id
            }
            
            # Log submission
            self.audit_logger(
                state.application.app_id,
                "application_submitted",
                "orchestrator",
                {"application_data": state.application.dict()}
            )
            
            return state
            
        except Exception as e:
            state.current_state = ApplicationState.ERROR
            state.error = f"Error in submission: {str(e)}"
            return state
    
    def _handle_data_collection(self, state: OrchestratorState) -> OrchestratorState:
        """Handle data collection from documents."""
        try:
            # Import data collector agent
            from agents.data_collector import DataCollectorAgent
            
            # Create collector agent
            collector = DataCollectorAgent()
            
            # Process documents
            for doc in state.application.documents:
                if doc.get("processed", False):
                    continue
                
                # Process document and extract data
                extracted_data = collector.process_document(
                    doc["file_path"],
                    doc["filename"],
                    doc["document_type"]
                )
                
                # Update document with extracted data
                doc["extracted_data"] = extracted_data
                doc["processed"] = True
                
                # Update application with extracted data
                for key, value in extracted_data.items():
                    if hasattr(state.application, key) and getattr(state.application, key) is None:
                        setattr(state.application, key, value)
            
            # Update state
            state.current_state = ApplicationState.VALIDATION
            state.agent_output = {
                "message": "Data collection completed",
                "extracted_fields": [
                    key for key, value in state.application.dict().items()
                    if value is not None and key != "documents" and key != "chat_history"
                ]
            }
            
            # Log data collection
            self.audit_logger(
                state.application.app_id,
                "data_collection_completed",
                "data_collector",
                {"extracted_fields": state.agent_output["extracted_fields"]}
            )
            
            return state
            
        except Exception as e:
            state.current_state = ApplicationState.ERROR
            state.error = f"Error in data collection: {str(e)}"
            return state
    
    def _handle_validation(self, state: OrchestratorState) -> OrchestratorState:
        """Handle data validation."""
        try:
            # Import validator agent
            from agents.validator import ValidatorAgent
            
            # Create validator agent
            validator = ValidatorAgent()
            
            # Validate application data
            is_valid, validation_errors = validator.validate(state.application.dict())
            
            # Update application with validation result
            state.application.validation_status = "valid" if is_valid else "invalid"
            
            if not is_valid:
                state.application.errors = validation_errors
            
            # Update application in database
            app = self.db_session.query(Application).filter(
                Application.application_id == state.application.app_id
            ).first()
            
            if app:
                app.validation_status = state.application.validation_status
                self.db_session.commit()
            
            # Determine next state
            if is_valid:
                state.current_state = ApplicationState.ASSESSMENT
            else:
                # Return to data collection if invalid
                state.current_state = ApplicationState.DATA_COLLECTION
            
            state.agent_output = {
                "message": "Validation completed",
                "is_valid": is_valid,
                "errors": validation_errors if not is_valid else []
            }
            
            # Log validation
            self.audit_logger(
                state.application.app_id,
                "validation_completed",
                "validator",
                {
                    "is_valid": is_valid,
                    "errors": validation_errors if not is_valid else []
                }
            )
            
            return state
            
        except Exception as e:
            state.current_state = ApplicationState.ERROR
            state.error = f"Error in validation: {str(e)}"
            return state
    
    def _handle_assessment(self, state: OrchestratorState) -> OrchestratorState:
        """Handle application assessment."""
        try:
            # Import assessor agent
            from agents.assessor import AssessorAgent
            
            # Create assessor agent
            assessor = AssessorAgent()
            
            # Assess application
            is_approved, reasons, assessment_details = assessor.assess_application(state.application.dict())
            
            # Update application with assessment result
            state.application.assessment_status = "approved" if is_approved else "rejected"
            state.application.risk_level = assessment_details.get("risk_level", "unknown")
            
            # Update application in database
            app = self.db_session.query(Application).filter(
                Application.application_id == state.application.app_id
            ).first()
            
            if app:
                app.assessment_status = state.application.assessment_status
                app.risk_level = state.application.risk_level
                self.db_session.commit()
            
            state.current_state = ApplicationState.COUNSELING
            state.agent_output = {
                "message": "Assessment completed",
                "is_approved": is_approved,
                "reasons": reasons,
                "assessment_details": assessment_details
            }
            
            # Log assessment
            self.audit_logger(
                state.application.app_id,
                "assessment_completed",
                "assessor",
                {
                    "is_approved": is_approved,
                    "reasons": reasons,
                    "assessment_details": assessment_details
                }
            )
            
            return state
            
        except Exception as e:
            state.current_state = ApplicationState.ERROR
            state.error = f"Error in assessment: {str(e)}"
            return state
    
    def _handle_counseling(self, state: OrchestratorState) -> OrchestratorState:
        """Handle counseling and recommendations."""
        try:
            # Import counselor agent
            from agents.counselor import CounselorAgent
            
            # Create counselor agent
            counselor = CounselorAgent()
            
            # Generate recommendations
            recommendations = counselor.generate_recommendations(state.application.dict())
            
            # Explain decision
            explanation = counselor.explain_decision(state.application.dict())
            
            # Update application with recommendations
            state.application.recommendations = recommendations
            
            # Mark as completed
            state.current_state = ApplicationState.COMPLETED
            state.agent_output = {
                "message": "Counseling completed",
                "recommendations": recommendations,
                "explanation": explanation
            }
            
            # Log counseling
            self.audit_logger(
                state.application.app_id,
                "counseling_completed",
                "counselor",
                {
                    "recommendations_count": len(recommendations),
                    "explanation": explanation
                }
            )
            
            return state
            
        except Exception as e:
            state.current_state = ApplicationState.ERROR
            state.error = f"Error in counseling: {str(e)}"
            return state
    
    def _handle_error(self, state: OrchestratorState) -> OrchestratorState:
        """Handle errors in the workflow."""
        # Log the error
        self.audit_logger(
            state.application.app_id,
            "workflow_error",
            "orchestrator",
            {"error": state.error}
        )
        
        # Determine if we can recover from the error
        if "submission" in state.error.lower():
            # Return to submission stage
            state.current_state = ApplicationState.SUBMISSION
        elif "data" in state.error.lower() or "document" in state.error.lower():
            # Return to data collection stage
            state.current_state = ApplicationState.DATA_COLLECTION
        elif "validat" in state.error.lower():
            # Return to validation stage
            state.current_state = ApplicationState.VALIDATION
        elif "assess" in state.error.lower():
            # Return to assessment stage
            state.current_state = ApplicationState.ASSESSMENT
        elif "counsel" in state.error.lower():
            # Return to counseling stage
            state.current_state = ApplicationState.COUNSELING
        else:
            # Cannot recover, mark as error
            state.current_state = ApplicationState.ERROR
        
        return state
    
    # Routing methods
    def _route_after_submission(self, state: OrchestratorState) -> str:
        """Determine next step after submission."""
        if state.current_state == ApplicationState.ERROR:
            return ApplicationState.ERROR
        return ApplicationState.DATA_COLLECTION
    
    def _route_after_data_collection(self, state: OrchestratorState) -> str:
        """Determine next step after data collection."""
        if state.current_state == ApplicationState.ERROR:
            return ApplicationState.ERROR
        return ApplicationState.VALIDATION
    
    def _route_after_validation(self, state: OrchestratorState) -> str:
        """Determine next step after validation."""
        if state.current_state == ApplicationState.ERROR:
            return ApplicationState.ERROR
        if state.current_state == ApplicationState.DATA_COLLECTION:
            return ApplicationState.DATA_COLLECTION
        return ApplicationState.ASSESSMENT
    
    def _route_after_assessment(self, state: OrchestratorState) -> str:
        """Determine next step after assessment."""
        if state.current_state == ApplicationState.ERROR:
            return ApplicationState.ERROR
        return ApplicationState.COUNSELING
    
    def _route_after_counseling(self, state: OrchestratorState) -> str:
        """Determine next step after counseling."""
        if state.current_state == ApplicationState.ERROR:
            return ApplicationState.ERROR
        return ApplicationState.COMPLETED
    
    def _route_after_error(self, state: OrchestratorState) -> str:
        """Determine next step after error handling."""
        return state.current_state.value
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        try:
            self.db_session.close()
        except:
            pass
