"""
Enhanced Validator Agent - Performs sophisticated validation on application data.
Handles data consistency checking, anomaly detection, and cross-document validation.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, date
import json

from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field

# Import utility for local LLM
from utils.llm_factory import get_llm
from database.db_setup import Application, get_db

# Load environment variables
load_dotenv()

class ValidatorAgent:
    """
    Enhanced validator agent that checks data completeness, consistency, and validity.
    Performs sophisticated validation including cross-document checks and anomaly detection.
    """
    
    def __init__(self):
        """Initialize the validator agent with necessary components."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ValidatorAgent")
        
        # Define validation rules
        self.validation_rules = {
            "name": self._validate_name,
            "email": self._validate_email,
            "phone": self._validate_phone,
            "emirates_id": self._validate_emirates_id,
            "income": self._validate_income,
            "family_size": self._validate_family_size,
            "address": self._validate_address,
            "employment_status": self._validate_employment_status,
            "employer": self._validate_employer,
            "job_title": self._validate_job_title,
            "employment_duration": self._validate_employment_duration,
            "monthly_expenses": self._validate_monthly_expenses,
            "assets_value": self._validate_assets_value,
            "liabilities_value": self._validate_liabilities_value,
        }
        
        # Define required fields
        self.required_fields = [
            "name",
            "income",
            "family_size",
            "address",
        ]
        
        # Define cross-field validation rules
        self.cross_field_validations = [
            self._validate_income_vs_expenses,
            self._validate_employment_consistency,
            self._validate_financial_consistency,
        ]
        
        # Initialize LLM for advanced validation
        self.llm = get_llm()
        
        # Initialize database session
        self.db = next(get_db())
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate application data against defined rules.
        
        Args:
            data: Dictionary of application data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.logger.info("Starting validation of application data")
        errors = []
        
        # Check if all required fields are present
        missing_fields = [field for field in self.required_fields if field not in data or data[field] is None]
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            self.logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate each field that is present
        for field, rule in self.validation_rules.items():
            if field in data and data[field] is not None:
                if not rule(data[field]):
                    error_msg = self._get_error_message(field, data[field])
                    errors.append(error_msg)
                    self.logger.warning(error_msg)
        
        # Perform cross-field validations
        for cross_validation in self.cross_field_validations:
            result, error_msg = cross_validation(data)
            if not result:
                errors.append(error_msg)
                self.logger.warning(error_msg)
        
        # Check for potential anomalies and fraud indicators
        anomalies = self._detect_anomalies(data)
        if anomalies:
            errors.extend(anomalies)
            for anomaly in anomalies:
                self.logger.warning(f"Anomaly detected: {anomaly}")
        
        is_valid = len(errors) == 0
        
        # Update validation status in database if application_id is provided
        if "application_id" in data and data["application_id"]:
            try:
                app = self.db.query(Application).filter(
                    Application.filename == data["filename"]
                ).first()
                
                if app:
                    app.validation_status = "valid" if is_valid else "invalid"
                    self.db.commit()
            except Exception as e:
                self.logger.error(f"Error updating validation status: {str(e)}")
                self.db.rollback()
        
        self.logger.info(f"Validation completed. Valid: {is_valid}")
        return is_valid, errors
    
    async def explain_validation_errors(self, application_id: str, user_query: str = None) -> Dict[str, Any]:
        """
        Explain validation errors to the user.
        
        Args:
            application_id: ID of the application
            user_query: Optional query from user
            
        Returns:
            Dictionary with explanation text and suggestions
        """
        try:
            # Get application from database
            app = self.db.query(Application).filter(
                Application.filename == application_id
            ).first()
            
            if not app:
                return {
                    "text": f"Application with ID {application_id} not found",
                    "suggestions": ["Check application ID and try again"]
                }
            
            # If application is valid, return success message
            if app.validation_status == "valid":
                return {
                    "text": "Your application has been validated successfully. No errors were found.",
                    "suggestions": [
                        "What happens next?",
                        "When will I receive a decision?",
                        "Can I make changes to my application?"
                    ]
                }
            
            # Get application data
            app_data = {
                "name": app.name,
                "email": app.email,
                "phone": app.phone,
                "emirates_id": app.emirates_id,
                "income": app.income,
                "family_size": app.family_size,
                "address": app.address,
                "employment_status": app.employment_status,
                "employer": app.employer,
                "job_title": app.job_title,
                "employment_duration": app.employment_duration,
                "monthly_expenses": app.monthly_expenses,
                "assets_value": app.assets_value,
                "liabilities_value": app.liabilities_value,
                "application_id": app.application_id
            }
            
            # Re-validate to get current errors
            is_valid, errors = self.validate(app_data)
            
            # If no specific query, provide general explanation
            if not user_query:
                explanation = "Your application has the following validation issues:\n\n"
                for error in errors:
                    explanation += f"• {error}\n"
                
                explanation += "\nPlease correct these issues and resubmit your application."
                
                return {
                    "text": explanation,
                    "suggestions": [
                        "How do I fix these issues?",
                        "What documents do I need to provide?",
                        "Can I update my application?"
                    ]
                }
            
            # If user has a specific query, use LLM to generate a targeted response
            prompt = f"""
            The user's application has the following validation errors:
            {json.dumps(errors)}
            
            The user is asking: "{user_query}"
            
            Provide a helpful, concise response that directly addresses their question. Focus on explaining what they need to do to fix the validation issues or provide the necessary information to complete their application.
            """
            
            try:
                llm_response = self.llm.invoke(prompt)
                return {
                    "text": llm_response,
                    "suggestions": [
                        "How do I upload new documents?",
                        "What formats are accepted?",
                        "When can I resubmit my application?"
                    ],
                    "agent": "validator"
                }
            except Exception as e:
                self.logger.error(f"Error generating LLM response: {str(e)}")
                return {
                    "text": "I'm having trouble providing a specific response to your question. Please try again or ask about a specific validation error.",
                    "suggestions": ["List all validation errors", "How to resubmit my application"]
                }
            
        except Exception as e:
            self.logger.error(f"Error explaining validation errors: {str(e)}")
            return {
                "text": "I encountered an error while retrieving your validation results. Please try again later.",
                "error": str(e)
            }
    
    # Field validation methods
    def _validate_name(self, value: str) -> bool:
        """Validate name field."""
        if not isinstance(value, str):
            return False
        
        # Name should be at least 2 characters, maximum 100
        if not (2 <= len(value.strip()) <= 100):
            return False
        
        # Name should contain at least one space (first and last name)
        if ' ' not in value.strip():
            return False
        
        # Name should not contain numbers or special characters (except hyphen, apostrophe)
        if re.search(r'[^a-zA-Z\s\-\']', value):
            return False
        
        return True
    
    def _validate_email(self, value: str) -> bool:
        """Validate email field."""
        if not isinstance(value, str):
            return False
        
        # Simple email validation using regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, value))
    
    def _validate_phone(self, value: str) -> bool:
        """Validate phone field."""
        if not isinstance(value, str):
            return False
        
        # Remove non-numeric characters for validation
        digits = re.sub(r'\D', '', value)
        
        # Phone number should have between 8 and 15 digits
        return 8 <= len(digits) <= 15
    
    def _validate_emirates_id(self, value: str) -> bool:
        """Validate Emirates ID field."""
        if not isinstance(value, str):
            return False
        
        # Remove non-alphanumeric characters
        id_clean = re.sub(r'[^0-9]', '', value)
        
        # Emirates ID should be 15 digits
        if len(id_clean) != 15:
            return False
        
        # Additional validation: Emirates ID starts with 784 (UAE country code)
        return id_clean.startswith('784')
    
    def _validate_income(self, value) -> bool:
        """Validate income field."""
        try:
            if not isinstance(value, (int, float)):
                return False
            
            # Income must be non-negative and less than 10 million AED (reasonable upper limit)
            return 0 <= value < 10_000_000
        except:
            return False
    
    def _validate_family_size(self, value) -> bool:
        """Validate family size field."""
        try:
            if not isinstance(value, int):
                return False
            
            # Family size must be between 1 and 30 (reasonable range)
            return 1 <= value <= 30
        except:
            return False
    
    def _validate_address(self, value) -> bool:
        """Validate address field."""
        if not isinstance(value, str):
            return False
        
        # Address must be at least 10 characters and less than 500
        return 10 <= len(value.strip()) <= 500
    
    def _validate_employment_status(self, value) -> bool:
        """Validate employment status field."""
        if not isinstance(value, str):
            return False
        
        # Valid employment statuses
        valid_statuses = [
            'employed', 'self-employed', 'unemployed', 'retired', 
            'student', 'homemaker', 'unable to work', 'part-time'
        ]
        
        return value.lower() in valid_statuses
    
    def _validate_employer(self, value) -> bool:
        """Validate employer field."""
        if value is None:
            return True  # Optional field
        
        if not isinstance(value, str):
            return False
        
        # Employer name should be at least 2 characters, maximum 200
        return 2 <= len(value.strip()) <= 200
    
    def _validate_job_title(self, value) -> bool:
        """Validate job title field."""
        if value is None:
            return True  # Optional field
        
        if not isinstance(value, str):
            return False
        
        # Job title should be at least 2 characters, maximum 100
        return 2 <= len(value.strip()) <= 100
    
    def _validate_employment_duration(self, value) -> bool:
        """Validate employment duration field (in months)."""
        if value is None:
            return True  # Optional field
        
        try:
            if not isinstance(value, int):
                return False
            
            # Employment duration must be non-negative and less than 720 months (60 years)
            return 0 <= value < 720
        except:
            return False
    
    def _validate_monthly_expenses(self, value) -> bool:
        """Validate monthly expenses field."""
        if value is None:
            return True  # Optional field
        
        try:
            if not isinstance(value, (int, float)):
                return False
            
            # Monthly expenses must be non-negative and less than 1 million AED
            return 0 <= value < 1_000_000
        except:
            return False
    
    def _validate_assets_value(self, value) -> bool:
        """Validate assets value field."""
        if value is None:
            return True  # Optional field
        
        try:
            if not isinstance(value, (int, float)):
                return False
            
            # Assets value must be non-negative and less than 100 million AED
            return 0 <= value < 100_000_000
        except:
            return False
    
    def _validate_liabilities_value(self, value) -> bool:
        """Validate liabilities value field."""
        if value is None:
            return True  # Optional field
        
        try:
            if not isinstance(value, (int, float)):
                return False
            
            # Liabilities value must be non-negative and less than 50 million AED
            return 0 <= value < 50_000_000
        except:
            return False
    
    # Cross-field validation methods
    def _validate_income_vs_expenses(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate income against expenses."""
        if 'income' in data and 'monthly_expenses' in data and data['income'] is not None and data['monthly_expenses'] is not None:
            # Monthly expenses should not exceed 90% of income (reasonable threshold)
            if data['monthly_expenses'] > data['income'] * 0.9:
                return False, "Monthly expenses exceed 90% of income, which suggests potential financial instability."
        
        return True, ""
    
    def _validate_employment_consistency(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate consistency between employment fields."""
        employment_status = data.get('employment_status')
        employer = data.get('employer')
        job_title = data.get('job_title')
        employment_duration = data.get('employment_duration')
        
        if employment_status and employment_status.lower() in ['unemployed', 'student', 'retired', 'homemaker']:
            # If unemployed/student/retired, shouldn't have current employer or job title
            if employer or job_title:
                return False, f"Employment status is '{employment_status}' but employer or job title is provided."
        
        elif employment_status and employment_status.lower() in ['employed', 'self-employed', 'part-time']:
            # If employed, should have employer and job title
            if not employer or not job_title:
                return False, f"Employment status is '{employment_status}' but employer or job title is missing."
        
        return True, ""
    
    def _validate_financial_consistency(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate consistency between financial fields."""
        income = data.get('income')
        assets_value = data.get('assets_value')
        liabilities_value = data.get('liabilities_value')
        
        if income and assets_value and liabilities_value:
            # Check if liabilities are extremely high compared to income and assets
            if liabilities_value > income * 10 and liabilities_value > assets_value * 2:
                return False, "Liabilities are unusually high compared to income and assets, suggesting potential financial distress."
        
        return True, ""
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[str]:
        """Detect potential anomalies and fraud indicators."""
        anomalies = []
        
        # Check for unusually high income
        if data.get('income') and data['income'] > 1_000_000:
            anomalies.append("Income is unusually high and requires verification.")
        
        # Check for unusually large family size
        if data.get('family_size') and data['family_size'] > 15:
            anomalies.append("Family size is unusually large and requires verification.")
        
        # Check for suspiciously short employment duration with high income
        if data.get('employment_duration') and data.get('income'):
            if data['employment_duration'] < 6 and data['income'] > 300_000:
                anomalies.append("High income with very short employment duration requires verification.")
        
        # Check for inconsistency between assets and income
        if data.get('assets_value') and data.get('income'):
            if data['assets_value'] > data['income'] * 20:
                anomalies.append("Assets value is unusually high compared to income and requires verification.")
        
        return anomalies
    
    def _get_error_message(self, field: str, value) -> str:
        """Generate descriptive error message for failed validation."""
        error_messages = {
            "name": f"Invalid name: '{value}'. Must be a valid full name (first and last name) without numbers or special characters.",
            "email": f"Invalid email address: '{value}'. Must be in the format 'name@domain.com'.",
            "phone": f"Invalid phone number: '{value}'. Must contain 8-15 digits.",
            "emirates_id": f"Invalid Emirates ID: '{value}'. Must be a 15-digit number starting with 784.",
            "income": f"Invalid income value: {value}. Must be a number between 0 and 10,000,000 AED.",
            "family_size": f"Invalid family size: {value}. Must be a number between 1 and 30.",
            "address": f"Invalid address: '{value}'. Must be between 10 and 500 characters.",
            "employment_status": f"Invalid employment status: '{value}'. Must be one of: employed, self-employed, unemployed, retired, student, homemaker, unable to work, part-time.",
            "employer": f"Invalid employer: '{value}'. Must be between 2 and 200 characters.",
            "job_title": f"Invalid job title: '{value}'. Must be between 2 and 100 characters.",
            "employment_duration": f"Invalid employment duration: {value}. Must be a non-negative number less than 720 months (60 years).",
            "monthly_expenses": f"Invalid monthly expenses: {value}. Must be a non-negative number less than 1,000,000 AED.",
            "assets_value": f"Invalid assets value: {value}. Must be a non-negative number less than 100,000,000 AED.",
            "liabilities_value": f"Invalid liabilities value: {value}. Must be a non-negative number less than 50,000,000 AED."
        }
        
        return error_messages.get(field, f"Validation failed for '{field}': {value}")
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        try:
            self.db.close()
        except:
            pass
