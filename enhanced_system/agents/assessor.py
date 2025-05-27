"""
Enhanced Assessor Agent - Evaluates applications for eligibility and risk.
Uses ML models for classification and decision making.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field

# Import utility for local LLM
from utils.llm_factory import get_llm
from database.db_setup import Application, get_db

# Import decision engine
from models.model_loader import ModelLoader

# Load environment variables
load_dotenv()

class AssessorAgent:
    """
    Enhanced assessor agent that evaluates applications for financial support eligibility.
    Uses machine learning models for risk assessment and decision making.
    """
    
    def __init__(self):
        """Initialize the assessor agent with necessary components."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AssessorAgent")
        
        # Risk and eligibility thresholds
        self.thresholds = {
            'income': 50000,  # AED
            'min_income_per_member': 10000,  # AED per family member
            'family_size': 8,  # Maximum family size
            'debt_to_income_ratio': 0.5,  # Maximum acceptable debt-to-income ratio
            'min_employment_duration': 6,  # Minimum employment duration in months
        }
        
        # Initialize ML models
        self.risk_model = self._initialize_risk_model()
        self.eligibility_model = self._initialize_eligibility_model()
        
        # Initialize database connection and decision engine
        self.db = next(get_db())
        self.model_loader = ModelLoader(self.db)
        self.logger.info("Decision engine initialized for assessment")
        
        # Initialize LLM for explanation generation
        self.llm = get_llm()
        
        # Initialize database session
        self.db = next(get_db())
    
    def _initialize_risk_model(self):
        """Initialize or load the risk assessment model."""
        model_path = os.path.join(os.path.dirname(__file__), "../models/risk_model.pkl")
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                self.logger.info("Loading existing risk model")
                return joblib.load(model_path)
        except Exception as e:
            self.logger.warning(f"Could not load risk model: {str(e)}")
        
        # Create a new model if loading fails
        self.logger.info("Creating new risk assessment model")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # We'll save this later when we have data to train it
        return model
    
    def _initialize_eligibility_model(self):
        """Initialize or load the eligibility classification model."""
        model_path = os.path.join(os.path.dirname(__file__), "../models/eligibility_model.pkl")
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                self.logger.info("Loading existing eligibility model")
                return joblib.load(model_path)
        except Exception as e:
            self.logger.warning(f"Could not load eligibility model: {str(e)}")
        
        # Create a new model if loading fails
        self.logger.info("Creating new eligibility model")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # We'll save this later when we have data to train it
        return model
    
    def assess_application(self, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Assess an application for financial support eligibility using the decision engine.
        
        Args:
            data: Dictionary containing application data
            
        Returns:
            Tuple of (is_approved, reasons, assessment_details)
        """
        self.logger.info(f"Assessing application for {data.get('name', 'Unknown')}")
        
        # Calculate key metrics for legacy purposes
        income_per_member = data.get('income', 0) / data.get('family_size', 1) if data.get('family_size', 0) > 0 else 0
        debt_to_income = data.get('liabilities_value', 0) / data.get('income', 1) if data.get('income', 0) > 0 else 0
        net_worth = data.get('assets_value', 0) - data.get('liabilities_value', 0)
        
        # Use the decision engine to evaluate the application
        try:
            self.logger.info("Using decision engine to evaluate application")
            decision_result = self.model_loader.evaluate_application(data)
            
            # Extract decision results
            status = decision_result.get("assessment_status", "review")
            risk_level = decision_result.get("risk_level", "medium")
            score = decision_result.get("score", 50)
            recommendations = decision_result.get("recommendations", [])
            
            # Map status to approval decision
            is_approved = status == "approved"
            
            # Initialize reasons based on score
            reasons = []
            if status == "rejected":
                reasons.append(f"Application score ({score}) below approval threshold")
            elif status == "review":
                reasons.append(f"Application requires review (score: {score})")
            
            # Add risk level reason if high
            if risk_level == "high":
                reasons.append("High risk application requires additional review")
                
            # Create comprehensive assessment details
            assessment_details = {
                'risk_level': risk_level,
                'income_per_member': income_per_member,
                'debt_to_income_ratio': debt_to_income,
                'net_worth': net_worth,
                'assessment_date': datetime.utcnow().isoformat(),
                'score': score,
                'recommendations': recommendations,
                'status': status
            }
            
            self.logger.info(f"Decision engine evaluation complete. Status: {status}, Score: {score}")
            
        except Exception as e:
            # Fall back to the legacy assessment method if decision engine fails
            self.logger.warning(f"Decision engine evaluation failed: {str(e)}. Falling back to legacy assessment.")
            
            # Determine risk level using model-based and rule-based approaches
            risk_level = self._calculate_risk_level(data)
            
            # Store assessment details
            assessment_details = {
                'risk_level': risk_level,
                'income_per_member': income_per_member,
                'debt_to_income_ratio': debt_to_income,
                'net_worth': net_worth,
                'assessment_date': datetime.utcnow().isoformat()
            }
            
            # Apply eligibility rules
            reasons = []
            is_approved = True
            
            # Check income threshold
            if data.get('income', 0) < self.thresholds['income']:
                reasons.append("Income below minimum threshold")
                is_approved = False
            
            # Check income per family member
            if income_per_member < self.thresholds['min_income_per_member']:
                reasons.append("Income per family member below minimum threshold")
                is_approved = False
            
            # Check family size
            if data.get('family_size', 0) > self.thresholds['family_size']:
                reasons.append("Family size exceeds maximum limit")
                is_approved = False
            
            # Check debt-to-income ratio
            if debt_to_income > self.thresholds['debt_to_income_ratio']:
                reasons.append("Debt-to-income ratio exceeds maximum threshold")
                is_approved = False
            
            # Check employment duration for employed applicants
            if data.get('employment_status') in ['employed', 'self-employed'] and \
               data.get('employment_duration', 0) < self.thresholds['min_employment_duration']:
                reasons.append("Employment duration below minimum requirement")
                is_approved = False
            
            # Apply model-based decision if available and no hard rules violated
            if is_approved and self.eligibility_model is not None:
                try:
                    model_decision = self._get_model_decision(data)
                    if not model_decision:
                        reasons.append("Application does not meet eligibility criteria based on predictive model")
                        is_approved = False
                except Exception as e:
                    self.logger.warning(f"Model-based decision failed: {str(e)}")
            
            else:
                processed_data[field] = 0 if field_type == int else 0.0
        
        # Ensure required string fields are present
        string_fields = ['employment_status', 'job_title', 'address']
        for field in string_fields:
            if field not in processed_data or not processed_data[field]:
                processed_data[field] = 'unknown'
        
        # Handle special cases
        if processed_data['family_size'] <= 0:
            processed_data['family_size'] = 1
        
        return processed_data
    
    def _generate_llm_explanation(self, is_approved: bool, assessment_details: Dict[str, Any], 
                                 application_data: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of the assessment decision using LLM.
        
        Args:
            is_approved: Whether the application was approved
            assessment_details: Assessment results from ML models
            application_data: Processed application data
            
        Returns:
            Explanation text
        """
        try:
            # Create prompt for LLM
            prompt = f"""
            Generate a detailed but concise explanation of the following social support application assessment.
            
            Applicant Information:
            - Name: {data.get('name', 'Unknown')}
            - Income: {data.get('income', 0)} AED
            - Family Size: {data.get('family_size', 0)}
            - Employment Status: {data.get('employment_status', 'Unknown')}
            
            Assessment Result:
            - Decision: {"Approved" if is_approved else "Not Approved"}
            - Risk Level: {assessment_details['risk_level'].upper()}
            
            {"Reasons for decision:" if reasons else ""}
            {chr(10).join([f"- {reason}" for reason in reasons]) if reasons else ""}
            
            Additional Assessment Details:
            - Income per Family Member: {assessment_details['income_per_member']:.2f} AED
            - Debt-to-Income Ratio: {assessment_details['debt_to_income_ratio']:.2f}
            - Net Worth: {assessment_details['net_worth']:.2f} AED
            
            Please provide an explanation in a professional tone that explains the decision and the factors that influenced it.
            If not approved, provide constructive guidance on what the applicant can do to improve their situation.
            Keep your response under 250 words.
            """
            
            # Get explanation from LLM
            explanation = self.llm.invoke(prompt)
            return explanation
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            
            # Fallback explanation
            if is_approved:
                return "Your application has been approved based on our assessment of your financial situation and needs."
            else:
                return f"Your application was not approved for the following reasons: {', '.join(reasons)}. Please review and address these issues to improve your eligibility."
    
    def get_recent_assessments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent assessments from the database.
        
        Args:
            limit: Maximum number of assessments to retrieve
            
        Returns:
            List of assessment dictionaries
        """
        try:
            applications = self.db.query(Application).order_by(
                Application.updated_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': app.id,
                    'application_id': app.application_id,
                    'name': app.name,
                    'income': app.income,
                    'family_size': app.family_size,
                    'address': app.address,
                    'validation_status': app.validation_status,
                    'assessment_status': app.assessment_status,
                    'risk_level': app.risk_level,
                    'created_at': app.created_at.isoformat(),
                    'updated_at': app.updated_at.isoformat()
                }
                for app in applications
            ]
        except Exception as e:
            self.logger.error(f"Error retrieving assessments: {str(e)}")
            return []
    
    def train_models(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train risk and eligibility models with historical data.
        
        Args:
            training_data: List of historical application data with outcomes
            
        Returns:
            Boolean indicating training success
        """
        try:
            if not training_data:
                self.logger.warning("No training data provided")
                return False
            
            # Prepare features and labels
            features = []
            risk_labels = []
            eligibility_labels = []
            
            for item in training_data:
                # Extract features
                feature_vector = self._extract_features(item)
                features.append(feature_vector)
                
                # Extract labels
                risk_labels.append(1 if item.get('risk_level') == 'high' else 0)
                eligibility_labels.append(1 if item.get('assessment_status') == 'approved' else 0)
            
            # Convert to numpy arrays
            X = np.array(features)
            y_risk = np.array(risk_labels)
            y_eligibility = np.array(eligibility_labels)
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train risk model
            self.risk_model.fit(X_scaled, y_risk)
            
            # Train eligibility model
            self.eligibility_model.fit(X_scaled, y_eligibility)
            
            # Save models
            model_dir = os.path.join(os.path.dirname(__file__), "../models")
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.risk_model, os.path.join(model_dir, "risk_model.pkl"))
            joblib.dump(self.eligibility_model, os.path.join(model_dir, "eligibility_model.pkl"))
            
            self.logger.info("Models trained and saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        try:
            self.db.close()
        except:
            pass
