"""
Decision Engine for the Enhanced Social Security Application System.
Responsible for determining application status (approved/rejected/pending)
and generating personalized recommendations based on application data.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Decision engine for determining application status and providing recommendations.
    Uses both rule-based approaches and ML-based scoring to make decisions.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the decision engine with optional database session."""
        self.db_session = db_session
        logger.info("Decision Engine initialized")
    
    def evaluate_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an application and determine its status and recommendations.
        
        Args:
            application_data: Dictionary containing application information
            
        Returns:
            Dictionary with evaluation results including status, score, and recommendations
        """
        logger.info(f"Evaluating application: {application_data.get('application_id', 'Unknown ID')}")
        
        # Calculate application score (0-100)
        score = self._calculate_score(application_data)
        
        # Determine application status based on score and rules
        status, risk_level = self._determine_status(score, application_data)
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(application_data, score, status)
        
        # Prepare evaluation results
        result = {
            "application_id": application_data.get("application_id"),
            "score": score,
            "assessment_status": status,
            "risk_level": risk_level,
            "recommendations": recommendations
        }
        
        logger.info(f"Application {application_data.get('application_id', 'Unknown')} evaluated: {status}, score: {score}")
        return result
    
    def _calculate_score(self, data: Dict[str, Any]) -> int:
        """
        Calculate application score based on various factors.
        
        Args:
            data: Application data
            
        Returns:
            Score between 0-100
        """
        # Initialize base score
        score = 50
        
        # Financial factors (income relative to family size)
        try:
            income = float(data.get("income", 0))
            family_size = int(data.get("family_size", 1))
            
            # Per capita income
            per_capita_income = income / max(1, family_size)
            
            # Adjust score based on per capita income
            if per_capita_income < 1000:
                score += 20  # Very low income, high need
            elif per_capita_income < 3000:
                score += 15  # Low income
            elif per_capita_income < 5000:
                score += 5   # Moderate income
            elif per_capita_income < 8000:
                score -= 5   # Above average income
            else:
                score -= 15  # High income
                
            # Employment status factor
            employment_status = data.get("employment_status", "").lower()
            if employment_status in ["unemployed", "unable to work"]:
                score += 15
            elif employment_status in ["part-time", "student"]:
                score += 10
            elif employment_status in ["self-employed"]:
                score += 5
                
            # Document verification
            documents = data.get("documents", [])
            verified_docs = [d for d in documents if d.get("status") == "verified"]
            if len(verified_docs) >= 3:
                score += 5
                
            # Duration adjustment (longer employment = more stable)
            employment_duration = int(data.get("employment_duration", 0))
            if employment_duration > 36:  # 3+ years
                score -= 5  # More stable, less urgent need
            elif employment_duration < 6:  # Less than 6 months
                score += 5  # Less stable, higher need
                
            # Assets vs Liabilities
            assets = float(data.get("assets_value", 0))
            liabilities = float(data.get("liabilities_value", 0))
            if liabilities > assets and liabilities > 0:
                debt_ratio = min(liabilities / max(assets, 1), 5)  # Cap at 5x
                score += int(5 * debt_ratio)  # Higher debt = higher need
            
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            # Fall back to average score on error
        
        # Ensure score is within bounds
        return max(0, min(100, score))
    
    def _determine_status(self, score: int, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Determine application status and risk level based on score and other factors.
        
        Args:
            score: Application score (0-100)
            data: Application data
            
        Returns:
            Tuple of (status, risk_level)
        """
        # Simple rule-based approach
        if score >= 75:
            status = "approved"
            risk_level = "low"
        elif score >= 60:
            status = "approved"
            risk_level = "medium"
        elif score >= 40:
            status = "review"
            risk_level = "medium"
        elif score >= 25:
            status = "review"
            risk_level = "high"
        else:
            status = "rejected"
            risk_level = "high"
            
        # Check validation status
        validation_status = data.get("validation_status", "").lower()
        if validation_status in ["flagged", "incomplete"]:
            status = "review"  # Override for flagged applications
            risk_level = "high"
            
        return status, risk_level
    
    def _generate_recommendations(self, data: Dict[str, Any], score: int, status: str) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations based on application data.
        
        Args:
            data: Application data
            score: Application score
            status: Application status
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Financial Support recommendation
        if score >= 40:
            eligibility = min(100, score + 10)
            priority = "high" if score >= 70 else "medium"
            recommendations.append({
                "id": 1,
                "category": "Financial Assistance",
                "description": "Based on your income and family size, you qualify for monthly financial support.",
                "priority": priority,
                "eligibility": eligibility
            })
        
        # Employment Support
        employment_status = data.get("employment_status", "").lower()
        if employment_status in ["unemployed", "part-time", "student"]:
            eligibility = 80
            recommendations.append({
                "id": 2,
                "category": "Employment Support",
                "description": "You may benefit from our job placement program to improve your long-term financial stability.",
                "priority": "high" if employment_status == "unemployed" else "medium",
                "eligibility": eligibility
            })
        
        # Housing Assistance
        monthly_expenses = float(data.get("monthly_expenses", 0))
        income = float(data.get("income", 1))
        if monthly_expenses > 0 and (monthly_expenses / income) > 0.5:
            recommendations.append({
                "id": 3,
                "category": "Housing Assistance",
                "description": "You may qualify for housing support benefits to reduce your living expenses.",
                "priority": "medium",
                "eligibility": 70
            })
        
        # Education Support
        if employment_status in ["student", "part-time", "unemployed"]:
            recommendations.append({
                "id": 4,
                "category": "Education Support",
                "description": "You may be eligible for education grants or subsidized training programs.",
                "priority": "medium" if employment_status == "student" else "low",
                "eligibility": 65
            })
            
        # Healthcare Support
        family_size = int(data.get("family_size", 1))
        if family_size > 2 or score > 60:
            recommendations.append({
                "id": 5,
                "category": "Healthcare Support",
                "description": "Your family may qualify for subsidized healthcare services.",
                "priority": "medium",
                "eligibility": 75
            })
        
        return recommendations
