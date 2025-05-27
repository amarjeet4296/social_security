"""
Enhanced Counselor Agent - Provides personalized guidance and recommendations.
Uses LLM for generating tailored advice and support options.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field

# Import utility for local LLM
from utils.llm_factory import get_llm
from database.db_setup import Application, Recommendation, Interaction, get_db
from database.chroma_manager import ChromaManager

# Load environment variables
load_dotenv()

class CounselorAgent:
    """
    Enhanced counselor agent that provides personalized guidance and recommendations.
    Uses LLM for generating tailored advice and contextual support.
    """
    
    def __init__(self):
        """Initialize the counselor agent with necessary components."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CounselorAgent")
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize database session
        self.db = next(get_db())
        
        # Initialize ChromaDB manager
        self.chroma_manager = ChromaManager()
        
        # Initialize policy documents if not already initialized
        self._initialize_policy_documents()
    
    def _initialize_policy_documents(self):
        """Initialize policy documents in ChromaDB if not already present."""
        try:
            # Check if policies already exist
            results = self.chroma_manager.search_documents("policies", "requirements", n_results=1)
            if results:
                self.logger.info("Policy documents already initialized")
                return
            
            # Add policy documents
            policies = [
                {
                    "text": "Income Requirements: Minimum annual income of 50,000 AED for single applicants. For families, add 10,000 AED per additional family member. Income must be verifiable through pay stubs, tax returns, or bank statements.",
                    "metadata": {"category": "income", "type": "requirement"}
                },
                {
                    "text": "Family Size Guidelines: Maximum family size of 8 members. Each additional family member must be documented with birth certificates or legal guardianship papers. Special circumstances may be considered with additional documentation.",
                    "metadata": {"category": "family", "type": "guideline"}
                },
                {
                    "text": "Required Documents: 1) Valid government-issued ID, 2) Proof of income (last 3 months), 3) Proof of residence (utility bills or lease), 4) Family documentation (birth certificates), 5) Employment verification letter. All documents must be current and notarized if required.",
                    "metadata": {"category": "documents", "type": "requirement"}
                },
                {
                    "text": "Application Process: 1) Submit initial application with basic information, 2) Provide required documentation within 30 days, 3) Undergo verification process (7-10 business days), 4) Receive decision within 15 business days. Rejected applications can be appealed within 30 days with additional documentation.",
                    "metadata": {"category": "process", "type": "guideline"}
                },
                {
                    "text": "Risk Assessment: Applications are evaluated based on income stability, family size, and documentation completeness. High-risk factors include: income below requirements, incomplete documentation, or family size exceeding guidelines. Mitigation strategies must be provided for high-risk applications.",
                    "metadata": {"category": "risk", "type": "guideline"}
                },
                {
                    "text": "Financial Support Options: 1) Direct Financial Assistance: Monthly stipend based on income gap and family size, 2) Housing Subsidy: Up to 30% of rent for qualifying families, 3) Education Grants: Coverage for tuition and supplies for dependent children, 4) Healthcare Assistance: Coverage for essential medical services not covered by insurance.",
                    "metadata": {"category": "support", "type": "options"}
                },
                {
                    "text": "Economic Enablement Programs: 1) Vocational Training: Free courses in high-demand skills, 2) Job Placement Assistance: Resume building, interview preparation, and employer matching, 3) Small Business Grants: Seed funding for approved business plans, 4) Professional Certification: Financial support for obtaining professional certifications.",
                    "metadata": {"category": "enablement", "type": "programs"}
                },
                {
                    "text": "Appeals Process: If your application is rejected, you may appeal within 30 days by submitting additional documentation addressing the specific reasons for rejection. Appeals are reviewed by a separate committee and decisions are typically made within 20 business days. A second appeal may be submitted if new evidence becomes available.",
                    "metadata": {"category": "appeals", "type": "process"}
                }
            ]
            
            # Add policies to ChromaDB
            for policy in policies:
                self.chroma_manager.add_documents(
                    [policy["text"]],
                    [policy["metadata"]],
                    "policies"
                )
            
            self.logger.info("Policy documents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing policy documents: {str(e)}")
    
    def generate_recommendations(
        self, 
        application_data: Dict[str, Any],
        assessment_status: Optional[str] = None,
        risk_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized recommendations based on application data, assessment status, and risk level.
        
        Args:
            application_data: Dictionary containing application data
            assessment_status: The assessment status (e.g., 'approved', 'rejected')
            risk_level: The calculated risk level (e.g., 'low', 'medium', 'high')
            
        Returns:
            Dictionary containing 'recommendations' list and 'summary' string.
        """
        recommendations_list = []
        application_id = application_data.get("application_id")
        
        # Basic summary, can be enhanced based on status and risk
        summary_message = f"Counseling session for application {application_id}."

        try:
            self.logger.info(f"Generating recommendations for application {application_id} (Status: {assessment_status}, Risk: {risk_level})")
            
            # TODO: Pass assessment_status and risk_level to helper methods if they need to adapt their logic
            # Generate financial support recommendations
            financial_recommendations = self._generate_financial_recommendations(application_data)
            recommendations_list.extend(financial_recommendations)
            
            # Generate economic enablement recommendations
            enablement_recommendations = self._generate_enablement_recommendations(application_data)
            recommendations_list.extend(enablement_recommendations)
            
            # Generate document and process recommendations
            process_recommendations = self._generate_process_recommendations(application_data)
            recommendations_list.extend(process_recommendations)

            # Enhance summary based on assessment status
            if assessment_status == "approved":
                summary_message += " Application approved. Recommendations focus on next steps and available support."
            elif assessment_status == "rejected":
                summary_message += " Application rejected. Recommendations focus on areas for improvement and alternative options."
            elif assessment_status:
                summary_message += f" Application status is {assessment_status}. Review recommendations for guidance."
            else:
                summary_message += " Application status pending or unknown. General recommendations provided."
            
            # Store recommendations in database if application_id is provided
            if application_id:
                self._store_recommendations(application_id, recommendations_list)
            
            return {
                "recommendations": recommendations_list,
                "summary": summary_message 
            }
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            error_recommendation = {
                "category": "Error",
                "priority": "high",
                "description": f"Error generating recommendations: {str(e)}",
                "action_items": ["Please try again later or contact support"]
            }
            return {
                "recommendations": [error_recommendation],
                "summary": f"Failed to generate recommendations for application {application_id} due to an error. Please contact support."
            }
    
    def explain_decision(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain the decision and provide next steps.
        
        Args:
            application_data: Dictionary containing application data
            
        Returns:
            Dictionary with explanation and next steps
        """
        status = application_data.get("assessment_status", "pending")
        risk_level = application_data.get("risk_level", "medium")
        application_id = application_data.get("application_id")
        
        try:
            self.logger.info(f"Explaining decision for application {application_id}")
            
            # Generate explanation using LLM
            if status == "rejected":
                explanation = self._generate_rejection_explanation(application_data)
                next_steps = [
                    "Review the feedback provided",
                    "Address any missing requirements",
                    "Submit additional documentation if needed",
                    "Consider reapplying after addressing concerns",
                    "You can appeal the decision within 30 days"
                ]
                timeline = "You can appeal or reapply after 30 days"
            elif status == "approved":
                explanation = self._generate_approval_explanation(application_data)
                next_steps = [
                    "Review the approval letter",
                    "Complete any remaining paperwork",
                    "Schedule a follow-up appointment if needed",
                    "Attend orientation for support programs"
                ]
                timeline = "Next steps should be completed within 14 days"
            else:
                explanation = "Your application is still being processed. Our team is reviewing your information and documents."
                next_steps = [
                    "Wait for the review process to complete",
                    "Respond to any requests for additional information",
                    "Keep your contact information up to date"
                ]
                timeline = "Review process typically takes 7-10 business days"
            
            return {
                "status": status,
                "explanation": explanation,
                "next_steps": next_steps,
                "timeline": timeline
            }
        except Exception as e:
            self.logger.error(f"Error explaining decision: {str(e)}")
            return {
                "status": status,
                "explanation": f"We are currently unable to provide a detailed explanation due to a system error. Your application status is: {status}.",
                "next_steps": ["Contact our support team for assistance"],
                "timeline": "Please allow 1-2 business days for a response"
            }
    
    async def provide_guidance(self, application_id: str, user_query: str) -> Dict[str, Any]:
        """
        Provide personalized guidance based on user query and application data.
        
        Args:
            application_id: ID of the application
            user_query: User's question or query
            
        Returns:
            Dictionary with guidance text and suggestions
        """
        try:
            self.logger.info(f"Providing guidance for application {application_id}, query: {user_query}")
            
            # Get application from database
            app = self.db.query(Application).filter(
                Application.filename == application_id
            ).first()
            
            if not app:
                return {
                    "text": f"Application with ID {application_id} not found",
                    "suggestions": ["Check application ID and try again"]
                }
            
            # Search for relevant policy information
            policy_results = self.chroma_manager.search_documents("policies", user_query, n_results=3)
            policy_texts = [result["text"] for result in policy_results]
            
            # Get recommendations for this application
            recommendations = self.db.query(Recommendation).filter(
                Recommendation.application_id == app.id
            ).all()
            
            recommendation_texts = [
                f"{rec.category} ({rec.priority}): {rec.description}"
                for rec in recommendations
            ]
            
            # Get recent interactions
            interactions = self.db.query(Interaction).filter(
                Interaction.application_id == app.id
            ).order_by(Interaction.timestamp.desc()).limit(5).all()
            
            chat_history = [
                {"role": "user" if interaction.interaction_type == "chat" else "assistant", 
                 "content": interaction.content}
                for interaction in interactions
                if interaction.interaction_type in ["chat", "chat_response"]
            ]
            
            # Prepare application data for LLM
            app_data = {
                "name": app.name,
                "email": app.email,
                "phone": app.phone,
                "income": app.income,
                "family_size": app.family_size,
                "address": app.address,
                "validation_status": app.validation_status,
                "assessment_status": app.assessment_status,
                "risk_level": app.risk_level
            }
            
            # Detect soft decline (review status) and prepare economic enablement information
            is_soft_decline = app.assessment_status.lower() in ["review", "pending"]
            economic_enablement_info = ""
            
            if is_soft_decline:
                # Generate specific economic enablement recommendations for soft decline cases
                enablement_recs = self._generate_soft_decline_recommendations({
                    "employment_status": app.employment_status,
                    "income": app.income,
                    "family_size": app.family_size,
                    "job_title": app.job_title,
                    "employment_duration": app.employment_duration
                })
                
                if enablement_recs:
                    economic_enablement_info = "\n\nECONOMIC ENABLEMENT OPPORTUNITIES (for applications under review):\n"
                    for rec in enablement_recs:
                        economic_enablement_info += f"- {rec['category']}: {rec['description']}\n"
                        economic_enablement_info += "  Action steps: " + ", ".join(rec['action_items'][:2]) + "\n"
            
            # Keywords that might trigger economic enablement information
            enablement_keywords = ["training", "job", "work", "career", "skill", "education", "employment", 
                                 "business", "startup", "entrepreneur", "upskill", "income", "money", 
                                 "financial", "option", "alternative", "help", "support", "guidance", "decline", "rejected"]
            
            # Check if query contains any enablement keywords
            should_provide_enablement = is_soft_decline and any(keyword in user_query.lower() for keyword in enablement_keywords)
            
            # Create prompt for LLM
            prompt = f"""
            You are a helpful AI assistant for a government social security department.
            You are helping an applicant with their application for financial support and economic enablement.
            
            APPLICATION INFORMATION:
            {json.dumps(app_data, indent=2)}
            
            RELEVANT POLICIES:
            {chr(10).join(policy_texts)}
            
            RECOMMENDATIONS:
            {chr(10).join(recommendation_texts)}
            {economic_enablement_info if should_provide_enablement else ''}
            
            RECENT CONVERSATION HISTORY:
            {json.dumps(chat_history[-3:], indent=2) if chat_history else "No previous conversation"}
            
            USER QUERY:
            {user_query}
            
            {'IMPORTANT INSTRUCTION: The user''s application is currently under review (soft decline). Proactively suggest economic enablement options like vocational training, job placement assistance, or business development support tailored to their situation. Make your response interactive by suggesting follow-up questions the user could ask about specific programs.' if is_soft_decline else ''}
            
            Please provide a helpful, accurate, and compassionate response to the user's query.
            Focus on providing practical guidance and clear next steps based on their application status and the available information.
            {'Include at least 2-3 specific economic enablement options that could help improve their situation while their application is being reviewed.' if should_provide_enablement else ''}
            Keep your response under 250 words and maintain a professional, supportive tone.
            End your response with 2-3 specific follow-up questions the user might want to ask.
            """
            
            # Test direct Ollama connection
            try:
                import requests
                self.logger.info("Testing direct Ollama connection...")
                test_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "mistral", "prompt": "Say hello world", "stream": False}
                )
                self.logger.info(f"Direct Ollama test response: {test_response.status_code} - {test_response.text[:100]}...")
            except Exception as ollama_test_error:
                self.logger.error(f"Error testing Ollama connection: {str(ollama_test_error)}")
            
            # Get response from LLM
            self.logger.info("Invoking LLM with prompt...")
            self.logger.info(f"LLM Config: {self.llm}")
            self.logger.info(f"Prompt: {prompt[:200]}...")
            
            try:
                llm_response = self.llm.invoke(prompt)
                self.logger.info(f"LLM response received: {llm_response[:100]}...")
            except Exception as llm_error:
                self.logger.error(f"Error invoking LLM: {str(llm_error)}")
                import traceback
                self.logger.error(traceback.format_exc())
                llm_response = f"Error getting AI response: {str(llm_error)}"
                raise
            
            # Generate suggestions based on query and application status
            suggestions = self._generate_suggestions(user_query, app_data)
            
            # Store interaction in database
            self._store_interaction(app.id, user_query, llm_response)
            
            return {
                "text": llm_response,
                "suggestions": suggestions,
                "agent": "counselor"
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"Error providing guidance: {str(e)}\n{error_trace}")
            return {
                "text": f"I'm sorry, I encountered an error while processing your request: {str(e)}. Please try again or contact our support team for assistance.",
                "suggestions": ["Contact support", "Check application status", "Submit documentation"],
                "error": str(e),
                "trace": error_trace
            }
    
    # Helper methods
    def _generate_financial_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate financial support recommendations."""
        recommendations = []
        income = data.get("income", 0)
        family_size = data.get("family_size", 1)
        income_per_member = income / family_size if family_size > 0 else income
        assessment_status = data.get("assessment_status", "pending")
        
        # Only generate if application was approved
        if assessment_status == "approved":
            # Direct financial assistance
            if income_per_member < 20000:
                recommendations.append({
                    "category": "Financial Assistance",
                    "priority": "high",
                    "description": "Monthly financial stipend to supplement income",
                    "action_items": [
                        "Complete financial assistance form",
                        "Provide bank account details for direct deposit",
                        "Submit proof of ongoing expenses"
                    ]
                })
            
            # Housing subsidy
            if income < 60000:
                recommendations.append({
                    "category": "Housing Subsidy",
                    "priority": "medium",
                    "description": "Partial housing subsidy to assist with rent or mortgage payments",
                    "action_items": [
                        "Submit housing contract or mortgage statement",
                        "Complete housing subsidy application",
                        "Provide proof of residence"
                    ]
                })
            
            # Education support for families with children
            if family_size > 2:
                recommendations.append({
                    "category": "Education Support",
                    "priority": "medium" if income < 70000 else "low",
                    "description": "Financial assistance for educational expenses of dependent children",
                    "action_items": [
                        "Submit school enrollment documents",
                        "Provide list of educational expenses",
                        "Complete education support application"
                    ]
                })
        
        return recommendations
    
    def _generate_enablement_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate economic enablement recommendations."""
        recommendations = []
        employment_status = data.get("employment_status", "unknown")
        employment_duration = data.get("employment_duration", 0)
        income = data.get("income", 0)
        
        # Vocational training
        if employment_status in ["unemployed", "student"] or income < 40000:
            recommendations.append({
                "category": "Vocational Training",
                "priority": "high" if employment_status == "unemployed" else "medium",
                "description": "Free vocational training in high-demand fields to improve employment prospects",
                "action_items": [
                    "Complete skills assessment test",
                    "Review available training programs",
                    "Select preferred training path",
                    "Schedule orientation session"
                ]
            })
        
        # Job placement assistance
        if employment_status in ["unemployed", "part-time"] or employment_duration < 12:
            recommendations.append({
                "category": "Job Placement",
                "priority": "high" if employment_status == "unemployed" else "medium",
                "description": "Job search assistance, resume building, and interview preparation",
                "action_items": [
                    "Submit current resume",
                    "Complete career interests assessment",
                    "Schedule meeting with employment counselor",
                    "Attend job readiness workshop"
                ]
            })
        
        # Small business support
        if employment_status in ["self-employed"] or data.get("job_title", "").lower() in ["entrepreneur", "business owner"]:
            recommendations.append({
                "category": "Small Business Support",
                "priority": "medium",
                "description": "Business development resources and potential micro-financing",
                "action_items": [
                    "Submit business plan",
                    "Complete entrepreneurship assessment",
                    "Attend business development workshop",
                    "Schedule consultation with business advisor"
                ]
            })
        
        # Professional certification
        if employment_status in ["employed", "part-time"] and income < 80000:
            recommendations.append({
                "category": "Professional Certification",
                "priority": "medium",
                "description": "Support for obtaining professional certifications to enhance career prospects",
                "action_items": [
                    "Identify relevant certifications for your field",
                    "Research certification requirements",
                    "Submit certification support application",
                    "Develop study plan with advisor"
                ]
            })
        
        return recommendations
    
    def _generate_soft_decline_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specialized recommendations for soft decline (review status) cases."""
        recommendations = []
        employment_status = data.get("employment_status", "unknown")
        employment_duration = data.get("employment_duration", 0)
        income = data.get("income", 0)
        family_size = data.get("family_size", 1)
        
        # Skill enhancement for improved application
        recommendations.append({
            "category": "Enhanced Skills Certification",
            "priority": "high",
            "description": "Obtain industry-recognized certifications to strengthen your application and improve income potential",
            "action_items": [
                "Schedule a personalized skills assessment consultation",
                "Review our catalog of government-sponsored certification programs",
                "Register for a training session with a career counselor",
                "Learn about stipends available during training periods"
            ]
        })
        
        # Application enhancement guidance
        recommendations.append({
            "category": "Application Enhancement Support",
            "priority": "high",
            "description": "Receive personalized guidance on strengthening your application with additional documentation and qualifications",
            "action_items": [
                "Schedule a one-on-one consultation with an application specialist",
                "Get document preparation assistance for stronger evidence",
                "Learn about qualifying factors that can improve your application",
                "Explore alternative support programs during the review period"
            ]
        })
        
        # Career transition support (based on employment status)
        if employment_status in ["unemployed", "part-time"] or income < 50000:
            recommendations.append({
                "category": "Career Transition Program",
                "priority": "high",
                "description": "Comprehensive support for transitioning to higher-paying employment opportunities in growth sectors",
                "action_items": [
                    "Enroll in our fast-track career transition program",
                    "Meet with an industry specialist in a high-demand field",
                    "Access our exclusive job placement network for program participants",
                    "Learn about temporary financial support during your transition"
                ]
            })
        
        # Digital skills enhancement
        recommendations.append({
            "category": "Digital Skills Enhancement",
            "priority": "medium",
            "description": "Free training in essential digital skills that increase employability across all sectors",
            "action_items": [
                "Take our digital literacy assessment",
                "Enroll in our technology basics or advanced courses",
                "Get free access to learning platforms and certifications",
                "Schedule hands-on workshops with technology mentors"
            ]
        })
        
        # Entrepreneurship bootcamp (for potential entrepreneurs)
        if employment_status in ["unemployed", "self-employed", "part-time"]:
            recommendations.append({
                "category": "Entrepreneurship Bootcamp",
                "priority": "medium",
                "description": "Intensive program to develop, launch, and grow a small business with potential grant funding",
                "action_items": [
                    "Attend our entrepreneurship potential assessment session",
                    "Develop a business concept with our mentors",
                    "Learn about micro-grants and startup funding options",
                    "Join our business incubator for ongoing support"
                ]
            })
        
        # Financial literacy program
        recommendations.append({
            "category": "Financial Empowerment Program",
            "priority": "medium",
            "description": "Comprehensive financial literacy training and personalized financial planning assistance",
            "action_items": [
                "Schedule a financial health check session",
                "Enroll in our budgeting and savings workshops",
                "Receive guidance on debt management and credit improvement",
                "Create a personalized financial stability plan"
            ]
        })
        
        # Healthcare skills training (especially for larger families)
        if family_size > 2:
            recommendations.append({
                "category": "Healthcare Career Pathway",
                "priority": "medium",
                "description": "Training for in-demand healthcare positions with stable employment and growth potential",
                "action_items": [
                    "Explore various healthcare career tracks with a counselor",
                    "Learn about accelerated certification programs",
                    "Discuss childcare support options during training",
                    "Connect with healthcare employers seeking candidates"
                ]
            })
        
        return recommendations
        
    def _generate_process_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate process and documentation recommendations."""
        recommendations = []
        assessment_status = data.get("assessment_status", "pending")
        validation_status = data.get("validation_status", "pending")
        
        # Documentation improvements for invalid applications
        if validation_status == "invalid":
            recommendations.append({
                "category": "Documentation",
                "priority": "high",
                "description": "Additional documentation required to validate your application",
                "action_items": [
                    "Review validation errors",
                    "Submit missing or corrected documents",
                    "Ensure all information is consistent across documents",
                    "Follow up with validation team"
                ]
            })
        
        # Appeal guidance for rejected applications
        if assessment_status == "rejected":
            recommendations.append({
                "category": "Appeal Process",
                "priority": "high",
                "description": "Guidance on appealing the rejection decision",
                "action_items": [
                    "Review rejection reasons carefully",
                    "Gather additional supporting documentation",
                    "Complete appeal application form",
                    "Submit appeal within 30 days of decision"
                ]
            })
        
        # Follow-up process for pending applications
        if assessment_status == "pending" and validation_status == "valid":
            recommendations.append({
                "category": "Application Follow-up",
                "priority": "medium",
                "description": "Steps to check on your application status and ensure timely processing",
                "action_items": [
                    "Check application portal for status updates",
                    "Respond promptly to any information requests",
                    "Contact support after 10 business days if no update",
                    "Ensure contact information is current"
                ]
            })
        
        return recommendations
    
    def _generate_approval_explanation(self, data: Dict[str, Any]) -> str:
        """Generate approval explanation using LLM."""
        try:
            # Create prompt for LLM
            prompt = f"""
            Generate a personalized approval explanation for a social support application with the following details:
            
            Applicant Name: {data.get('name', 'Applicant')}
            Income: {data.get('income', 0)} AED
            Family Size: {data.get('family_size', 1)}
            Risk Level: {data.get('risk_level', 'medium')}
            
            The explanation should be professional, compassionate, and informative. Include:
            1. Congratulations on approval
            2. Brief explanation of benefits they qualify for
            3. Next steps in the process
            4. Timeline for receiving support
            
            Keep your response under 200 words and maintain a professional, supportive tone.
            """
            
            # Get explanation from LLM
            explanation = self.llm.invoke(prompt)
            return explanation
        except Exception as e:
            self.logger.error(f"Error generating approval explanation: {str(e)}")
            return "Congratulations! Your application for social support has been approved. You will receive detailed information about your benefits and next steps in your approval packet."
    
    def _generate_rejection_explanation(self, data: Dict[str, Any]) -> str:
        """Generate rejection explanation using LLM."""
        try:
            # Create prompt for LLM
            prompt = f"""
            Generate a personalized rejection explanation for a social support application with the following details:
            
            Applicant Name: {data.get('name', 'Applicant')}
            Income: {data.get('income', 0)} AED
            Family Size: {data.get('family_size', 1)}
            Risk Level: {data.get('risk_level', 'medium')}
            
            Rejection Reasons:
            - Income threshold: {50000} AED (minimum required)
            - Income per family member: {10000} AED (minimum required)
            - Your income per family member: {data.get('income', 0) / data.get('family_size', 1) if data.get('family_size', 0) > 0 else 0} AED
            
            The explanation should be professional, compassionate, and constructive. Include:
            1. Empathetic acknowledgment of the decision
            2. Clear explanation of why they didn't qualify
            3. Information about the appeals process
            4. Alternative resources or programs they might qualify for
            
            Keep your response under 200 words and maintain a professional, supportive tone.
            """
            
            # Get explanation from LLM
            explanation = self.llm.invoke(prompt)
            return explanation
        except Exception as e:
            self.logger.error(f"Error generating rejection explanation: {str(e)}")
            return "We regret to inform you that your application does not currently meet our eligibility criteria. You have the right to appeal this decision within 30 days or to reapply when your circumstances change."
    
    def _generate_suggestions(self, query: str, app_data: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions based on query and application data."""
        suggestions = []
        
        # Default suggestions based on application status
        if app_data.get("assessment_status") == "approved":
            suggestions = [
                "What benefits am I eligible for?",
                "When will I receive my first payment?",
                "How do I update my information?"
            ]
        elif app_data.get("assessment_status") == "rejected":
            suggestions = [
                "How do I appeal this decision?",
                "What documents should I provide for appeal?",
                "Are there other programs I might qualify for?"
            ]
        elif app_data.get("validation_status") == "invalid":
            suggestions = [
                "What documents are missing?",
                "How do I correct my application?",
                "When can I resubmit?"
            ]
        else:
            suggestions = [
                "What is the status of my application?",
                "How long does processing take?",
                "Do you need additional information from me?"
            ]
        
        # Add query-specific suggestions
        if "document" in query.lower() or "upload" in query.lower():
            suggestions.append("What file formats are accepted?")
        
        if "training" in query.lower() or "job" in query.lower() or "work" in query.lower():
            suggestions.append("What job training programs do you offer?")
        
        if "payment" in query.lower() or "money" in query.lower() or "benefit" in query.lower():
            suggestions.append("How are benefit amounts calculated?")
        
        # Return 3 suggestions maximum
        return suggestions[:3]
    
    def _store_recommendations(self, application_id: str, recommendations: List[Dict[str, Any]]) -> None:
        """Store recommendations in the database."""
        try:
            # Get application from database
            app = self.db.query(Application).filter(
                Application.filename == application_id
            ).first()
            
            if not app:
                self.logger.warning(f"Application with ID {application_id} not found")
                return
            
            # Delete existing recommendations
            self.db.query(Recommendation).filter(
                Recommendation.application_id == app.id
            ).delete()
            
            # Add new recommendations
            for rec in recommendations:
                recommendation = Recommendation(
                    application_id=app.id,
                    category=rec["category"],
                    priority=rec["priority"],
                    description=rec["description"],
                    action_items=rec.get("action_items", []),
                    created_at=datetime.utcnow()
                )
                self.db.add(recommendation)
            
            self.db.commit()
            self.logger.info(f"Stored {len(recommendations)} recommendations for application {application_id}")
        except Exception as e:
            self.logger.error(f"Error storing recommendations: {str(e)}")
            self.db.rollback()
    
    def _store_interaction(self, application_id: int, query: str, response: str) -> None:
        """Store chat interaction in the database."""
        try:
            # Store user query
            user_interaction = Interaction(
                application_id=application_id,
                interaction_type="chat",
                content=query,
                agent="user",
                timestamp=datetime.utcnow()
            )
            self.db.add(user_interaction)
            
            # Store system response
            system_interaction = Interaction(
                application_id=application_id,
                interaction_type="chat_response",
                content=response,
                agent="counselor",
                timestamp=datetime.utcnow()
            )
            self.db.add(system_interaction)
            
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Error storing interaction: {str(e)}")
            self.db.rollback()
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        try:
            self.db.close()
        except:
            pass
