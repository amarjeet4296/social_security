"""
ML Models for Social Security Application System
Includes models for financial need classification, fraud risk detection, 
employment stability prediction, and support amount recommendation.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# ML libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report

# Import local modules
from config.system_config import MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml_models")


class ModelManager:
    """
    Manages the ML models for the social security application system.
    Includes training, prediction, and model management functionality.
    """
    


    def __init__(self):
        """Initialize the model manager."""
        logger.info("Initializing ModelManager")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Initialize model paths
        self.model_paths = {
            "financial_need": os.path.join(MODEL_DIR, "financial_need_classifier.joblib"),
            "fraud_risk": os.path.join(MODEL_DIR, "fraud_risk_detector.joblib"),
            "employment_stability": os.path.join(MODEL_DIR, "employment_stability_predictor.joblib"),
            "support_amount": os.path.join(MODEL_DIR, "support_amount_regressor.joblib")
        }
        
        # Initialize models dictionary
        self.models = {}
        
        # Load or create models
        self._load_or_create_models()
        
        logger.info("ModelManager initialized successfully")

    def _load_or_create_models(self):
        """Load existing models or create new ones if they don't exist."""
        for model_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                logger.info(f"Loading existing model: {model_name}")
                try:
                    self.models[model_name] = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}")
                    self.models[model_name] = self._create_model(model_name)
            else:
                logger.info(f"Creating new model: {model_name}")
                self.models[model_name] = self._create_model(model_name)


    def _create_model(self, model_name: str):
        """Create a new model based on the model name."""
        if model_name == "financial_need":
            # Random Forest for Financial Need Classification
            # Justification: Random Forest is robust to outliers and missing values, handles both numerical and categorical data, provides feature importance for explainability, and is well-suited for tabular social security application data with mixed feature types and potential class imbalance.
            # (High, Medium, Low need categories)
            model = Pipeline([
                ('preprocessor', self._create_preprocessor()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=7,
                    class_weight='balanced',
                    random_state=42
                ))
            ])
        
        elif model_name == "fraud_risk":
            # Random Forest for Fraud Risk Detection
            model = Pipeline([
                ('preprocessor', self._create_preprocessor()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                ))
            ])
        
        elif model_name == "employment_stability":
            # Logistic Regression for Employment Stability
            model = Pipeline([
                ('preprocessor', self._create_preprocessor()),
                ('classifier', LogisticRegression(
                    C=1.0,
                    solver='lbfgs',
                    multi_class='auto',
                    max_iter=1000,
                    random_state=42
                ))
            ])
        
        elif model_name == "support_amount":
            # Support Vector Regression for Support Amount
            model = Pipeline([
                ('preprocessor', self._create_preprocessor()),
                ('regressor', SVR(
                    kernel='rbf',
                    C=100,
                    epsilon=0.1,
                    gamma='scale'
                ))
            ])
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Train model with synthetic data
        self._train_with_synthetic_data(model, model_name)
        
        # Save the model
        joblib.dump(model, self.model_paths[model_name])
        
        return model

    def _train_with_synthetic_data(self, model, model_name: str):
        """Train the model with synthetic data."""
        logger.info(f"Training {model_name} with synthetic data")
        
        # Generate synthetic data
        X, y = self._generate_synthetic_data(model_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        if model_name == "support_amount":
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            logger.info(f"{model_name} Mean Squared Error: {mse:.4f}")
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            logger.info(f"{model_name} Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    def _generate_synthetic_data(self, model_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic data for training."""
        # Define number of samples
        n_samples = 1000
        
        # Generate random data
        np.random.seed(42)
        
        # Base features for all models
        data = {
            'income': np.random.uniform(3000, 50000, n_samples),
            'family_size': np.random.randint(1, 10, n_samples),
            'monthly_expenses': np.random.uniform(1000, 30000, n_samples),
            'assets_value': np.random.uniform(0, 2000000, n_samples),
            'liabilities_value': np.random.uniform(0, 1000000, n_samples),
            'employment_duration': np.random.randint(0, 240, n_samples),  # 0 to 20 years in months
            'employment_status': np.random.choice(
                ['employed', 'self-employed', 'unemployed', 'retired', 'student'],
                n_samples
            ),
            'job_title': np.random.choice(
                ['manager', 'clerk', 'engineer', 'doctor', 'teacher', 'driver', 'laborer', 'business_owner', 'none'],
                n_samples
            )
        }
        
        # Create dataframe
        X = pd.DataFrame(data)
        
        # Generate target variable based on model type
        if model_name == "financial_need":
            # Create a rule-based target for financial need
            # High need: low income, high family size, high expenses
            # Medium need: moderate income and expenses
            # Low need: high income, low expenses
            
            # Calculate income per family member
            X['income_per_member'] = X['income'] / X['family_size']
            
            # Calculate expense to income ratio
            X['expense_ratio'] = X['monthly_expenses'] / X['income']
            
            # Calculate net worth
            X['net_worth'] = X['assets_value'] - X['liabilities_value']
            
            # Create target based on these calculated features
            y = np.zeros(n_samples, dtype=int)
            
            # High need (2): low income per member, high expense ratio, or negative net worth
            high_need = (
                (X['income_per_member'] < 5000) |
                (X['expense_ratio'] > 0.7) |
                (X['net_worth'] < 0)
            )
            y[high_need] = 2
            
            # Low need (0): high income per member, low expense ratio, positive net worth
            low_need = (
                (X['income_per_member'] > 15000) &
                (X['expense_ratio'] < 0.3) &
                (X['net_worth'] > 500000)
            )
            y[low_need] = 0
            
            # Medium need (1): everything else
            medium_need = ~(high_need | low_need)
            y[medium_need] = 1
            
            # Drop calculated features
            X = X.drop(['income_per_member', 'expense_ratio', 'net_worth'], axis=1)
        
        elif model_name == "fraud_risk":
            # Create a rule-based target for fraud risk
            # Fraud indicators: inconsistent data, unusual patterns
            
            # Create some inconsistency flags
            X['income_vs_expenses'] = X['income'] < X['monthly_expenses']
            X['high_assets_low_income'] = (X['assets_value'] > 1000000) & (X['income'] < 10000)
            X['unemployed_with_income'] = (X['employment_status'] == 'unemployed') & (X['income'] > 5000)
            
            # Create target based on these flags
            y = np.zeros(n_samples, dtype=int)
            
            # High risk (1): multiple inconsistency flags
            high_risk = (
                X['income_vs_expenses'].astype(int) +
                X['high_assets_low_income'].astype(int) +
                X['unemployed_with_income'].astype(int)
            ) >= 2
            
            y[high_risk] = 1
            
            # Drop calculated features
            X = X.drop(['income_vs_expenses', 'high_assets_low_income', 'unemployed_with_income'], axis=1)
        
        elif model_name == "employment_stability":
            # Create a rule-based target for employment stability
            # Stable: long employment duration, professional jobs
            # Unstable: short duration, frequent changes
            
            # Professional job titles
            professional_jobs = ['manager', 'engineer', 'doctor', 'teacher', 'business_owner']
            
            # Create stability score
            stability_score = (
                (X['employment_duration'] / 12) +  # Years of employment
                (X['job_title'].isin(professional_jobs)).astype(int) * 2 +  # Professional job bonus
                (X['employment_status'] == 'employed').astype(int) * 2 +  # Employment status bonus
                (X['income'] / 10000)  # Income factor
            )
            
            # Create binary target (1 = stable, 0 = unstable)
            y = (stability_score > 5).astype(int)
        
        elif model_name == "support_amount":
            # Create a regression target for support amount
            # Support amount based on income, family size, expenses
            
            # Calculate income per family member
            income_per_member = X['income'] / X['family_size']
            
            # Calculate expense to income ratio
            expense_ratio = X['monthly_expenses'] / X['income']
            
            # Base support amount
            base_support = 5000
            
            # Adjust based on income per member (inverse relationship)
            income_factor = np.maximum(0, 1 - (income_per_member / 20000))
            
            # Adjust based on expense ratio (direct relationship)
            expense_factor = np.minimum(1, expense_ratio)
            
            # Adjust based on family size (direct relationship)
            family_factor = np.log1p(X['family_size']) / np.log1p(10)  # Normalize to 0-1
            
            # Calculate support amount
            y = base_support * (
                0.5 + 
                0.3 * income_factor + 
                0.1 * expense_factor + 
                0.1 * family_factor
            )
            
            # Add some noise
            y += np.random.normal(0, 500, n_samples)
            
            # Ensure positive values
            y = np.maximum(0, y)
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return X, y
    
    def predict_financial_need(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict financial need category for an application.
        Returns need level (high, medium, low) and confidence.
        """
        try:
            # Convert application data to DataFrame
            df = self._application_to_dataframe(application_data)
            
            # Make prediction
            model = self.models["financial_need"]
            need_level_idx = model.predict(df)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(df)[0]
            confidence = probabilities[need_level_idx]
            
            # Map index to need level
            need_levels = ["low", "medium", "high"]
            need_level = need_levels[need_level_idx]
            
            return {
                "need_level": need_level,
                "confidence": confidence,
                "probabilities": {level: prob for level, prob in zip(need_levels, probabilities)}
            }
        
        except Exception as e:
            logger.error(f"Error predicting financial need: {str(e)}")
            return {
                "need_level": "medium",  # Default to medium as fallback
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_fraud_risk(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for an application.
        Returns risk score (0-100) and risk level (low, medium, high).
        """
        try:
            # Convert application data to DataFrame
            df = self._application_to_dataframe(application_data)
            
            # Make prediction
            model = self.models["fraud_risk"]
            is_risky = model.predict(df)[0]
            
            # Get prediction probability
            risk_prob = model.predict_proba(df)[0][is_risky]
            
            # Convert to risk score (0-100)
            risk_score = risk_prob * 100
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "low"
            elif risk_score < 70:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "confidence": risk_prob
            }
        
        except Exception as e:
            logger.error(f"Error predicting fraud risk: {str(e)}")
            return {
                "risk_score": 50,  # Default to medium risk as fallback
                "risk_level": "medium",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_employment_stability(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict employment stability for an application.
        Returns stability score (0-100) and stability status.
        """
        try:
            # Convert application data to DataFrame
            df = self._application_to_dataframe(application_data)
            
            # Make prediction
            model = self.models["employment_stability"]
            is_stable = model.predict(df)[0]
            
            # Get prediction probability
            stability_prob = model.predict_proba(df)[0][is_stable]
            
            # Convert to stability score (0-100)
            stability_score = stability_prob * 100
            
            # Determine stability status
            stability_status = "stable" if is_stable == 1 else "unstable"
            
            return {
                "stability_score": stability_score,
                "stability_status": stability_status,
                "confidence": stability_prob
            }
        
        except Exception as e:
            logger.error(f"Error predicting employment stability: {str(e)}")
            return {
                "stability_score": 50,  # Default to medium stability as fallback
                "stability_status": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_support_amount(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict recommended support amount for an application.
        Returns support amount and rationale.
        """
        try:
            # Convert application data to DataFrame
            df = self._application_to_dataframe(application_data)
            
            # Make prediction
            model = self.models["support_amount"]
            support_amount = model.predict(df)[0]
            
            # Round to nearest 100
            support_amount = round(support_amount / 100) * 100
            
            # Calculate income per family member as context
            income = application_data.get("income", 0)
            family_size = application_data.get("family_size", 1)
            income_per_member = income / max(1, family_size)
            
            # Generate simple rationale
            if income_per_member < 5000:
                rationale = "Low income per family member indicates high need for support."
            elif income_per_member < 10000:
                rationale = "Moderate income per family member indicates standard support needs."
            else:
                rationale = "Higher income per family member indicates lower support needs."
            
            return {
                "support_amount": support_amount,
                "income_per_member": income_per_member,
                "rationale": rationale
            }
        
        except Exception as e:
            logger.error(f"Error predicting support amount: {str(e)}")
            
            # Fallback to rule-based calculation
            income = application_data.get("income", 0)
            family_size = application_data.get("family_size", 1)
            income_per_member = income / max(1, family_size)
            
            # Simple rule-based calculation
            if income_per_member < 5000:
                support_amount = 5000
            elif income_per_member < 10000:
                support_amount = 3000
            else:
                support_amount = 1000
            
            return {
                "support_amount": support_amount,
                "income_per_member": income_per_member,
                "rationale": "Calculated using fallback rules due to prediction error.",
                "error": str(e)
            }
    
    def _application_to_dataframe(self, application_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert application data dictionary to DataFrame for model input."""
        # Create a single-row DataFrame with required columns
        df = pd.DataFrame({
            'income': [application_data.get('income', 0)],
            'family_size': [application_data.get('family_size', 1)],
            'monthly_expenses': [application_data.get('monthly_expenses', 0)],
            'assets_value': [application_data.get('assets_value', 0)],
            'liabilities_value': [application_data.get('liabilities_value', 0)],
            'employment_duration': [application_data.get('employment_duration', 0)],
            'employment_status': [application_data.get('employment_status', 'unemployed')],
            'job_title': [application_data.get('job_title', 'none')]
        })
        
        return df

    def assess_application(self, application_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive assessment of an application.
        Returns approval status, reasons, and detailed assessment.
        """
        assessment_details = {}
        reasons = []
        
        try:
            # Run all prediction models
            financial_need = self.predict_financial_need(application_data)
            fraud_risk = self.predict_fraud_risk(application_data)
            employment_stability = self.predict_employment_stability(application_data)
            support_amount = self.predict_support_amount(application_data)
            
            # Store all assessment results
            assessment_details = {
                "financial_need": financial_need,
                "fraud_risk": fraud_risk,
                "employment_stability": employment_stability,
                "support_recommendation": support_amount,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Determine risk level
            if fraud_risk["risk_level"] == "high":
                risk_level = "high"
                reasons.append("High fraud risk detected in the application.")
            elif fraud_risk["risk_level"] == "medium":
                risk_level = "medium"
                reasons.append("Medium fraud risk detected.")
            else:
                risk_level = "low"
            
            assessment_details["risk_level"] = risk_level
            
            # Determine approval based on models
            is_approved = True
            
            # Reject if high risk
            if risk_level == "high":
                is_approved = False
            
            # Consider financial need
            if financial_need["need_level"] == "low" and support_amount["support_amount"] < 1000:
                is_approved = False
                reasons.append("Low financial need doesn't meet minimum support threshold.")
            
            # Consider employment stability for large support amounts
            if support_amount["support_amount"] > 5000 and employment_stability["stability_status"] == "unstable":
                is_approved = False
                reasons.append("Unstable employment status doesn't qualify for high support amount.")
            
            # Check income and family size requirements
            income = application_data.get("income", 0)
            family_size = application_data.get("family_size", 1)
            income_per_member = income / max(1, family_size)
            
            if income_per_member > 20000:
                is_approved = False
                reasons.append("Income per family member exceeds maximum threshold for support.")
            
            # Generate recommendations
            recommendations = []
            
            # Financial support recommendation
            if is_approved:
                recommendations.append({
                    "category": "financial_support",
                    "priority": "high",
                    "description": f"Recommended financial support of {support_amount['support_amount']} AED based on need assessment.",
                    "action_items": ["Process payment", "Schedule follow-up review in 6 months"]
                })
            
            # Employment recommendations
            if employment_stability["stability_status"] == "unstable":
                recommendations.append({
                    "category": "employment",
                    "priority": "high",
                    "description": "Employment instability detected. Recommend job placement services and skills training.",
                    "action_items": ["Refer to job placement program", "Schedule career counseling session"]
                })
            
            # Financial management recommendations
            if application_data.get("liabilities_value", 0) > application_data.get("assets_value", 0):
                recommendations.append({
                    "category": "financial_management",
                    "priority": "medium",
                    "description": "Negative net worth detected. Recommend financial counseling and debt management assistance.",
                    "action_items": ["Schedule financial counseling session", "Provide debt management resources"]
                })
            
            # Education recommendations
            if application_data.get("job_title", "") == "none" or application_data.get("employment_status", "") == "unemployed":
                recommendations.append({
                    "category": "education",
                    "priority": "high",
                    "description": "Recommend skills development and educational programs to improve employment prospects.",
                    "action_items": ["Enroll in vocational training program", "Provide scholarship opportunities"]
                })
            
            # Add recommendations to assessment details
            assessment_details["recommendations"] = recommendations
            
            # Add income_per_member to assessment details
            assessment_details["income_per_member"] = income_per_member
            
            # If not approved and no reasons, add a generic reason
            if not is_approved and not reasons:
                reasons.append("Application does not meet eligibility criteria based on comprehensive assessment.")
            
            return is_approved, reasons, assessment_details
        
        except Exception as e:
            logger.error(f"Error in comprehensive assessment: {str(e)}")
            
            # Fallback assessment
            is_approved = False
            reasons.append(f"Error during assessment: {str(e)}")
            assessment_details["error"] = str(e)
            assessment_details["risk_level"] = "high"  # Default to high risk on error
            
            return is_approved, reasons, assessment_details
