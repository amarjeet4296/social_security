"""
System Configuration - Central configuration for the social security application system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Ensure directories exist
for directory in [DATA_DIR, UPLOADS_DIR, MODEL_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "user": os.getenv("DB_USER", "amarjeet"),
    "password": os.getenv("DB_PASSWORD", "9582924264"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "social")
}

# ChromaDB configuration
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
CHROMA_COLLECTIONS = {
    "policies": "policy_documents",
    "applications": "application_embeddings",
    "documents": "document_embeddings"
}

# LLM configuration
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # ollama, lmstudio, openwebui
    "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434"),
    "model_name": os.getenv("LLM_MODEL_NAME", "llama3"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048"))
}

# Agent configuration
AGENT_CONFIG = {
    "orchestrator": {
        "reasoning_framework": "react",
        "memory_type": "buffer"
    },
    "data_extraction": {
        "ocr_engine": "tesseract",
        "table_extraction": "camelot"
    },
    "validator": {
        "consistency_threshold": 0.8,
        "required_fields": ["income", "family_size", "address", "emirates_id"]
    },
    "assessor": {
        "models": {
            "financial_need": "financial_need_classifier.joblib",
            "fraud_risk": "fraud_risk_detector.joblib",
            "employment_stability": "employment_stability_predictor.joblib",
            "support_amount": "support_amount_regressor.joblib"
        },
        "thresholds": {
            "approval_threshold": 0.7,
            "risk_threshold": 0.3
        }
    },
    "counselor": {
        "max_history": 10,
        "response_template_dir": os.path.join(BASE_DIR, "templates", "counselor")
    }
}

# API configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("API_DEBUG", "False").lower() == "true",
    "workers": int(os.getenv("API_WORKERS", "4"))
}

# UI configuration
UI_CONFIG = {
    "theme": "light",
    "enable_chat": True,
    "upload_limit_mb": 10,
    "max_files": 5
}

# Document types
DOCUMENT_TYPES = {
    "emirates_id": {
        "allowed_extensions": [".jpg", ".jpeg", ".png", ".pdf"],
        "required": True,
        "max_size_mb": 5
    },
    "bank_statement": {
        "allowed_extensions": [".pdf", ".jpg", ".jpeg", ".png"],
        "required": True,
        "max_size_mb": 10
    },
    "resume": {
        "allowed_extensions": [".pdf", ".docx", ".doc"],
        "required": False,
        "max_size_mb": 5
    },
    "assets_liabilities": {
        "allowed_extensions": [".xlsx", ".csv"],
        "required": True,
        "max_size_mb": 5
    },
    "credit_report": {
        "allowed_extensions": [".pdf"],
        "required": True,
        "max_size_mb": 10
    }
}

# Observability configuration
OBSERVABILITY_CONFIG = {
    "langsmith_api_key": os.getenv("LANGSMITH_API_KEY", ""),
    "langsmith_project": os.getenv("LANGSMITH_PROJECT", "social_security_application"),
    "logging_level": os.getenv("LOGGING_LEVEL", "INFO"),
    "log_file": os.path.join(BASE_DIR, "logs", "system.log")
}

# Create logs directory
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
