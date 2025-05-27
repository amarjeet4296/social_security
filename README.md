# Enhanced Social Security Application System

A comprehensive multi-agent system for processing, validating, and assessing social security applications with AI assistance.

## Overview

This system provides an end-to-end solution for managing social security applications, from initial submission to final assessment. It features a user-friendly web interface, robust backend processing, and AI-powered assistance throughout the application process.

## Architecture

The system uses a multi-agent architecture with the following components:

### Frontend
- **Streamlit Application**: User interface for application submission, document upload, status checking, and AI chat assistance

### Backend
- **FastAPI Server**: RESTful API endpoints for application processing
- **PostgreSQL Database**: Stores application data, documents, and assessment results
- **Ollama Integration**: Local LLM service for AI chat assistance

### Agent System
- **Data Collector**: Gathers and validates application information
- **Validator**: Verifies application details and uploaded documents
- **Assessor**: Evaluates applications against eligibility criteria
- **Counselor**: Provides guidance and assistance to applicants
- **Orchestrator**: Coordinates workflow between different agents

## System Features

- **Application Submission**: User-friendly form for submitting new applications
- **Document Management**: Upload and validation of supporting documents
- **Status Tracking**: Real-time tracking of application status
- **AI Chat Assistance**: Conversational interface for applicant questions
- **Robust Fallback Mechanisms**: System remains functional even when certain components are unavailable

## Setup Instructions

### Prerequisites
- Python 3.8+
- PostgreSQL
- Ollama (for LLM functionality)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd social_app_1
   ```

2. Install the required dependencies:
   ```
   pip install -r enhanced_system/requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=social_security
   DB_USER=your_username
   DB_PASSWORD=your_password
   OLLAMA_API_URL=http://localhost:11434
   ```

4. Initialize the PostgreSQL database:
   ```
   psql -U your_username -d social_security -f setup/init_db.sql
   ```

5. Start Ollama with the Mistral model:
   ```
   ollama run mistral
   ```

## Usage

1. Start the FastAPI backend server:
   ```
   cd enhanced_system
   uvicorn api.main:app --host 0.0.0.0 --port 8080
   ```

2. Launch the Streamlit frontend:
   ```
   cd enhanced_system
   streamlit run app.py
   ```

3. Access the application at `http://localhost:8501` in your web browser

## Fallback Implementations

The system includes several fallback mechanisms to ensure functionality even when certain components are unavailable:

- **Mock LLM**: Provides contextual responses when Ollama is unavailable
- **Direct Chat Implementation**: Works without requiring the backend API
- **Mock Data Generation**: Supplies application status data when the API is unavailable

## Development and Contribution

### Project Structure
```
social_app_1/
├── .env                          # Environment variables
├── app.py                        # Main application entry point
├── enhanced_system/
│   ├── agents/                   # Agent implementations
│   │   ├── data_collector.py
│   │   ├── validator.py
│   │   ├── assessor.py
│   │   ├── counselor.py
│   │   └── orchestrator.py
│   ├── api/                      # FastAPI backend
│   │   └── main.py
│   ├── models/                   # Data models and AI models
│   │   ├── decision_engine.py
│   │   └── model_loader.py
│   ├── ui/                       # Streamlit UI components
│   └── requirements.txt          # Project dependencies
└── README.md                     # This file
```

## License

[Specify license information]
