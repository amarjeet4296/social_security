"""
FastAPI backend for Enhanced Social Security Application System.
Handles application submission, document upload, status checking, chat, and decision explanation.
Integrates with multi-agent workflow and PostgreSQL DB.
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Optional
import uuid
import shutil

from agents.data_collector import DataCollectorAgent
from agents.validator import ValidatorAgent
from agents.assessor import AssessorAgent
from agents.counselor import CounselorAgent
from database.db_setup import SessionLocal, Application, Document

# For chat/LLM
from agents.llm import get_llm_response, get_mock_response

app = FastAPI(title="Enhanced Social Security API")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for DB session
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Schemas ---
class ApplicationCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    emirates_id: Optional[str] = None
    income: float
    family_size: int
    address: str
    employment_status: str
    employer: Optional[str] = None
    job_title: Optional[str] = None
    employment_duration: Optional[int] = None
    monthly_expenses: Optional[float] = None
    assets_value: Optional[float] = None
    liabilities_value: Optional[float] = None

class ApplicationResponse(BaseModel):
    application_id: str
    name: str
    income: float
    family_size: int
    address: str
    assessment_status: str
    validation_status: str
    risk_level: Optional[str]
    created_at: str
    updated_at: str
    score: Optional[float] = None
    documents: Optional[list] = []
    recommendations: Optional[list] = []

class DocumentResponse(BaseModel):
    document_id: str
    document_type: str
    filename: str
    status: str
    uploaded_at: str

class ChatRequest(BaseModel):
    application_id: str
    message: str

class ChatResponse(BaseModel):
    text: str
    suggestions: Optional[list] = []

# --- Initialize Agents ---
data_collector = DataCollectorAgent()
validator = ValidatorAgent()
assessor = AssessorAgent()
counselor = CounselorAgent()

# --- API Endpoints ---
@app.post("/api/applications", response_model=ApplicationResponse)
def submit_application(app: ApplicationCreate, db: Session = Depends(get_db)):
    """Create new application and run agentic workflow."""
    # 1. Create DB record
    new_app = Application(
        app_id=str(uuid.uuid4()),
        name=app.name,
        email=app.email,
        phone=app.phone,
        emirates_id=app.emirates_id,
        income=app.income,
        family_size=app.family_size,
        address=app.address,
        employment_status=app.employment_status,
        employer=app.employer,
        job_title=app.job_title,
        employment_duration=app.employment_duration,
        monthly_expenses=app.monthly_expenses,
        assets_value=app.assets_value,
        liabilities_value=app.liabilities_value,
    )
    db.add(new_app)
    db.commit()
    db.refresh(new_app)

    # 2. Run agentic workflow
    collected = data_collector.collect(app.dict())
    validated = validator.validate(collected)
    assessed = assessor.assess(validated)
    # Save assessment
    new_app.assessment_status = assessed.get("assessment_status", "pending")
    new_app.validation_status = assessed.get("validation_status", "pending")
    new_app.risk_level = assessed.get("risk_level")
    db.commit()
    db.refresh(new_app)

    # 3. Response
    return ApplicationResponse(
        application_id=new_app.app_id,
        name=new_app.name,
        income=new_app.income,
        family_size=new_app.family_size,
        address=new_app.address,
        assessment_status=new_app.assessment_status,
        validation_status=new_app.validation_status,
        risk_level=new_app.risk_level,
        created_at=str(new_app.created_at),
        updated_at=str(new_app.updated_at),
        documents=[],
        recommendations=[]
    )

@app.post("/api/documents", response_model=DocumentResponse)
def upload_document(
    application_id: str = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a supporting document for an application."""
    app_obj = db.query(Application).filter_by(app_id=application_id).first()
    if not app_obj:
        raise HTTPException(status_code=404, detail="Application not found")
    # Store file
    doc_folder = os.path.join("uploads", application_id)
    os.makedirs(doc_folder, exist_ok=True)
    file_path = os.path.join(doc_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Save DB record
    doc = Document(
        application_id=app_obj.id,
        document_type=document_type,
        filename=file.filename,
        status="uploaded"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return DocumentResponse(
        document_id=str(doc.id),
        document_type=doc.document_type,
        filename=doc.filename,
        status=doc.status,
        uploaded_at=str(doc.created_at)
    )

@app.get("/api/applications/{application_id}", response_model=ApplicationResponse)
def get_application_status(application_id: str, db: Session = Depends(get_db)):
    """Get application status and details."""
    app_obj = db.query(Application).filter_by(app_id=application_id).first()
    if not app_obj:
        raise HTTPException(status_code=404, detail="Application not found")
    # Get documents
    docs = db.query(Document).filter_by(application_id=app_obj.id).all()
    doc_list = [
        {
            "document_id": str(d.id),
            "document_type": d.document_type,
            "filename": d.filename,
            "status": d.status,
            "uploaded_at": str(d.created_at)
        } for d in docs
    ]
    # Recommendations (stub)
    recommendations = []
    if app_obj.assessment_status == "rejected":
        rec = counselor.generate_recommendations(app_obj.to_dict(), app_obj.assessment_status, app_obj.risk_level)
        recommendations = rec.get("recommendations", [])
    return ApplicationResponse(
        application_id=app_obj.app_id,
        name=app_obj.name,
        income=app_obj.income,
        family_size=app_obj.family_size,
        address=app_obj.address,
        assessment_status=app_obj.assessment_status,
        validation_status=app_obj.validation_status,
        risk_level=app_obj.risk_level,
        created_at=str(app_obj.created_at),
        updated_at=str(app_obj.updated_at),
        documents=doc_list,
        recommendations=recommendations
    )

@app.get("/api/applications/{application_id}/explanation")
def get_application_explanation(application_id: str, db: Session = Depends(get_db)):
    """Get explanation for decision (approval or rejection)."""
    app_obj = db.query(Application).filter_by(app_id=application_id).first()
    if not app_obj:
        raise HTTPException(status_code=404, detail="Application not found")
    explanation = counselor.explain_decision(app_obj.to_dict())
    return {"explanation": explanation.get("explanation"), "next_steps": explanation.get("next_steps"), "timeline": explanation.get("timeline")}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """Chat endpoint for AI support assistant."""
    app_obj = db.query(Application).filter_by(app_id=req.application_id).first()
    if not app_obj:
        raise HTTPException(status_code=404, detail="Application not found")
    # Try Ollama, fallback to mock
    try:
        response = get_llm_response(req.message, app_obj.to_dict())
    except Exception:
        response = get_mock_response(req.message, app_obj.to_dict())
    return ChatResponse(text=response, suggestions=[])

# Static for uploaded docs (optional)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
