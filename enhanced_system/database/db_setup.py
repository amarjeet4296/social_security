from typing import Dict, List
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey, JSON, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
Base = declarative_base()
metadata = MetaData()

# PostgreSQL connection configuration
DB_USER = os.getenv('DB_USER', 'amarjeet')
DB_PASSWORD = os.getenv('DB_PASSWORD', '9582924264')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'social')

# Construct database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800  # Recycle connections after 30 minutes
)

SessionLocal = sessionmaker(bind=engine)

# Database Models
class Application(Base):
    """Enhanced Application model for storing application data"""
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    # Using existing filename field to store application_id as well
    filename = Column(String(255), nullable=False)
    
    # Original required fields
    income = Column(Float, nullable=False)
    family_size = Column(Integer, nullable=False)
    address = Column(Text, nullable=False)
    validation_status = Column(String(50), nullable=False)
    assessment_status = Column(String(50), nullable=False)
    risk_level = Column(String(50), nullable=False)
    
    # Additional fields (nullable to maintain compatibility)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    emirates_id = Column(String(50), nullable=True)
    
    # Employment Information
    employment_status = Column(String(50), nullable=True)
    employer = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    employment_duration = Column(Integer, nullable=True)  # In months
    
    # Financial Information
    monthly_expenses = Column(Float, nullable=True)
    assets_value = Column(Float, nullable=True)
    liabilities_value = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="application")
    interactions = relationship("Interaction", back_populates="application")
    recommendations = relationship("Recommendation", back_populates="application")
    
    def __repr__(self):
        return f"<Application(id={self.id}, name='{self.name}', status='{self.assessment_status}')>"


class Document(Base):
    """Model for storing uploaded documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    document_type = Column(String(50), nullable=False)  # e.g., "bank_statement", "emirates_id", "resume"
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    mime_type = Column(String(100), nullable=False)
    extracted_data = Column(JSON, nullable=True)  # Stores OCR/extracted data
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    application = relationship("Application", back_populates="documents")
    
    def __repr__(self):
        return f"<Document(id={self.id}, type='{self.document_type}', filename='{self.filename}')>"


class Interaction(Base):
    """Model for storing user-system interactions"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    interaction_type = Column(String(50), nullable=False)  # e.g., "chat", "form_submission", "document_upload"
    content = Column(Text, nullable=True)  # Message content or action details
    agent = Column(String(50), nullable=True)  # Which agent handled this interaction
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    application = relationship("Application", back_populates="interactions")
    
    def __repr__(self):
        return f"<Interaction(id={self.id}, type='{self.interaction_type}', agent='{self.agent}')>"


class Recommendation(Base):
    """Model for storing support recommendations"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    category = Column(String(50), nullable=False)  # e.g., "financial", "training", "job_matching"
    priority = Column(String(20), nullable=False)  # "high", "medium", "low"
    description = Column(Text, nullable=False)
    action_items = Column(JSON, nullable=True)  # JSON array of action items
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    application = relationship("Application", back_populates="recommendations")
    
    def __repr__(self):
        return f"<Recommendation(id={self.id}, category='{self.category}', priority='{self.priority}')>"


class PolicyDocument(Base):
    """Model for storing policy documents"""
    __tablename__ = "policy_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(255), nullable=True)  # Reference ID in ChromaDB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PolicyDocument(id={self.id}, title='{self.title}', category='{self.category}')>"


class AuditLog(Base):
    """Model for storing system audit logs"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=True)
    action = Column(String(255), nullable=False)
    actor = Column(String(50), nullable=False)  # User ID or agent name
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', actor='{self.actor}')>"


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database and create tables if they don't exist"""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise


if __name__ == "__main__":
    # Initialize database when script is run directly
    init_db()
