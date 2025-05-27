"""
Streamlit frontend for the Enhanced Social Security Application System.
Provides user interface for application submission, document upload, and AI chat.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import datetime
import time
import uuid
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8080")

# Page configuration
st.set_page_config(
    page_title="Social Security Support System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .status-approved {
        color: #10B981;
        font-weight: bold;
    }
    .status-rejected {
        color: #EF4444;
        font-weight: bold;
    }
    .status-pending {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-processing {
        color: #3B82F6;
        font-weight: bold;
    }
    .chat-user {
        background-color: #E5E7EB;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-ai {
        background-color: #DBEAFE;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .priority-high {
        color: #EF4444;
        font-weight: bold;
    }
    .priority-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .priority-low {
        color: #10B981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "application_id" not in st.session_state:
    st.session_state.application_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"
if "application_data" not in st.session_state:
    st.session_state.application_data = None
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# Helper functions
def format_status(status):
    """Format application status with appropriate CSS class."""
    if status == "approved":
        return f'<span class="status-approved">Approved</span>'
    elif status == "rejected":
        return f'<span class="status-rejected">Rejected</span>'
    elif status == "pending":
        return f'<span class="status-pending">Pending</span>'
    elif status == "processing":
        return f'<span class="status-processing">Processing</span>'
    else:
        return status

def format_priority(priority):
    """Format recommendation priority with appropriate CSS class."""
    if priority == "high":
        return f'<span class="priority-high">High</span>'
    elif priority == "medium":
        return f'<span class="priority-medium">Medium</span>'
    elif priority == "low":
        return f'<span class="priority-low">Low</span>'
    else:
        return priority

def submit_application(data):
    """Submit application to API."""
    try:
        response = requests.post(f"{API_URL}/api/applications", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting application: {str(e)}")
        return None

def upload_document(file, application_id, document_type):
    """Upload document to API."""
    try:
        files = {"file": file}
        data = {
            "application_id": application_id,
            "document_type": document_type
        }
        response = requests.post(f"{API_URL}/api/documents", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def get_application_status(application_id):
    """Get application status from API or provide mock data if API is unavailable."""
    try:
        # Try to get from API first
        try:
            response = requests.get(f"{API_URL}/api/applications/{application_id}", timeout=3)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not connect to API. Using demo data for application status.")
            
        # Provide mock data for demonstration/testing purposes
        if application_id:
            # Use the application_id as a seed for consistent mock data
            import hashlib
            seed = int(hashlib.md5(application_id.encode()).hexdigest(), 16) % 10000
            import random
            random.seed(seed)
            
            statuses = ["pending", "approved", "rejected", "review"]
            validation_statuses = ["pending", "verified", "incomplete", "flagged"]
            risk_levels = ["low", "medium", "high"]
            
            # Generate mock application data
            created_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))
            
            return {
                "application_id": application_id,
                "name": f"Demo User {application_id[-4:]}",
                "income": random.randint(2000, 20000),
                "family_size": random.randint(1, 7),
                "address": "123 Demo Street, Demo City",
                "assessment_status": random.choice(statuses),
                "validation_status": random.choice(validation_statuses),
                "risk_level": random.choice(risk_levels),
                "created_at": created_date.isoformat(),
                "updated_at": (created_date + datetime.timedelta(days=random.randint(0, 5))).isoformat(),
                "score": random.randint(40, 95),
                "documents": [
                    {"document_id": str(uuid.uuid4()), "document_type": "emirates_id", "filename": "emirates_id.pdf", "status": "verified"},
                    {"document_id": str(uuid.uuid4()), "document_type": "income_statement", "filename": "income.pdf", "status": random.choice(["pending", "verified"])},
                    {"document_id": str(uuid.uuid4()), "document_type": "bank_statement", "filename": "bank.pdf", "status": random.choice(["pending", "verified"])},
                ],
                "recommendations": [
                    {
                        "id": 1,
                        "category": "Financial Assistance",
                        "description": "Based on your income and family size, you qualify for monthly financial support.",
                        "priority": "high" if random.random() < 0.3 else "medium",
                        "eligibility": random.randint(70, 100)
                    },
                    {
                        "id": 2,
                        "category": "Employment Support",
                        "description": "You may benefit from our job placement program to improve your long-term financial stability.",
                        "priority": "medium",
                        "eligibility": random.randint(50, 90)
                    },
                    {
                        "id": 3,
                        "category": "Housing Assistance",
                        "description": "You may qualify for housing support benefits to reduce your living expenses.",
                        "priority": "low" if random.random() < 0.7 else "medium",
                        "eligibility": random.randint(30, 80)
                    }
                ]
            }
    except Exception as e:
        st.error(f"Error getting application status: {str(e)}")
        return None

def send_chat_message(application_id, message):
    """Send chat message to API."""
    try:
        data = {
            "application_id": application_id,
            "message": message
        }
        st.write(f"Sending request to {API_URL}/api/chat with application_id: {application_id}")
        response = requests.post(f"{API_URL}/api/chat", json=data)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        st.error(f"Error sending message: {str(e)}\n\nDetails: {error_trace}")
        return None

def get_application_explanation(application_id):
    """Get application explanation from API."""
    try:
        response = requests.get(f"{API_URL}/api/applications/{application_id}/explanation")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting application explanation: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/government.png", width=80)
st.sidebar.title("Social Security Support")

# Application ID input
application_id_input = st.sidebar.text_input("Application ID (if you have one)")
if application_id_input and application_id_input != st.session_state.application_id:
    st.session_state.application_id = application_id_input
    app_data = get_application_status(application_id_input)
    if app_data:
        st.session_state.application_data = app_data
        st.sidebar.success(f"Application loaded: {app_data['name']}")
    else:
        st.sidebar.error("Application not found")
        st.session_state.application_id = None
        st.session_state.application_data = None

# Navigation menu
pages = {
    "Home": "🏠 Home",
    "Apply": "📝 Apply for Support",
    "Documents": "📄 Upload Documents",
    "Status": "🔍 Application Status",
    "Chat": "💬 Chat Assistance",
    "About": "ℹ️ About"
}

selected_page = st.sidebar.radio("Navigation", list(pages.values()))
st.session_state.current_tab = list(pages.keys())[list(pages.values()).index(selected_page)]

# Sidebar application status summary
if st.session_state.application_data:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Application Summary")
    app_data = st.session_state.application_data
    st.sidebar.markdown(f"**Name:** {app_data['name']}")
    st.sidebar.markdown(f"**Status:** {app_data['assessment_status'].capitalize()}")
    st.sidebar.markdown(f"**Validation:** {app_data['validation_status'].capitalize()}")
    st.sidebar.markdown(f"**Documents:** {len(app_data.get('documents', []))}")
    st.sidebar.markdown(f"**Submitted:** {app_data['created_at'].split('T')[0]}")

# Home page
if st.session_state.current_tab == "Home":
    st.markdown('<h1 class="main-header">Welcome to the Social Security Support System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Our AI-powered platform helps you access social support services quickly and efficiently.
        
        ### What We Offer
        
        - **Financial Support**: Direct assistance for eligible individuals and families
        - **Economic Enablement**: Training, job placement, and small business support
        - **Personalized Guidance**: AI-powered counseling and recommendations
        
        ### How It Works
        
        1. Submit your application with basic information
        2. Upload supporting documents
        3. Receive an instant assessment and recommendations
        4. Chat with our AI assistant for personalized guidance
        
        Get started by clicking on **"Apply for Support"** in the sidebar!
        """)
        
        # Quick links
        st.markdown("### Quick Links")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("Apply Now", use_container_width=True):
                st.session_state.current_tab = "Apply"
                st.experimental_rerun()
        
        with col_b:
            if st.button("Check Status", use_container_width=True):
                st.session_state.current_tab = "Status"
                st.experimental_rerun()
        
        with col_c:
            if st.button("Get Help", use_container_width=True):
                st.session_state.current_tab = "Chat"
                st.experimental_rerun()
    
    with col2:
        # Decorative image
        st.image("https://img.icons8.com/color/480/000000/welfare.png", width=250)
        
        # Statistics
        st.markdown("### Quick Processing")
        st.metric("Average Processing Time", "3 minutes", "-99% vs traditional")
        st.metric("Approval Rate", "94%", "+15%")
        st.metric("User Satisfaction", "98%", "+25%")

# Apply page
elif st.session_state.current_tab == "Apply":
    st.markdown('<h1 class="main-header">Apply for Social Support</h1>', unsafe_allow_html=True)
    
    with st.form("application_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", help="Enter your full name as it appears on your ID")
            email = st.text_input("Email Address", help="Enter a valid email address for communications")
            phone = st.text_input("Phone Number", help="Enter your phone number including country code")
        
        with col2:
            emirates_id = st.text_input("Emirates ID", help="Enter your 15-digit Emirates ID number")
            family_size = st.number_input("Family Size", min_value=1, max_value=30, value=1, help="Number of family members including yourself")
            address = st.text_area("Residential Address", help="Enter your current residential address")
        
        st.markdown("### Financial Information")
        col3, col4 = st.columns(2)
        
        with col3:
            income = st.number_input("Monthly Income (AED)", min_value=0, value=0, help="Your average monthly income in AED")
            monthly_expenses = st.number_input("Monthly Expenses (AED)", min_value=0, value=0, help="Your average monthly expenses in AED")
        
        with col4:
            assets_value = st.number_input("Total Assets Value (AED)", min_value=0, value=0, help="Estimated value of all your assets")
            liabilities_value = st.number_input("Total Liabilities Value (AED)", min_value=0, value=0, help="Total value of all your debts and liabilities")
        
        st.markdown("### Employment Information")
        col5, col6 = st.columns(2)
        
        with col5:
            employment_status = st.selectbox(
                "Employment Status",
                ["employed", "self-employed", "unemployed", "retired", "student", "homemaker", "unable to work", "part-time"],
                help="Select your current employment status"
            )
        
        with col6:
            employer = st.text_input("Employer Name", help="Name of your current employer (if employed)")
            job_title = st.text_input("Job Title", help="Your current job title (if employed)")
            employment_duration = st.number_input("Employment Duration (months)", min_value=0, value=0, help="How long you've been at your current job in months")
        
        submit_button = st.form_submit_button("Submit Application")
    
    if submit_button:
        if not name or not address or income < 0 or family_size < 1:
            st.error("Please fill in all required fields")
        else:
            # Prepare application data
            application_data = {
                "name": name,
                "email": email,
                "phone": phone,
                "emirates_id": emirates_id,
                "income": income,
                "family_size": family_size,
                "address": address,
                "employment_status": employment_status,
                "employer": employer,
                "job_title": job_title,
                "employment_duration": employment_duration,
                "monthly_expenses": monthly_expenses,
                "assets_value": assets_value,
                "liabilities_value": liabilities_value
            }
            
            # Submit application
            response = submit_application(application_data)
            
            if response:
                st.session_state.application_id = response["application_id"]
                st.success(f"Application submitted successfully! Your application ID is: {response['application_id']}")
                st.info("Please note your application ID for future reference. You can now upload supporting documents.")
                
                # Button to go to documents page
                if st.button("Upload Supporting Documents"):
                    st.session_state.current_tab = "Documents"
                    st.experimental_rerun()

# Documents page
elif st.session_state.current_tab == "Documents":
    st.markdown('<h1 class="main-header">Upload Supporting Documents</h1>', unsafe_allow_html=True)
    
    if not st.session_state.application_id:
        st.warning("Please enter your Application ID in the sidebar or submit a new application first")
    else:
        st.info(f"Uploading documents for Application ID: {st.session_state.application_id}")
        
        # Document upload section
        document_type = st.selectbox(
            "Document Type",
            ["emirates_id", "bank_statement", "resume", "assets_liabilities", "income_statement", "other"],
            help="Select the type of document you're uploading"
        )
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv"],
            help="Upload supporting documents in PDF, image, or spreadsheet format"
        )
        
        if uploaded_file and st.button("Upload Document"):
            # Upload document
            response = upload_document(uploaded_file, st.session_state.application_id, document_type)
            
            if response:
                st.success(f"Document uploaded successfully: {uploaded_file.name}")
                
                # Add to uploaded documents list
                if "uploaded_documents" not in st.session_state:
                    st.session_state.uploaded_documents = []
                
                st.session_state.uploaded_documents.append({
                    "filename": uploaded_file.name,
                    "document_type": document_type,
                    "upload_time": datetime.datetime.now().isoformat()
                })
                
                # Refresh application data
                app_data = get_application_status(st.session_state.application_id)
                if app_data:
                    st.session_state.application_data = app_data
        
        # Display uploaded documents
        if st.session_state.application_data and "documents" in st.session_state.application_data:
            st.markdown("### Uploaded Documents")
            
            docs = st.session_state.application_data["documents"]
            
            if not docs:
                st.info("No documents uploaded yet")
            else:
                for doc in docs:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{doc['filename']}**")
                    with col2:
                        st.write(f"Type: {doc.get('document_type', 'Unknown')}")
                    with col3:
                        # Safely handle the case where uploaded_at doesn't exist
                        if 'uploaded_at' in doc:
                            st.write(f"Uploaded: {doc['uploaded_at'].split('T')[0]}")
                        else:
                            st.write("Uploaded: Not available")
                    st.markdown("---")
        
        # Check application status button
        if st.button("Check Application Status"):
            st.session_state.current_tab = "Status"
            st.experimental_rerun()

# Status page
elif st.session_state.current_tab == "Status":
    st.markdown('<h1 class="main-header">Application Status</h1>', unsafe_allow_html=True)
    
    if not st.session_state.application_id:
        st.warning("Please enter your Application ID in the sidebar to check your status")
    else:
        # Refresh application data
        if st.button("Refresh Status"):
            app_data = get_application_status(st.session_state.application_id)
            if app_data:
                st.session_state.application_data = app_data
                st.success("Status updated")
        
        if not st.session_state.application_data:
            app_data = get_application_status(st.session_state.application_id)
            if app_data:
                st.session_state.application_data = app_data
        
        if st.session_state.application_data:
            app_data = st.session_state.application_data
            
            # Status overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Validation Status")
                st.markdown(format_status(app_data["validation_status"]), unsafe_allow_html=True)
                
                if app_data["validation_status"] == "invalid":
                    st.error("Your application has validation issues that need to be addressed")
                elif app_data["validation_status"] == "valid":
                    st.success("Your application has passed validation")
                else:
                    st.info("Your application is being validated")
            
            with col2:
                st.markdown("### Assessment Status")
                st.markdown(format_status(app_data["assessment_status"]), unsafe_allow_html=True)
                
                if app_data["assessment_status"] == "rejected":
                    st.error("Your application has been rejected")
                elif app_data["assessment_status"] == "approved":
                    st.success("Your application has been approved")
                else:
                    st.info("Your application is being assessed")
            
            with col3:
                st.markdown("### Risk Level")
                risk_level = app_data.get("risk_level", "unknown")
                
                if risk_level == "high":
                    st.error(f"Risk Level: {risk_level.upper()}")
                elif risk_level == "medium":
                    st.warning(f"Risk Level: {risk_level.upper()}")
                elif risk_level == "low":
                    st.success(f"Risk Level: {risk_level.upper()}")
                else:
                    st.info(f"Risk Level: {risk_level.upper()}")
            
            # Application timeline
            st.markdown("### Application Timeline")
            
            created_date = datetime.datetime.fromisoformat(app_data["created_at"].replace("Z", "+00:00"))
            updated_date = datetime.datetime.fromisoformat(app_data["updated_at"].replace("Z", "+00:00"))
            
            timeline_data = [
                {"Date": created_date, "Event": "Application Submitted", "Status": "Completed"},
                {"Date": created_date + datetime.timedelta(minutes=1), "Event": "Document Processing", "Status": "Completed" if app_data.get("documents") else "Pending"},
                {"Date": created_date + datetime.timedelta(minutes=2), "Event": "Validation", "Status": "Completed" if app_data["validation_status"] in ["valid", "invalid"] else "Pending"},
                {"Date": updated_date, "Event": "Assessment", "Status": "Completed" if app_data["assessment_status"] in ["approved", "rejected"] else "Pending"},
                {"Date": updated_date + datetime.timedelta(minutes=1), "Event": "Decision", "Status": "Completed" if app_data["assessment_status"] in ["approved", "rejected"] else "Pending"}
            ]
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=timeline_df["Date"],
                    y=timeline_df["Event"],
                    mode="markers+lines",
                    marker=dict(
                        size=16,
                        color=["green" if status == "Completed" else "orange" for status in timeline_df["Status"]],
                        symbol="circle"
                    ),
                    line=dict(color="royalblue", width=2)
                )
            ])
            
            fig.update_layout(
                title="Application Processing Timeline",
                xaxis_title="Date",
                yaxis_title="Process Stage",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Decision explanation
            if app_data["assessment_status"] in ["approved", "rejected"]:
                st.markdown("### Decision Explanation")
                
                explanation = get_application_explanation(st.session_state.application_id)
                
                if explanation:
                    st.info(explanation.get("explanation", "No explanation available"))
                    
                    st.markdown("#### Next Steps")
                    for step in explanation.get("next_steps", []):
                        st.markdown(f"- {step}")
                    
                    st.markdown(f"**Timeline:** {explanation.get('timeline', 'No timeline available')}")
            
            # Recommendations
            if "recommendations" in app_data and app_data["recommendations"]:
                st.markdown("### Recommendations")
                
                for rec in app_data["recommendations"]:
                    with st.expander(f"{rec['category']} - Priority: {format_priority(rec['priority'])}", expanded=rec['priority'] == 'high'):
                        st.markdown(f"**{rec['description']}**")
                        
                        st.markdown("**Action Items:**")
                        for item in rec.get("action_items", []):
                            st.markdown(f"- {item}")
            
            # Chat button
            if st.button("Chat with Support Assistant"):
                st.session_state.current_tab = "Chat"
                st.experimental_rerun()

# Chat page
elif st.session_state.current_tab == "Chat":
    st.markdown('<h1 class="main-header">AI Support Assistant</h1>', unsafe_allow_html=True)
    
    if not st.session_state.application_id:
        st.warning("Please enter your Application ID in the sidebar to chat with the support assistant")
    else:
        # Option to use direct chat (bypass backend)
        use_direct_chat = st.checkbox("Use direct chat (recommended)", value=True, help="Use a simpler chat implementation that doesn't rely on backend API")
        
        # Option to disable Ollama attempts
        use_ollama = not st.checkbox("Skip Ollama connection (faster responses)", value=True, help="Use pre-programmed responses instead of connecting to Ollama")
        
        if use_direct_chat:
            # Direct chat implementation (integrated from simple_chat.py)
            # Initialize chat history if needed
            if "simple_chat_history" not in st.session_state:
                st.session_state.simple_chat_history = []
            
            # Display chat history
            for message in st.session_state.simple_chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-user"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai"><strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Input for new message
            with st.form("direct_chat_form", clear_on_submit=True):
                user_message = st.text_area("Your message:", height=100, key="direct_chat_input")
                submitted = st.form_submit_button("Send")
                
                if submitted and user_message:
                    # Add user message to chat history
                    st.session_state.simple_chat_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Generate AI response using direct implementation
                    def get_simple_response(user_query, use_ollama=False):
                        """Generate a simple response based on the user query."""
                        try:
                            # Try direct Ollama connection only if enabled
                            if use_ollama:
                                try:
                                    with st.spinner("Connecting to Ollama..."):
                                        response = requests.post(
                                            "http://localhost:11434/api/generate",
                                            json={
                                                "model": "mistral",
                                                "prompt": f"""You are a helpful AI assistant for the Social Security Support System.
                                                The user has a question about their social security application: {user_query}
                                                Provide a helpful, concise response:""",
                                                "stream": False
                                            },
                                            timeout=15
                                        )
                                    
                                    if response.status_code == 200:
                                        return response.json().get("response", "No response received from AI system.")
                                except Exception as e:
                                    # If Ollama fails, continue to fallback responses
                                    st.error(f"Could not connect to Ollama: {str(e)}")
                            else:
                                st.info("Using pre-programmed responses (Ollama connection disabled).")
                            
                            # Fallback to simple rule-based responses
                            user_query = user_query.lower()
                            
                            if "income" in user_query or "financial" in user_query or "money" in user_query:
                                return "Based on your income and financial situation, you may be eligible for additional support. I recommend submitting your latest income statements and employment records to strengthen your application."
                            
                            elif "document" in user_query or "upload" in user_query:
                                return "To complete your application, please upload the following documents: proof of identity (Emirates ID or passport), proof of income (salary slips or bank statements), and proof of residence (utility bills or rental agreement)."
                            
                            elif "status" in user_query or "progress" in user_query:
                                return "Your application is currently being processed. The typical processing time is 5-7 business days. You'll receive notifications as your application progresses through validation and assessment."
                            
                            elif "eligible" in user_query or "qualify" in user_query:
                                return "Eligibility for social security benefits depends on several factors including income level, family size, employment status, and residency status. Based on the information in your application, our system will determine your eligibility and provide recommendations."
                            
                            elif "help" in user_query or "assistance" in user_query:
                                return "I'm here to help with your social security application. I can provide information on eligibility criteria, required documents, application status, and recommendations based on your specific situation."
                            
                            else:
                                return "Thank you for your query. I'm your AI assistant for the Social Security Support System. I can help with application submissions, document requirements, eligibility criteria, and checking application status. Please let me know how I can assist you further."
                        
                        except Exception as e:
                            return f"I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists. Error details: {str(e)}"
                    
                    ai_response = get_simple_response(user_message, use_ollama)
                    
                    # Add AI response to chat history
                    st.session_state.simple_chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    # Store the input to clear it on the next rerun
                    st.session_state["last_message"] = user_message
                    
                    # Rerun to update the UI
                    st.rerun()
            
            # Option to clear chat history
            if st.button("Clear Chat History", key="simple_clear_chat"):
                st.session_state.simple_chat_history = []
                st.rerun()
        else:         
            # Original chat implementation using backend API
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-user"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-ai"><strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                user_message = st.text_area("Your message:", height=100)
                submitted = st.form_submit_button("Send")
                
                if submitted and user_message:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Send message to API
                    response = send_chat_message(st.session_state.application_id, user_message)
                    
                    if response:
                        # Add AI response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["text"]
                        })
                        
                        # Display suggestions
                        if "suggestions" in response and response["suggestions"]:
                            suggestion_buttons = []
                            for suggestion in response["suggestions"]:
                                suggestion_buttons.append(suggestion)
                    
                    # Rerun to update chat display
                    st.rerun()
        
        # Display suggestions if available
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "assistant":
            last_response = send_chat_message(st.session_state.application_id, "show suggestions")
            
            if last_response and "suggestions" in last_response and last_response["suggestions"]:
                st.markdown("**Suggested Questions:**")
                cols = st.columns(len(last_response["suggestions"]))
                
                for i, suggestion in enumerate(last_response["suggestions"]):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            # Add suggestion to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggestion
                            })
                            
                            # Send message to API
                            response = send_chat_message(st.session_state.application_id, suggestion)
                            
                            if response:
                                # Add AI response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response["text"]
                                })
                            
                            # Rerun to update chat display
                            st.experimental_rerun()

# About page
elif st.session_state.current_tab == "About":
    st.markdown('<h1 class="main-header">About the System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Enhanced Social Security Application System
    
    Our AI-powered system revolutionizes the social security application process, reducing processing time from 5-20 working days to just minutes.
    
    ### System Features
    
    - **Multimodal Document Processing**: Extract information from various document types
    - **Advanced Validation**: Ensure data completeness and consistency
    - **Risk Assessment**: AI-powered evaluation of applications
    - **Personalized Recommendations**: Tailored support options based on individual needs
    - **Interactive Chat**: Real-time assistance through our AI chatbot
    
    ### Technology Stack
    
    - **Backend**: Python, FastAPI
    - **Frontend**: Streamlit
    - **Database**: PostgreSQL, ChromaDB
    - **ML/AI**: Scikit-learn, LangChain, LlamaIndex
    - **Reasoning**: ReAct framework for agent reasoning
    - **Orchestration**: LangGraph for agent workflow
    - **LLM**: Locally hosted models via Ollama
    
    ### Privacy & Security
    
    Your data is securely stored and processed in compliance with data protection regulations. We implement state-of-the-art security measures to ensure the confidentiality and integrity of your personal information.
    """)
    
    # System architecture diagram
    st.markdown("### System Architecture")
    
    architecture_diagram = """
    graph TD
        A[User Interface] -->|Submit Application| B[API Layer]
        B -->|Process Application| C[Orchestrator Agent]
        C -->|Extract Data| D[Data Collector Agent]
        C -->|Validate Data| E[Validator Agent]
        C -->|Assess Eligibility| F[Assessor Agent]
        C -->|Generate Recommendations| G[Counselor Agent]
        D -->|Store Data| H[(PostgreSQL Database)]
        E -->|Update Status| H
        F -->|Store Assessment| H
        G -->|Store Recommendations| H
        G -->|Policy Information| I[(ChromaDB)]
        A -->|Chat Messages| J[Chat Interface]
        J -->|Query Agents| C
    """
    
    st.graphviz_chart(architecture_diagram)
    
    # Team information
    st.markdown("### System Development")
    st.markdown("""
    This system was developed as part of an initiative to modernize social security services using AI technology.
    
    The system aims to address key pain points in the traditional application process:
    - Manual data gathering
    - Semi-automated validations
    - Inconsistent information
    - Time-consuming reviews
    - Subjective decision-making
    
    Our goal is to provide fast, fair, and transparent access to social support services.
    """)

# Display footer
st.markdown("---")
st.markdown("© 2025 Social Security Support System | Powered by AI | Privacy Policy | Terms of Service")
