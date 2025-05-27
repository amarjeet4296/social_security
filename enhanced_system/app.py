import streamlit as st
import pandas as pd
import asyncio
import tempfile
import os
import logging
from datetime import datetime

# Import agent modules
from enhanced_system.agents.data_collector import DataCollector
from enhanced_system.agents.validator import ValidatorAgent
from enhanced_system.agents.assessor import AssessorAgent
from enhanced_system.database.models import Application
from enhanced_system.agents.counselor import CounselorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set page config
st.set_page_config(
    page_title="Multi-Agent Document Processing System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['filename', 'income', 'family_size', 'address', 'validation_status', 'assessment_status', 'risk_level'])

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"
    
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
    
if 'current_result' not in st.session_state:
    st.session_state.current_result = {}
    
# Initialize agents
logging.info("Initializing agents")
collector = DataCollector()
validator = ValidatorAgent()
assessor = AssessorAgent()

# Initialize counselor only when needed to save resources
try:
    counselor = CounselorAgent()
    st.session_state.counselor_available = True
    logging.info("Counselor agent initialized successfully")
except Exception as e:
    st.session_state.counselor_available = False
    logging.error(f"Error initializing counselor: {str(e)}")

# Title and application description
st.title("🤖 Multi-Agent Document Processing System")

st.markdown("""
## About This System

This application uses a multi-agent architecture to process documents, validate data, assess eligibility, and provide guidance:

1. **Data Collector Agent**: Extracts key information from documents using OCR
2. **Validator Agent**: Ensures data meets quality standards
3. **Assessor Agent**: Evaluates applications and performs risk scoring
4. **Counselor Agent**: Provides personalized guidance using LLM technology

All data is persisted in a database for tracking and future reference.
""")

# Create tabs for different functionalities
tabs = ["Upload", "Process", "History", "Guidance", "System Info"]
selected_tab = st.radio("Navigation", tabs, horizontal=True, index=tabs.index(st.session_state.current_tab))
st.session_state.current_tab = selected_tab

# Separator
st.markdown("---")

# Upload Tab - Document Upload and Initial Processing
if selected_tab == "Upload":
    st.header("📤 Document Upload")
    st.write("Upload a document to begin the processing pipeline.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display file details
        st.write("File Details:")
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Process button - routes to Data Collector Agent
        if st.button("Submit Document"):
            st.session_state.current_tab = "Process"
            st.session_state.uploaded_file = uploaded_file
            st.experimental_rerun()

# Process Tab - Data Collection, Validation, and Assessment
elif selected_tab == "Process":
    st.header("⚙️ Document Processing Pipeline")
    
    if not hasattr(st.session_state, 'uploaded_file'):
        st.warning("No file has been uploaded. Please go to the Upload tab first.")
        if st.button("Go to Upload Tab"):
            st.session_state.current_tab = "Upload"
            st.experimental_rerun()
    else:
        uploaded_file = st.session_state.uploaded_file
        st.write(f"Processing file: {uploaded_file.name}")
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_area = st.empty()
        
        # Process the document through all agents
        with st.spinner("Processing document..."):
            try:
                # DATA COLLECTOR AGENT
                status_area.info("🔍 Data Collector Agent: Extracting information from document...")
                progress_bar.progress(10)
                
                # Read file bytes
                file_bytes = uploaded_file.read()
                
                # Process the document
                result = asyncio.run(collector.process_document(file_bytes, uploaded_file.name))
                progress_bar.progress(30)
                
                # Add filename to result
                result['filename'] = uploaded_file.name
                
                # Create a database entry immediately after processing
                application = Application(
                    filename=result['filename'],
                    income=result.get('income', 0),
                    family_size=result.get('family_size', 0),
                    address=result.get('address', ''),
                    validation_status="",  # Will be updated after validation
                    assessment_status="",  # Will be updated after assessment
                    risk_level=""
                )
                assessor.db.add(application)
                assessor.db.commit()
                progress_bar.progress(40)
                
                # VALIDATOR AGENT
                status_area.info("✅ Validator Agent: Checking data quality...")
                progress_bar.progress(50)
                
                # Validate the result
                is_valid, validation_errors = validator.validate(result)
                
                # Update validation status in database
                application.validation_status = "✅ Valid" if is_valid else "❌ Invalid"
                assessor.db.commit()
                
                # Add validation status to result for display
                result['validation_status'] = application.validation_status
                progress_bar.progress(70)
                
                # ASSESSOR AGENT
                status_area.info("🧮 Assessor Agent: Evaluating application and risk...")
                progress_bar.progress(80)
                
                if is_valid:
                    # Assess the application
                    is_approved, reasons, assessment_details = assessor.assess_application(result)
                    
                    # Update assessment status in database
                    application.assessment_status = "✅ Approved" if is_approved else "❌ Rejected"
                    application.risk_level = assessment_details['risk_level']
                    assessor.db.commit()
                    
                    # Add assessment results to result dict
                    result['assessment_status'] = application.assessment_status
                    result['risk_level'] = application.risk_level
                    
                    # Show assessment results
                    if is_approved:
                        status_area.success("✅ Application approved!")
                    else:
                        status_area.warning("❌ Application rejected for the following reasons:")
                        for reason in reasons:
                            st.error(reason)
                    
                    st.info(f"Risk Level: {assessment_details['risk_level'].upper()}")
                    st.info(f"Income per Family Member: {assessment_details['income_per_member']:.2f} AED")
                else:
                    status_area.warning("Document processed but validation failed:")
                    for error in validation_errors:
                        st.error(error)
                
                # Add to DataFrame
                new_row = pd.DataFrame([result])
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                st.session_state.current_result = result
                st.session_state.processing_complete = True
                progress_bar.progress(100)
                
                # Success message and next steps
                st.success("Document processing complete!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View Processing History"):
                        st.session_state.current_tab = "History"
                        st.experimental_rerun()
                with col2:
                    if st.button("Get Guidance"):
                        st.session_state.current_tab = "Guidance"
                        st.experimental_rerun()
                
            except Exception as e:
                progress_bar.progress(100)
                st.error(f"Error processing document: {str(e)}")
                logging.error(f"Document processing error: {str(e)}")

# History Tab - View processed documents and database records
elif selected_tab == "History":
    st.header("📋 Processing History")
    
    # Sub-tabs for different views
    history_tabs = st.tabs(["Session History", "Database Records"])
    
    with history_tabs[0]:
        st.subheader("Documents Processed in Current Session")
        if not st.session_state.df.empty:
            st.dataframe(st.session_state.df)
            
            # Add download button for DataFrame
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="processed_documents.csv",
                mime="text/csv"
            )
        else:
            st.info("No documents processed in this session. Upload a file to begin.")
    
    with history_tabs[1]:
        st.subheader("Recent Assessments from Database")
        recent_assessments = assessor.get_recent_assessments()
        if recent_assessments:
            st.dataframe(pd.DataFrame(recent_assessments))
        else:
            st.info("No assessments in the database yet.")

# Guidance Tab - Counselor Agent interaction
elif selected_tab == "Guidance":
    st.header("🧠 Counselor Agent Guidance")
    
    if not st.session_state.processing_complete:
        st.warning("Please process a document first to receive personalized guidance.")
        if st.button("Go to Upload Tab"):
            st.session_state.current_tab = "Upload"
            st.experimental_rerun()
    else:
        st.subheader("Ask for Guidance About Your Application")
        
        # Display application details for reference
        if st.session_state.current_result:
            with st.expander("View Your Application Details", expanded=False):
                for key, value in st.session_state.current_result.items():
                    st.write(f"**{key}:** {value}")
        
        # Counselor interaction section
        if not st.session_state.counselor_available:
            st.error("Counselor Agent is not available. Please ensure Ollama is running with the Mistral model loaded.")
            st.info("You can install Ollama using: `brew install ollama` and load the model with: `ollama pull mistral`")
        else:
            # Try to initialize the counselor
            try:
                if not hasattr(counselor, 'agent'):
                    counselor = CounselorAgent()
                    
                default_query = "What can I do to improve my application?"
                user_query = st.text_input("Enter your question for the counselor agent:", value=default_query)
                
                if st.button("Get Guidance"):
                    with st.spinner("Counselor is preparing guidance..."):
                        try:
                            guidance = counselor.provide_guidance(st.session_state.current_result, user_query)
                            st.markdown("### Counselor Guidance")
                            st.json(guidance)
                        except Exception as e:
                            st.error(f"Error getting guidance: {str(e)}")
                            st.info("Please ensure Ollama is running and the Mistral model is loaded.")
            except Exception as e:
                st.error(f"Counselor agent initialization error: {str(e)}")
                st.info("Please ensure Ollama is running and the Mistral model is loaded.")

# System Info Tab - Information about the system architecture and agents
elif selected_tab == "System Info":
    st.header("ℹ️ System Architecture and Information")
    
    st.subheader("Multi-Agent Architecture")
    st.markdown("""
    This application uses a sophisticated multi-agent architecture to process documents and provide guidance:
    
    1. **Decision Orchestrator** (this Streamlit app)
       - Coordinates the flow between all agents
       - Provides user interface for document upload and interaction
    
    2. **Data Collector Agent**
       - Extracts text from documents using OCR
       - Identifies key information like income, family size, and address
       - Uses tesseract and PDF processing libraries
    
    3. **Validator Agent**
       - Validates extracted data against business rules
       - Ensures data quality and completeness
       - Flags issues for human review
    
    4. **Assessor Agent**
       - Evaluates applications based on validated data
       - Performs risk scoring and eligibility assessment
       - Stores results in PostgreSQL database
    
    5. **Counselor Agent**
       - Provides guidance using LLM technology (Ollama/Mistral)
       - Accesses policy documents stored in ChromaDB
       - Generates personalized recommendations
    
    **Data Persistence Layer**
    - PostgreSQL for structured application data
    - ChromaDB for policy documents and semantic search
    
    **Logging and Monitoring**
    - Error logging and debugging information
    - Application state tracking
    """)
    
    # System diagnostics
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Agents Status:**")
        st.write("- Data Collector: ✅ Running")
        st.write("- Validator: ✅ Running")
        st.write("- Assessor: ✅ Running")
        st.write(f"- Counselor: {'✅ Running' if st.session_state.counselor_available else '❌ Not Available'}")
    
    with col2:
        st.write("**Database Status:**")
        try:
            count = len(assessor.get_recent_assessments(limit=1000))
            st.write(f"- PostgreSQL: ✅ Connected ({count} records)")
        except:
            st.write("- PostgreSQL: ❌ Connection Issue")
        
        st.write(f"- ChromaDB: {'✅ Connected' if st.session_state.counselor_available else '❌ Not Verified'}")
    
    # Version information
    st.subheader("Version Information")
    st.write(f"- Application Version: 1.0.0")
    st.write(f"- Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Reset application button
    if st.button("Reset Application State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Application state has been reset!")
        st.experimental_rerun()

# Note: The main application flow is now handled by the tabbed interface above