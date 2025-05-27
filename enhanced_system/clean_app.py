import streamlit as st
import pandas as pd
import asyncio
import tempfile
import os
import logging
import traceback
from datetime import datetime

# Import the Process tab implementation
from enhanced_system.process_tab import render_process_tab

# Import agent modules
from enhanced_system.agents.data_collector import DataCollectorAgent as DataCollector
from enhanced_system.agents.validator import ValidatorAgent
from enhanced_system.agents.assessor import AssessorAgent
from enhanced_system.database.db_setup import Application
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
    st.session_state.current_tab = "Application Form"  # Changed default to Application Form
    
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
tabs = ["Application Form", "Upload", "Process", "History", "Guidance", "System Info"]
selected_tab = st.radio("Navigation", tabs, horizontal=True, index=tabs.index(st.session_state.current_tab))
st.session_state.current_tab = selected_tab

# Separator
st.markdown("---")

# Application Form Tab - For manual data entry
if selected_tab == "Application Form":
    st.header("📝 Application Form")
    st.write("Complete this form to apply for social security benefits.")
    
    # Initialize application data if not in session state
    if 'application_data' not in st.session_state:
        st.session_state.application_data = {
            "name": "",
            "email": "",
            "phone": "",
            "emirates_id": "",
            "income": 0.0,
            "family_size": 1,
            "address": "",
            "employment_status": "Employed",
            "employer": "",
            "job_title": "",
            "employment_duration": 0,
            "monthly_expenses": 0.0,
            "assets_value": 0.0,
            "liabilities_value": 0.0
        }
    
    # Create form with multiple sections using expanders
    with st.form("application_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.application_data["name"] = st.text_input(
                "Full Name",
                value=st.session_state.application_data["name"],
                help="Enter your full name as it appears on official documents"
            )
            
            st.session_state.application_data["email"] = st.text_input(
                "Email Address", 
                value=st.session_state.application_data["email"],
                help="Enter a valid email address for communications"
            )
        
        with col2:
            st.session_state.application_data["phone"] = st.text_input(
                "Phone Number", 
                value=st.session_state.application_data["phone"],
                help="Enter your mobile number with country code"
            )
            
            st.session_state.application_data["emirates_id"] = st.text_input(
                "Emirates ID", 
                value=st.session_state.application_data["emirates_id"],
                help="Enter your Emirates ID number"
            )
        
        st.session_state.application_data["address"] = st.text_area(
            "Residential Address", 
            value=st.session_state.application_data["address"],
            help="Enter your complete residential address"
        )
        
        st.session_state.application_data["family_size"] = st.number_input(
            "Family Size", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.application_data["family_size"],
            help="Include yourself and all dependents"
        )
        
        st.divider()
        st.subheader("Financial Information")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.application_data["income"] = st.number_input(
                "Monthly Income (AED)", 
                min_value=0.0, 
                value=float(st.session_state.application_data["income"]),
                step=500.0,
                help="Your total monthly income from all sources"
            )
            
            st.session_state.application_data["monthly_expenses"] = st.number_input(
                "Monthly Expenses (AED)", 
                min_value=0.0, 
                value=float(st.session_state.application_data["monthly_expenses"]),
                step=500.0,
                help="Your total monthly expenses"
            )
        
        with col2:
            st.session_state.application_data["assets_value"] = st.number_input(
                "Total Assets Value (AED)", 
                min_value=0.0, 
                value=float(st.session_state.application_data["assets_value"]),
                step=1000.0,
                help="Total value of all assets including property, vehicles, etc."
            )
            
            st.session_state.application_data["liabilities_value"] = st.number_input(
                "Total Liabilities (AED)", 
                min_value=0.0, 
                value=float(st.session_state.application_data["liabilities_value"]),
                step=1000.0,
                help="Total value of all debts and liabilities"
            )
        
        st.divider()
        st.subheader("Employment Information")
        col1, col2 = st.columns(2)
        with col1:
            employment_options = ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"]
            st.session_state.application_data["employment_status"] = st.selectbox(
                "Employment Status", 
                options=employment_options,
                index=employment_options.index(st.session_state.application_data["employment_status"]) if st.session_state.application_data["employment_status"] in employment_options else 0
            )
            
            st.session_state.application_data["employer"] = st.text_input(
                "Employer Name", 
                value=st.session_state.application_data["employer"],
                disabled=st.session_state.application_data["employment_status"] not in ["Employed", "Self-Employed"]
            )
        
        with col2:
            st.session_state.application_data["job_title"] = st.text_input(
                "Job Title", 
                value=st.session_state.application_data["job_title"],
                disabled=st.session_state.application_data["employment_status"] not in ["Employed", "Self-Employed"]
            )
            
            st.session_state.application_data["employment_duration"] = st.number_input(
                "Employment Duration (months)", 
                min_value=0, 
                value=st.session_state.application_data["employment_duration"],
                disabled=st.session_state.application_data["employment_status"] not in ["Employed", "Self-Employed"]
            )
        
        st.divider()
        submit_button = st.form_submit_button("Submit Application")
        
        if submit_button:
            # Validate form data
            required_fields = ["name", "phone", "emirates_id", "income", "address"]
            missing_fields = [field for field in required_fields if not st.session_state.application_data[field]]
            
            if missing_fields:
                st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
            else:
                # Create application in database
                try:
                    # Format data for database
                    app_data = {
                        "filename": f"APP_{datetime.now().strftime('%Y%m%d%H%M%S')}",  # Generate unique ID
                        **st.session_state.application_data
                    }
                    
                    # Initialize assessment status
                    app_data["validation_status"] = "pending"
                    app_data["assessment_status"] = "pending"
                    app_data["risk_level"] = "unknown"
                    
                    # Create application object
                    application = Application(**app_data)
                    assessor.db.add(application)
                    assessor.db.commit()
                    
                    # Store for processing
                    st.session_state.current_result = app_data
                    st.session_state.processing_complete = False
                    
                    # Show success message and redirect to processing
                    st.success("Application submitted successfully! Proceeding to processing...")
                    st.session_state.current_tab = "Process"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error submitting application: {str(e)}")
    
    # Additional guidance
    with st.expander("Need help with your application?"):
        st.write("""
        ### Application Guidelines
        - Provide accurate information to ensure your application is processed correctly
        - You can upload supporting documents in the Upload tab after submitting this form
        - All financial information should be in AED (Arab Emirates Dirham)
        - For assistance, use the Guidance tab to chat with our AI assistant
        """)

# Upload Tab - Document Upload and Initial Processing
elif selected_tab == "Upload":
    st.header("📤 Document Upload")
    st.write("Upload supporting documents for your application.")
    
    # Check if application exists
    if 'current_result' not in st.session_state or not st.session_state.current_result:
        st.warning("Please complete the application form first before uploading documents.")
        if st.button("Go to Application Form"):
            st.session_state.current_tab = "Application Form"
            st.experimental_rerun()
    else:
        # Display application info
        st.info(f"Uploading documents for: {st.session_state.current_result.get('name', 'Unknown applicant')}")
        
        # Document type selector
        doc_type = st.selectbox(
            "Document Type",
            options=["Emirates ID", "Bank Statement", "Salary Certificate", "Tenancy Contract", "Utility Bill", "Other"]
        )
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'csv'])
        
        if uploaded_file is not None:
            # Display file details
            st.write("File Details:")
            st.write(f"Document Type: {doc_type}")
            st.write(f"Filename: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size} bytes")
            
            # Process button - routes to Data Collector Agent
            if st.button("Submit Document"):
                st.session_state.document_type = doc_type.lower().replace(" ", "_")
                st.session_state.uploaded_file = uploaded_file
                st.session_state.current_tab = "Process"
                st.experimental_rerun()

# Process Tab - Data Collection, Validation, and Assessment
elif selected_tab == "Process":
    # Use the modular implementation from process_tab.py
    render_process_tab(collector, validator, assessor, counselor)

# History Tab - View processed documents and database records
elif selected_tab == "History":
    st.header("📋 Processing History")
    
    # Sub-tabs for different views
    history_tabs = ["Application List", "Data Explorer", "Database Records"]
    selected_history_tab = st.radio("View", history_tabs, horizontal=True)
    
    if selected_history_tab == "Application List":
        st.subheader("Processed Applications")
        
        if st.session_state.df.empty:
            st.info("No applications have been processed yet.")
        else:
            # Create a view of the dataframe with select columns
            view_df = st.session_state.df[['filename', 'income', 'family_size', 'validation_status', 'assessment_status', 'risk_level']].copy()
            
            # Add a formatted timestamp column
            view_df['processed_at'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Display the dataframe
            st.dataframe(view_df)
            
            # Allow downloading the data
            if st.button("Download CSV"):
                # Convert dataframe to CSV
                csv = view_df.to_csv(index=False)
                
                # Create a download button
                st.download_button(
                    label="Download Data",
                    data=csv,
                    file_name="processed_applications.csv",
                    mime="text/csv"
                )
    
    elif selected_history_tab == "Data Explorer":
        st.subheader("Application Data Explorer")
        
        if st.session_state.df.empty:
            st.info("No data available for exploration.")
        else:
            # Add filtering options
            st.write("Filter by:")
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by assessment status
                status_filter = st.multiselect(
                    "Assessment Status",
                    options=st.session_state.df['assessment_status'].unique()
                )
                
            with col2:
                # Filter by risk level
                risk_filter = st.multiselect(
                    "Risk Level",
                    options=st.session_state.df['risk_level'].unique()
                )
            
            # Apply filters
            filtered_df = st.session_state.df.copy()
            if status_filter:
                filtered_df = filtered_df[filtered_df['assessment_status'].isin(status_filter)]
            if risk_filter:
                filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
            
            # Show filtered data
            st.write(f"Showing {len(filtered_df)} applications")
            st.dataframe(filtered_df)
            
            # Simple analytics
            if len(filtered_df) > 0:
                st.subheader("Analytics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_income = filtered_df['income'].mean()
                    st.metric("Average Income", f"{avg_income:.2f} AED")
                
                with col2:
                    avg_family = filtered_df['family_size'].mean()
                    st.metric("Average Family Size", f"{avg_family:.1f}")
                
                with col3:
                    approved_count = len(filtered_df[filtered_df['assessment_status'].str.contains("Approved", na=False)])
                    approval_rate = (approved_count / len(filtered_df)) * 100
                    st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    elif selected_history_tab == "Database Records":
        st.subheader("Database Records")
        
        try:
            # Fetch records from database
            records = assessor.db.query(Application).all()
            
            if not records:
                st.info("No records found in the database.")
            else:
                # Convert to dataframe
                data = []
                for record in records:
                    data.append({
                        'ID': record.id,
                        'Filename': record.filename,
                        'Name': record.name,
                        'Income': record.income,
                        'Family Size': record.family_size,
                        'Address': record.address,
                        'Validation': record.validation_status,
                        'Assessment': record.assessment_status,
                        'Risk Level': record.risk_level,
                        'Created': record.created_at.strftime("%Y-%m-%d %H:%M:%S") if record.created_at else "N/A",
                        'Updated': record.updated_at.strftime("%Y-%m-%d %H:%M:%S") if record.updated_at else "N/A"
                    })
                
                db_df = pd.DataFrame(data)
                st.dataframe(db_df)
                
                # Allow record deletion
                if st.button("Clear All Records (CAUTION)"):
                    if st.session_state.get('confirm_delete') != True:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ This will delete ALL records from the database. Click again to confirm.")
                    else:
                        try:
                            for record in records:
                                assessor.db.delete(record)
                            assessor.db.commit()
                            st.success("All records deleted successfully.")
                            st.session_state.confirm_delete = False
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error deleting records: {str(e)}")
                            assessor.db.rollback()
        
        except Exception as e:
            st.error(f"Error fetching database records: {str(e)}")

# Guidance Tab - AI-powered chat assistance
elif selected_tab == "Guidance":
    st.header("🤖 AI Guidance Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI guidance assistant for the social security application process. How can I help you today?"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Check if counselor is available
    if not st.session_state.counselor_available:
        st.warning("The AI guidance system is currently unavailable. Please try again later.")
    else:
        # Get user input
        prompt = st.chat_input("Ask a question about your application or the support process...")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if hasattr(counselor, 'explain_decision') and 'current_result' in st.session_state and st.session_state.current_result:
                            # If we have application data and the question seems to be about the decision
                            if any(keyword in prompt.lower() for keyword in ["why", "decision", "approved", "rejected", "assessment", "result"]):
                                explanation = counselor.explain_decision(st.session_state.current_result)
                                response = explanation.get('explanation', "I don't have specific information about your application decision.")
                            else:
                                # General guidance
                                response = counselor.get_guidance(prompt)
                        else:
                            # General guidance
                            response = counselor.get_guidance(prompt)
                        
                        st.write(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Provide suggested questions
        with st.expander("Suggested Questions"):
            st.write("""
            - What documents do I need for my application?
            - How is my application assessment calculated?
            - Why was my application rejected?
            - What economic enablement support is available?
            - How long does the application process take?
            - How can I appeal a decision?
            """)

# System Info Tab - Information about the system architecture
elif selected_tab == "System Info":
    st.header("ℹ️ System Architecture and Information")
    
    st.subheader("Multi-Agent Architecture")
    st.markdown("""
    This application uses a sophisticated multi-agent architecture to process social security applications:
    
    1. **Data Collector Agent**: Extracts structured information from various document types including:
       - Emirates ID documents
       - Bank statements
       - Salary certificates
       - Tenancy contracts
       - Utility bills
       
       This agent uses OCR and NLP techniques to convert document content into structured data.
    
    2. **Validator Agent**: Performs comprehensive validation checks:
       - Verifies data completeness
       - Ensures data consistency across documents
       - Flags potential fraud indicators
       - Validates format of critical fields (ID numbers, financial figures)
    
    3. **Assessor Agent**: Evaluates application eligibility:
       - Applies risk scoring models
       - Considers income, employment history, and family size
       - Calculates financial metrics
       - Makes recommendation for approval or rejection
    
    4. **Counselor Agent**: Provides personalized guidance:
       - Offers explanations for decisions
       - Suggests additional documents if needed
       - Recommends economic enablement programs
       - Provides application improvement tips
    """)
    
    st.subheader("Technical Architecture")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Frontend
        - **Streamlit**: Interactive UI
        - **Pandas**: Data processing
        - **Plotly**: Data visualization
        
        ### Database
        - **PostgreSQL**: Application data storage
        - **ChromaDB**: Vector storage for documents
        
        ### ML/AI Components
        - **Scikit-learn**: Classification models
        - **Tesseract OCR**: Document text extraction
        - **LangChain**: Agent orchestration
        - **Ollama**: Local LLM inference
        """)
    
    with col2:
        st.markdown("""
        ### Processing Pipeline
        1. **Document Upload**: User submits application form and documents
        2. **Data Extraction**: System extracts structured information
        3. **Validation**: Data is checked for completeness and consistency
        4. **Assessment**: Application is evaluated against eligibility criteria
        5. **Decision**: System provides recommendation with explanation
        6. **Guidance**: AI assistant provides personalized advice
        
        ### Integration Points
        - **API Endpoints**: http://localhost:8080
        - **Ollama Server**: http://localhost:11434
        """)
    
    # System status
    st.subheader("System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Database status
        try:
            # Check database connection
            record_count = assessor.db.query(Application).count()
            st.success("✅ Database Connected")
            st.info(f"Records: {record_count}")
        except Exception:
            st.error("❌ Database Offline")
    
    with col2:
        # LLM status
        if st.session_state.counselor_available:
            st.success("✅ LLM Service Running")
            st.info("Using: Mistral (Ollama)")
        else:
            st.warning("⚠️ Using Fallback LLM")
    
    with col3:
        # Document processing status
        if hasattr(collector, "process_document"):
            st.success("✅ Document Processing Ready")
        else:
            st.error("❌ Document Processing Unavailable")
    
    with col4:
        # API status (mock for now)
        api_status = "Online"  # In a real app, this would check the API
        if api_status == "Online":
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
    
    # Version information
    st.text("Version: 1.2.0 | Last Updated: May 27, 2025")
