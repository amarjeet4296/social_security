"""
Process Tab implementation for Social Security Application System
This module contains the streamlit code for the Process tab
that handles document processing, validation, and assessment.
"""

import streamlit as st
import pandas as pd
import traceback
import logging
from datetime import datetime

def render_process_tab(collector, validator, assessor, counselor):
    """Render the Process tab content for application processing"""
    st.header("⚙️ Application Processing Pipeline")
    
    # Initialize processing state
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    
    # Check if we have data to process
    has_application = 'current_result' in st.session_state and st.session_state.current_result
    has_document = hasattr(st.session_state, 'uploaded_file')
    
    if not has_application and not has_document:
        st.warning("No application or document has been submitted. Please complete the application form or upload documents first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Application Form"):
                st.session_state.current_tab = "Application Form"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Upload Tab"):
                st.session_state.current_tab = "Upload"
                st.experimental_rerun()
    else:
        # Application summary
        if has_application:
            with st.expander("Application Summary", expanded=True):
                app_data = st.session_state.current_result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Personal Information**")
                    st.write(f"Name: {app_data.get('name', 'N/A')}")
                    st.write(f"ID: {app_data.get('emirates_id', 'N/A')}")
                    st.write(f"Family Size: {app_data.get('family_size', 'N/A')}")
                with col2:
                    st.write("**Financial Information**")
                    st.write(f"Income: {app_data.get('income', 0):.2f} AED")
                    st.write(f"Assets: {app_data.get('assets_value', 0):.2f} AED")
                    st.write(f"Liabilities: {app_data.get('liabilities_value', 0):.2f} AED")
                with col3:
                    st.write("**Employment Information**")
                    st.write(f"Status: {app_data.get('employment_status', 'N/A')}")
                    st.write(f"Employer: {app_data.get('employer', 'N/A')}")
                    st.write(f"Duration: {app_data.get('employment_duration', 0)} months")
        
        # Document information if available
        if has_document:
            with st.expander("Document Information", expanded=True):
                uploaded_file = st.session_state.uploaded_file
                doc_type = getattr(st.session_state, 'document_type', 'unknown')
                st.write(f"**Document Type**: {doc_type.replace('_', ' ').title()}")
                st.write(f"**Filename**: {uploaded_file.name}")
                st.write(f"**Size**: {uploaded_file.size} bytes")
        
        # Start processing button
        if not st.session_state.processing_started:
            if st.button("Start Processing"):
                st.session_state.processing_started = True
                st.experimental_rerun()
        
        # Process the application and/or document
        if st.session_state.processing_started:
            process_application_data(collector, validator, assessor, counselor)


def process_application_data(collector, validator, assessor, counselor):
    """Process application data through the agent workflow"""
    # Progress indicator
    progress_bar = st.progress(0)
    status_area = st.empty()
    
    # Create columns for results
    result_col1, result_col2 = st.columns(2)
    
    # Check data sources
    has_application = 'current_result' in st.session_state and st.session_state.current_result
    has_document = hasattr(st.session_state, 'uploaded_file')
    
    with st.spinner("Processing application..."):
        try:
            # Initial processing - either from form data or document
            if has_application:
                result = st.session_state.current_result.copy()
                status_area.info("📋 Processing application data...")
            elif has_document:
                # Process document with DATA COLLECTOR AGENT
                status_area.info("🔍 Data Collector Agent: Extracting information from document...")
                progress_bar.progress(10)
                
                # Read file bytes
                uploaded_file = st.session_state.uploaded_file
                file_bytes = uploaded_file.read()
                
                # Get document type
                doc_type = getattr(st.session_state, 'document_type', 'unknown')
                
                # Process the document - first check the expected signature of the process_document method
                try:
                    # Try with file path, filename, document_type signature
                    with open('temp_file', 'wb') as f:
                        f.write(file_bytes)
                    result = collector.process_document('temp_file', uploaded_file.name, doc_type)
                except TypeError:
                    try:
                        # Try with file_bytes, filename, document_type signature
                        result = collector.process_document(file_bytes, uploaded_file.name, doc_type)
                    except TypeError:
                        # Try with file_bytes, filename only
                        result = collector.process_document(file_bytes, uploaded_file.name)
                
                progress_bar.progress(30)
                
                # Add filename to result
                result['filename'] = uploaded_file.name
            
            # Ensure we have a valid application record
            from database.db_setup import Application
            
            if 'app_id' not in result and 'filename' in result:
                app = assessor.db.query(Application).filter_by(filename=result['filename']).first()
                
                if not app:
                    # Create a new application record
                    application = Application(
                        filename=result['filename'],
                        name=result.get('name', ''),
                        email=result.get('email', ''),
                        phone=result.get('phone', ''),
                        emirates_id=result.get('emirates_id', ''),
                        income=result.get('income', 0),
                        family_size=result.get('family_size', 0),
                        address=result.get('address', ''),
                        employment_status=result.get('employment_status', ''),
                        employer=result.get('employer', ''),
                        job_title=result.get('job_title', ''),
                        employment_duration=result.get('employment_duration', 0),
                        monthly_expenses=result.get('monthly_expenses', 0),
                        assets_value=result.get('assets_value', 0),
                        liabilities_value=result.get('liabilities_value', 0),
                        validation_status="pending",
                        assessment_status="pending",
                        risk_level="unknown"
                    )
                    assessor.db.add(application)
                    assessor.db.commit()
                    result['app_id'] = application.id
                else:
                    result['app_id'] = app.id
                    
                    # Update application with new information from document if applicable
                    if has_document:
                        # Only update fields if they're present in the result
                        for field in ['income', 'family_size', 'address', 'employment_status',
                                     'employer', 'job_title', 'employment_duration',
                                     'assets_value', 'liabilities_value']:
                            if field in result and result[field]:
                                setattr(app, field, result[field])
                        assessor.db.commit()
            
            progress_bar.progress(40)
            
            # VALIDATOR AGENT
            status_area.info("✅ Validator Agent: Checking data quality...")
            progress_bar.progress(50)
            
            # Validate the result
            is_valid, validation_errors = validator.validate(result)
            
            # Update validation status in database
            app = assessor.db.query(Application).filter_by(filename=result['filename']).first()
            if app:
                app.validation_status = "valid" if is_valid else "invalid"
                assessor.db.commit()
            
            # Add validation status to result for display
            result['validation_status'] = "✅ Valid" if is_valid else "❌ Invalid"
            progress_bar.progress(70)
            
            # Show validation results
            with result_col1:
                st.subheader("Validation Results")
                if is_valid:
                    st.success("✅ Application data validated successfully")
                else:
                    st.error("❌ Validation failed")
                    for error in validation_errors:
                        st.warning(error)
            
            # ASSESSOR AGENT
            status_area.info("🧮 Assessor Agent: Evaluating application and risk...")
            progress_bar.progress(80)
            
            if is_valid:
                # Assess the application
                is_approved, reasons, assessment_details = assessor.assess_application(result)
                
                # Update assessment status in database
                if app:
                    app.assessment_status = "approved" if is_approved else "rejected"
                    app.risk_level = assessment_details['risk_level']
                    assessor.db.commit()
                
                # Add assessment results to result dict
                result['assessment_status'] = "✅ Approved" if is_approved else "❌ Rejected"
                result['risk_level'] = assessment_details['risk_level']
                result['assessment_details'] = assessment_details
                
                # Show assessment results
                with result_col2:
                    st.subheader("Assessment Results")
                    if is_approved:
                        st.success("✅ Application approved!")
                    else:
                        st.error("❌ Application rejected")
                        for reason in reasons:
                            st.warning(reason)
                    
                    # Display risk metrics
                    st.metric("Risk Level", assessment_details['risk_level'].upper())
                    st.metric("Income per Family Member", 
                            f"{assessment_details['income_per_member']:.2f} AED",
                            delta=f"{assessment_details['income_per_member'] - 10000:.2f}" 
                            if 'income_per_member' in assessment_details else None)
                    
                    # If there are other metrics in assessment_details, display them
                    for key, value in assessment_details.items():
                        if key not in ['risk_level', 'income_per_member', 'assessment_date']:
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            else:
                with result_col2:
                    st.subheader("Assessment Results")
                    st.error("Assessment skipped due to validation failures")
                    st.info("Please correct the validation errors and try again")
            
            # COUNSELOR AGENT - Generate recommendations
            if is_valid and 'counselor_available' in st.session_state and st.session_state.counselor_available:
                status_area.info("🧠 Counselor Agent: Generating recommendations...")
                progress_bar.progress(90)
                
                # Generate recommendations
                recommendations = counselor.generate_recommendations(result)
                result['recommendations'] = recommendations
                
                # Display recommendations
                st.subheader("Recommendations")
                if recommendations:
                    for rec in recommendations:
                        with st.expander(f"{rec['category'].title()} - {rec['priority'].upper()} Priority"):
                            st.write(rec['description'])
                            if 'action_items' in rec and rec['action_items']:
                                st.write("**Action Items:**")
                                for item in rec['action_items']:
                                    st.write(f"- {item}")
                else:
                    st.info("No specific recommendations available at this time.")
            
            # Update session state
            st.session_state.current_result = result
            st.session_state.processing_complete = True
            
            # Add to dataframe for history
            new_row = pd.DataFrame([result])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            
            # Complete
            progress_bar.progress(100)
            status_area.success("✅ Processing complete!")
            
            # Reset processing state button
            if st.button("Process Another Application"):
                st.session_state.processing_started = False
                if hasattr(st.session_state, 'uploaded_file'):
                    delattr(st.session_state, 'uploaded_file')
                if hasattr(st.session_state, 'document_type'):
                    delattr(st.session_state, 'document_type')
                st.session_state.current_tab = "Application Form"
                st.experimental_rerun()
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View History"):
                    st.session_state.current_tab = "History"
                    st.experimental_rerun()
            with col2:
                if st.button("Chat with AI Assistant") and st.session_state.counselor_available:
                    st.session_state.current_tab = "Guidance"
                    st.experimental_rerun()
                    
        except Exception as e:
            progress_bar.progress(100)
            status_area.error(f"Error during processing: {str(e)}")
            st.error(traceback.format_exc())
            logging.error(f"Processing error: {str(e)}")
            logging.error(traceback.format_exc())
