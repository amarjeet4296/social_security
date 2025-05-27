"""
Enhanced Data Collector Agent - Extracts data from multiple document types.
Supports multimodal data processing (text, images, tabular data).
"""

import os
import re
import io
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image as PILImage
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import json
from typing import Dict, Any, List, Tuple, Optional
import uuid
import logging
from datetime import datetime

from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Import utility for local LLM
from utils.llm_factory import get_llm
from database.db_setup import Document, get_db

# Load environment variables
load_dotenv()

class DataCollectorAgent:
    """
    Enhanced data collector agent that extracts information from various document types.
    Supports multimodal data processing including text, images, and tabular data.
    """
    
    def __init__(self):
        """Initialize the data collector agent with necessary components."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DataCollectorAgent")
        
        # Set up Tesseract OCR
        tesseract_path = '/opt/homebrew/bin/tesseract'
        if not os.path.exists(tesseract_path):
            tesseract_path = '/usr/local/bin/tesseract'  # Fallback path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # OCR configurations
        self.arabic_config = r'--oem 3 --psm 6 -l ara+eng'
        self.english_config = r'--oem 3 --psm 6 -l eng'
        
        # Initialize LLM for advanced text parsing
        self.llm = get_llm()
        
        # Initialize database session
        self.db = next(get_db())
    
    def process_document(self, file_path: str, filename: str, document_type: str) -> Dict[str, Any]:
        """
        Process a document and extract relevant information based on document type.
        
        Args:
            file_path: Path to the document file
            filename: Name of the document file
            document_type: Type of document (e.g., "bank_statement", "emirates_id", "resume")
            
        Returns:
            Dictionary of extracted data
        """
        try:
            self.logger.info(f"Processing document: {filename} (Type: {document_type})")
            
            # Determine file type and use appropriate processing method
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                extracted_data = self._process_image(file_path, document_type)
            elif filename.lower().endswith('.pdf'):
                extracted_data = self._process_pdf(file_path, document_type)
            elif filename.lower().endswith(('.xlsx', '.xls', '.csv')):
                extracted_data = self._process_tabular(file_path, document_type)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            # Store document in database
            self._store_document_metadata(file_path, filename, document_type, extracted_data)
            
            self.logger.info(f"Successfully processed document: {filename}")
            return extracted_data
        except Exception as e:
            self.logger.error(f"Error processing document {filename}: {str(e)}")
            return {"error": str(e)}
    
    def _store_document_metadata(self, file_path: str, filename: str, document_type: str, extracted_data: Dict[str, Any]) -> None:
        """Store document metadata in the database."""
        try:
            # Determine MIME type
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                mime_type = f"image/{filename.split('.')[-1].lower()}"
            elif filename.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            elif filename.lower().endswith('.xlsx'):
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif filename.lower().endswith('.xls'):
                mime_type = "application/vnd.ms-excel"
            elif filename.lower().endswith('.csv'):
                mime_type = "text/csv"
            else:
                mime_type = "application/octet-stream"
            
            # Create document record
            document = Document(
                application_id=None,  # Will be updated when linked to an application
                document_type=document_type,
                filename=filename,
                file_path=file_path,
                mime_type=mime_type,
                extracted_data=extracted_data,
                uploaded_at=datetime.utcnow()
            )
            
            self.db.add(document)
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Error storing document metadata: {str(e)}")
            self.db.rollback()
    
    def _process_image(self, image_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process an image document and extract data.
        
        Args:
            image_path: Path to the image file
            document_type: Type of document
            
        Returns:
            Dictionary of extracted data
        """
        # Open image
        image = PILImage.open(image_path)
        
        # Determine language configuration
        config = self._get_language_config(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config=config)
        
        # Extract data based on document type
        if document_type == "emirates_id":
            return self._extract_emirates_id_data(text, image)
        elif document_type == "bank_statement":
            return self._extract_bank_statement_data(text)
        elif document_type == "resume":
            return self._extract_resume_data(text)
        else:
            # Generic extraction for other document types
            return self._extract_generic_data(text)
    
    def _process_pdf(self, pdf_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process a PDF document and extract data.
        
        Args:
            pdf_path: Path to the PDF file
            document_type: Type of document
            
        Returns:
            Dictionary of extracted data
        """
        # Try text-based extraction first
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # If sufficient text is extracted, process it
            if len(text) > 100:
                if document_type == "bank_statement":
                    return self._extract_bank_statement_data(text)
                elif document_type == "resume":
                    return self._extract_resume_data(text)
                else:
                    return self._extract_generic_data(text)
        except Exception as e:
            self.logger.warning(f"Text-based PDF extraction failed: {str(e)}")
        
        # Fallback to OCR for image-based PDFs
        try:
            images = convert_from_path(pdf_path)
            combined_text = ""
            
            for image in images:
                config = self._get_language_config(image)
                combined_text += pytesseract.image_to_string(image, config=config)
            
            # Extract data based on document type
            if document_type == "bank_statement":
                return self._extract_bank_statement_data(combined_text)
            elif document_type == "resume":
                return self._extract_resume_data(combined_text)
            else:
                return self._extract_generic_data(combined_text)
        except Exception as e:
            self.logger.error(f"OCR-based PDF extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _process_tabular(self, file_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process a tabular document (Excel, CSV) and extract data.
        
        Args:
            file_path: Path to the tabular file
            document_type: Type of document
            
        Returns:
            Dictionary of extracted data
        """
        try:
            # Read tabular data
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            # Extract data based on document type
            if document_type == "assets_liabilities":
                return self._extract_assets_liabilities_data(df)
            elif document_type == "income_statement":
                return self._extract_income_statement_data(df)
            else:
                # Generic tabular data extraction
                return self._extract_generic_tabular_data(df)
        except Exception as e:
            self.logger.error(f"Error processing tabular data: {str(e)}")
            return {"error": str(e)}
    
    def _get_language_config(self, image: PILImage.Image) -> str:
        """
        Determine the language configuration for OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tesseract configuration string
        """
        try:
            # Sample a small portion of the image for language detection
            sample = pytesseract.image_to_string(image, config=self.english_config)
            
            # Check for Arabic characters
            if re.search(r'[\u0600-\u06FF]', sample):
                return self.arabic_config
        except Exception as e:
            self.logger.warning(f"Language detection error: {str(e)}")
        
        # Default to English
        return self.english_config
    
    def _extract_emirates_id_data(self, text: str, image: Optional[PILImage.Image] = None) -> Dict[str, Any]:
        """
        Extract data from Emirates ID.
        
        Args:
            text: Extracted text from OCR
            image: Optional image for additional processing
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Extract name
        name_match = re.search(r'Name[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            data['name'] = name_match.group(1).strip()
        
        # Extract Emirates ID number
        id_match = re.search(r'(?:ID|Number)[:\s-]*(\d{3}-\d{4}-\d{7}-\d|784-\d{4}-\d{7}-\d|\d{15})', text, re.IGNORECASE)
        if id_match:
            data['emirates_id'] = id_match.group(1).strip()
        
        # Extract expiry date
        expiry_match = re.search(r'Expiry[:\s]*(\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        if expiry_match:
            data['id_expiry'] = expiry_match.group(1).strip()
        
        # Use LLM to extract additional information
        if len(text) > 50:
            prompt = f"""
            Extract the following information from this Emirates ID text. Return ONLY a JSON object with these fields (leave empty if not found):
            1. name (full name)
            2. nationality
            3. gender
            4. date_of_birth (in format DD/MM/YYYY)
            5. emirates_id (ID number)
            
            Text from ID:
            {text}
            """
            
            try:
                llm_response = self.llm.invoke(prompt)
                llm_data = json.loads(llm_response)
                
                # Update data with LLM extraction results
                for key, value in llm_data.items():
                    if value and (key not in data or not data[key]):
                        data[key] = value
            except Exception as e:
                self.logger.warning(f"LLM extraction error: {str(e)}")
        
        return data
    
    def _extract_bank_statement_data(self, text: str) -> Dict[str, Any]:
        """
        Extract data from a bank statement.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Extract account holder name
        name_match = re.search(r'(?:Account Holder|Name)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            data['name'] = name_match.group(1).strip()
        
        # Extract account number
        account_match = re.search(r'(?:Account Number|A/C No)[:\s]*([0-9-]+)', text, re.IGNORECASE)
        if account_match:
            data['account_number'] = account_match.group(1).strip()
        
        # Extract income (total credits)
        credits = re.findall(r'(?:Credit|Deposit|Salary)[:\s]*(?:AED|د.إ)\s*([\d,]+\.\d{2}|[\d,]+)', text, re.IGNORECASE)
        if credits:
            # Convert to numbers and sum
            total_credits = sum(float(c.replace(',', '')) for c in credits)
            data['income'] = total_credits
        
        # Extract expenses (total debits)
        debits = re.findall(r'(?:Debit|Withdrawal)[:\s]*(?:AED|د.إ)\s*([\d,]+\.\d{2}|[\d,]+)', text, re.IGNORECASE)
        if debits:
            # Convert to numbers and sum
            total_debits = sum(float(d.replace(',', '')) for d in debits)
            data['monthly_expenses'] = total_debits
        
        # Use LLM to extract additional information
        prompt = f"""
        Extract the following financial information from this bank statement. Return ONLY a JSON object with these fields (use numbers without currency symbols, leave empty if not found):
        1. account_holder_name
        2. average_monthly_income (approximate monthly income)
        3. total_expenses (total expenses/debits)
        4. opening_balance
        5. closing_balance
        
        Text from bank statement:
        {text[:2000]}  # Limit text size for LLM
        """
        
        try:
            llm_response = self.llm.invoke(prompt)
            llm_data = json.loads(llm_response)
            
            # Update data with LLM extraction results
            for key, value in llm_data.items():
                if key == 'account_holder_name' and value:
                    data['name'] = value
                elif key == 'average_monthly_income' and value and not data.get('income'):
                    try:
                        data['income'] = float(str(value).replace(',', ''))
                    except ValueError:
                        pass
                elif key == 'total_expenses' and value and not data.get('monthly_expenses'):
                    try:
                        data['monthly_expenses'] = float(str(value).replace(',', ''))
                    except ValueError:
                        pass
        except Exception as e:
            self.logger.warning(f"LLM extraction error: {str(e)}")
        
        return data
    
    def _extract_resume_data(self, text: str) -> Dict[str, Any]:
        """
        Extract data from a resume/CV.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Extract name (usually at the top of resume)
        lines = text.split('\n')
        if lines:
            data['name'] = lines[0].strip()
        
        # Extract email
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            data['email'] = email_match.group(0)
        
        # Extract phone number
        phone_match = re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match:
            data['phone'] = phone_match.group(0)
        
        # Use LLM to extract employment information
        prompt = f"""
        Extract the following employment information from this resume. Return ONLY a JSON object with these fields (leave empty if not found):
        1. current_job_title
        2. current_employer
        3. employment_duration_months (total work experience in months)
        4. skills (list of key skills)
        5. education (highest education level)
        
        Text from resume:
        {text[:3000]}  # Limit text size for LLM
        """
        
        try:
            llm_response = self.llm.invoke(prompt)
            llm_data = json.loads(llm_response)
            
            # Update data with LLM extraction results
            if 'current_job_title' in llm_data and llm_data['current_job_title']:
                data['job_title'] = llm_data['current_job_title']
            
            if 'current_employer' in llm_data and llm_data['current_employer']:
                data['employer'] = llm_data['current_employer']
            
            if 'employment_duration_months' in llm_data and llm_data['employment_duration_months']:
                try:
                    data['employment_duration'] = int(llm_data['employment_duration_months'])
                except ValueError:
                    pass
            
            # Additional fields
            if 'skills' in llm_data and llm_data['skills']:
                data['skills'] = llm_data['skills']
            
            if 'education' in llm_data and llm_data['education']:
                data['education'] = llm_data['education']
        except Exception as e:
            self.logger.warning(f"LLM extraction error: {str(e)}")
        
        return data
    
    def _extract_assets_liabilities_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract data from assets and liabilities spreadsheet.
        
        Args:
            df: Pandas DataFrame containing the tabular data
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Look for assets in the dataframe
        assets_cols = [col for col in df.columns if 'asset' in col.lower()]
        if assets_cols:
            # Sum numeric values in assets columns
            assets_df = df[assets_cols].select_dtypes(include=[np.number])
            if not assets_df.empty:
                data['assets_value'] = assets_df.sum().sum()
        else:
            # Try to identify asset rows
            asset_rows = df[df.apply(lambda row: any('asset' in str(val).lower() for val in row), axis=1)]
            if not asset_rows.empty:
                # Find numeric columns
                numeric_cols = asset_rows.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data['assets_value'] = asset_rows[numeric_cols].sum().sum()
        
        # Look for liabilities in the dataframe
        liab_cols = [col for col in df.columns if any(term in col.lower() for term in ['liab', 'debt', 'loan'])]
        if liab_cols:
            # Sum numeric values in liability columns
            liab_df = df[liab_cols].select_dtypes(include=[np.number])
            if not liab_df.empty:
                data['liabilities_value'] = liab_df.sum().sum()
        else:
            # Try to identify liability rows
            liab_rows = df[df.apply(lambda row: any(term in str(val).lower() for val in row for term in ['liab', 'debt', 'loan']), axis=1)]
            if not liab_rows.empty:
                # Find numeric columns
                numeric_cols = liab_rows.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data['liabilities_value'] = liab_rows[numeric_cols].sum().sum()
        
        return data
    
    def _extract_income_statement_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract data from income statement.
        
        Args:
            df: Pandas DataFrame containing the tabular data
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Look for income/revenue/salary in the dataframe
        income_rows = df[df.apply(lambda row: any(term in str(val).lower() for val in row for term in ['income', 'revenue', 'salary']), axis=1)]
        if not income_rows.empty:
            # Find numeric columns
            numeric_cols = income_rows.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['income'] = income_rows[numeric_cols].sum().sum()
        
        # Look for expenses in the dataframe
        expense_rows = df[df.apply(lambda row: any(term in str(val).lower() for val in row for term in ['expense', 'cost', 'expenditure']), axis=1)]
        if not expense_rows.empty:
            # Find numeric columns
            numeric_cols = expense_rows.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['monthly_expenses'] = expense_rows[numeric_cols].sum().sum()
        
        return data
    
    def _extract_generic_data(self, text: str) -> Dict[str, Any]:
        """
        Extract generic data from document text.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Extract name
        name_match = re.search(r'(?:Name|Full Name)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            data['name'] = name_match.group(1).strip()
        
        # Extract email
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            data['email'] = email_match.group(0)
        
        # Extract phone
        phone_match = re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match:
            data['phone'] = phone_match.group(0)
        
        # Extract address
        address_match = re.search(r'(?:Address|Location|Residence)[:\s]*([^\n]+(?:\n[^\n]+){0,2})', text, re.IGNORECASE)
        if address_match:
            data['address'] = address_match.group(1).strip()
        
        # Extract income
        income_match = re.search(r'(?:Income|Salary|Revenue)[:\s]*(?:AED|د.إ)?\s*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
        if income_match:
            try:
                data['income'] = float(income_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Extract family size
        family_match = re.search(r'(?:Family Size|Family Members|Dependents)[:\s]*(\d+)', text, re.IGNORECASE)
        if family_match:
            try:
                data['family_size'] = int(family_match.group(1))
            except ValueError:
                pass
        
        return data
    
    def _extract_generic_tabular_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract generic data from tabular data.
        
        Args:
            df: Pandas DataFrame containing the tabular data
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        # Check column names for relevant data
        for col in df.columns:
            col_lower = col.lower()
            if 'income' in col_lower or 'salary' in col_lower:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not numeric_vals.empty:
                    data['income'] = float(numeric_vals.iloc[0])
            
            elif 'expense' in col_lower:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not numeric_vals.empty:
                    data['monthly_expenses'] = float(numeric_vals.iloc[0])
            
            elif 'family' in col_lower and 'size' in col_lower:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not numeric_vals.empty:
                    data['family_size'] = int(numeric_vals.iloc[0])
            
            elif 'asset' in col_lower:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not numeric_vals.empty:
                    data['assets_value'] = float(numeric_vals.iloc[0])
            
            elif 'liab' in col_lower or 'debt' in col_lower:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not numeric_vals.empty:
                    data['liabilities_value'] = float(numeric_vals.iloc[0])
        
        return data
    
    def extract_missing_fields(self, text: str, missing_fields: List[str]) -> Dict[str, Any]:
        """
        Extract specific missing fields from text.
        
        Args:
            text: Text to extract fields from
            missing_fields: List of field names to extract
            
        Returns:
            Dictionary of extracted fields
        """
        # Create a customized prompt for the LLM
        prompt = f"""
        Extract the following information from this text. Return ONLY a JSON object with these fields (leave empty if not found):
        {', '.join(missing_fields)}
        
        Text:
        {text[:4000]}  # Limit text size for LLM
        """
        
        try:
            llm_response = self.llm.invoke(prompt)
            return json.loads(llm_response)
        except Exception as e:
            self.logger.warning(f"LLM extraction error: {str(e)}")
            return {}
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        try:
            self.db.close()
        except:
            pass
