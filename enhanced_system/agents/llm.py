"""
LLM integration for Enhanced Social Security Application System.
Provides get_llm_response (Ollama) and get_mock_response (fallback).
"""
import requests
import logging

def get_llm_response(message, app_data=None):
    """Call Ollama LLM (Mistral) for a response."""
    try:
        prompt = f"""
        You are a helpful AI assistant for the Social Security Support System.\n\nApplication data: {app_data}\nUser message: {message}\nProvide a helpful, concise response.
        """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("response", "[No response from Ollama]")
        else:
            logging.error(f"Ollama error: {response.status_code} {response.text}")
            raise RuntimeError("Ollama LLM error")
    except Exception as e:
        logging.error(f"LLM call failed: {str(e)}")
        raise

def get_mock_response(message, app_data=None):
    """Fallback response if LLM unavailable."""
    msg = message.lower()
    if "income" in msg or "financial" in msg or "money" in msg:
        return "Based on your income and financial situation, you may be eligible for additional support. Please submit your latest income statements and employment records."
    elif "document" in msg or "upload" in msg:
        return "To complete your application, please upload: proof of identity, proof of income, and proof of residence."
    elif "status" in msg or "progress" in msg:
        return "Your application is being processed. Typical processing time is 5-7 business days."
    elif "eligible" in msg or "qualify" in msg:
        return "Eligibility depends on income, family size, employment, and residency. Our system will determine your eligibility and provide recommendations."
    elif "help" in msg or "assistance" in msg:
        return "I'm here to help with your application. I can provide information on eligibility, required documents, status, and recommendations."
    else:
        return "Thank you for your query. I'm your AI assistant for the Social Security Support System. Please let me know how I can assist you further."
