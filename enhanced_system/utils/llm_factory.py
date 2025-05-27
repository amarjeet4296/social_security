"""
LLM Factory utility for creating and configuring language models.
Supports various model providers with consistent interface.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()

class MockLLM(LLM):
    """A mock LLM for fallback when Ollama is not available."""
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Generate a mock response based on the prompt."""
        import logging
        logging.info(f"MockLLM received prompt: {prompt[:100]}...")
        
        # Generate a contextual response based on the prompt
        if "income" in prompt.lower() or "financial" in prompt.lower():
            return "Based on your income and financial situation, you may be eligible for additional support. I recommend submitting your latest income statements and employment records to strengthen your application."
        
        if "document" in prompt.lower() or "upload" in prompt.lower():
            return "To complete your application, please upload the following documents: proof of identity (Emirates ID or passport), proof of income (salary slips or bank statements), and proof of residence (utility bills or rental agreement)."
        
        if "status" in prompt.lower() or "progress" in prompt.lower():
            return "Your application is currently being processed. The typical processing time is 5-7 business days. You'll receive notifications as your application progresses through validation and assessment."
        
        if "help" in prompt.lower() or "assistance" in prompt.lower():
            return "I'm here to help with your social security application. I can provide information on eligibility criteria, required documents, application status, and recommendations based on your specific situation."
        
        # Default response if no specific context is detected
        return "Thank you for your query. I'm your AI assistant for the Social Security Support System. I can help with application submissions, document requirements, eligibility criteria, and checking application status. Please let me know how I can assist you further."
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mock"

def get_llm(
    model_name: str = "mistral", 
    temperature: float = 0.1,
    provider: str = "ollama",
    use_mock: bool = False
) -> LLM:
    """
    Factory function to create an LLM instance based on specified parameters.
    
    Args:
        model_name: Name of the model to use
        temperature: Sampling temperature (0.0 to 1.0)
        provider: Model provider (ollama, openai, etc.)
        use_mock: Whether to use the mock LLM (for testing or when Ollama is unavailable)
        
    Returns:
        Configured LLM instance
    """
    # Use mock LLM if requested or if MOCK_LLM env var is set
    if use_mock or os.getenv("MOCK_LLM", "").lower() == "true":
        return MockLLM()
    
    if provider.lower() == "ollama":
        try:
            return _create_ollama_llm(model_name, temperature)
        except Exception as e:
            import logging
            logging.error(f"Error creating Ollama LLM: {str(e)}. Falling back to MockLLM.")
            return MockLLM()
    else:
        # Default to Ollama if provider not supported
        try:
            return _create_ollama_llm(model_name, temperature)
        except Exception as e:
            import logging
            logging.error(f"Error creating default LLM: {str(e)}. Falling back to MockLLM.")
            return MockLLM()

def _create_ollama_llm(model_name: str, temperature: float) -> LLM:
    """
    Create an Ollama-based LLM.
    
    Args:
        model_name: Name of the Ollama model
        temperature: Sampling temperature
        
    Returns:
        Configured Ollama LLM
    """
    # Set Ollama host from environment or use default
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Configure callbacks for streaming output
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=ollama_host,
        callback_manager=callback_manager,
        # Additional parameters
        num_ctx=4096,  # Context window size
        repeat_penalty=1.1,  # Penalty for repetition
        verbose=True,  # Enable verbose mode for debugging
    )

def list_available_models() -> Dict[str, Any]:
    """
    List available models from the configured provider.
    
    Returns:
        Dictionary of available models and their details
    """
    try:
        import requests
        
        # Get Ollama host from environment or use default
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # List models from Ollama
        response = requests.get(f"{ollama_host}/api/tags")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to list models: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error listing models: {str(e)}"}

def download_model(model_name: str) -> Dict[str, Any]:
    """
    Download a model to the local Ollama server.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        Status dictionary
    """
    try:
        import requests
        
        # Get Ollama host from environment or use default
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Request model pull
        response = requests.post(
            f"{ollama_host}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            return {"status": "success", "message": f"Model {model_name} download initiated"}
        else:
            return {"error": f"Failed to download model: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error downloading model: {str(e)}"}
