import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaManager:
    """
    Manager for ChromaDB vector database operations.
    Handles storage and retrieval of policy documents and other embeddings.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB manager with specified persistence directory.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Initialize embedding function (using OpenAI by default, can be changed)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize collections
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize required collections if they don't exist"""
        # Collection for policy documents
        self.policies_collection = self._get_or_create_collection("policies")
        
        # Collection for application templates
        self.templates_collection = self._get_or_create_collection("templates")
        
        # Collection for support recommendations
        self.recommendations_collection = self._get_or_create_collection("recommendations")
    
    def _get_or_create_collection(self, name: str):
        """Get a collection if it exists, or create a new one"""
        try:
            return self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            return self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, collection_name: str = "policies") -> List[str]:
        """
        Add documents to a specified collection with metadata.
        
        Args:
            texts: List of text documents to add
            metadatas: List of metadata dictionaries for each document
            collection_name: Name of the collection to add documents to
            
        Returns:
            List of document IDs
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
        
        # Generate IDs if not present in metadata
        ids = [meta.get("id", str(uuid.uuid4())) for meta in metadatas]
        
        # Get the appropriate collection
        collection = self._get_or_create_collection(collection_name)
        
        # Add documents
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search_documents(self, collection_name: str, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            n_results: Number of results to return
            filter_dict: Optional filter to apply to the search
            
        Returns:
            List of result dictionaries with text and metadata
        """
        # Get the appropriate collection
        collection = self._get_or_create_collection(collection_name)
        
        # Perform search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def delete_document(self, collection_name: str, document_id: str) -> None:
        """
        Delete a document from a collection by ID.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to delete
        """
        collection = self._get_or_create_collection(collection_name)
        collection.delete(ids=[document_id])
    
    def update_document(self, collection_name: str, document_id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        """
        Update a document in a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to update
            text: New text content
            metadata: New metadata (optional)
        """
        collection = self._get_or_create_collection(collection_name)
        
        # Delete the old document
        collection.delete(ids=[document_id])
        
        # Add the updated document
        collection.add(
            documents=[text],
            metadatas=[metadata] if metadata else None,
            ids=[document_id]
        )
    
    def get_document(self, collection_name: str, document_id: str) -> Dict:
        """
        Get a document by ID.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to retrieve
            
        Returns:
            Dictionary with document text and metadata
        """
        collection = self._get_or_create_collection(collection_name)
        
        result = collection.get(ids=[document_id])
        
        if result["documents"]:
            return {
                "text": result["documents"][0],
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "id": document_id
            }
        
        return None
