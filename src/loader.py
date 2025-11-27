"""
Document Loader - Loading component for RAG system
Handles loading various document formats with metadata preservation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, Document


class DocumentLoader:
    """
    Document loader for various file formats with metadata handling.
    Part of Q1 indexation pipeline - Loading step.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize document loader.
        
        Args:
            data_dir: Directory containing documents to load
        """
        self.data_dir = Path(data_dir)
        self.supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.html'}
    
    def load_documents(self, filename_as_id: bool = True) -> List[Document]:
        """
        Load all documents from the data directory.
        
        Args:
            filename_as_id: Use filename as document ID
            
        Returns:
            List of loaded Document objects
        """
        print(f"📂 Loading documents from {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Use LlamaIndex SimpleDirectoryReader
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            filename_as_id=filename_as_id,
            recursive=True
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"   ✅ Loaded {len(documents)} document(s)")
        
        # Enhance metadata
        for doc in documents:
            self._enhance_metadata(doc)
        
        return documents
    
    def load_single_document(self, file_path: str) -> Document:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Load single file
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)]
        ).load_data()
        
        if not documents:
            raise ValueError(f"Failed to load document: {file_path}")
        
        doc = documents[0]
        self._enhance_metadata(doc)
        
        return doc
    
    def _enhance_metadata(self, document: Document):
        """
        Enhance document metadata with additional information.
        
        Args:
            document: Document to enhance
        """
        # Add file size
        if 'file_path' in document.metadata:
            file_path = Path(document.metadata['file_path'])
            if file_path.exists():
                document.metadata['file_size'] = file_path.stat().st_size
                document.metadata['file_extension'] = file_path.suffix.lower()
        
        # Add content statistics
        document.metadata['content_length'] = len(document.text)
        document.metadata['word_count'] = len(document.text.split())
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        Get information about documents in the data directory.
        
        Returns:
            Dictionary with document statistics
        """
        if not self.data_dir.exists():
            return {"error": f"Directory not found: {self.data_dir}"}
        
        files = []
        total_size = 0
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                size = file_path.stat().st_size
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.data_dir)),
                    "size": size,
                    "extension": file_path.suffix.lower()
                })
                total_size += size
        
        return {
            "directory": str(self.data_dir),
            "total_files": len(files),
            "total_size": total_size,
            "supported_extensions": list(self.supported_extensions),
            "files": files
        }
