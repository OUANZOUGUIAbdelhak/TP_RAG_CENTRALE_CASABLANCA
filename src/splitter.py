"""
Document Splitter - Splitting component for RAG system
Handles text chunking with metadata preservation, optimized for Markdown.
"""

from typing import List, Dict, Any, Optional
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import Document, TextNode


class DocumentSplitter:
    """
    Document splitter optimized for Markdown format with metadata preservation.
    Part of Q1 indexation pipeline - Splitting step.
    """
    
    def __init__(self, 
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 separator: str = " "):
        """
        Initialize document splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Create sentence splitter
        self.sentence_splitter = SentenceSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create markdown splitter for markdown files
        self.markdown_splitter = MarkdownNodeParser()
    
    def split_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Split documents into chunks with metadata preservation.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of text nodes (chunks)
        """
        print(f"✂️  Splitting {len(documents)} documents into chunks...")
        print(f"   Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        all_nodes = []
        
        for doc in documents:
            # Choose splitter based on file type
            if self._is_markdown_document(doc):
                nodes = self._split_markdown_document(doc)
            else:
                nodes = self._split_regular_document(doc)
            
            all_nodes.extend(nodes)
        
        print(f"   ✅ Created {len(all_nodes)} chunks")
        return all_nodes
    
    def _is_markdown_document(self, document: Document) -> bool:
        """
        Check if document is a Markdown file.
        
        Args:
            document: Document to check
            
        Returns:
            True if document is Markdown
        """
        file_name = document.metadata.get('file_name', '')
        file_path = document.metadata.get('file_path', '')
        
        return (file_name.endswith('.md') or 
                file_path.endswith('.md') or
                '# ' in document.text[:100])  # Check for markdown headers
    
    def _split_markdown_document(self, document: Document) -> List[TextNode]:
        """
        Split Markdown document preserving structure.
        
        Args:
            document: Markdown document to split
            
        Returns:
            List of text nodes
        """
        try:
            # Use markdown splitter first to preserve structure
            markdown_nodes = self.markdown_splitter.get_nodes_from_documents([document])
            
            # Further split large markdown sections if needed
            final_nodes = []
            for node in markdown_nodes:
                if len(node.text) > self.chunk_size:
                    # Split large sections with sentence splitter
                    sub_nodes = self.sentence_splitter.get_nodes_from_documents([
                        Document(text=node.text, metadata=node.metadata)
                    ])
                    final_nodes.extend(sub_nodes)
                else:
                    final_nodes.append(node)
            
            return final_nodes
            
        except Exception as e:
            print(f"⚠️  Markdown splitting failed, using regular splitter: {e}")
            return self._split_regular_document(document)
    
    def _split_regular_document(self, document: Document) -> List[TextNode]:
        """
        Split regular document using sentence splitter.
        
        Args:
            document: Document to split
            
        Returns:
            List of text nodes
        """
        return self.sentence_splitter.get_nodes_from_documents([document])
    
    def get_chunk_statistics(self, nodes: List[TextNode]) -> Dict[str, Any]:
        """
        Get statistics about the created chunks.
        
        Args:
            nodes: List of text nodes
            
        Returns:
            Dictionary with chunk statistics
        """
        if not nodes:
            return {"total_chunks": 0}
        
        chunk_lengths = [len(node.text) for node in nodes]
        word_counts = [len(node.text.split()) for node in nodes]
        
        # Group by source document
        sources = {}
        for node in nodes:
            source = node.metadata.get('file_name', 'Unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            "total_chunks": len(nodes),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "chunks_per_source": sources,
            "total_sources": len(sources)
        }
    
    def preview_chunks(self, nodes: List[TextNode], max_chunks: int = 3):
        """
        Preview first few chunks for debugging.
        
        Args:
            nodes: List of text nodes
            max_chunks: Maximum number of chunks to preview
        """
        print(f"\n📋 PREVIEW OF CHUNKS (showing first {max_chunks}):")
        print("-" * 60)
        
        for i, node in enumerate(nodes[:max_chunks]):
            source = node.metadata.get('file_name', 'Unknown')
            print(f"\nChunk {i+1} from {source}:")
            print(f"Length: {len(node.text)} chars, {len(node.text.split())} words")
            
            # Show first 200 characters
            preview_text = node.text[:200]
            if len(node.text) > 200:
                preview_text += "..."
            
            print(f"Content: {preview_text}")
            print("-" * 40)


class AdvancedSplitter(DocumentSplitter):
    """
    Advanced document splitter with additional features.
    """
    
    def __init__(self, 
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 preserve_headers: bool = True,
                 min_chunk_size: int = 100):
        """
        Initialize advanced splitter.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            preserve_headers: Whether to preserve markdown headers
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.preserve_headers = preserve_headers
        self.min_chunk_size = min_chunk_size
    
    def split_with_context(self, documents: List[Document]) -> List[TextNode]:
        """
        Split documents while preserving contextual information.
        
        Args:
            documents: Documents to split
            
        Returns:
            List of enhanced text nodes with context
        """
        nodes = self.split_documents(documents)
        
        # Add contextual information
        for i, node in enumerate(nodes):
            # Add position information
            node.metadata['chunk_index'] = i
            node.metadata['total_chunks'] = len(nodes)
            
            # Add neighboring context references
            if i > 0:
                node.metadata['previous_chunk_preview'] = nodes[i-1].text[:100]
            if i < len(nodes) - 1:
                node.metadata['next_chunk_preview'] = nodes[i+1].text[:100]
        
        return nodes
