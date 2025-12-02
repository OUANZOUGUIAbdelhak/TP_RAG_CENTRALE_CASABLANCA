"""
Q1: Document Indexation System
===============================
This module handles document indexing for the RAG system.

Pipeline:
1. Loading: Load documents using a data loader
2. Splitting: Divide documents into chunks (optimized for Markdown)
3. Advanced Metadata Extraction: Extract titles and questions answered (using LLM)
4. Embedding: Calculate embeddings using HuggingFace models
5. Storage: Store embeddings + metadata in ChromaDB vector store
"""

import os
import re
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llm_fallback import create_llm_with_fallback


class DocumentIndexer:
    """
    Q1: Document indexing system using ChromaDB and HuggingFace embeddings.
    
    This class implements the complete indexation pipeline:
    - Document loading from a directory
    - Text splitting with metadata preservation
    - Embedding generation using HuggingFace models
    - Storage in ChromaDB vector store
    """
    
    def __init__(self,
                 data_dir: str = "./data",
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 collection_name: str = "rag_collection",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-2.0-flash",
                 use_advanced_rag: bool = True):
        """
        Initialize the document indexer.
        
        Args:
            data_dir: Directory containing documents to index
            vectorstore_dir: Directory for ChromaDB persistence
            embedding_model_name: HuggingFace embedding model name
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            collection_name: Name of the ChromaDB collection
            groq_api_key: Groq API key for LLM-based metadata extraction (optional)
            groq_model: Groq model name for metadata extraction
            gemini_api_key: Gemini API key for fallback (optional)
            gemini_model: Gemini model name for fallback
            use_advanced_rag: Whether to use advanced RAG pipeline with title/QA extraction
        """
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.use_advanced_rag = use_advanced_rag
        
        # Initialize embedding model (HuggingFace)
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        print(f"✅ Embedding model loaded successfully")
        
        # Initialize LLM for advanced metadata extraction (if enabled) with fallback
        self.llm = None
        if use_advanced_rag:
            self.llm = create_llm_with_fallback(
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
                load_from_config=True
            )
            if not self.llm:
                print(f"⚠️  Failed to initialize any LLM (Groq or Gemini)")
                print(f"⚠️  Falling back to basic RAG pipeline (no title/QA extraction)")
                self.use_advanced_rag = False
        else:
            print(f"📝 Basic RAG pipeline (no metadata extraction)")
        
        # Initialize ChromaDB client
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        
        # Store index reference
        self.index = None
    
    def build_index(self) -> VectorStoreIndex:
        """
        Build the vector index from documents in data_dir.
        
        This implements the complete Q1 pipeline with Advanced RAG:
        1. Loading: Load documents with SimpleDirectoryReader
        2. Splitting: Split into chunks with SentenceSplitter
        3. Advanced Metadata Extraction: Extract titles and questions answered (if enabled)
        4. Embedding: Generate embeddings with HuggingFace model
        5. Storage: Store in ChromaDB vector store with enriched metadata
        
        Returns:
            VectorStoreIndex: The built index
        """
        print("\n" + "="*80)
        print("🚀 Q1: BUILDING DOCUMENT INDEX")
        if self.use_advanced_rag:
            print("   🔥 Using Advanced RAG Pipeline (Title + QA Extraction)")
        else:
            print("   📝 Using Basic RAG Pipeline")
        print("="*80 + "\n")
        
        # Step 1: Loading - Load documents from data directory
        print("📂 Step 1: Loading documents...")
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            filename_as_id=True,  # Use filename as document ID
            recursive=True  # Search subdirectories
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"   ✅ Loaded {len(documents)} document(s)")
        
        # Verify document content - check if PDFs actually have text
        print("\n   🔍 Verifying document content...")
        for i, doc in enumerate(documents[:3], 1):  # Check first 3 documents
            text_length = len(doc.text) if doc.text else 0
            file_name = doc.metadata.get('file_name', 'Unknown')
            print(f"      Doc {i}: {file_name} - {text_length} characters")
            if text_length < 100:
                print(f"      ⚠️  WARNING: Document has very little text ({text_length} chars)")
                print(f"      ⚠️  This might indicate PDF text extraction failed")
                # Show first 200 chars of what was extracted
                if doc.text:
                    print(f"      Preview: {doc.text[:200]}...")
                else:
                    print(f"      No text content found!")
        
        # Filter out documents with no meaningful content
        valid_documents = []
        for doc in documents:
            if doc.text and len(doc.text.strip()) > 50:  # At least 50 characters
                valid_documents.append(doc)
            else:
                print(f"   ⚠️  Skipping document with insufficient content: {doc.metadata.get('file_name', 'Unknown')}")
        
        if not valid_documents:
            raise ValueError("No documents with valid text content found. PDF text extraction may have failed.")
        
        documents = valid_documents
        print(f"   ✅ {len(documents)} document(s) with valid content ready for indexing")
        
        # Step 2/3: Splitting + Advanced Metadata Extraction
        print(f"\n✂️  Step 2/3: Splitting documents and extracting metadata (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        # Build the ingestion pipeline with strict prompts for concise outputs
        if self.use_advanced_rag and self.llm:
            print("   🔥 Setting up Advanced RAG pipeline with metadata extractors...")
            
            # Create title extractor prompt - generate clear, descriptive, and complete titles
            title_prompt_template = (
                "{context_str}\n\n"
                "Generate a clear, descriptive, and complete title that accurately summarizes the content of this text excerpt.\n\n"
                "Guidelines:\n"
                "- Create a comprehensive title that fully captures the main topic or theme\n"
                "- Use descriptive language that clearly conveys what the content is about\n"
                "- Make the title complete and meaningful - do not truncate or abbreviate unnecessarily\n"
                "- The title should be informative enough for someone to understand the content's subject\n"
                "- Output ONLY the title text, nothing else\n"
                "- No prefixes like 'Title:' or 'The title is'\n"
                "- No quotes around the title\n"
                "- No explanations or additional text\n\n"
                "Title:"
            )
            title_extractor = TitleExtractor(
                llm=self.llm,
                nodes=5,
                node_template=title_prompt_template
            )
            
            # Create QA extractor prompt - generate meaningful, high-quality questions and answers
            qa_prompt_template = (
                "{context_str}\n\n"
                "Generate meaningful, high-quality questions that this text excerpt can answer, along with their corresponding answers.\n\n"
                "Guidelines:\n"
                "- Create questions that are substantive and directly related to the content\n"
                "- Ensure questions are clear, specific, and well-formed\n"
                "- Provide complete, accurate answers based on the content\n"
                "- Focus on questions that would be valuable for someone seeking information from this text\n"
                "- Generate 3-5 questions with their answers\n"
                "- Output format:\n"
                "Q1: [question]\nA1: [answer]\n\nQ2: [question]\nA2: [answer]\n\nQ3: [question]\nA3: [answer]\n\n"
                "- Make questions and answers comprehensive and informative\n"
                "- No truncation or unnecessary abbreviation\n\n"
                "Questions and Answers:"
            )
            qa_extractor = QuestionsAnsweredExtractor(
                llm=self.llm,
                questions=3,
                prompt_template=qa_prompt_template
            )
            
            # Create ingestion pipeline with all transformations
            pipeline = IngestionPipeline(
                transformations=[text_splitter, title_extractor, qa_extractor]
            )
            
            print("   ✅ Advanced pipeline configured (Title + QA extraction with quality-focused prompts)")
            print(f"   ⏳ Processing {len(documents)} document(s) (this may take a while)...")
            
            # Process documents one by one to ensure all are processed
            # This prevents the pipeline from stopping after the first document
            all_nodes = []
            for doc_idx, doc in enumerate(documents, 1):
                doc_name = doc.metadata.get('file_name', f'Document {doc_idx}')
                print(f"      Processing document {doc_idx}/{len(documents)}: {doc_name}")
                try:
                    # Process each document separately
                    doc_nodes = pipeline.run(documents=[doc], in_place=False, show_progress=False)
                    all_nodes.extend(doc_nodes)
                    print(f"         ✅ Generated {len(doc_nodes)} nodes from {doc_name}")
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a 401/authentication error
                    is_auth_error = "401" in error_str or "invalid api key" in error_str or "unauthorized" in error_str
                    
                    if is_auth_error:
                        # Suppress detailed error for auth failures - just note the fallback
                        print(f"         ⚠️  LLM authentication failed, using fallback processing...")
                    else:
                        print(f"         ⚠️  Error processing {doc_name}: {e}")
                    
                    # Fallback: just split the document if metadata extraction fails
                    try:
                        fallback_nodes = text_splitter.get_nodes_from_documents([doc], show_progress=False)
                        all_nodes.extend(fallback_nodes)
                        print(f"         ✅ Fallback: Generated {len(fallback_nodes)} nodes (no metadata)")
                    except Exception as e2:
                        print(f"         ❌ Failed to process {doc_name}: {e2}")
                        continue
            
            nodes = all_nodes
            print(f"   ✅ Generated {len(nodes)} total enriched nodes with metadata")
            
            # Post-process nodes to clean up metadata and ensure proper keys
            print("   🔧 Post-processing metadata to ensure quality and proper formatting...")
            cleaned_count = 0
            for node in nodes:
                # Clean up title - remove extra text, ensure it's stored as 'title', preserve complete titles
                title = node.metadata.get('title') or node.metadata.get('document_title', '')
                if title:
                    # Remove common prefixes and explanations
                    title = str(title).strip()
                    # Remove phrases like "The title is", "Title:", etc.
                    title = re.sub(r'^(The title is|Title:|Le titre|Titre:|Title\s*:)\s*', '', title, flags=re.IGNORECASE)
                    # Remove quotes
                    title = title.strip().strip('"').strip("'").strip()
                    # Remove any trailing explanations (common patterns)
                    title = re.sub(r'\s*\.\s*This title.*$', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\s*\.\s*Ce titre.*$', '', title, flags=re.IGNORECASE)
                    # Split by newlines and take first line (in case LLM added extra text)
                    title = title.split('\n')[0].strip()
                    # Store complete title without truncation
                    node.metadata['title'] = title
                    cleaned_count += 1
                    # Remove document_title if it exists
                    if 'document_title' in node.metadata:
                        del node.metadata['document_title']
                
                # Clean up questions - preserve complete, meaningful Q&A pairs
                questions = node.metadata.get('questions_this_excerpt_can_answer', '')
                if questions:
                    # If it's a string, try to extract Q&A pairs or questions
                    if isinstance(questions, str):
                        # Try to extract Q1:, Q2:, Q3: patterns with answers (A1:, A2:, A3:)
                        qa_pattern = r'Q\d+:\s*([^\n]+?)(?:\nA\d+:\s*([^\n]+?))?(?=\nQ\d+:|$)'
                        qa_matches = re.findall(qa_pattern, questions, re.MULTILINE)
                        if qa_matches:
                            # Format as Q&A pairs, preserving complete content
                            formatted_qa = []
                            for q, a in qa_matches:
                                q = q.strip()
                                if a:
                                    a = a.strip()
                                    formatted_qa.append(f"Q: {q}\nA: {a}")
                                else:
                                    formatted_qa.append(f"Q: {q}")
                            questions = '\n\n'.join(formatted_qa)
                        else:
                            # Try to extract just questions (Q1:, Q2:, Q3:)
                            q_pattern = r'Q\d+:\s*([^\n]+?)(?=\nQ\d+:|$)'
                            q_matches = re.findall(q_pattern, questions, re.MULTILINE)
                            if q_matches:
                                # Clean each question, preserving complete content
                                cleaned_questions = []
                                for q in q_matches:
                                    q = q.strip()
                                    cleaned_questions.append(q)
                                questions = '\n'.join([f"Q{i+1}: {q}" for i, q in enumerate(cleaned_questions)])
                            else:
                                # If no pattern found, preserve the original content
                                questions = questions.strip()
                    elif isinstance(questions, list):
                        # Format list as Q&A pairs, preserving complete content
                        formatted_qa = []
                        for i, q in enumerate(questions, 1):
                            q_str = str(q).strip()
                            formatted_qa.append(f"Q{i}: {q_str}")
                        questions = '\n'.join(formatted_qa)
                    
                    # Store complete Q&A without truncation
                    node.metadata['questions_this_excerpt_can_answer'] = questions
            
            print(f"   ✅ Post-processed {len(nodes)} nodes ({cleaned_count} titles cleaned)")
            
            # Show sample nodes with metadata from different documents
            if nodes:
                print("\n   🔍 Sample nodes with metadata:")
                sample_indices = [0, len(nodes)//4, len(nodes)//2] if len(nodes) > 3 else [0]
                for idx in sample_indices[:3]:
                    sample_node = nodes[idx]
                    title = sample_node.metadata.get('title', 'N/A')
                    questions = sample_node.metadata.get('questions_this_excerpt_can_answer', 'N/A')
                    print(f"      Node {idx+1}:")
                    print(f"         Title ({len(title.split())} words): {title[:60]}")
                    print(f"         Questions: {str(questions)[:80]}...")
        else:
            # Basic pipeline: just split
            print("   📝 Using basic text splitting (no metadata extraction)")
            nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
            print(f"   ✅ Generated {len(nodes)} nodes")
        
        # Step 4: Setup ChromaDB collection
        print(f"\n💾 Step 4: Setting up ChromaDB vector store...")
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(f"   ✅ ChromaDB collection '{self.collection_name}' ready")
        
        # Step 5: Create index with enriched nodes
        print(f"\n🔨 Step 5: Creating vector index (embedding + storage)...")
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,  # HuggingFace embeddings
            show_progress=True  # Show progress bar
        )
        
        print(f"   ✅ Index created and stored in ChromaDB!")
        
        # Verify metadata storage
        if self.use_advanced_rag and nodes:
            print(f"\n📊 Metadata Summary:")
            nodes_with_title = sum(1 for n in nodes if n.metadata.get('title') or n.metadata.get('document_title'))
            nodes_with_qa = sum(1 for n in nodes if n.metadata.get('questions_this_excerpt_can_answer'))
            print(f"   - Nodes with title: {nodes_with_title}/{len(nodes)}")
            print(f"   - Nodes with QA metadata: {nodes_with_qa}/{len(nodes)}")
            
            # Show sample titles
            if nodes_with_title > 0:
                print(f"\n   📝 Sample titles:")
                for i, node in enumerate(nodes[:3], 1):
                    title = node.metadata.get('title') or node.metadata.get('document_title', 'N/A')
                    print(f"      {i}. {title[:60]}...")
        
        print("\n" + "="*80)
        print("✅ Q1: INDEX BUILD COMPLETE!")
        print("="*80 + "\n")
        
        return self.index
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index (if built).
        
        Returns:
            VectorStoreIndex or None
        """
        return self.index


def check_data_folder(data_dir: str = "./data") -> bool:
    """
    Check if data folder exists and contains PDF files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if folder exists and contains PDFs, False otherwise
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    pdf_files = list(data_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   - {pdf.name} ({size_mb:.2f} MB)")
    
    return True









print("")
