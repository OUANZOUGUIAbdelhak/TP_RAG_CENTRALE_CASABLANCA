"""
Q3: Question-Answering System with LLM
=======================================
This module implements the Q&A system using an LLM.

Features:
- Uses retrieved documents from vector database as context
- Employs an open-source LLM (via Groq API)
- Custom prompt template for optimal answer generation
- Synthesizes information from multiple sources
"""

import os
# Disable PostHog telemetry (prevents timeout errors)
os.environ["LLAMA_TELEMETRY_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from pathlib import Path
from typing import Dict, Any, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import yaml
from llm_fallback import create_llm_with_fallback


class QASystem:
    """
    Q3: Question-Answering system using LLM with RAG.
    
    This class combines document retrieval with LLM-based answer generation.
    It uses:
    - Retrieved paragraphs from the vector database as context
    - An open-source LLM (Groq) to synthesize answers
    - A custom prompt template optimized for RAG
    """
    
    def __init__(self,
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 collection_name: str = "rag_collection",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-2.0-flash",
                 use_gemini: bool = False):
        """
        Initialize the Q&A system.
        
        Args:
            vectorstore_dir: Directory containing the ChromaDB vector store
            embedding_model_name: HuggingFace embedding model name
            collection_name: ChromaDB collection name
            groq_api_key: Groq API key for LLM (if None, loads from config)
            groq_model: Groq model name to use
            gemini_api_key: Gemini API key for fallback (if None, loads from config)
            gemini_model: Gemini model name to use
            use_gemini: Whether to use Gemini instead of Groq (deprecated - use fallback instead)
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize embedding model
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        print(f"✅ Embedding model loaded")
        
        # Initialize LLM with fallback mechanism (Groq -> Gemini)
        self.llm = create_llm_with_fallback(
            groq_api_key=groq_api_key,
            groq_model=groq_model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            load_from_config=True
        )
        
        # Load index
        self.index = self._load_index()
        
        # Create query engine with custom prompt
        self.query_engine = None
        if self.llm:
            self.query_engine = self._create_query_engine()
    
    
    def _load_index(self) -> VectorStoreIndex:
        """Load the vector index from ChromaDB."""
        if not self.vectorstore_dir.exists():
            raise ValueError(f"Vector store not found: {self.vectorstore_dir}. Please build index first.")
        
        print(f"📂 Loading index from {self.vectorstore_dir}...")
        
        chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        chroma_collection = chroma_client.get_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        print(f"✅ Index loaded successfully")
        return index
    
    def _create_query_engine(self):
        """
        Q3: Create query engine with custom prompt template.
        
        The prompt template is optimized for RAG and includes:
        - Context from retrieved documents
        - Instructions for synthesizing information
        - Guidelines for handling insufficient context
        """
        # Custom prompt template for Q3
        # This template ensures the LLM:
        # 1. Uses only the provided context (retrieved documents)
        # 2. Uses metadata (titles, questions) internally for better understanding
        # 3. Provides well-formatted, ChatGPT-style responses
        qa_prompt_template = PromptTemplate(
            "You are a highly knowledgeable and articulate AI assistant. "
            "Your goal is to provide clear, well-structured, natural, and insightful answers — "
            "as if you are explaining confidently from your own understanding.\n\n"

            "📚 INFORMATION SOURCE:\n"
            "Use ONLY the information from the context below. "
            "The context contains actual document content - focus on the substantive text content, "
            "not file paths, directory names, or metadata.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"

            "⚠️ CRITICAL INSTRUCTIONS:\n"
            "• IGNORE file paths, directory names, folder structures, or system metadata.\n"
            "• FOCUS ONLY on the actual document content - the substantive text about the topic.\n"
            "• If the context only contains file paths or metadata without actual content, "
            "say 'I cannot answer this question as the retrieved documents do not contain relevant content.'\n"
            "• Base your answer ONLY on the actual document text content provided.\n\n"

            "🎯 RESPONSE STYLE REQUIREMENTS:\n"
            "• Do NOT mention or reference 'context', 'documents', 'files', 'chunks', or 'metadata'.\n"
            "• Do NOT write phrases like 'According to the document' or 'Based on the context'.\n"
            "• Do NOT reference file paths, directory names, or system information.\n"
            "• Respond naturally, as if you already know the information.\n"
            "• Be confident, informative, and professional — do not sound robotic.\n"
            "• Use heading structure (##, ###) and bullet points where helpful.\n"
            "• Add light, relevant emojis only where they improve readability (✨ 💡 📌 📊 📝 ⚠️ 🚀).\n"
            "• Do NOT explain how you derived the answer or mention backend processes.\n"
            "• Provide meaningful, narrative-style answers, not just bullet point extractions.\n"
            "• Structure long answers with introduction, key points, and conclusion.\n\n"

            "🚫 AVOID:\n"
            "• Avoid generic section titles like Introduction, Conclusion, Summary.\n"
            "• Prefer narrative descriptions over bullet-point extractions.\n"
            "• Write in a smooth, well-connected flow, using transitional phrases.\n"
            "• Keep tone professional, articulate, and reflective—like a research scientist or academic advisor is describing the person.\n"
            "• NEVER mention file paths, directories, or system metadata in your answer.\n\n"

            "💡 Final Goal:\n"
            "Provide a polished, human-like, insightful response based on the actual document content — "
            "suitable for a well-written profile summary, professional explanation, or academic-level answer.\n\n"

            "📝 Question: {query_str}\n\n"
            "💬 Answer:"
        )


    

        
        try:
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=10,  # Retrieve top 10 most relevant chunks
                response_mode="compact",  # Compact response mode for efficiency
                text_qa_template=qa_prompt_template  # Our custom prompt
            )
            print(f"✅ Query engine created with custom prompt template")
            return query_engine
        except TypeError:
            # Fallback if text_qa_template parameter doesn't work
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=10,
                response_mode="compact"
            )
            print(f"✅ Query engine created (using default prompt)")
            return query_engine
    
    def answer(self, question: str, document_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Q3: Answer a question using RAG (Retrieval + LLM generation).
        
        Process:
        1. Retrieve relevant paragraphs from vector database
        2. Provide context to LLM via custom prompt
        3. LLM synthesizes information and generates answer
        
        Args:
            question: User's question
            document_path: Optional path to filter by specific document (e.g., "adm.pdf")
            
        Returns:
            Dictionary containing:
                - answer: Generated answer from LLM
                - sources: List of source documents with scores
                - confidence: Overall confidence score
        """
        if not self.query_engine:
            return {
                "answer": "LLM not available. Please configure Groq API key in Config.yaml",
                "sources": [],
                "confidence": 0.0
            }
        
        print(f"🤔 Processing question: {question}")
        if document_path:
            print(f"📄 Filtering by document: {document_path}")
            print(f"   📝 Raw document_path received: '{document_path}'")
        
        try:
            # If document_path is provided, create a filtered query engine
            if document_path:
                # Normalize the document path to match how it's stored in metadata
                # The file_name in metadata might be stored as "data/filename.pdf" or just "filename.pdf"
                # We need to match both possibilities
                normalized_path = document_path.replace('\\', '/')
                
                # Try different path formats that might be stored in metadata
                possible_paths = []
                if normalized_path.startswith('data/'):
                    possible_paths.append(normalized_path)
                    possible_paths.append(normalized_path.replace('data/', ''))
                    possible_paths.append(normalized_path.replace('/', '\\'))
                else:
                    possible_paths.append(normalized_path)
                    possible_paths.append(f"data/{normalized_path}")
                    # Fix: Extract backslash replacement outside f-string
                    windows_path = normalized_path.replace('/', '\\')
                    possible_paths.append(f"data\\{windows_path}")
                
                # Get the custom prompt template from the existing query engine
                custom_template = None
                if hasattr(self.query_engine, '_text_qa_template'):
                    custom_template = self.query_engine._text_qa_template
                elif hasattr(self.query_engine, 'retriever') and hasattr(self.query_engine.retriever, '_text_qa_template'):
                    custom_template = self.query_engine.retriever._text_qa_template
                
                # Create a retriever with metadata filtering
                # We'll filter nodes after retrieval since ChromaDB filter syntax may vary
                retriever = self.index.as_retriever(similarity_top_k=20)  # Get more results to filter from
                
                # Retrieve nodes and filter by file_name
                retrieved_nodes = retriever.retrieve(question)
                
                # Extract just the filename from the document_path (e.g., "adm.pdf" from "data/adm.pdf")
                doc_filename = Path(normalized_path).name
                
                print(f"   🔍 Looking for document: '{doc_filename}'")
                print(f"   📋 Checking {len(retrieved_nodes)} retrieved nodes...")
                print(f"   📝 Possible paths to match: {possible_paths}")
                
                # Filter nodes to only include those from the specified document
                filtered_nodes = []
                all_filenames_seen = set()  # Track all filenames we see for debugging
                
                for i, node in enumerate(retrieved_nodes):
                    node_file_name = node.metadata.get('file_name', '')
                    # Normalize the node file name for comparison
                    node_filename = Path(node_file_name.replace('\\', '/')).name
                    all_filenames_seen.add(node_filename)
                    
                    # Debug: Print first few nodes to see what we're comparing
                    if i < 5:
                        print(f"      Node {i+1}: file_name='{node_file_name}', extracted_filename='{node_filename}'")
                    
                    # Multiple matching strategies:
                    # 1. Exact filename match (case-insensitive)
                    filename_matches = node_filename.lower() == doc_filename.lower()
                    
                    # 2. Check if filename without extension matches (in case of case differences)
                    doc_name_no_ext = Path(doc_filename).stem.lower()
                    node_name_no_ext = Path(node_filename).stem.lower()
                    name_matches = doc_name_no_ext == node_name_no_ext
                    
                    # 3. Check if any of the possible paths match
                    path_matches = False
                    for path in possible_paths:
                        path_filename = Path(path.replace('\\', '/')).name
                        # Check exact filename match
                        if path_filename.lower() == node_filename.lower():
                            path_matches = True
                            break
                        # Check if path is contained in file_name (for cases like "data/adm.pdf")
                        normalized_node_path = node_file_name.replace('\\', '/').lower()
                        normalized_check_path = path.replace('\\', '/').lower()
                        if normalized_check_path in normalized_node_path or normalized_node_path.endswith(normalized_check_path):
                            path_matches = True
                            break
                        # Also check filename part
                        if path_filename.lower() == node_filename.lower():
                            path_matches = True
                            break
                    
                    matches = filename_matches or name_matches or path_matches
                    
                    if matches:
                        filtered_nodes.append(node)
                        if len(filtered_nodes) <= 3:
                            print(f"      ✅ MATCH FOUND: {node_file_name}")
                
                # Debug: Show all unique filenames found
                print(f"   📚 All filenames in retrieved nodes: {sorted(all_filenames_seen)}")
                
                print(f"   📊 Found {len(filtered_nodes)} matching nodes out of {len(retrieved_nodes)} retrieved")
                
                # If we found filtered nodes, use them; otherwise return an error
                if filtered_nodes:
                    print(f"   ✅ Filtered to {len(filtered_nodes)} chunks from document: {document_path}")
                    # Limit to top 10 filtered nodes (keep them sorted by relevance)
                    filtered_nodes = filtered_nodes[:10]
                    
                    # Create a query engine with the filtered nodes
                    # We'll use a custom retriever that returns only filtered nodes
                    from llama_index.core.query_engine import RetrieverQueryEngine
                    
                    # Create a custom retriever that returns only filtered nodes
                    class FilteredRetriever:
                        def __init__(self, nodes):
                            self.nodes = nodes
                        
                        def retrieve(self, query_str):
                            return self.nodes
                    
                    filtered_retriever = FilteredRetriever(filtered_nodes)
                    
                    # Get the prompt template from the existing query engine
                    try:
                        # Try to get the prompt template from the query engine
                        if custom_template:
                            template = custom_template
                        else:
                            # Use the default template from the QA prompt
                            template = PromptTemplate(
                                "You are a highly knowledgeable and articulate AI assistant. "
                                "Your goal is to provide clear, well-structured, natural, and insightful answers — "
                                "as if you are explaining confidently from your own understanding.\n\n"
                                "📚 INFORMATION SOURCE:\n"
                                "Use ONLY the information from the context below. "
                                "The context contains actual document content - focus on the substantive text content, "
                                "not file paths, directory names, or metadata.\n"
                                "---------------------\n"
                                "{context_str}\n"
                                "---------------------\n\n"
                                "⚠️ CRITICAL INSTRUCTIONS:\n"
                                "• IGNORE file paths, directory names, folder structures, or system metadata.\n"
                                "• FOCUS ONLY on the actual document content - the substantive text about the topic.\n"
                                "• If the context only contains file paths or metadata without actual content, "
                                "say 'I cannot answer this question as the retrieved documents do not contain relevant content.'\n"
                                "• Base your answer ONLY on the actual document text content provided.\n\n"
                                "📝 Question: {query_str}\n\n"
                                "💬 Answer:"
                            )
                    except:
                        template = None
                    
                    # Create query engine with filtered retriever
                    filtered_query_engine = RetrieverQueryEngine(
                        retriever=filtered_retriever,
                        llm=self.llm,
                        response_mode="compact",
                        text_qa_template=template
                    )
                    
                    response = filtered_query_engine.query(question)
                else:
                    # No matching nodes found - return error instead of searching all documents
                    print(f"   ❌ No chunks found for document: {document_path}")
                    print(f"   💡 Document might not be indexed or path doesn't match")
                    
                    # Get list of unique document names from retrieved nodes for debugging
                    available_docs = list(set(Path(n.metadata.get('file_name', 'Unknown')).name for n in retrieved_nodes[:10] if n.metadata.get('file_name')))
                    
                    error_msg = (
                        f"I couldn't find any content from the document '{doc_filename}' in the index.\n\n"
                        f"Please make sure:\n"
                        f"1. The document is indexed (build the index if needed)\n"
                        f"2. The document name matches exactly\n\n"
                    )
                    if available_docs:
                        error_msg += f"Available documents in index: {', '.join(available_docs[:5])}"
                    
                    return {
                        "answer": error_msg,
                        "sources": [],
                        "confidence": 0.0
                    }
            else:
                # Use the default query engine (no filtering)
                response = self.query_engine.query(question)
            
            # Extract source documents first to check content quality
            source_nodes = getattr(response, 'source_nodes', [])
            
            # Debug: Check if retrieved content is meaningful
            if source_nodes:
                print(f"   📊 Retrieved {len(source_nodes)} source chunks")
                # Check first source for content quality
                first_node = source_nodes[0]
                content_preview = first_node.text[:200] if first_node.text else "NO TEXT"
                print(f"   🔍 First chunk preview: {content_preview}...")
                
                # Warn if content looks like metadata only
                if first_node.text and (
                    'data\\' in first_node.text or 
                    'data/' in first_node.text or 
                    'RAG_TO_MODIFY' in first_node.text or
                    len(first_node.text.strip()) < 50
                ):
                    print(f"   ⚠️  WARNING: Retrieved content appears to be mostly metadata/paths")
                    print(f"   ⚠️  This may indicate indexing issues - consider rebuilding the index")
            
            # Extract answer (LLM-generated response)
            if hasattr(response, 'response'):
                answer = str(response.response).strip()
            else:
                answer = str(response).strip()
            
            # Extract source documents (metadata used internally, not exposed in response)
            sources = []
            for node in source_nodes[:5]:
                source_path = node.metadata.get('file_name', 'Unknown')
                if 'data' in source_path:
                    source_path = source_path.replace('data\\', '').replace('data/', '')
                
                score = getattr(node, 'score', 0.0)
                similarity = max(0.0, min(1.0, 1.0 - score))
                
                # Note: Metadata (title, questions_answered) is available in node.metadata
                # and used by the LLM for context, but we don't expose it in the response
                # The LLM can incorporate this information naturally into the answer content
                
                source = {
                    "source": source_path,
                    "page": node.metadata.get('page_label', 'N/A'),
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "score": round(score, 4),
                    "similarity": round(similarity, 4),
                    "similarity_percent": round(similarity * 100, 1)
                }
                
                # Metadata is NOT added to the response - it's used internally only
                sources.append(source)
            
            # Calculate confidence from source similarities
            confidence = sum(s["similarity"] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def answer_with_details(self, question: str):
        """
        Q3: Answer a question and print detailed results.
        
        Args:
            question: User's question
        """
        result = self.answer(question)
        
        print("\n" + "="*80)
        print("💡 ANSWER")
        print("="*80 + "\n")
        print(result["answer"])
        
        if result["sources"]:
            print("\n" + "="*80)
            print("📚 SOURCES")
            print("="*80 + "\n")
            
            for i, source in enumerate(result["sources"], 1):
                print(f"Source #{i}: {source['source']} (Page: {source['page']})")
                print(f"Similarity: {source['similarity_percent']}%")
                print(f"Content: {source['content']}")
                print("-"*80 + "\n")
                # Note: Metadata (title, questions_answered) is used internally by the LLM
                # but not shown in the response to keep it clean and user-friendly
            
            print(f"🎯 Overall Confidence: {result['confidence']*100:.1f}%")

