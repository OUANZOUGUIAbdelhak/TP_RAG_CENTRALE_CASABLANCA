"""
Question-Answer System - Q3: Système de question-réponse basé sur un LLM
Handles LLM-based question answering using retrieved context.
"""

from typing import Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.groq import Groq


class QASystem:
    """
    Q3: Question-Answer system using LLM with retrieved context.
    Creates optimized prompts and synthesizes answers from document context.
    """
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 similarity_top_k: int = 5):
        """
        Initialize the QA system.
        
        Args:
            index: VectorStoreIndex for document retrieval
            groq_api_key: Groq API key for LLM
            groq_model: Groq model name
            similarity_top_k: Number of documents to retrieve for context
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        
        # Initialize LLM (Groq)
        self.llm = None
        if groq_api_key:
            try:
                print(f"🔧 Initializing Groq LLM: {groq_model}")
                self.llm = Groq(model=groq_model, api_key=groq_api_key)
                print(f"✅ Groq LLM initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize Groq LLM: {e}")
                self.llm = None
        else:
            print("⚠️  No Groq API key provided. QA system will work in retrieval-only mode.")
        
        # Create optimized prompt template
        self.qa_prompt_template = self._create_prompt_template()
        
        # Initialize query engine
        self.query_engine = None
        self._setup_query_engine()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create an optimized prompt template for the LLM.
        
        Returns:
            PromptTemplate instance with instructions for the LLM
        """
        template_text = """
                        Tu es un assistant IA spécialisé dans l'analyse de documents. 
                        Ton rôle est de répondre aux questions en te basant UNIQUEMENT sur le contexte fourni.

                        CONTEXTE DOCUMENTAIRE:
                        ---------------------
                        {context_str}
                        ---------------------

                        INSTRUCTIONS:
                        1. Réponds à la question en français de manière claire et structurée
                        2. Base-toi EXCLUSIVEMENT sur les informations du contexte fourni
                        3. Si le contexte ne contient pas assez d'informations, dis-le explicitement
                        4. Cite les sources quand c'est pertinent
                        5. Structure ta réponse avec des paragraphes et des listes si nécessaire
                        6. Sois précis et factuel

                        QUESTION: {query_str}

                        RÉPONSE:
                    """
                                
        return PromptTemplate(template_text)
    
    def _setup_query_engine(self):
        """
        Setup the query engine with LLM and prompt template.
        """
        if not self.llm:
            print("⚠️  No LLM available. Query engine will work in retrieval-only mode.")
            return
        
        try:
            print(f"🤖 Setting up query engine...")
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.similarity_top_k,
                response_mode="compact",
                text_qa_template=self.qa_prompt_template
            )
            print(f"✅ Query engine ready with custom prompt template")
        except TypeError:
            # Fallback if text_qa_template parameter doesn't work
            print(f"⚠️  Using fallback query engine configuration...")
            try:
                self.query_engine = self.index.as_query_engine(
                    llm=self.llm,
                    similarity_top_k=self.similarity_top_k,
                    response_mode="compact"
                )
                print(f"✅ Query engine ready (using default prompt)")
            except Exception as e:
                print(f"❌ Failed to create query engine: {e}")
                self.query_engine = None
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the LLM and retrieved context.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.query_engine:
            # Fallback: retrieve documents without LLM
            return self._retrieve_only_answer(question)
        
        try:
            print(f"🔍 Processing question: {question[:100]}...")
            
            # Query with LLM
            response = self.query_engine.query(question)
            
            # Extract the answer
            if hasattr(response, 'response'):
                answer = str(response.response).strip()
            else:
                answer = str(response).strip()
            
            # Extract source nodes
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_path = node.metadata.get('file_name', 'Unknown')
                    if 'data' in source_path:
                        source_path = source_path.replace('data\\', '').replace('data/', '')
                    
                    score = getattr(node, 'score', 0.0)
                    similarity = max(0.0, min(1.0, 1.0 - score))
                    
                    sources.append({
                        "source": source_path,
                        "page": node.metadata.get('page_label', 'N/A'),
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": round(score, 4),
                        "similarity": round(similarity, 4),
                        "similarity_percent": round(similarity * 100, 1)
                    })
            
            # Calculate confidence
            confidence = sum(s["similarity"] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3),
                "method": "llm_synthesis"
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Erreur lors du traitement de la question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "method": "error"
            }
    
    def _retrieve_only_answer(self, question: str) -> Dict[str, Any]:
        """
        Fallback method: retrieve documents without LLM synthesis.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with retrieved content and sources
        """
        print(f"🔍 Retrieving documents for: {question[:100]}...")
        
        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(question)
        
        sources = []
        for node in nodes:
            source_path = node.metadata.get('file_name', 'Unknown')
            if 'data' in source_path:
                source_path = source_path.replace('data\\', '').replace('data/', '')
            
            score = getattr(node, 'score', 0.0)
            similarity = max(0.0, min(1.0, 1.0 - score))
            
            sources.append({
                "source": source_path,
                "page": node.metadata.get('page_label', 'N/A'),
                "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "score": round(score, 4),
                "similarity": round(similarity, 4),
                "similarity_percent": round(similarity * 100, 1)
            })
        
        # Simple answer from top result
        answer = f"Contenu le plus pertinent trouvé:\n\n{nodes[0].text}" if nodes else "Aucun document pertinent trouvé."
        
        confidence = sources[0]["similarity"] if sources else 0.0
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "method": "retrieval_only"
        }
    
    def batch_questions(self, questions: list) -> Dict[str, Any]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            Dictionary with all results
        """
        print(f"\n🧪 Processing {len(questions)} questions...")
        
        results = {}
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            result = self.answer_question(question)
            results[question] = result
            
            print(f"Q: {question}")
            print(f"A: {result['answer'][:100]}...")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Sources: {len(result['sources'])}")
        
        return results
    
    def print_qa_result(self, result: Dict[str, Any]):
        """
        Print formatted QA result.
        
        Args:
            result: Result dictionary from answer_question
        """
        print(f"\n" + "="*80)
        print(f"QUESTION: {result['question']}")
        print("="*80)
        
        print(f"\nRÉPONSE:")
        print(f"{result['answer']}")
        
        print(f"\nMÉTADONNÉES:")
        print(f"Confiance: {result['confidence']:.1%}")
        print(f"Méthode: {result['method']}")
        print(f"Sources utilisées: {len(result['sources'])}")
        
        if result['sources']:
            print(f"\nSOURCES:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['source']} (Page {source['page']}) - Similarité: {source['similarity_percent']}%")
        
        print(f"\n" + "="*80)
