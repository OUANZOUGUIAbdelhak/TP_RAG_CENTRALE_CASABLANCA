"""
RAG Chatbot - Q5 (Bonus): Construction de chatbot
Chatbot with conversation memory using the RAG QA system.
"""

from typing import List, Dict, Any, Optional
from src.qa_system import QASystem


class RAGChatbot:
    """
    Q5 (Bonus): Chatbot with conversation memory.
    Includes conversation history in prompts for contextual responses.
    """
    
    def __init__(self, qa_system: QASystem, max_history: int = 5):
        """
        Initialize the RAG chatbot.
        
        Args:
            qa_system: QASystem instance for answering questions
            max_history: Maximum number of conversation turns to remember
        """
        self.qa_system = qa_system
        self.max_history = max_history
        self.conversation_history = []
        self.session_id = None
    
    def start_session(self, session_id: str = None):
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"session_{len(self.conversation_history)}"
        self.conversation_history = []
        print(f"💬 Nouvelle session de chat démarrée: {self.session_id}")
    
    def chat(self, message: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a chat message with conversation context.
        
        Args:
            message: User's message
            verbose: Print debug information
            
        Returns:
            Dictionary with response and metadata
        """
        if verbose:
            print(f"💬 Message utilisateur: {message}")
        
        # Build contextualized query with conversation history
        contextualized_query = self._build_contextualized_query(message)
        
        if verbose and contextualized_query != message:
            print(f"📝 Requête contextualisée: {contextualized_query[:200]}...")
        
        # Get response from QA system
        qa_result = self.qa_system.answer_question(contextualized_query)
        
        # Store conversation turn
        conversation_turn = {
            "user_message": message,
            "assistant_response": qa_result["answer"],
            "sources": qa_result["sources"],
            "confidence": qa_result["confidence"]
        }
        
        self.conversation_history.append(conversation_turn)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Prepare response
        response = {
            "message": qa_result["answer"],
            "sources": qa_result["sources"],
            "confidence": qa_result["confidence"],
            "conversation_turn": len(self.conversation_history),
            "session_id": self.session_id
        }
        
        return response
    
    def _build_contextualized_query(self, current_message: str) -> str:
        """
        Build a contextualized query including conversation history.
        
        Args:
            current_message: Current user message
            
        Returns:
            Contextualized query string
        """
        if not self.conversation_history:
            return current_message
        
        # Build context from recent conversation
        context_parts = []
        
        # Add recent conversation turns (most recent first)
        recent_turns = self.conversation_history[-3:]  # Last 3 turns for context
        
        if recent_turns:
            context_parts.append("Contexte de la conversation précédente:")
            
            for i, turn in enumerate(recent_turns, 1):
                context_parts.append(f"Tour {i}:")
                context_parts.append(f"Utilisateur: {turn['user_message']}")
                context_parts.append(f"Assistant: {turn['assistant_response'][:150]}...")
                context_parts.append("")
        
        # Add current question
        context_parts.append("Question actuelle:")
        context_parts.append(current_message)
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation.
        
        Returns:
            Conversation summary with statistics
        """
        if not self.conversation_history:
            return {
                "session_id": self.session_id,
                "total_turns": 0,
                "avg_confidence": 0.0,
                "topics_discussed": []
            }
        
        # Calculate statistics
        confidences = [turn["confidence"] for turn in self.conversation_history]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Extract topics (simple keyword extraction)
        all_messages = [turn["user_message"] for turn in self.conversation_history]
        topics = self._extract_topics(all_messages)
        
        return {
            "session_id": self.session_id,
            "total_turns": len(self.conversation_history),
            "avg_confidence": round(avg_confidence, 3),
            "topics_discussed": topics[:5],  # Top 5 topics
            "last_sources": self.conversation_history[-1]["sources"] if self.conversation_history else []
        }
    
    def _extract_topics(self, messages: List[str]) -> List[str]:
        """
        Extract main topics from conversation messages.
        
        Args:
            messages: List of user messages
            
        Returns:
            List of identified topics
        """
        # Simple keyword-based topic extraction
        all_text = " ".join(messages).lower()
        
        # Common topic keywords (extend based on your domain)
        topic_keywords = {
            "intelligence artificielle": ["ia", "intelligence artificielle", "machine learning", "deep learning"],
            "technologie": ["technologie", "tech", "innovation", "numérique"],
            "données": ["données", "data", "base de données", "information"],
            "sécurité": ["sécurité", "protection", "cybersécurité", "confidentialité"],
            "développement": ["développement", "programmation", "code", "logiciel"],
            "recherche": ["recherche", "étude", "analyse", "investigation"]
        }
        
        identified_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                identified_topics.append(topic)
        
        return identified_topics
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print(f"🗑️ Historique de conversation effacé pour la session {self.session_id}")
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export conversation history.
        
        Returns:
            Complete conversation data
        """
        return {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "summary": self.get_conversation_summary()
        }
    
    def print_conversation_history(self):
        """Print formatted conversation history."""
        if not self.conversation_history:
            print("📭 Aucun historique de conversation")
            return
        
        print(f"\n" + "="*80)
        print(f"💬 HISTORIQUE DE CONVERSATION - Session: {self.session_id}")
        print("="*80)
        
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\n--- Tour {i} ---")
            print(f"👤 Utilisateur: {turn['user_message']}")
            print(f"🤖 Assistant: {turn['assistant_response'][:200]}...")
            print(f"📊 Confiance: {turn['confidence']:.1%}")
            print(f"📚 Sources: {len(turn['sources'])}")
        
        print(f"\n" + "="*80)
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        """
        print(f"\n🤖 Chatbot RAG interactif démarré!")
        print(f"💡 Tapez 'quit' pour quitter, 'history' pour voir l'historique")
        print(f"💡 Tapez 'clear' pour effacer l'historique, 'summary' pour un résumé")
        print("-" * 60)
        
        self.start_session()
        
        while True:
            try:
                user_input = input("\n👤 Vous: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Au revoir!")
                    break
                elif user_input.lower() == 'history':
                    self.print_conversation_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary()
                    print(f"\n📊 Résumé: {summary}")
                    continue
                elif not user_input:
                    print("⚠️ Veuillez saisir une question.")
                    continue
                
                # Process message
                print("🔍 Recherche en cours...")
                response = self.chat(user_input)
                
                print(f"\n🤖 Assistant: {response['message']}")
                print(f"📊 Confiance: {response['confidence']:.1%} | Sources: {len(response['sources'])}")
                
            except KeyboardInterrupt:
                print("\n👋 Session interrompue. Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")


class MultiSessionChatbot:
    """
    Extended chatbot supporting multiple conversation sessions.
    """
    
    def __init__(self, qa_system: QASystem):
        """
        Initialize multi-session chatbot.
        
        Args:
            qa_system: QASystem instance
        """
        self.qa_system = qa_system
        self.sessions = {}
        self.active_session = None
    
    def create_session(self, session_id: str) -> RAGChatbot:
        """
        Create a new chat session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            RAGChatbot instance for the session
        """
        if session_id in self.sessions:
            print(f"⚠️ Session {session_id} existe déjà")
            return self.sessions[session_id]
        
        chatbot = RAGChatbot(self.qa_system)
        chatbot.start_session(session_id)
        self.sessions[session_id] = chatbot
        self.active_session = session_id
        
        print(f"✅ Session {session_id} créée")
        return chatbot
    
    def switch_session(self, session_id: str) -> Optional[RAGChatbot]:
        """
        Switch to an existing session.
        
        Args:
            session_id: Session to switch to
            
        Returns:
            RAGChatbot instance or None if not found
        """
        if session_id not in self.sessions:
            print(f"❌ Session {session_id} introuvable")
            return None
        
        self.active_session = session_id
        print(f"🔄 Basculé vers la session {session_id}")
        return self.sessions[session_id]
    
    def list_sessions(self) -> Dict[str, Any]:
        """
        List all active sessions.
        
        Returns:
            Dictionary of session summaries
        """
        return {
            session_id: chatbot.get_conversation_summary()
            for session_id, chatbot in self.sessions.items()
        }
