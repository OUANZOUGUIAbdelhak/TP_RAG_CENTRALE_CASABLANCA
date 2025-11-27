#!/usr/bin/env python3
"""
CLI - Command Line Interface for the RAG System

This is the main entry point! Run all functionality from here.

Usage:
    python cli.py build         # Build the index from PDFs (Q1)
    python cli.py search        # Search documents (Q2)
    python cli.py ask           # Ask a question (Q3)
    python cli.py evaluate      # Evaluate the system (Q4)
    python cli.py chat          # Start interactive chatbot (Q5)
    python cli.py --help        # Show help

No hardcoded values - everything comes from Config.yaml!
"""

import sys
import os
import argparse
import yaml

# Add src to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from indexer import DocumentIndexer, check_data_folder
from retriever import DocumentRetriever
from qa_system import QASystem
from evaluator import RAGEvaluator
from chatbot import RAGChatbot


def load_config():
    """
    Load configuration from Config.yaml.
    Returns dict with all settings.
    """
    try:
        with open('Config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Warning: Could not load Config.yaml: {e}")
        print("Using default configuration...\n")
        return {
            'paths': {
                'data_dir': './data',
                'vectorstore_dir': './vectorstore'
            },
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'document_processing': {
                'chunk_size': 500,
                'chunk_overlap': 50
            },
            'retrieval': {
                'default_k': 5
            }
        }


def cmd_build(args, config):
    """
    Command: Build the index from PDF documents (Q1)
    """
    print("\n" + "="*80)
    print("🏗️  BUILD INDEX (Q1)")
    print("="*80 + "\n")
    
    data_dir = config['paths']['data_dir']
    vectorstore_dir = config['paths']['vectorstore_dir']
    
    # Check if we have PDFs
    print("🔍 Checking for PDF files...")
    if not check_data_folder(data_dir):
        print("\n❌ No PDF files found!")
        print(f"💡 Please add 3-4 PDF files to: {data_dir}")
        return 1
    
    print()
    
    # Confirm before building (unless --force flag is used)
    if not args.force:
        response = input("🚀 Ready to build? This might take a few minutes. (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("👋 Cancelled.")
            return 0
        print()
    
    # Build the index
    try:
        indexer = DocumentIndexer(
            data_dir=data_dir,
            vectorstore_dir=vectorstore_dir,
            embedding_model_name=config['embedding']['model_name'],
            chunk_size=config['document_processing']['chunk_size'],
            chunk_overlap=config['document_processing']['chunk_overlap']
        )
        
        indexer.build_index()
        
        print("\n✅ Build complete! You can now use search, ask, evaluate, or chat commands.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        return 1


def cmd_search(args, config):
    """
    Command: Search for documents (Q2)
    """
    print("\n" + "="*80)
    print("🔍 SEARCH DOCUMENTS (Q2)")
    print("="*80 + "\n")
    
    vectorstore_dir = config['paths']['vectorstore_dir']
    k = args.k or config['retrieval']['default_k']
    
    # Get query from args or prompt user
    if args.query:
        query = ' '.join(args.query)
    else:
        query = input("Enter your search query: ").strip()
        if not query:
            print("❌ No query provided.")
            return 1
    
    print()
    
    # Perform search
    try:
        retriever = DocumentRetriever(
            vectorstore_dir=vectorstore_dir,
            embedding_model_name=config['embedding']['model_name']
        )
        
        retriever.search_and_print_results(query, k=k)
        return 0
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        print("💡 Have you built the index first? Run: python cli.py build")
        return 1


def cmd_ask(args, config):
    """
    Command: Ask a question and get an answer (Q3)
    """
    print("\n" + "="*80)
    print("❓ ASK A QUESTION (Q3)")
    print("="*80 + "\n")
    
    vectorstore_dir = config['paths']['vectorstore_dir']
    
    # Get question from args or prompt user
    if args.question:
        question = ' '.join(args.question)
    else:
        question = input("Enter your question: ").strip()
        if not question:
            print("❌ No question provided.")
            return 1
    
    print()
    
    # Get answer
    try:
        qa_system = QASystem(
            vectorstore_dir=vectorstore_dir,
            embedding_model_name=config['embedding']['model_name']
        )
        
        qa_system.answer_with_details(question)
        return 0
        
    except Exception as e:
        print(f"❌ Error getting answer: {e}")
        print("💡 Have you built the index first? Run: python cli.py build")
        return 1


def cmd_evaluate(args, config):
    """
    Command: Evaluate system performance (Q4)
    """
    print("\n" + "="*80)
    print("📊 EVALUATE SYSTEM (Q4)")
    print("="*80 + "\n")
    
    vectorstore_dir = config['paths']['vectorstore_dir']
    
    # Sample test cases - YOU SHOULD CUSTOMIZE THESE!
    print("⚠️  Using default test cases. For better evaluation, modify the test cases in cli.py")
    print()
    
    test_cases = [
        {
            'question': 'What is the main topic?',
            'expected_keywords': ['topic', 'subject', 'about']
        },
        {
            'question': 'Can you summarize the key points?',
            'expected_keywords': ['summary', 'points', 'main']
        },
        {
            'question': 'What are the important concepts?',
            'expected_keywords': ['concept', 'important', 'key']
        }
    ]
    
    # Run evaluation
    try:
        evaluator = RAGEvaluator(
            vectorstore_dir=vectorstore_dir,
            embedding_model_name=config['embedding']['model_name']
        )
        
        if args.quick:
            # Quick quality check
            evaluator.quick_quality_check()
        else:
            # Full evaluation
            evaluator.evaluate_end_to_end(test_cases)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("💡 Have you built the index first? Run: python cli.py build")
        return 1


def cmd_chat(args, config):
    """
    Command: Start interactive chatbot (Q5 - Bonus)
    """
    print("\n" + "="*80)
    print("💬 INTERACTIVE CHATBOT (Q5)")
    print("="*80 + "\n")
    
    vectorstore_dir = config['paths']['vectorstore_dir']
    max_history = config.get('chatbot', {}).get('max_history', 5)
    
    # Start chatbot
    try:
        chatbot = RAGChatbot(
            vectorstore_dir=vectorstore_dir,
            embedding_model_name=config['embedding']['model_name'],
            max_history=max_history
        )
        
        chatbot.interactive_session()
        return 0
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("💡 Have you built the index first? Run: python cli.py build")
        return 1


def main():
    """
    Main entry point - parse arguments and run the appropriate command.
    """
    parser = argparse.ArgumentParser(
        description="RAG System - Retrieval Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py build              Build index from PDFs
  python cli.py search "machine learning"    Search documents
  python cli.py ask "What is AI?"            Ask a question
  python cli.py evaluate --quick             Quick evaluation
  python cli.py chat                         Start chatbot
        """
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command (Q1)
    parser_build = subparsers.add_parser('build', help='Build index from PDF documents (Q1)')
    parser_build.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    # Search command (Q2)
    parser_search = subparsers.add_parser('search', help='Search for documents (Q2)')
    parser_search.add_argument('query', nargs='*', help='Search query')
    parser_search.add_argument('-k', type=int, help='Number of results to return')
    
    # Ask command (Q3)
    parser_ask = subparsers.add_parser('ask', help='Ask a question (Q3)')
    parser_ask.add_argument('question', nargs='*', help='Your question')
    
    # Evaluate command (Q4)
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate system performance (Q4)')
    parser_eval.add_argument('--quick', action='store_true', help='Quick quality check only')
    
    # Chat command (Q5)
    parser_chat = subparsers.add_parser('chat', help='Start interactive chatbot (Q5 - Bonus)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        print("\n💡 Start with: python cli.py build")
        return 0
    
    # Load configuration
    config = load_config()
    
    # Route to appropriate command
    commands = {
        'build': cmd_build,
        'search': cmd_search,
        'ask': cmd_ask,
        'evaluate': cmd_evaluate,
        'chat': cmd_chat
    }
    
    if args.command in commands:
        return commands[args.command](args, config)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
