"""
LLM Fallback System
===================
Provides a fallback mechanism that tries Groq first, then falls back to Gemini
if Groq is unavailable or fails.
"""

import os
from typing import Optional, Any
from pathlib import Path
import yaml

# Disable PostHog telemetry
os.environ["LLAMA_TELEMETRY_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

try:
    from llama_index.llms.groq import Groq
except ImportError:
    Groq = None

try:
    # Try the newer Google Generative AI SDK import (v0.8+)
    try:
        from google import genai
        GEMINI_AVAILABLE = True
        GEMINI_NEW_SDK = True
    except ImportError:
        # Fallback to older import style (v0.3-0.7)
        try:
            import google.generativeai as genai
            GEMINI_AVAILABLE = True
            GEMINI_NEW_SDK = False
        except ImportError:
            GEMINI_AVAILABLE = False
            GEMINI_NEW_SDK = False
            genai = None
except Exception:
    GEMINI_AVAILABLE = False
    GEMINI_NEW_SDK = False
    genai = None

from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    LLM,
)


class GeminiLLM(LLM):
    """
    LlamaIndex-compatible wrapper for Google Gemini API.
    This allows Gemini to be used as a drop-in replacement for Groq.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Gemini API key
            model: Model name (e.g., "gemini-2.0-flash")
            temperature: Sampling temperature
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        # Initialize Gemini client (handle both new and old SDK)
        try:
            if GEMINI_NEW_SDK:
                # New SDK: genai.Client()
                self.client = genai.Client(api_key=api_key)
            else:
                # Old SDK: genai.configure() and genai.GenerativeModel()
                genai.configure(api_key=api_key)
                self.client = None  # Will use genai.GenerativeModel() directly
            print(f"✅ Gemini client initialized: {model}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini client: {e}")
            raise
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model,
            temperature=self.temperature,
            context_window=32768,  # Gemini 3 Pro context window
            num_output=8192,
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Complete a prompt using Gemini.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            CompletionResponse with generated text
        """
        try:
            if GEMINI_NEW_SDK and self.client:
                # New SDK: client.models.generate_content()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            else:
                # Old SDK: genai.GenerativeModel().generate_content()
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
            
            text = response.text if hasattr(response, 'text') else str(response)
            
            return CompletionResponse(
                text=text,
                raw={"response": response}
            )
        except Exception as e:
            print(f"❌ Gemini completion error: {e}")
            raise
    
    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """
        Chat completion using Gemini.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional arguments
            
        Returns:
            CompletionResponse with generated text
        """
        # Convert LlamaIndex messages to Gemini format
        # Gemini expects a single string or structured content
        prompt_parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            
            if role == "user" or role == "assistant":
                prompt_parts.append(f"{role.capitalize()}: {content}")
            else:
                prompt_parts.append(content)
        
        prompt = "\n\n".join(prompt_parts)
        
        try:
            if GEMINI_NEW_SDK and self.client:
                # New SDK: client.models.generate_content()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            else:
                # Old SDK: genai.GenerativeModel().generate_content()
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
            
            text = response.text if hasattr(response, 'text') else str(response)
            
            return CompletionResponse(
                text=text,
                raw={"response": response}
            )
        except Exception as e:
            print(f"❌ Gemini chat error: {e}")
            raise
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """
        Stream completion (not fully implemented for Gemini).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Yields:
            CompletionResponse chunks
        """
        # For now, return non-streaming completion
        response = self.complete(prompt, **kwargs)
        yield response


def create_llm_with_fallback(
    groq_api_key: Optional[str] = None,
    groq_model: str = "llama-3.3-70b-versatile",
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.0-flash",
    load_from_config: bool = True
) -> Optional[Any]:
    """
    Create an LLM with automatic fallback: tries Groq first, then Gemini.
    
    Args:
        groq_api_key: Groq API key (optional if load_from_config=True)
        groq_model: Groq model name
        gemini_api_key: Gemini API key (optional if load_from_config=True)
        gemini_model: Gemini model name
        load_from_config: Whether to load API keys from Config.yaml
        
    Returns:
        LLM instance (Groq or Gemini) or None if both fail
    """
    # Load config if requested
    if load_from_config:
        config = _load_config()
        if config:
            groq_config = config.get('groq', {})
            gemini_config = config.get('gemini', {})
            
            if not groq_api_key:
                groq_api_key = groq_config.get('api_key')
            if not groq_model:
                groq_model = groq_config.get('model', 'llama-3.3-70b-versatile')
            
            if not gemini_api_key:
                gemini_api_key = gemini_config.get('api_key')
            if not gemini_model:
                gemini_model = gemini_config.get('model', 'gemini-2.0-flash')
    
    # Check environment variables as fallback (useful for deployment/CI)
    # Only use env vars if config value is missing or is a placeholder
    if not groq_api_key or groq_api_key in ["your_key_here", "your_key", ""]:
        groq_api_key = os.getenv("GROQ_API_KEY") or groq_api_key
    
    if not gemini_api_key or gemini_api_key in ["your_gemini_api_key_here", "your_key", ""]:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or gemini_api_key
    
    # Check if Groq API key is valid (not a placeholder)
    groq_key_is_valid = groq_api_key and groq_api_key not in ["your_key_here", "your_key", ""]
    
    # Skip Groq entirely if API key is invalid/placeholder and go straight to Gemini
    if not groq_key_is_valid:
        print(f"ℹ️  Groq API key not configured, using Gemini directly...")
    # Try Groq first (only if API key is valid)
    elif groq_api_key and Groq:
        try:
            print(f"🔧 Attempting to initialize Groq LLM: {groq_model}")
            llm = Groq(model=groq_model, api_key=groq_api_key)
            print(f"✅ Groq LLM initialized successfully")
            return llm
        except Exception as e:
            print(f"⚠️  Failed to initialize Groq LLM: {e}")
            print(f"🔄 Falling back to Gemini...")
    else:
        # Groq skipped or not available - go to Gemini
        if not groq_key_is_valid:
            pass  # Already printed message above
        elif not Groq:
            print(f"⚠️  Groq package not available, using Gemini...")
        else:
            print(f"🔄 Attempting to use Gemini...")
    
    # Fallback to Gemini
    if gemini_api_key and GEMINI_AVAILABLE:
        try:
            print(f"🔧 Attempting to initialize Gemini LLM: {gemini_model}")
            llm = GeminiLLM(api_key=gemini_api_key, model=gemini_model)
            print(f"✅ Gemini LLM initialized successfully (fallback)")
            return llm
        except Exception as e:
            print(f"❌ Failed to initialize Gemini LLM: {e}")
            print(f"❌ Both Groq and Gemini failed. LLM queries will not work.")
            return None
    else:
        if not gemini_api_key:
            print(f"⚠️  No Gemini API key provided")
        if not GEMINI_AVAILABLE:
            print(f"⚠️  Gemini package not available. Install with: pip install google-generativeai")
        print(f"❌ No LLM available. LLM queries will not work.")
        return None


def _load_config() -> Optional[dict]:
    """Load configuration from Config.yaml."""
    try:
        config_paths = [
            Path("Config.yaml"),
            Path(__file__).parent.parent / "Config.yaml",
            Path(__file__).parent.parent.parent / "Config.yaml",
        ]
        
        for config_path in config_paths:
            abs_path = config_path.resolve()
            if abs_path.exists():
                with open(abs_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
    
    return None

