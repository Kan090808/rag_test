#!/usr/bin/env python3
"""
Test script to verify that RAG Agent uses config values properly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import config and RAG agent
    import config
    from rag_agent import RAGAgent
    
    print("Testing config integration...")
    print(f"Config TOP_K_DOCUMENTS: {config.TOP_K_DOCUMENTS}")
    
    # Create RAG agent instance
    rag_agent = RAGAgent(
        openrouter_api_key="test-key",  # Using dummy key for testing
        model_name=config.MODEL_NAME,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        log_file=config.LOG_FILE,
        system_prompt=config.SYSTEM_PROMPT
    )
    
    print(f"RAG Agent default_top_k: {rag_agent.default_top_k}")
    
    # Verify the values match
    if rag_agent.default_top_k == config.TOP_K_DOCUMENTS:
        print("✅ SUCCESS: RAG Agent is using config TOP_K_DOCUMENTS value correctly!")
    else:
        print(f"❌ ERROR: Expected {config.TOP_K_DOCUMENTS}, got {rag_agent.default_top_k}")
    
    print(f"Model name: {rag_agent.model_name}")
    print(f"Max tokens: {rag_agent.max_tokens}")
    print(f"Temperature: {rag_agent.temperature}")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
