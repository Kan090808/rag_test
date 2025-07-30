#!/usr/bin/env python3
"""
Test script to verify that RAG Agent uses all config values properly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import config and RAG agent
    import config
    from rag_agent import RAGAgent
    
    print("Testing config integration for all parameters...")
    print("=" * 50)
    
    print("Config values:")
    print(f"  MODEL_NAME: {config.MODEL_NAME}")
    print(f"  MAX_TOKENS: {config.MAX_TOKENS}")
    print(f"  TEMPERATURE: {config.TEMPERATURE}")
    print(f"  TOP_K_DOCUMENTS: {config.TOP_K_DOCUMENTS}")
    print(f"  LOG_FILE: {config.LOG_FILE}")
    print()
    
    # Test 1: Create RAG agent with explicit parameters (should override config)
    print("Test 1: Creating RAG agent with explicit parameters...")
    rag_agent_explicit = RAGAgent(
        openrouter_api_key="test-key",
        model_name="custom-model",
        max_tokens=500,
        temperature=0.5,
        log_file="custom.log",
        system_prompt="Custom prompt"
    )
    
    print(f"  Model name: {rag_agent_explicit.model_name}")
    print(f"  Max tokens: {rag_agent_explicit.max_tokens}")
    print(f"  Temperature: {rag_agent_explicit.temperature}")
    print(f"  Top K: {rag_agent_explicit.default_top_k}")
    print(f"  Log file: {rag_agent_explicit.log_file}")
    print()
    
    # Test 2: Create RAG agent with default parameters (should use config)
    print("Test 2: Creating RAG agent with default parameters (using config)...")
    rag_agent_config = RAGAgent(openrouter_api_key="test-key")
    
    print(f"  Model name: {rag_agent_config.model_name}")
    print(f"  Max tokens: {rag_agent_config.max_tokens}")
    print(f"  Temperature: {rag_agent_config.temperature}")
    print(f"  Top K: {rag_agent_config.default_top_k}")
    print(f"  Log file: {rag_agent_config.log_file}")
    print()
    
    # Verify config values are used correctly
    success = True
    errors = []
    
    if rag_agent_config.model_name != config.MODEL_NAME:
        errors.append(f"Model name mismatch: expected {config.MODEL_NAME}, got {rag_agent_config.model_name}")
        success = False
    
    if rag_agent_config.max_tokens != config.MAX_TOKENS:
        errors.append(f"Max tokens mismatch: expected {config.MAX_TOKENS}, got {rag_agent_config.max_tokens}")
        success = False
    
    if rag_agent_config.temperature != config.TEMPERATURE:
        errors.append(f"Temperature mismatch: expected {config.TEMPERATURE}, got {rag_agent_config.temperature}")
        success = False
    
    if rag_agent_config.default_top_k != config.TOP_K_DOCUMENTS:
        errors.append(f"Top K mismatch: expected {config.TOP_K_DOCUMENTS}, got {rag_agent_config.default_top_k}")
        success = False
    
    if rag_agent_config.log_file != config.LOG_FILE:
        errors.append(f"Log file mismatch: expected {config.LOG_FILE}, got {rag_agent_config.log_file}")
        success = False
    
    print("=" * 50)
    if success:
        print("✅ SUCCESS: All config values are being used correctly!")
    else:
        print("❌ ERRORS found:")
        for error in errors:
            print(f"  - {error}")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
