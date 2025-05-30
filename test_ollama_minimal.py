#!/usr/bin/env python3
"""Minimal test for Ollama integration"""

import os
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

import litellm
import json

# Test basic Ollama completion
print("Testing Ollama with qwen3:8b...")

try:
    response = litellm.completion(
        model="ollama/qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
            {"role": "user", "content": 'Respond with JSON: {"number": 5}'}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Test pricing scenario
    print("\nTesting pricing scenario...")
    
    response = litellm.completion(
        model="ollama/qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a pricing agent. Respond only with a JSON object containing a price between 1 and 10."},
            {"role": "user", "content": "Set a price. Respond with: {\"price\": X}"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure Ollama is running and qwen3:8b is installed:")
    print("  ollama serve")
    print("  ollama pull qwen3:8b")