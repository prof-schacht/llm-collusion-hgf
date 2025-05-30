#!/usr/bin/env python3
"""
Quick test with Fireworks to debug connection and response times
"""

import os
import time
import litellm
from dotenv import load_dotenv

load_dotenv()

def test_fireworks_connection():
    """Test basic Fireworks AI connection"""
    print("Testing Fireworks AI Connection")
    print("="*40)
    
    model = os.getenv('FIREWORKS_MODEL')
    print(f"Model: {model}")
    print(f"API Key: {'Set' if os.getenv('FIREWORKS_AI_API_KEY') else 'Not set'}")
    
    # Test 1: Simple completion
    print("\nTest 1: Simple JSON response...")
    start = time.time()
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                {"role": "user", "content": "Return this exact JSON: {\"price\": 5.0}"}
            ],
            temperature=0.1,
            max_tokens=50,
            timeout=30
        )
        
        elapsed = time.time() - start
        print(f"✓ Response received in {elapsed:.1f} seconds")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test 2: Trading decision
    print("\nTest 2: Trading decision...")
    start = time.time()
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a trader. Respond only with JSON."},
                {"role": "user", "content": """Current prices: {"trader_0": 5.0}
Choose your price (1-10). Return ONLY: {"action": "set_price", "price": 5.5}"""}
            ],
            temperature=0.1,
            max_tokens=50,
            timeout=30
        )
        
        elapsed = time.time() - start
        print(f"✓ Response received in {elapsed:.1f} seconds")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        
    print("\n" + "="*40)
    print("Connection test complete!")
    print("\nIf responses are slow, consider:")
    print("- Using a smaller model")
    print("- Reducing max_tokens")
    print("- Simplifying prompts")

if __name__ == "__main__":
    test_fireworks_connection()