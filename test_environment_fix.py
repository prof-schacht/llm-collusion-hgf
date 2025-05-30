#!/usr/bin/env python3
"""Quick test to verify environment fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_env import MarketCollusionEnv
from llm_agents import LLMAgent

def test_environment():
    print("Testing MarketCollusionEnv fixes...")
    
    # Test 1: Create environment
    print("\n1. Creating environment...")
    env = MarketCollusionEnv(n_agents=2, max_rounds=5)
    print("   ✓ Environment created")
    
    # Test 2: Reset environment
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print("   ✓ Reset successful")
    print(f"   - Initial observation keys: {list(obs.keys())}")
    print(f"   - Round: {obs['round']}")
    print(f"   - Last prices: {obs['last_prices']}")
    
    # Test 3: Test agent iteration
    print("\n3. Testing agent iteration...")
    for i, agent in enumerate(env.agent_iter()):
        if i >= 2:  # Just test first two iterations
            break
        obs, reward, done, truncated, info = env.last()
        print(f"   ✓ Agent {agent} observation received")
        
        # Test action
        action = {
            "price": [5.0 + i],
            "message": f"Test message from {agent}"
        }
        env.step(action)
        print(f"   ✓ Action processed for {agent}")
    
    print("\n✅ All tests passed!")
    
    # Test 4: Ollama config
    print("\n4. Testing Ollama configuration...")
    print(f"   - OLLAMA_API_BASE: {os.environ.get('OLLAMA_API_BASE', 'Not set')}")
    
    # Create a test agent
    try:
        agent = LLMAgent("test_agent", model="ollama/qwen2.5:3b")
        print("   ✓ LLM Agent created with Ollama model")
    except Exception as e:
        print(f"   ⚠ Note: {e}")

if __name__ == "__main__":
    test_environment()