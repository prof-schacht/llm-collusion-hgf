#!/usr/bin/env python3
"""Quick test of the fixed implementation"""

from market_env import MarketCollusionEnv
from llm_agents import LLMAgent
import numpy as np

# Create environment
env = MarketCollusionEnv(n_agents=2, max_rounds=3)

# Create agents with fallback behavior for testing
agents = [
    LLMAgent(f"seller_{i}", model="ollama/qwen3:8b", personality="profit_maximizer")
    for i in range(2)
]

print("Testing HGF with Ollama (using fallback for quick test)...")
print("="*50)

# Reset environment
obs, info = env.reset()

# Run a few rounds
for round_num in range(3):
    print(f"\nRound {round_num + 1}:")
    
    for agent_idx, agent_name in enumerate(env.agents):
        obs, reward, done, truncated, info = env.last()
        
        if done or truncated:
            break
            
        # For quick testing, use fallback action
        agent = agents[agent_idx]
        action = agent.fallback_action(obs)
        
        print(f"  {agent_name}: Price=${action['price'][0]:.2f}")
        
        env.step(action)

print("\n✅ Environment and agent setup working correctly!")
print("\nTo test with actual LLM responses:")
print("1. Ensure Ollama is running: ollama serve")
print("2. Run: python run_experiment.py --model ollama/qwen3:8b --episodes 5 --no-wandb")
print("3. Or use the dashboard: streamlit run dashboard.py")