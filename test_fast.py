#!/usr/bin/env python3
"""Fast test configuration for Ollama"""

import os
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

from market_env import MarketCollusionEnv
from llm_agents import LLMAgent
from safety_wrapper import HierarchicalSafetyWrapper

print("Running FAST experiment (3 rounds, 1 episode)...")
print("="*60)

# Create environment with minimal rounds
env = MarketCollusionEnv(n_agents=2, max_rounds=3)

# Create agents
agents = [
    LLMAgent(f"seller_{i}", model="ollama/qwen3:8b", personality="profit_maximizer")
    for i in range(2)
]

# Reset
obs, info = env.reset()

# Run 3 rounds
round_num = 0
for step, agent_name in enumerate(env.agent_iter()):
    obs, reward, done, truncated, info = env.last()
    
    if done or truncated:
        break
        
    # Get agent action
    agent_idx = int(agent_name.split("_")[1])
    agent = agents[agent_idx]
    
    print(f"\nStep {step+1} - {agent_name} deciding...")
    action = agent.act(obs)
    
    price = action["price"][0]
    print(f"  → Price: ${price:.2f}")
    if action.get("message"):
        print(f"  → Message: '{action['message']}'")
    
    env.step(action)
    
    # Print round summary
    if (step + 1) % 2 == 0:
        round_num += 1
        if env.price_history:
            avg_price = env.price_history[-1]["avg_price"]
            print(f"\nRound {round_num} complete - Average price: ${avg_price:.2f}")

print("\n" + "="*60)
print("TEST COMPLETE!")
print(f"Final average price: ${env.price_history[-1]['avg_price']:.2f}" if env.price_history else "No prices recorded")
print("="*60)