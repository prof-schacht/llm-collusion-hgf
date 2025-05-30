#!/usr/bin/env python3
"""Test dynamic pricing improvements"""

import os
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from market_env import MarketCollusionEnv
from llm_agents import create_diverse_agents
from safety_wrapper import HierarchicalSafetyWrapper

print("Testing improved dynamic pricing...")
print("="*60)

# Create environment
env = MarketCollusionEnv(n_agents=2, max_rounds=8)

# Create diverse agents
agents = create_diverse_agents(n_agents=2, model="ollama/qwen3:8b")
print(f"Agent personalities: {[a.personality for a in agents]}")

# Reset
obs, info = env.reset()
print(f"\nInitial prices: seller_0=${obs['last_prices'][0]:.2f}, seller_1=${obs['last_prices'][1]:.2f}")

# Run episode
print("\nRunning episode:")
print("-" * 40)

for step, agent_name in enumerate(env.agent_iter()):
    obs, reward, done, truncated, info = env.last()
    
    if done or truncated:
        break
        
    # Get agent action
    agent_idx = int(agent_name.split("_")[1])
    agent = agents[agent_idx]
    
    action = agent.act(obs)
    
    price = action["price"][0]
    print(f"\nRound {obs['round']+1} - {agent_name} ({agent.personality}):")
    print(f"  Price: ${price:.2f}")
    if action.get("message"):
        print(f"  Message: '{action['message']}'")
    
    env.step(action)
    
    # Show market state after both agents act
    if (step + 1) % 2 == 0 and env.price_history:
        last_round = env.price_history[-1]
        print(f"\n  Market state:")
        print(f"    Average price: ${last_round['avg_price']:.2f}")
        print(f"    Price spread: ${max(last_round['prices'].values()) - min(last_round['prices'].values()):.2f}")
        print(f"    Profits: {', '.join(f'{k}=${v:.2f}' for k,v in last_round['profits'].items())}")

print("\n" + "="*60)
print("Price evolution:")
for i, round_data in enumerate(env.price_history):
    prices = round_data['prices']
    print(f"Round {i+1}: seller_0=${prices['seller_0']:.2f}, seller_1=${prices['seller_1']:.2f}, avg=${round_data['avg_price']:.2f}")

# Calculate price variance
all_prices = []
for round_data in env.price_history:
    all_prices.extend(round_data['prices'].values())
    
import numpy as np
price_variance = np.var(all_prices)
print(f"\nOverall price variance: {price_variance:.3f}")
print("(Higher variance = more dynamic pricing)")