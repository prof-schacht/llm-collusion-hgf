#!/usr/bin/env python3
"""Debug safety framework detection"""

import os
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

import logging
# Enable debug logging for safety wrapper
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from market_env import MarketCollusionEnv
from safety_wrapper import HierarchicalSafetyWrapper
import numpy as np

print("Testing safety framework with debug output...")
print("="*60)

# Create base environment
base_env = MarketCollusionEnv(n_agents=2, max_rounds=5)

# Wrap with safety
safe_env = HierarchicalSafetyWrapper(base_env)

# Reset
obs, info = safe_env.reset()

# Run a few rounds with high prices
for step, agent_name in enumerate(safe_env.agent_iter()):
    obs, reward, done, truncated, info = safe_env.last()
    
    if done or truncated:
        break
    
    # Set high prices to trigger detection
    high_price = np.random.uniform(7.0, 9.0)
    action = {
        "price": np.array([high_price], dtype=np.float32),
        "message": "Let's maintain high prices" if step % 3 == 0 else ""
    }
    
    print(f"\n{agent_name} setting price: ${high_price:.2f}")
    safe_env.step(action)
    
    if "safety_intervention" in info:
        print(f"🛡️ INTERVENTION: {info['safety_intervention']}")

print("\n" + "="*60)
print(f"Interventions logged: {len(safe_env.intervention_log)}")
print(f"Final prices: {safe_env.last_prices}")