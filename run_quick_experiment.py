#!/usr/bin/env python3
"""Quick experiment runner for testing with Ollama"""

import os
import sys
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

# Suppress verbose logging
import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from run_experiment import run_comparison_experiment

# Quick configuration
config = {
    "n_episodes": 2,  # Just 2 episodes
    "llm_model": "ollama/qwen3:8b",
    "use_wandb": False
}

# Override default rounds
from market_env import MarketCollusionEnv
MarketCollusionEnv.__init__.__defaults__ = (2, 10.0, 3)  # 3 rounds instead of 10

print("Running quick experiment (2 episodes, 3 rounds each)...")
print("This should take about 30-60 seconds with Ollama\n")

results = run_comparison_experiment(config)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
print(f"\nBaseline (no safety):")
print(f"  Average price: ${results['baseline']['avg_final_price']:.2f}")
print(f"  Messages sent: {results['baseline']['total_messages']}")

print(f"\nWith HGF Safety:")
print(f"  Average price: ${results['with_safety']['avg_final_price']:.2f}")
print(f"  Messages sent: {results['with_safety']['total_messages']}")
print(f"  Interventions: {results['with_safety']['intervention_rate']:.1f} per episode")

price_reduction = results['baseline']['avg_final_price'] - results['with_safety']['avg_final_price']
print(f"\n✅ Safety framework reduced prices by ${price_reduction:.2f}")