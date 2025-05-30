#!/usr/bin/env python3
"""Improved experiment runner with better prompts for dynamic pricing"""

import os
import sys
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

# Suppress verbose logging
import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from run_experiment import run_comparison_experiment
from market_env import MarketCollusionEnv

# Override default rounds
MarketCollusionEnv.__init__.__defaults__ = (2, 10.0, 10)  # 10 rounds for better dynamics

# Configuration
config = {
    "n_episodes": 5,
    "llm_model": "ollama/qwen3:8b",
    "use_wandb": False
}

print("Running improved experiment with dynamic pricing...")
print("This experiment uses:")
print("- Better prompts that encourage price exploration")
print("- Diverse agent personalities (adaptive vs profit_maximizer)")
print("- Smart fallback strategies when LLM fails")
print("- 10 rounds per episode to show price evolution")
print("\n" + "="*60 + "\n")

results = run_comparison_experiment(config)

print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)

# Detailed results
print(f"\nBaseline (no safety):")
print(f"  Average final price: ${results['baseline']['avg_final_price']:.2f}")
print(f"  Price range: ${results['baseline'].get('min_price', 0):.2f} - ${results['baseline'].get('max_price', 0):.2f}")
print(f"  Messages sent: {results['baseline']['total_messages']}")

# Calculate price variance from episodes
baseline_prices = []
for ep in results['baseline']['episodes']:
    for round_data in ep['prices']:
        baseline_prices.extend(round_data['prices'].values())

import numpy as np
baseline_variance = np.var(baseline_prices) if baseline_prices else 0

print(f"  Price variance: {baseline_variance:.3f}")

print(f"\nWith HGF Safety:")
print(f"  Average final price: ${results['with_safety']['avg_final_price']:.2f}")
print(f"  Messages sent: {results['with_safety']['total_messages']}")
print(f"  Interventions: {results['with_safety']['intervention_rate']:.1f} per episode")

# Calculate safety variance
safety_prices = []
for ep in results['with_safety']['episodes']:
    for round_data in ep['prices']:
        safety_prices.extend(round_data['prices'].values())

safety_variance = np.var(safety_prices) if safety_prices else 0
print(f"  Price variance: {safety_variance:.3f}")

price_reduction = results['baseline']['avg_final_price'] - results['with_safety']['avg_final_price']
print(f"\n✅ Safety framework reduced prices by ${price_reduction:.2f}")
print(f"✅ Baseline variance: {baseline_variance:.3f} (higher = more dynamic)")

# Save results
import json
with open("results/improved_experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/improved_experiment_results.json")