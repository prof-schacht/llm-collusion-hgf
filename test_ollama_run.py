#!/usr/bin/env python3
"""Test script for Ollama with verbose output"""

import os
import sys

# Set up environment
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'

from run_experiment import run_single_experiment

# Run a minimal experiment
print("Running minimal experiment with Ollama...")
print("="*60)

results = run_single_experiment(
    n_episodes=2,
    enable_safety=False,
    model="ollama/qwen3:8b",
    use_wandb=False
)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
print(f"Average final price: ${results['avg_final_price']:.2f}")
print(f"Total messages sent: {results['total_messages']}")
print(f"Episodes completed: {len(results['episodes'])}")

if results['episodes'] and results['episodes'][0]['prices']:
    print(f"\nFirst episode price evolution:")
    for i, price_data in enumerate(results['episodes'][0]['prices'][:5]):
        print(f"  Round {i+1}: ${price_data['avg_price']:.2f}")