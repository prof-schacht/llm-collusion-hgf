#!/usr/bin/env python3
"""Demo of dynamic pricing behavior"""

import numpy as np
from market_env import MarketCollusionEnv
import matplotlib.pyplot as plt

print("Demonstrating dynamic pricing behavior...")
print("="*60)

# Create environment
env = MarketCollusionEnv(n_agents=2, max_rounds=10)
obs, info = env.reset()

# Simulate dynamic pricing strategies
strategies = {
    "seller_0": "adaptive",  # Starts low, gradually increases
    "seller_1": "aggressive"  # Starts high, reacts to competition
}

# Price history for each agent
price_history = {
    "seller_0": [],
    "seller_1": []
}

print("\nSimulating market dynamics:")
print("-" * 40)

for round_num in range(10):
    for agent_idx, agent_name in enumerate(["seller_0", "seller_1"]):
        obs, reward, done, truncated, info = env.last()
        
        if done or truncated:
            break
            
        # Simulate different pricing strategies
        if agent_name == "seller_0":
            # Adaptive: Start low, gradually increase if profitable
            if round_num < 3:
                price = np.random.uniform(4.0, 5.5)
            elif round_num < 6:
                # Try to cooperate
                price = min(obs['last_prices'][0] + 0.5, 7.0)
            else:
                # Stabilize at profitable level
                price = np.random.uniform(6.5, 7.5)
        else:
            # Aggressive: Start high, react to competition
            if round_num < 2:
                price = np.random.uniform(7.0, 8.0)
            elif obs['last_prices'][0] < 5.0:
                # Match low prices
                price = obs['last_prices'][0] + np.random.uniform(-0.2, 0.2)
            else:
                # Try to maintain high prices
                price = np.random.uniform(6.0, 7.5)
        
        action = {
            "price": np.array([price], dtype=np.float32),
            "message": ""
        }
        
        price_history[agent_name].append(price)
        env.step(action)
    
    if env.price_history:
        last_round = env.price_history[-1]
        print(f"\nRound {round_num + 1}:")
        print(f"  Prices: seller_0=${last_round['prices']['seller_0']:.2f}, seller_1=${last_round['prices']['seller_1']:.2f}")
        print(f"  Profits: seller_0=${last_round['profits']['seller_0']:.2f}, seller_1=${last_round['profits']['seller_1']:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), price_history["seller_0"], 'b-o', label="Seller 0 (Adaptive)")
plt.plot(range(1, 11), price_history["seller_1"], 'r-s', label="Seller 1 (Aggressive)")
plt.axhline(y=3.0, color='gray', linestyle='--', label="Cost")
plt.axhline(y=4.0, color='green', linestyle='--', alpha=0.5, label="Competitive Price")
plt.xlabel("Round")
plt.ylabel("Price ($)")
plt.title("Dynamic Pricing Behavior Example")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(2, 9)

# Save plot
plt.savefig("dynamic_pricing_example.png")
print(f"\n✅ Plot saved as 'dynamic_pricing_example.png'")

# Calculate metrics
price_variance = np.var(price_history["seller_0"] + price_history["seller_1"])
avg_price = np.mean(price_history["seller_0"] + price_history["seller_1"])
print(f"\nMetrics:")
print(f"  Average price: ${avg_price:.2f}")
print(f"  Price variance: {price_variance:.3f}")
print(f"  Price range: ${min(price_history['seller_0'] + price_history['seller_1']):.2f} - ${max(price_history['seller_0'] + price_history['seller_1']):.2f}")