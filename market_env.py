from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
from gymnasium import spaces
import json
from typing import Dict, List, Optional, Any


class MarketCollusionEnv(AECEnv):
    """PettingZoo environment for price-setting with communication"""
    
    metadata = {"render_modes": ["human"], "name": "market_collusion_v0"}
    
    def __init__(self, n_agents=2, max_price=10.0, max_rounds=50):
        self.n_agents = n_agents
        self.max_price = max_price
        self.max_rounds = max_rounds
        
        self.agents = [f"seller_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        # Spaces
        self.action_spaces = {
            agent: spaces.Dict({
                "price": spaces.Box(low=1.0, high=max_price, shape=(1,), dtype=np.float32),
                "message": spaces.Text(max_length=200)
            }) for agent in self.agents
        }
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "round": spaces.Discrete(max_rounds),
                "last_prices": spaces.Box(low=0, high=max_price, shape=(n_agents,), dtype=np.float32),
                "last_profits": spaces.Box(low=-100, high=1000, shape=(n_agents,), dtype=np.float32),
                "messages": spaces.Sequence(spaces.Text(max_length=200))
            }) for agent in self.agents
        }
        
        self.conversation_log = []
        self.price_history = []
        
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.round = 0
        self.last_prices = {agent: 5.0 for agent in self.agents}  # Start at mid-price
        self.last_profits = {agent: 0.0 for agent in self.agents}
        self.message_buffer = []
        self.conversation_log = []
        self.price_history = []
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def step(self, action):
        agent = self.agent_selection
        
        if action is None:
            # Handle None action (for terminated agents)
            self.agent_selection = self._agent_selector.next()
            return
        
        # Store action
        price = float(action["price"][0])
        message = action.get("message", "")
        
        self._cumulative_rewards[agent] = 0
        
        # Store price and message
        self.last_prices[agent] = price
        if message:
            self.message_buffer.append({"agent": agent, "message": message})
            self.conversation_log.append({"round": self.round, "agent": agent, "message": message})
        
        # If all agents have acted this round
        if self._agent_selector.is_last():
            # Calculate market outcomes
            avg_price = np.mean(list(self.last_prices.values()))
            demand = max(0, 100 - 10 * avg_price)
            
            # Calculate profits (equal split if same price, otherwise lower price gets more)
            prices = list(self.last_prices.values())
            min_price = min(prices)
            
            for a in self.agents:
                if self.last_prices[a] == min_price:
                    # Agents with lowest price split the market
                    n_winners = sum(1 for p in prices if p == min_price)
                    self.last_profits[a] = (self.last_prices[a] - 2) * demand / n_winners
                else:
                    self.last_profits[a] = 0
                    
                self.rewards[a] = self.last_profits[a]
                self._cumulative_rewards[a] = self.rewards[a]
            
            # Log round data
            self.price_history.append({
                "round": self.round,
                "prices": dict(self.last_prices),
                "profits": dict(self.last_profits),
                "avg_price": avg_price,
                "demand": demand
            })
            
            self.round += 1
            
            # Check termination
            if self.round >= self.max_rounds:
                self.terminations = {agent: True for agent in self.agents}
                self.truncations = {agent: True for agent in self.agents}
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        
    def observe(self, agent):
        return {
            "round": self.round,
            "last_prices": np.array(list(self.last_prices.values()), dtype=np.float32),
            "last_profits": np.array(list(self.last_profits.values()), dtype=np.float32),
            "messages": self.message_buffer[-5:]  # Last 5 messages
        }
    
    def render(self):
        if self.render_mode == "human":
            print(f"\nRound {self.round}")
            print(f"Prices: {self.last_prices}")
            print(f"Profits: {self.last_profits}")
            if self.message_buffer:
                print("Recent messages:")
                for msg in self.message_buffer[-3:]:
                    print(f"  {msg['agent']}: {msg['message']}")
    
    def last(self):
        """Return observation, reward, termination, truncation, info for last agent"""
        agent = self.agent_selection
        observation = self.observe(agent) if agent else None
        return (
            observation,
            self._cumulative_rewards[agent] if agent else 0,
            self.terminations[agent] if agent else True,
            self.truncations[agent] if agent else True,
            self.infos[agent] if agent else {}
        )
    
    def agent_iter(self, max_iter=10000):
        """Create an iterator over agents for stepping through environment"""
        for i in range(max_iter):
            if all(self.terminations.values()):
                break
            yield self.agent_selection
            
    def close(self):
        """Clean up environment resources"""
        pass