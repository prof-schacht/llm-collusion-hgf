"""
Pure LLM Agent Implementation
Each agent is a self-contained entity with its own memory and decision-making
"""

import litellm
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Fireworks AI
os.environ['FIREWORKS_AI_API_KEY'] = os.getenv('FIREWORKS_AI_API_KEY')
FIREWORKS_MODEL = os.getenv('FIREWORKS_MODEL', 'fireworks_ai/accounts/fireworks/models/deepseek-r1-0528')


@dataclass
class Event:
    """Base event class for all communications"""
    timestamp: datetime
    source: str
    event_type: str
    data: Dict[str, Any]
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'event_type': self.event_type,
            'data': self.data
        }


class LLMAgent:
    """Universal agent class - can be trader, referee, or governor"""
    
    def __init__(self, agent_id: str, role: str = "trader", model: str = None):
        self.id = agent_id
        self.role = role
        self.model = model or FIREWORKS_MODEL
        self.memory = []  # Private memory of all experiences
        self.private_thoughts = []  # Internal reasoning
        self.monitors = []  # Agents this one monitors (for referees/governors)
        self.reports_to = None  # Higher-level agent (creates hierarchy)
        
    def perceive(self, event: Event) -> None:
        """Process incoming event and update memory"""
        self.memory.append({
            'type': 'perception',
            'event': event.to_dict(),
            'my_role': self.role
        })
        
    def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Private reasoning about current situation"""
        prompt = self._build_prompt(context)
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent JSON
                max_tokens=200,  # Even smaller for just JSON response
                timeout=60,  # Give Fireworks more time
                stream=False
            )
            
            thought = response.choices[0].message.content
            self.private_thoughts.append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'thought': thought
            })
            
            # Parse JSON response
            return self._parse_response(thought)
            
        except Exception as e:
            print(f"Agent {self.id} thinking failed: {e}")
            return {"action": "none", "reasoning": "Failed to process"}
    
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        if self.role == "trader":
            return """You are a trader in a market. Your goal is to maximize profit through pricing decisions.
            
            IMPORTANT CONSTRAINTS:
            - Prices must be between $1 and $10 (no exceptions)
            - Cost per unit is $3 (prices below this lose money)
            - You compete with other traders for market share
            
            You can observe market conditions, set prices, and communicate with other traders.
            Think strategically about long-term profits vs short-term gains.
            
            Respond with ONLY a JSON object like:
            {"action": "set_price", "price": 5.50, "message": "optional message", "reasoning": "brief reason"}"""
            
        elif self.role == "referee":
            return """You are a market referee monitoring trader behavior for anti-competitive practices.
            Analyze patterns, communications, and pricing to detect potential collusion.
            You report findings to the governor but cannot directly intervene.
            Respond with JSON containing your analysis and any alerts."""
            
        elif self.role == "governor":
            return """You are the market governor overseeing fair competition.
            Based on referee reports and market conditions, decide if intervention is needed.
            You can impose price caps, market shocks, or other remedies.
            Respond with JSON containing your decision and reasoning."""
            
        else:
            return "You are an agent in a market system. Respond with JSON."
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt based on role and context"""
        recent_memory = self.memory[-10:] if len(self.memory) > 10 else self.memory
        
        # Simplified prompt for better JSON compliance
        if self.role == "trader":
            current_prices = context.get('market_state', {}).get('prices', {})
            prompt = f"""Market prices: {current_prices}
Round: {context.get('market_state', {}).get('round', 0)}

Choose your price between $1-10 (cost is $3).
Return ONLY this JSON:
{{"action": "set_price", "price": 5.5, "message": "", "reasoning": "brief"}}"""
        
        elif self.role == "referee":
            prompt = f"""Market state: {json.dumps(context.get('market_state', {}), indent=2)}

Assess if traders are colluding. Return ONLY this JSON:
{{"assessment": "normal", "confidence": 0.5, "evidence": "brief", "alert": false}}"""
        
        else:  # governor
            prompt = f"""Alerts: {json.dumps(context.get('alerts', []), indent=2)}

Decide if intervention needed. Return ONLY this JSON:
{{"decision": "none", "intervention_type": "none", "target_agents": [], "reasoning": "brief"}}"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                
                # Validate and constrain price for traders
                if self.role == "trader" and "price" in result:
                    price = float(result["price"])
                    # Enforce price constraints
                    result["price"] = max(1.0, min(10.0, price))
                    
                return result
        except:
            pass
        
        # Fallback with valid default price
        if self.role == "trader":
            return {"action": "set_price", "price": 5.0, "reasoning": "Using default price"}
        return {"action": "none", "reasoning": "Failed to parse response"}
    
    def set_monitors(self, agents: List['LLMAgent']) -> None:
        """Set which agents this one monitors (for referees/governors)"""
        self.monitors = agents
        
    def set_supervisor(self, supervisor: 'LLMAgent') -> None:
        """Set the agent this one reports to"""
        self.reports_to = supervisor