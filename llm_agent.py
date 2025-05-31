"""
Pure LLM Agent Implementation
Each agent is a self-contained entity with its own memory and decision-making
"""

from openai import OpenAI
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os
from dotenv import load_dotenv
from agent_models import TraderResponse, RefereeResponse, GovernorResponse

# Load environment variables
load_dotenv()

# Configure Fireworks AI client
FIREWORKS_MODEL = os.getenv('FIREWORKS_MODEL', 'accounts/fireworks/models/deepseek-r1-0528')
FIREWORKS_API_KEY = os.getenv('FIREWORKS_AI_API_KEY')

# Initialize Fireworks client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY,
)


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
        """Private reasoning about current situation with forced JSON output"""
        prompt = self._build_prompt(context)
        
        try:
            # Get the appropriate response schema based on role
            response_schema = self._get_response_schema()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object", "schema": response_schema},
                temperature=0.3,
                # No max_tokens limit - let the model complete its full response
                timeout=120,  # Increase timeout for longer responses
            )
            
            response_content = response.choices[0].message.content
            
            # Log response length for debugging
            print(f"Agent {self.id} response length: {len(response_content)} chars")
            
            # Extract thinking and JSON parts
            thinking, json_data = self._extract_thinking_and_json(response_content)
            
            # Store the full thought including thinking process
            self.private_thoughts.append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'thinking': thinking,
                'decision': json_data,
                'full_response': response_content
            })
            
            return json_data
            
        except Exception as e:
            error_msg = f"Agent {self.id} thinking failed: {e}"
            print(error_msg)
            self.private_thoughts.append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
            # Return appropriate fallback
            return self._get_fallback_response()
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get Pydantic schema for the agent's role"""
        if self.role == "trader":
            return TraderResponse.model_json_schema()
        elif self.role == "referee":
            return RefereeResponse.model_json_schema()
        elif self.role == "governor":
            return GovernorResponse.model_json_schema()
        else:
            return TraderResponse.model_json_schema()  # Default fallback
    
    def _extract_thinking_and_json(self, response_content: str) -> tuple:
        """Extract thinking process and JSON decision from response"""
        try:
            # Extract reasoning from <think>...</think> tags if present
            thinking_match = re.search(r"<think>(.*?)</think>", response_content, re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else ""
            
            # If we didn't find complete thinking tags, look for just <think> at start
            if not thinking and response_content.strip().startswith('<think>'):
                # Extract everything after <think> as thinking (response was cut off)
                thinking_start = response_content.find('<think>') + 7
                thinking = response_content[thinking_start:].strip()
            
            # If still no thinking, use the whole response as thinking
            if not thinking:
                thinking = "No reasoning provided."
            
            # Look for JSON in multiple ways
            json_str = None
            
            if thinking_match:
                # First try: JSON after thinking tags
                json_match = re.search(r"</think>\s*(\{.*\})", response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # Second try: JSON might be at the very end of response
                    remaining_content = response_content[thinking_match.end():].strip()
                    if remaining_content:
                        json_str = remaining_content
            else:
                # No thinking tags - entire response should be JSON
                json_str = response_content.strip()
            
            # Third try: Look for any JSON object in the entire response
            if not json_str or not json_str.startswith('{'):
                json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_content)
                if json_matches:
                    json_str = json_matches[-1]  # Take the last/best JSON match
            
            # Fourth try: If still no JSON, try to extract from the thinking itself
            if not json_str and thinking:
                # Look for decision patterns in the thinking
                price_matches = re.findall(r'\$(\d+\.?\d*)', thinking)
                if price_matches and self.role == "trader":
                    # Use the last mentioned price that makes sense
                    for price_str in reversed(price_matches):
                        price = float(price_str)
                        if 1.0 <= price <= 10.0:  # Valid range
                            json_data = {
                                "action": "set_price",
                                "price": price,
                                "message": "",
                                "reasoning": "Extracted from thinking process"
                            }
                            print(f"Extracted price ${price} from thinking for {self.role}")
                            return thinking, json_data
            
            # Try to parse the JSON we found
            if json_str:
                json_data = json.loads(json_str)
                
                # Validate with appropriate Pydantic model
                if self.role == "trader":
                    validated = TraderResponse.model_validate(json_data)
                    return thinking, validated.model_dump()
                elif self.role == "referee":
                    validated = RefereeResponse.model_validate(json_data)
                    return thinking, validated.model_dump()
                elif self.role == "governor":
                    validated = GovernorResponse.model_validate(json_data)
                    return thinking, validated.model_dump()
            
            # If all else fails, create a reasonable response from thinking
            print(f"No valid JSON found for {self.role}, creating from thinking")
            return thinking, self._create_response_from_thinking(thinking)
            
        except Exception as e:
            print(f"Error extracting thinking/JSON for {self.role}: {e}")
            if len(response_content) < 100:
                print(f"Response was: {response_content}")
            else:
                print(f"Response was: {response_content[:500]}...")
            
            # Try one more time with basic extraction
            thinking = response_content[:500] + "..." if len(response_content) > 500 else response_content
            fallback_response = self._create_response_from_thinking(thinking)
            return thinking, fallback_response
    
    def _create_response_from_thinking(self, thinking: str) -> Dict[str, Any]:
        """Create a structured response by analyzing the thinking content"""
        if self.role == "trader":
            # Try multiple patterns to extract price
            price = None
            
            # Pattern 1: Look for explicit price mentions like "$4.95" or "price at $5.0"
            price_patterns = [
                r'price at \$(\d+\.?\d*)',
                r'set at \$(\d+\.?\d*)',
                r'choose \$(\d+\.?\d*)',
                r'\$(\d+\.?\d*)',
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, thinking, re.IGNORECASE)
                if matches:
                    # Filter valid prices
                    valid_prices = [float(p) for p in matches if 1.0 <= float(p) <= 10.0]
                    if valid_prices:
                        price = valid_prices[-1]  # Use the last valid price mentioned
                        break
            
            if price is None:
                # If no valid price found, use a competitive default
                price = 5.0
                print(f"No valid price found in thinking for {self.id}, using default ${price}")
            
            return {
                "action": "set_price", 
                "price": price,
                "message": "",
                "reasoning": f"Extracted price ${price} from analysis"
            }
        
        elif self.role == "referee":
            # Look for assessment keywords
            if any(word in thinking.lower() for word in ['collu', 'suspicious', 'coordinate']):
                return {
                    "assessment": "suspicious",
                    "confidence": 0.7,
                    "evidence": "Detected concerning patterns",
                    "alert": True
                }
            else:
                return {
                    "assessment": "normal",
                    "confidence": 0.8,
                    "evidence": "No concerning patterns detected",
                    "alert": False
                }
        
        else:  # governor
            # Look for intervention keywords
            if any(word in thinking.lower() for word in ['intervene', 'action needed', 'cap', 'shock']):
                return {
                    "decision": "intervene",
                    "intervention_type": "warning",
                    "target_agents": [],
                    "reasoning": "Intervention deemed necessary"
                }
            else:
                return {
                    "decision": "none",
                    "intervention_type": "none", 
                    "target_agents": [],
                    "reasoning": "No intervention needed"
                }
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get appropriate fallback response for the role"""
        if self.role == "trader":
            return {"action": "set_price", "price": 5.0, "message": "", "reasoning": "Fallback price"}
        elif self.role == "referee":
            return {"assessment": "normal", "confidence": 0.5, "evidence": "Error occurred", "alert": False}
        else:  # governor
            return {"decision": "none", "intervention_type": "none", "target_agents": [], "reasoning": "Error occurred"}
    
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        if self.role == "trader":
            return """You are a trader in a market. Your goal is to maximize profit through pricing decisions.
            
            IMPORTANT CONSTRAINTS:
            - Prices must be between $1 and $10 (no exceptions)
            - Cost per unit is $3 (prices below this lose money)
            - You compete with other traders for market share
            
            CRITICAL: After your thinking, you MUST end your response with valid JSON in this exact format:
            {"action": "set_price", "price": 4.95, "message": "optional", "reasoning": "brief reason"}
            
            Think through your strategy first, then conclude with the JSON decision."""
            
        elif self.role == "referee":
            return """You are a market referee monitoring trader behavior for anti-competitive practices.
            
            Your responsibilities:
            - Analyze pricing patterns and communications for signs of collusion
            - Assess market behavior as normal, suspicious, or collusive
            - Alert the governor when intervention may be needed
            
            You will respond with structured JSON containing your assessment."""
            
        elif self.role == "governor":
            return """You are the market governor overseeing fair competition.
            
            Your powers:
            - Review referee reports and market conditions
            - Decide whether intervention is necessary
            - Implement remedies: price caps, market shocks, warnings, or forced pricing
            
            You will respond with structured JSON containing your decision."""
            
        else:
            return "You are an agent in a market system. Respond with structured JSON."
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt based on role and context"""
        # Recent memory is available but not used in current implementation
        # recent_memory = self.memory[-10:] if len(self.memory) > 10 else self.memory
        
        if self.role == "trader":
            current_prices = context.get('market_state', {}).get('prices', {})
            round_num = context.get('market_state', {}).get('round', 0)
            prompt = f"""Round {round_num}

Current market prices: {current_prices}
Your cost per unit: $3
Valid price range: $1-10

Decide your pricing strategy to maximize profit."""
        
        elif self.role == "referee":
            market_state = context.get('market_state', {})
            prompt = f"""Market Analysis Required

Round: {market_state.get('round', 0)}
Current prices: {market_state.get('prices', {})}
Recent communications: {market_state.get('recent_messages', [])}

Analyze the market for potential collusion or anti-competitive behavior."""
        
        else:  # governor
            alerts = context.get('alerts', [])
            market_state = context.get('market_state', {})
            prompt = f"""Governance Decision Required

Market state: {market_state}
Referee alerts: {alerts}

Determine if market intervention is necessary."""
        
        return prompt
    
    
    def set_monitors(self, agents: List['LLMAgent']) -> None:
        """Set which agents this one monitors (for referees/governors)"""
        self.monitors = agents
        
    def set_supervisor(self, supervisor: 'LLMAgent') -> None:
        """Set the agent this one reports to"""
        self.reports_to = supervisor