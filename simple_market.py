"""
Minimal market environment - just tracks state
All dynamics determined by LLM agents
"""

from typing import Dict, List, Optional
from datetime import datetime
from event_bus import EventBus, Event, EventTypes
from llm_agent import LLMAgent


class SimpleMarket:
    """Minimal market that just tracks state - no rules"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.round = 0
        self.state = {
            'round': 0,
            'prices': {},
            'messages': [],
            'interventions': []
        }
        
    def update_price(self, agent_id: str, price: float) -> None:
        """Record a price update"""
        self.state['prices'][agent_id] = price
        
        # Post event
        self.event_bus.post(Event(
            timestamp=datetime.now(),
            source='market',
            event_type=EventTypes.PRICE_SET,
            data={'agent': agent_id, 'price': price, 'round': self.round}
        ))
        
    def add_message(self, from_agent: str, to_agent: str, message: str) -> None:
        """Record a message between agents"""
        msg_data = {
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'round': self.round,
            'timestamp': datetime.now().isoformat()
        }
        self.state['messages'].append(msg_data)
        
        # Post event
        self.event_bus.post(Event(
            timestamp=datetime.now(),
            source=from_agent,
            event_type=EventTypes.MESSAGE_SENT,
            data=msg_data
        ))
        
    def apply_intervention(self, intervention: Dict) -> None:
        """Record an intervention"""
        self.state['interventions'].append({
            'round': self.round,
            'intervention': intervention,
            'timestamp': datetime.now().isoformat()
        })
        
    def complete_round(self) -> None:
        """Mark round as complete and notify subscribers"""
        self.round += 1
        
        # Post round complete event with current state
        self.event_bus.post(Event(
            timestamp=datetime.now(),
            source='market',
            event_type=EventTypes.ROUND_COMPLETE,
            data={
                'round': self.round - 1,
                'prices': self.state['prices'].copy(),
                'message_count': len([m for m in self.state['messages'] if m['round'] == self.round - 1])
            }
        ))
        
    def get_state(self) -> Dict:
        """Get current market state"""
        return {
            'round': self.round,
            'prices': self.state['prices'].copy(),
            'recent_messages': self.state['messages'][-5:],
            'active_interventions': [i for i in self.state['interventions'] if i['round'] >= self.round - 2]
        }