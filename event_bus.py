"""
Simple Event Bus for agent communication
All interactions happen through events - no direct coupling
"""

from typing import Dict, List, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from llm_agent import Event, LLMAgent


class EventBus:
    """Central message bus for all agent communications"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[LLMAgent]] = {}
        self.event_log: List[Event] = []
        self.pending_events: List[Event] = []
        
    def subscribe(self, agent: LLMAgent, event_types: List[str]) -> None:
        """Subscribe an agent to specific event types"""
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            if agent not in self.subscribers[event_type]:
                self.subscribers[event_type].append(agent)
    
    def post(self, event: Event) -> None:
        """Post an event to the bus"""
        self.pending_events.append(event)
        
    def process_events(self) -> List[Event]:
        """Process all pending events and notify subscribers"""
        processed = []
        
        while self.pending_events:
            event = self.pending_events.pop(0)
            self.event_log.append(event)
            processed.append(event)
            
            # Notify all subscribers
            if event.event_type in self.subscribers:
                for agent in self.subscribers[event.event_type]:
                    agent.perceive(event)
                    
            # Also notify subscribers to "all" events
            if "all" in self.subscribers:
                for agent in self.subscribers["all"]:
                    agent.perceive(event)
                    
        return processed
    
    def get_history(self, event_types: List[str] = None, source: str = None) -> List[Event]:
        """Get filtered event history"""
        history = self.event_log
        
        if event_types:
            history = [e for e in history if e.event_type in event_types]
            
        if source:
            history = [e for e in history if e.source == source]
            
        return history


# Common event types
class EventTypes:
    # Trader events
    PRICE_SET = "price_set"
    MESSAGE_SENT = "message_sent"
    
    # Referee events  
    ALERT_RAISED = "alert_raised"
    ASSESSMENT_COMPLETE = "assessment_complete"
    
    # Governor events
    INTERVENTION_ORDERED = "intervention_ordered"
    MARKET_RESET = "market_reset"
    
    # Market events
    ROUND_COMPLETE = "round_complete"
    MARKET_UPDATE = "market_update"