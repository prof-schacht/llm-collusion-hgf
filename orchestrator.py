"""
Main orchestrator that creates and manages the agent ecosystem
"""

from typing import List, Dict, Optional
from event_bus import EventBus, EventTypes
from llm_agent import LLMAgent, Event
from simple_market import SimpleMarket
from datetime import datetime
import json


class MarketOrchestrator:
    """Orchestrates the multi-agent market simulation"""
    
    def __init__(self, n_traders: int = 2, n_referees: int = 1, has_governor: bool = True):
        self.event_bus = EventBus()
        self.market = SimpleMarket(self.event_bus)
        self.agents = {}
        
        # Create traders
        self.traders = []
        for i in range(n_traders):
            trader = LLMAgent(f"trader_{i}", role="trader")
            self.agents[trader.id] = trader
            self.traders.append(trader)
            # Traders see prices, messages, and market updates
            self.event_bus.subscribe(trader, [
                EventTypes.PRICE_SET,
                EventTypes.MESSAGE_SENT,
                EventTypes.ROUND_COMPLETE,
                EventTypes.INTERVENTION_ORDERED
            ])
        
        # Create referees
        self.referees = []
        for i in range(n_referees):
            referee = LLMAgent(f"referee_{i}", role="referee")
            referee.set_monitors(self.traders)  # Monitor all traders
            self.agents[referee.id] = referee
            self.referees.append(referee)
            # Referees see everything
            self.event_bus.subscribe(referee, ["all"])
        
        # Create governor
        self.governor = None
        if has_governor:
            self.governor = LLMAgent("governor", role="governor")
            self.governor.set_monitors(self.referees)  # Monitor referees
            self.agents[self.governor.id] = self.governor
            # Governor sees referee alerts and market state
            self.event_bus.subscribe(self.governor, [
                EventTypes.ALERT_RAISED,
                EventTypes.ROUND_COMPLETE
            ])
            
            # Set reporting hierarchy
            for referee in self.referees:
                referee.set_supervisor(self.governor)
    
    def run_round(self) -> Dict:
        """Run a single market round"""
        round_data = {'actions': [], 'events': []}
        
        # Phase 1: Traders make decisions
        for trader in self.traders:
            context = {
                'market_state': self.market.get_state(),
                'my_id': trader.id,
                'phase': 'trading'
            }
            
            try:
                decision = trader.think(context)
                round_data['actions'].append({
                    'agent': trader.id,
                    'decision': decision
                })
            except Exception as e:
                print(f"  Warning: {trader.id} failed to decide: {e}")
                # Use default action
                decision = {"action": "set_price", "price": 5.0, "reasoning": "Error - using default"}
                round_data['actions'].append({
                    'agent': trader.id,
                    'decision': decision,
                    'error': str(e)
                })
            
            # Execute trader decision
            if decision.get('action') == 'set_price':
                self.market.update_price(trader.id, decision['price'])
                
            if decision.get('message'):
                # For simplicity, broadcast to all traders
                for other in self.traders:
                    if other.id != trader.id:
                        self.market.add_message(trader.id, other.id, decision['message'])
        
        # Process all events
        self.event_bus.process_events()
        
        # Phase 2: Referees assess
        alerts = []
        for referee in self.referees:
            context = {
                'market_state': self.market.get_state(),
                'my_id': referee.id,
                'phase': 'assessment',
                'monitoring': [t.id for t in referee.monitors]
            }
            
            try:
                assessment = referee.think(context)
            except Exception as e:
                print(f"  Warning: {referee.id} assessment failed: {e}")
                assessment = {"assessment": "normal", "alert": False, "confidence": 0.5}
                
            round_data['actions'].append({
                'agent': referee.id,
                'assessment': assessment
            })
            
            # Raise alert if needed
            if assessment.get('alert'):
                alert_event = Event(
                    timestamp=datetime.now(),
                    source=referee.id,
                    event_type=EventTypes.ALERT_RAISED,
                    data={
                        'assessment': assessment['assessment'],
                        'confidence': assessment.get('confidence', 0),
                        'evidence': assessment.get('evidence', '')
                    }
                )
                self.event_bus.post(alert_event)
                alerts.append(alert_event)
        
        # Process referee events
        self.event_bus.process_events()
        
        # Phase 3: Governor decides on intervention
        if self.governor and alerts:
            context = {
                'market_state': self.market.get_state(),
                'alerts': [a.to_dict() for a in alerts],
                'my_id': self.governor.id,
                'phase': 'governance'
            }
            
            decision = self.governor.think(context)
            round_data['actions'].append({
                'agent': self.governor.id,
                'decision': decision
            })
            
            # Apply intervention if decided
            if decision.get('decision') == 'intervene':
                intervention_event = Event(
                    timestamp=datetime.now(),
                    source=self.governor.id,
                    event_type=EventTypes.INTERVENTION_ORDERED,
                    data={
                        'type': decision.get('intervention_type', 'warning'),
                        'targets': decision.get('target_agents', []),
                        'reasoning': decision.get('reasoning', '')
                    }
                )
                self.event_bus.post(intervention_event)
                self.market.apply_intervention(decision)
        
        # Complete round
        self.market.complete_round()
        self.event_bus.process_events()
        
        # Collect all events from this round
        round_data['events'] = [e.to_dict() for e in self.event_bus.event_log[-10:]]
        
        return round_data
    
    def get_conversation_log(self) -> List[Dict]:
        """Get all agent thoughts and communications"""
        log = []
        
        for agent_id, agent in self.agents.items():
            log.append({
                'agent': agent_id,
                'role': agent.role,
                'thoughts': agent.private_thoughts[-5:],  # Last 5 thoughts
                'memory_size': len(agent.memory)
            })
            
        return log