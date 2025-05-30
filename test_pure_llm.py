#!/usr/bin/env python3
"""
Test the pure LLM architecture with a simple scenario
"""

import os
from dotenv import load_dotenv
from llm_agent import LLMAgent, Event
from event_bus import EventBus, EventTypes
from simple_market import SimpleMarket
from datetime import datetime

# Load environment
load_dotenv()

def test_basic_setup():
    """Test basic agent creation and event flow"""
    print("Testing Pure LLM Architecture")
    print("="*40)
    
    # Create event bus
    event_bus = EventBus()
    
    # Create a trader
    trader = LLMAgent("trader_0", role="trader")
    print(f"✓ Created trader: {trader.id}")
    
    # Create a referee
    referee = LLMAgent("referee_0", role="referee")
    referee.set_monitors([trader])
    print(f"✓ Created referee: {referee.id}")
    
    # Create a governor
    governor = LLMAgent("governor", role="governor")
    governor.set_monitors([referee])
    referee.set_supervisor(governor)
    print(f"✓ Created governor: {governor.id}")
    
    # Subscribe to events
    event_bus.subscribe(trader, [EventTypes.PRICE_SET, EventTypes.MESSAGE_SENT])
    event_bus.subscribe(referee, ["all"])
    event_bus.subscribe(governor, [EventTypes.ALERT_RAISED])
    print("✓ Set up event subscriptions")
    
    # Create market
    market = SimpleMarket(event_bus)
    print("✓ Created market")
    
    # Test event flow
    print("\nTesting event flow:")
    
    # Trader sets a price
    market.update_price("trader_0", 7.50)
    events = event_bus.process_events()
    print(f"  - Price set event posted: {len(events)} events processed")
    
    # Check if agents received events
    print(f"  - Trader memory size: {len(trader.memory)}")
    print(f"  - Referee memory size: {len(referee.memory)}")
    
    # Test agent thinking (with a smaller model for testing)
    print("\nTesting agent reasoning:")
    context = {'market_state': market.get_state(), 'test': True}
    
    # For testing, use a smaller model if available
    if "ollama" in os.environ.get('OLLAMA_API_BASE', ''):
        trader.model = "ollama/qwen3:8b"
        print("  Using Ollama for testing...")
    
    try:
        thought = trader.think(context)
        print(f"  - Trader decision: {thought}")
    except Exception as e:
        print(f"  - Trader thinking skipped (model not available): {e}")
    
    print("\n✅ Basic architecture test complete!")
    
    # Show hierarchy
    print("\nAgent Hierarchy:")
    print("  Governor")
    print("     ↓")
    print("  Referee (monitors: trader_0)")
    print("     ↓")
    print("  Trader")


def test_minimal_round():
    """Test a minimal market round"""
    print("\n\nTesting Minimal Market Round")
    print("="*40)
    
    from orchestrator import MarketOrchestrator
    
    # Create minimal setup
    orchestrator = MarketOrchestrator(n_traders=2, n_referees=1, has_governor=True)
    
    print("Market setup:")
    print(f"  - Traders: {[t.id for t in orchestrator.traders]}")
    print(f"  - Referees: {[r.id for r in orchestrator.referees]}")
    print(f"  - Governor: {orchestrator.governor.id if orchestrator.governor else 'None'}")
    
    print("\nNote: To run actual market rounds, use:")
    print("  python run_pure_llm.py --rounds 5 --traders 2")
    

if __name__ == "__main__":
    test_basic_setup()
    test_minimal_round()