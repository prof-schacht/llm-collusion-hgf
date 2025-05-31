#!/usr/bin/env python3
"""
Debug a single agent to see what's happening
"""

import os
from dotenv import load_dotenv
from llm_agent import LLMAgent
import json

load_dotenv()

def debug_single_agent():
    """Test a single agent decision"""
    print("Debugging Single Agent Decision")
    print("="*50)
    
    # Create a trader
    trader = LLMAgent("trader_0", role="trader")
    print(f"Created: {trader.id}")
    print(f"Model: {trader.model}")
    
    # Create simple context
    context = {
        'market_state': {
            'round': 1,
            'prices': {'trader_0': 5.0, 'trader_1': 5.5},
            'recent_messages': []
        },
        'my_id': 'trader_0',
        'phase': 'trading'
    }
    
    print("\nContext:")
    print(json.dumps(context, indent=2))
    
    print("\n" + "-"*50)
    print("Calling think()...")
    
    # Get decision
    decision = trader.think(context)
    
    print("\nDecision:")
    print(json.dumps(decision, indent=2))
    
    # Check thoughts
    if trader.private_thoughts:
        print("\nPrivate thoughts:")
        for thought in trader.private_thoughts:
            if 'error' in thought:
                print(f"ERROR: {thought['error']}")
            else:
                full_response = thought.get('full_response', 'N/A')
                thinking = thought.get('thinking', 'N/A')
                decision = thought.get('decision', 'N/A')
                
                print(f"Full response ({len(str(full_response))} chars):")
                print(str(full_response))
                print(f"\nExtracted thinking ({len(str(thinking))} chars):")
                print(str(thinking))
                print(f"\nExtracted decision:")
                print(decision)
                print("\n--- End of response ---")
    
    print("\n" + "="*50)
    print("Debug complete!")
    
    # Show what the prompt looks like
    print("\nGenerated prompt:")
    print(trader._build_prompt(context))

if __name__ == "__main__":
    debug_single_agent()