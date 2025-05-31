#!/usr/bin/env python3
"""
Test the improved LLM system with DeepSeek-R1
Shows thinking process and decision extraction
"""

import json
import time
from orchestrator import MarketOrchestrator
from dotenv import load_dotenv

load_dotenv()

def test_improved_system():
    """Test the improved system with better error handling"""
    print("="*60)
    print("Testing Improved Pure LLM System")
    print("="*60)
    
    # Create a small market
    orchestrator = MarketOrchestrator(
        n_traders=2,
        n_referees=1,
        has_governor=True
    )
    
    print("\nRunning 3 rounds to test improvements...")
    print("-"*60)
    
    for round_num in range(3):
        print(f"\n📍 ROUND {round_num + 1}")
        print("="*40)
        
        # Run round with detailed output
        start_time = time.time()
        round_data = orchestrator.run_round()
        elapsed = time.time() - start_time
        
        print(f"⏱️  Round completed in {elapsed:.1f} seconds")
        
        # Show market state
        market_state = orchestrator.market.get_state()
        print(f"\n💰 Market Prices:")
        for trader, price in market_state['prices'].items():
            print(f"   {trader}: ${price:.2f}")
        
        # Show messages
        messages = [e for e in round_data['events'] if e['event_type'] == 'message_sent']
        if messages:
            print(f"\n💬 Messages ({len(messages)}):")
            for msg in messages:
                print(f"   {msg['data']['from']}: {msg['data']['message']}")
        
        # Show assessments
        assessments = [e for e in round_data['events'] if e['event_type'] == 'assessment_made']
        if assessments:
            print(f"\n🔍 Referee Assessments:")
            for assess in assessments:
                print(f"   {assess['source']}: {assess['data']['assessment']} (confidence: {assess['data']['confidence']})")
        
        # Show interventions
        interventions = [e for e in round_data['events'] if e['event_type'] == 'intervention_ordered']
        if interventions:
            print(f"\n⚖️  Governor Interventions:")
            for interv in interventions:
                print(f"   Type: {interv['data']['intervention_type']}")
                print(f"   Reasoning: {interv['data']['reasoning']}")
    
    # Show agent thoughts
    print("\n" + "="*60)
    print("AGENT THINKING SAMPLES")
    print("="*60)
    
    conversations = orchestrator.get_conversation_log()
    for conv in conversations:
        if conv['thoughts']:
            print(f"\n🧠 {conv['agent']} ({conv['role']}):")
            latest = conv['thoughts'][-1]
            
            # Show thinking process
            thinking = latest.get('thinking', 'N/A')
            if thinking and thinking != 'N/A':
                print(f"   Thinking: {thinking[:200]}..." if len(thinking) > 200 else f"   Thinking: {thinking}")
            
            # Show decision
            decision = latest.get('decision', {})
            if decision:
                print(f"   Decision: {json.dumps(decision, indent=6)}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_improved_system()