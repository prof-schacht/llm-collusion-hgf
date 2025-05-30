#!/usr/bin/env python3
"""
Run pure LLM-driven market simulation
No rules, no heuristics - just emergent behavior
"""

import json
import os
from datetime import datetime
from orchestrator import MarketOrchestrator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def run_experiment(n_rounds: int = 10, n_traders: int = 2):
    """Run a pure LLM market experiment"""
    
    print("="*60)
    print("Pure LLM Market Simulation")
    print("="*60)
    print(f"Model: {os.getenv('FIREWORKS_MODEL')}")
    print(f"Traders: {n_traders}")
    print(f"Rounds: {n_rounds}")
    print("="*60)
    
    # Create market ecosystem
    orchestrator = MarketOrchestrator(
        n_traders=n_traders,
        n_referees=1,
        has_governor=True
    )
    
    # Run simulation
    experiment_data = {
        'config': {
            'n_traders': n_traders,
            'n_rounds': n_rounds,
            'model': os.getenv('FIREWORKS_MODEL'),
            'timestamp': datetime.now().isoformat()
        },
        'rounds': []
    }
    
    for round_num in range(n_rounds):
        print(f"\nRound {round_num + 1}/{n_rounds}")
        print("-" * 40)
        
        # Run round
        print(f"  Running traders...", end='', flush=True)
        round_data = orchestrator.run_round()
        experiment_data['rounds'].append(round_data)
        print(" done!", flush=True)
        
        # Display summary
        market_state = orchestrator.market.get_state()
        if market_state['prices']:
            prices = list(market_state['prices'].values())
            avg_price = sum(prices) / len(prices) if prices else 0
            print(f"Prices: {market_state['prices']}")
            print(f"Average: ${avg_price:.2f}")
            
        # Show interventions
        for action in round_data['actions']:
            if action['agent'] == 'governor' and action['decision'].get('decision') == 'intervene':
                print(f"🛡️ INTERVENTION: {action['decision']['intervention_type']}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    filename = f"results/pure_llm_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n✅ Experiment complete. Results saved to {filename}")
    
    # Final analysis
    print("\n" + "="*60)
    print("CONVERSATION INSIGHTS")
    print("="*60)
    
    conversations = orchestrator.get_conversation_log()
    for conv in conversations:
        if conv['thoughts']:  # Only show agents who had thoughts
            print(f"\n{conv['agent']} ({conv['role']}):")
            last_thought = conv['thoughts'][-1] if conv['thoughts'] else {}
            if last_thought:
                print(f"  Last thought: {last_thought.get('thought', 'N/A')[:200]}...")
    
    return experiment_data


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='Run pure LLM market simulation')
    parser.add_argument('--rounds', type=int, default=5, help='Number of rounds')
    parser.add_argument('--traders', type=int, default=2, help='Number of traders')
    args = parser.parse_args()
    
    try:
        run_experiment(n_rounds=args.rounds, n_traders=args.traders)
    except Exception as e:
        print(f"\nExperiment ended with error: {e}")
    finally:
        # Clean up any pending async tasks
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
        except:
            pass