"""
Streamlit dashboard for Pure LLM Market Simulation
Real-time visualization of emergent behaviors
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from orchestrator import MarketOrchestrator
from dotenv import load_dotenv

# Load environment
load_dotenv()

st.set_page_config(page_title="Pure LLM Market Monitor", layout="wide")

st.title("🧠 Pure LLM Market Simulation")
st.markdown("Watch emergent behaviors develop without any rules or heuristics")

# Sidebar configuration
with st.sidebar:
    st.header("Experiment Configuration")
    
    n_traders = st.slider("Number of Traders", 2, 6, 2)
    n_referees = st.slider("Number of Referees", 1, 3, 1)
    has_governor = st.checkbox("Include Governor", value=True)
    n_rounds = st.slider("Rounds to Run", 1, 20, 5)
    
    st.markdown("---")
    
    # Model selection
    model_option = st.selectbox(
        "Model",
        ["Fireworks AI (DeepSeek-R1)", "Ollama (Local)", "Custom"],
        index=0
    )
    
    if model_option == "Custom":
        custom_model = st.text_input("Model Name", "gpt-4")
    else:
        custom_model = None
    
    st.markdown("---")
    
    if st.button("🚀 Start Simulation", type="primary"):
        st.session_state.running = True
        st.session_state.round = 0
        
        # Create orchestrator
        orchestrator = MarketOrchestrator(
            n_traders=n_traders,
            n_referees=n_referees,
            has_governor=has_governor
        )
        
        # Override model if custom selected
        if model_option == "Ollama (Local)":
            st.warning("Note: Using Ollama requires local setup")
            for agent in orchestrator.agents.values():
                agent.model = "ollama/qwen3:8b"
        elif model_option == "Custom" and custom_model:
            for agent in orchestrator.agents.values():
                agent.model = custom_model
                
        st.session_state.orchestrator = orchestrator
        st.session_state.history = {
            'rounds': [],
            'prices': [],
            'messages': [],
            'interventions': [],
            'thoughts': []
        }
        
    if st.button("🛑 Stop", type="secondary"):
        st.session_state.running = False

# Main area
if 'running' not in st.session_state:
    st.session_state.running = False

if not st.session_state.running and 'history' not in st.session_state:
    # Show introduction
    st.markdown("""
    ## 🎯 What This Shows
    
    This dashboard visualizes **pure emergent behavior** from LLM agents:
    
    - **No Rules**: Market dynamics emerge from LLM reasoning
    - **No Heuristics**: Detection happens through LLM analysis
    - **No Thresholds**: Interventions based on LLM judgment
    - **Private Memory**: Each agent develops its own strategy
    
    ### 🔍 What to Watch For
    
    1. **Price Evolution**: Do agents naturally collude or compete?
    2. **Communication Patterns**: What language emerges?
    3. **Detection**: Can referees identify anti-competitive behavior?
    4. **Interventions**: How does the governor respond?
    
    ### 🚀 Getting Started
    
    1. Configure the experiment in the sidebar
    2. Click "Start Simulation" to begin
    3. Watch behaviors emerge in real-time
    """)
    
else:
    # Running simulation
    if st.session_state.running and st.session_state.round < n_rounds:
        # Run next round
        orchestrator = st.session_state.orchestrator
        
        with st.spinner(f"Running round {st.session_state.round + 1}/{n_rounds}..."):
            try:
                round_data = orchestrator.run_round()
                st.session_state.history['rounds'].append(round_data)
            except Exception as e:
                st.error(f"Error in round: {e}")
                st.session_state.running = False
                st.stop()
            
            # Extract data for visualization
            market_state = orchestrator.market.get_state()
            if market_state['prices']:
                avg_price = sum(market_state['prices'].values()) / len(market_state['prices'])
                st.session_state.history['prices'].append({
                    'round': st.session_state.round,
                    'prices': market_state['prices'].copy(),
                    'avg': avg_price
                })
            
            # Count messages and interventions
            messages_this_round = [e for e in round_data['events'] if e['event_type'] == 'message_sent']
            interventions_this_round = [e for e in round_data['events'] if e['event_type'] == 'intervention_ordered']
            
            st.session_state.history['messages'].extend(messages_this_round)
            st.session_state.history['interventions'].extend(interventions_this_round)
            
            st.session_state.round += 1
            st.rerun()
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_prices = st.session_state.history['prices'][-1]['prices'] if st.session_state.history['prices'] else {}
        avg_price = st.session_state.history['prices'][-1]['avg'] if st.session_state.history['prices'] else 0
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col2:
        total_messages = len(st.session_state.history['messages'])
        st.metric("Messages Sent", total_messages)
    
    with col3:
        total_interventions = len(st.session_state.history['interventions'])
        st.metric("Interventions", total_interventions)
    
    with col4:
        rounds_complete = st.session_state.round
        st.metric("Rounds Complete", f"{rounds_complete}/{n_rounds}")
    
    # Price chart
    if st.session_state.history['prices']:
        st.subheader("📈 Price Evolution")
        
        price_df = pd.DataFrame(st.session_state.history['prices'])
        
        fig = go.Figure()
        
        # Add line for each trader
        for trader_id in current_prices.keys():
            trader_prices = [p['prices'].get(trader_id, 0) for p in st.session_state.history['prices']]
            fig.add_trace(go.Scatter(
                x=list(range(len(trader_prices))),
                y=trader_prices,
                mode='lines+markers',
                name=trader_id
            ))
        
        # Add average line
        avg_prices = [p['avg'] for p in st.session_state.history['prices']]
        fig.add_trace(go.Scatter(
            x=list(range(len(avg_prices))),
            y=avg_prices,
            mode='lines',
            name='Average',
            line=dict(dash='dash', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Messages and Interventions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💬 Recent Messages")
        if st.session_state.history['messages']:
            for msg in st.session_state.history['messages'][-5:]:
                st.info(f"**{msg['data']['from']}**: {msg['data']['message']}")
        else:
            st.info("No messages yet")
    
    with col2:
        st.subheader("🛡️ Interventions")
        if st.session_state.history['interventions']:
            for intervention in st.session_state.history['interventions']:
                st.warning(f"**Round {intervention['data']['round']}**: {intervention['data']['type']} - {intervention['data']['reasoning']}")
        else:
            st.success("No interventions triggered")
    
    # Agent Thoughts (Advanced View)
    with st.expander("🧠 Agent Thoughts (Advanced)"):
        if st.session_state.round > 0 and 'orchestrator' in st.session_state:
            conversations = st.session_state.orchestrator.get_conversation_log()
            
            tabs = st.tabs([agent['agent'] for agent in conversations if agent['thoughts']])
            
            for i, (tab, agent_data) in enumerate(zip(tabs, [a for a in conversations if a['thoughts']])):
                with tab:
                    st.write(f"**Role**: {agent_data['role']}")
                    st.write(f"**Memory Size**: {agent_data['memory_size']} events")
                    
                    if agent_data['thoughts']:
                        latest_thought = agent_data['thoughts'][-1]
                        st.json(latest_thought.get('thought', 'No thoughts recorded'))
    
    # Save results button
    if st.session_state.round >= n_rounds:
        st.success("✅ Simulation Complete!")
        
        if st.button("💾 Save Results"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/dashboard_experiment_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            
            experiment_data = {
                'config': {
                    'n_traders': n_traders,
                    'n_referees': n_referees,
                    'has_governor': has_governor,
                    'n_rounds': n_rounds,
                    'model': os.getenv('FIREWORKS_MODEL'),
                    'timestamp': datetime.now().isoformat()
                },
                'history': st.session_state.history,
                'final_state': st.session_state.orchestrator.market.get_state() if 'orchestrator' in st.session_state else {}
            }
            
            with open(filename, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            st.success(f"Results saved to {filename}")

# Footer
st.markdown("---")
st.caption("Pure LLM Architecture - No rules, just reasoning")