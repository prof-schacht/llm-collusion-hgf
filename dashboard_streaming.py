"""
Streaming Dashboard for Pure LLM Market Simulation
Shows real-time updates as agents think and make decisions
"""

import streamlit as st
import json
import os
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from orchestrator import MarketOrchestrator
from dotenv import load_dotenv

# Load environment
load_dotenv()

st.set_page_config(page_title="Pure LLM Market Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🧠 Pure LLM Market Simulation - Live View")
st.markdown("Watch emergent behaviors develop in real-time")

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.running = False
    st.session_state.round = 0
    st.session_state.history = {
        'rounds': [],
        'prices': [],
        'messages': [],
        'interventions': [],
        'thoughts': [],
        'events': []
    }

# Sidebar configuration
with st.sidebar:
    st.header("Experiment Configuration")
    
    n_traders = st.slider("Number of Traders", 2, 6, 2)
    n_referees = st.slider("Number of Referees", 1, 3, 1)
    has_governor = st.checkbox("Include Governor", value=True)
    n_rounds = st.slider("Rounds to Run", 1, 20, 5)
    
    st.markdown("---")
    
    if st.button("🚀 Start Simulation", type="primary", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.round = 0
        st.session_state.orchestrator = MarketOrchestrator(
            n_traders=n_traders,
            n_referees=n_referees,
            has_governor=has_governor
        )
        st.session_state.history = {
            'rounds': [],
            'prices': [],
            'messages': [],
            'interventions': [],
            'thoughts': [],
            'events': []
        }
        
    if st.button("🛑 Stop", type="secondary", disabled=not st.session_state.running):
        st.session_state.running = False

# Main content area
if st.session_state.orchestrator is None:
    # Introduction
    st.markdown("""
    ### 🎯 What This Shows
    
    This dashboard visualizes **pure emergent behavior** from LLM agents in real-time:
    
    - **Live Agent Thinking**: See what each agent is considering
    - **Real-time Decisions**: Watch prices update as decisions are made
    - **Communication Flow**: Monitor messages between agents
    - **Oversight Activity**: Track referee assessments and governor interventions
    
    Configure your experiment and click "Start Simulation" to begin.
    """)
else:
    # Create layout for live updates
    metrics_container = st.container()
    status_container = st.container()
    
    # Metrics row
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        metric1 = col1.empty()
        metric2 = col2.empty()
        metric3 = col3.empty()
        metric4 = col4.empty()
    
    # Status and activity feed
    with status_container:
        st.markdown("### 📊 Live Market Activity")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Agent Thinking", "Messages", "Oversight"])
        
        with tab1:
            chart_placeholder = st.empty()
        
        with tab2:
            thinking_container = st.container()
            thinking_placeholder = thinking_container.empty()
        
        with tab3:
            messages_placeholder = st.empty()
        
        with tab4:
            oversight_placeholder = st.empty()
    
    # Activity log at bottom
    st.markdown("### 📜 Activity Log")
    activity_container = st.container()
    log_placeholder = activity_container.empty()
    
    # Run simulation with live updates
    if st.session_state.running and st.session_state.round < n_rounds:
        orchestrator = st.session_state.orchestrator
        
        # Show current round
        metric4.metric("Round", f"{st.session_state.round + 1}/{n_rounds}")
        
        # Create placeholders for live agent updates
        with thinking_placeholder.container():
            st.markdown(f"#### 🤔 Round {st.session_state.round + 1} - Agent Thinking")
            
            # Show each phase with live updates
            phases = ['trading', 'assessment', 'governance']
            phase_names = {
                'trading': '💰 Trading Phase',
                'assessment': '🔍 Assessment Phase', 
                'governance': '⚖️ Governance Phase'
            }
            
            for phase in phases:
                if phase == 'governance' and not has_governor:
                    continue
                    
                st.markdown(f"**{phase_names[phase]}**")
                
                # Get relevant agents for this phase
                if phase == 'trading':
                    agents = [a for a in orchestrator.agents.values() if a.role == 'trader']
                elif phase == 'assessment':
                    agents = [a for a in orchestrator.agents.values() if a.role == 'referee']
                else:
                    agents = [a for a in orchestrator.agents.values() if a.role == 'governor']
                
                # Show thinking for each agent
                cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with cols[i]:
                        with st.spinner(f"{agent.id} thinking..."):
                            # Simulate the agent thinking (in real implementation, 
                            # this would be done in the orchestrator)
                            st.caption(f"**{agent.id}**")
                            
                            # Add a small delay to show the process
                            time.sleep(0.5)
                            
                            # In real implementation, we'd capture the thinking here
                            st.success("✓ Decision made")
        
        # Run the actual round
        try:
            with st.spinner("Processing round..."):
                round_data = orchestrator.run_round()
                st.session_state.history['rounds'].append(round_data)
                
                # Extract and display results
                market_state = orchestrator.market.get_state()
                
                # Update metrics
                if market_state['prices']:
                    prices = list(market_state['prices'].values())
                    avg_price = sum(prices) / len(prices)
                    min_price = min(prices)
                    max_price = max(prices)
                    
                    metric1.metric("Average Price", f"${avg_price:.2f}")
                    metric2.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
                    
                    # Add to history
                    st.session_state.history['prices'].append({
                        'round': st.session_state.round,
                        'prices': market_state['prices'].copy(),
                        'avg': avg_price
                    })
                
                # Count messages and interventions
                messages_count = len([e for e in round_data['events'] if e['event_type'] == 'message_sent'])
                interventions_count = len([e for e in round_data['events'] if e['event_type'] == 'intervention_ordered'])
                
                metric3.metric("Messages", messages_count)
                
                # Update chart
                if st.session_state.history['prices']:
                    df = pd.DataFrame(st.session_state.history['prices'])
                    
                    fig = go.Figure()
                    
                    # Add lines for each trader
                    for trader_id in market_state['prices'].keys():
                        trader_prices = [p['prices'].get(trader_id, 0) for p in st.session_state.history['prices']]
                        fig.add_trace(go.Scatter(
                            x=list(range(len(trader_prices))),
                            y=trader_prices,
                            mode='lines+markers',
                            name=trader_id,
                            line=dict(width=2)
                        ))
                    
                    # Add average line
                    avg_prices = [p['avg'] for p in st.session_state.history['prices']]
                    fig.add_trace(go.Scatter(
                        x=list(range(len(avg_prices))),
                        y=avg_prices,
                        mode='lines',
                        name='Average',
                        line=dict(dash='dash', width=3, color='white')
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Round",
                        yaxis_title="Price ($)",
                        yaxis_range=[0, 10],
                        height=300,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Show recent messages
                recent_messages = [e for e in round_data['events'] if e['event_type'] == 'message_sent'][-5:]
                if recent_messages:
                    with messages_placeholder.container():
                        for msg in recent_messages:
                            st.info(f"**{msg['data']['from']}**: {msg['data']['message']}")
                
                # Show interventions
                if interventions_count > 0:
                    interventions = [e for e in round_data['events'] if e['event_type'] == 'intervention_ordered']
                    with oversight_placeholder.container():
                        for intervention in interventions:
                            st.warning(f"🛡️ **Intervention**: {intervention['data']['type']}")
                
                # Update activity log
                with log_placeholder.container():
                    # Show last 10 events
                    all_events = []
                    for r in st.session_state.history['rounds'][-3:]:  # Last 3 rounds
                        all_events.extend(r.get('events', []))
                    
                    for event in all_events[-10:]:
                        # Handle timestamp that might be string or datetime
                        timestamp_raw = event.get('timestamp', datetime.now())
                        if isinstance(timestamp_raw, str):
                            # Parse ISO format string back to datetime
                            try:
                                from datetime import datetime as dt
                                timestamp_dt = dt.fromisoformat(timestamp_raw)
                                timestamp = timestamp_dt.strftime('%H:%M:%S')
                            except:
                                # If parsing fails, use the first part of the string
                                timestamp = timestamp_raw.split('T')[-1][:8] if 'T' in timestamp_raw else timestamp_raw[:8]
                        else:
                            timestamp = timestamp_raw.strftime('%H:%M:%S')
                        
                        event_type = event.get('event_type', 'unknown')
                        source = event.get('source', 'system')
                        
                        if event_type == 'price_set':
                            st.caption(f"{timestamp} - {source} set price to ${event['data']['price']:.2f}")
                        elif event_type == 'message_sent':
                            st.caption(f"{timestamp} - {source} sent message")
                        elif event_type == 'assessment_made':
                            st.caption(f"{timestamp} - {source} assessed market as {event['data']['assessment']}")
                
                st.session_state.round += 1
                time.sleep(1)  # Brief pause between rounds
                st.rerun()
                
        except Exception as e:
            st.error(f"Error in round: {str(e)}")
            st.session_state.running = False
    
    # Show completion message
    elif st.session_state.round >= n_rounds:
        st.success("✅ Simulation Complete!")
        
        # Final summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rounds", n_rounds)
        
        with col2:
            total_messages = sum(len([e for e in r['events'] if e['event_type'] == 'message_sent']) 
                               for r in st.session_state.history['rounds'])
            st.metric("Total Messages", total_messages)
        
        with col3:
            total_interventions = sum(len([e for e in r['events'] if e['event_type'] == 'intervention_ordered']) 
                                    for r in st.session_state.history['rounds'])
            st.metric("Total Interventions", total_interventions)
        
        # Save results button
        if st.button("💾 Save Results"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/streaming_experiment_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            
            experiment_data = {
                'config': {
                    'n_traders': n_traders,
                    'n_referees': n_referees,
                    'has_governor': has_governor,
                    'n_rounds': n_rounds,
                    'timestamp': datetime.now().isoformat()
                },
                'history': st.session_state.history,
                'final_state': orchestrator.market.get_state()
            }
            
            with open(filename, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            st.success(f"Results saved to {filename}")

# Footer
st.markdown("---")
st.caption("Pure LLM Architecture - No rules, just reasoning")