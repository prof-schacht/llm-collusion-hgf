"""
Real-time Streaming Dashboard with Persistent Display
Uses Streamlit's container management for live updates without full page reruns
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
from collections import deque
import threading
from contextlib import contextmanager

# Load environment
load_dotenv()

st.set_page_config(page_title="Real-time Market Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🔴 LIVE: Pure LLM Market Simulation")
st.markdown("Real-time streaming with persistent display")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.orchestrator = None
    st.session_state.running = False
    st.session_state.round = 0
    st.session_state.conversation_history = deque(maxlen=200)
    st.session_state.price_history = []
    st.session_state.current_prices = {}
    st.session_state.recent_events = deque(maxlen=10)
    st.session_state.simulation_complete = False

# Sidebar configuration
with st.sidebar:
    st.header("🎮 Simulation Control")
    
    n_traders = st.slider("Number of Traders", 2, 6, 2)
    n_referees = st.slider("Number of Referees", 1, 3, 1)
    has_governor = st.checkbox("Include Governor", value=True)
    n_rounds = st.slider("Rounds to Run", 1, 20, 10)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start", type="primary", disabled=st.session_state.running)
        if start_button:
            st.session_state.running = True
            st.session_state.round = 0
            st.session_state.orchestrator = MarketOrchestrator(
                n_traders=n_traders,
                n_referees=n_referees,
                has_governor=has_governor
            )
            st.session_state.conversation_history.clear()
            st.session_state.price_history = []
            st.session_state.current_prices = {}
            st.session_state.recent_events.clear()
            st.session_state.simulation_complete = False
            
    with col2:
        if st.button("⏸️ Stop", type="secondary", disabled=not st.session_state.running):
            st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    
    # Stats placeholders
    round_metric = st.empty()
    message_metric = st.empty()
    
    if st.session_state.running or st.session_state.round > 0:
        round_metric.metric("Round", f"{st.session_state.round}/{n_rounds}")
        message_metric.metric("Messages", len([m for m in st.session_state.conversation_history if m.get('type') == 'message']))

# Main layout
col_left, col_right = st.columns([2, 1])

# Left side - Conversation stream
with col_left:
    st.markdown("### 💬 Live Conversation Stream")
    conversation_container = st.container(height=600)
    # Create placeholder inside the container for proper rendering
    with conversation_container:
        conversation_placeholder = st.empty()

# Right side - Persistent market info
with col_right:
    st.markdown("### 📈 Market Dynamics")
    chart_placeholder = st.empty()
    
    st.markdown("### 🎯 Current Prices")
    price_placeholder = st.empty()
    
    st.markdown("### ⚡ Recent Events")
    events_container = st.container(height=200)
    with events_container:
        events_placeholder = st.empty()

# Helper functions
def update_conversation_display():
    """Update the conversation display with current history"""
    # Clear and rebuild the conversation display  
    with conversation_placeholder.container():
        for msg in reversed(list(st.session_state.conversation_history)):
            time_str = msg.get('time', '')
            agent = msg.get('agent', 'Unknown')
            phase = msg.get('phase', '')
            content = msg.get('content', '')
            msg_type = msg.get('type', 'thinking')
            
            # Style based on agent role
            if 'trader' in agent.lower():
                color = "🔵"
                bg_color = "#e6f2ff"
            elif 'referee' in agent.lower():
                color = "🟡"
                bg_color = "#fff9e6"
            elif 'governor' in agent.lower():
                color = "🔴"
                bg_color = "#ffe6e6"
            else:
                color = "⚪"
                bg_color = "#f0f0f0"
            
            # Format the message based on type
            if msg_type == 'thinking':
                st.markdown(
                    f"""
                    <div style='background-color: {bg_color}; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <strong>{color} {agent}</strong> <span style='color: #666; font-size: 0.9em;'>[{time_str}] {phase}</span><br>
                        <em style='color: #555;'>Thinking:</em> <span style='color: #333;'>{content[:300]}...</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif msg_type == 'decision':
                st.markdown(
                    f"""
                    <div style='background-color: {bg_color}; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>
                        <strong>{color} {agent}</strong> <span style='color: #666; font-size: 0.9em;'>[{time_str}]</span><br>
                        <strong style='color: #2e7d32;'>Decision:</strong> <span style='color: #333;'>{content}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif msg_type == 'message':
                st.markdown(
                    f"""
                    <div style='background-color: #f0f8ff; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <strong>{color} {agent}</strong> → <em style='color: #1976d2;'>"{content}"</em>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif msg_type == 'system':
                st.markdown(
                    f"""
                    <div style='background-color: #333; color: white; padding: 10px; margin: 5px 0; border-radius: 5px; text-align: center;'>
                        <strong>📢 {content}</strong>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

def update_chart():
    """Update the price chart"""
    if st.session_state.price_history:
        fig = go.Figure()
        
        # Get unique traders
        traders = set()
        for price_data in st.session_state.price_history:
            traders.update(price_data['prices'].keys())
        
        # Define colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Add line for each trader
        for i, trader in enumerate(sorted(traders)):
            prices = []
            rounds = []
            for j, price_data in enumerate(st.session_state.price_history):
                if trader in price_data['prices']:
                    prices.append(price_data['prices'][trader])
                    rounds.append(j + 1)
            
            if prices:
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=prices,
                    mode='lines+markers',
                    name=trader,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    connectgaps=False
                ))
        
        # Add average line
        avg_prices = []
        for price_data in st.session_state.price_history:
            prices = list(price_data['prices'].values())
            if prices:
                avg_prices.append(sum(prices) / len(prices))
        
        if avg_prices:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(avg_prices) + 1)),
                y=avg_prices,
                mode='lines',
                name='Average',
                line=dict(dash='dash', width=2, color='gray')
            ))
        
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Price ($)",
            yaxis_range=[0, 10],
            height=300,
            margin=dict(l=0, r=0, t=20, b=20),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        chart_placeholder.info("Chart will appear after first round...")

def update_prices():
    """Update current prices display"""
    if st.session_state.current_prices:
        # Use Streamlit columns for better rendering
        with price_placeholder.container():
            cols = st.columns(len(st.session_state.current_prices))
            for i, (trader, price) in enumerate(sorted(st.session_state.current_prices.items())):
                with cols[i]:
                    st.metric(trader, f"${price:.2f}")
    else:
        price_placeholder.info("Prices will appear after first round...")

def update_events():
    """Update recent events display"""
    if st.session_state.recent_events:
        with events_placeholder.container():
            for event in list(st.session_state.recent_events):
                if event['type'] == 'decision':
                    st.caption(f"🎯 **{event['agent']}**: {event['content'][:60]}...")
                elif event['type'] == 'message':
                    st.caption(f"💬 **{event['agent']}**: _{event['content'][:60]}_...")
    else:
        events_placeholder.info("Events will appear as simulation runs...")

def add_conversation_item(item):
    """Add item to conversation and update display"""
    st.session_state.conversation_history.append(item)
    update_conversation_display()
    
    # Also update events if relevant
    if item['type'] in ['decision', 'message']:
        st.session_state.recent_events.append(item)
        update_events()

def run_simulation():
    """Run the simulation with real-time updates"""
    orchestrator = st.session_state.orchestrator
    
    for round_num in range(st.session_state.round, n_rounds):
        if not st.session_state.running:
            break
            
        st.session_state.round = round_num
        
        # Update round metric
        round_metric.metric("Round", f"{round_num + 1}/{n_rounds}")
        
        # Add round start
        add_conversation_item({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'trading',
            'content': f'Starting Round {round_num + 1} - Trading Phase',
            'type': 'system'
        })
        
        # Trading Phase
        for trader in orchestrator.traders:
            if not st.session_state.running:
                break
                
            # Show thinking start
            add_conversation_item({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': trader.id,
                'phase': 'trading',
                'content': 'Analyzing market conditions...',
                'type': 'thinking'
            })
            
            # Get decision
            context = {
                'market_state': orchestrator.market.get_state(),
                'phase': 'trading'
            }
            
            decision = trader.think(context)
            
            # Show thinking
            if trader.private_thoughts:
                latest_thought = trader.private_thoughts[-1]
                thinking = latest_thought.get('thinking', '')
                if thinking:
                    add_conversation_item({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': trader.id,
                        'phase': 'trading',
                        'content': thinking,
                        'type': 'thinking'
                    })
            
            # Show decision
            if decision.get('action') == 'set_price':
                add_conversation_item({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': trader.id,
                    'phase': 'trading',
                    'content': f"Set price to ${decision['price']:.2f} - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
                
                # Update market
                orchestrator.market.update_price(trader.id, decision['price'])
                
                # Show message if any
                if decision.get('message'):
                    add_conversation_item({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': trader.id,
                        'phase': 'trading',
                        'content': decision['message'],
                        'type': 'message'
                    })
            
            # Update message count
            message_metric.metric("Messages", len([m for m in st.session_state.conversation_history if m.get('type') == 'message']))
            time.sleep(0.5)
        
        # Update market displays after trading
        market_state = orchestrator.market.get_state()
        st.session_state.current_prices = market_state['prices'].copy()
        update_prices()
        
        # Update price history and chart
        if market_state['prices']:
            st.session_state.price_history.append({
                'round': round_num + 1,
                'prices': market_state['prices'].copy()
            })
            update_chart()
        
        # Assessment Phase
        add_conversation_item({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'assessment',
            'content': 'Starting Assessment Phase',
            'type': 'system'
        })
        
        alerts = []
        for referee in orchestrator.referees:
            if not st.session_state.running:
                break
                
            context = {
                'market_state': orchestrator.market.get_state(),
                'phase': 'assessment'
            }
            
            assessment = referee.think(context)
            
            # Show assessment
            add_conversation_item({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': referee.id,
                'phase': 'assessment',
                'content': f"Assessment: {assessment['assessment']} (confidence: {assessment['confidence']})",
                'type': 'decision'
            })
            
            if assessment.get('alert'):
                alerts.append(assessment)
            
            time.sleep(0.5)
        
        # Governance Phase (if needed)
        if orchestrator.governor and alerts:
            add_conversation_item({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': 'System',
                'phase': 'governance',
                'content': 'Starting Governance Phase',
                'type': 'system'
            })
            
            context = {
                'market_state': orchestrator.market.get_state(),
                'alerts': alerts,
                'phase': 'governance'
            }
            
            decision = orchestrator.governor.think(context)
            
            # Show decision
            if decision.get('decision') == 'intervene':
                add_conversation_item({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': orchestrator.governor.id,
                    'phase': 'governance',
                    'content': f"INTERVENTION: {decision['intervention_type']} - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
        
        # Complete round
        orchestrator.market.complete_round()
        st.session_state.round += 1
        
        # Pause between rounds
        time.sleep(2)
    
    # Simulation complete
    st.session_state.simulation_complete = True
    st.session_state.running = False

# Initial display
update_conversation_display()
update_chart()
update_prices()
update_events()

# Run simulation if started
if st.session_state.running and not st.session_state.simulation_complete:
    run_simulation()

# Show completion
if st.session_state.simulation_complete:
    st.success("✅ Simulation Complete!")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rounds", st.session_state.round)
    with col2:
        st.metric("Total Messages", len([m for m in st.session_state.conversation_history if m.get('type') == 'message']))
    with col3:
        st.metric("Total Events", len(st.session_state.conversation_history))