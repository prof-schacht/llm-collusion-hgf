"""
Fixed Live Streaming Dashboard for Pure LLM Market Simulation
Maintains persistent display of charts and information
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

# Load environment
load_dotenv()

st.set_page_config(page_title="Live Market Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🔴 LIVE: Pure LLM Market Simulation")
st.markdown("Real-time agent conversations and market dynamics")

# Initialize session state with persistent elements
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.orchestrator = None
    st.session_state.running = False
    st.session_state.round = 0
    st.session_state.conversation_history = deque(maxlen=100)
    st.session_state.price_history = []
    st.session_state.current_prices = {}
    st.session_state.recent_events = []
    st.session_state.chart_data = None

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
        if st.button("▶️ Start", type="primary", disabled=st.session_state.running):
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
            st.session_state.recent_events = []
            st.session_state.chart_data = None
            
    with col2:
        if st.button("⏸️ Stop", type="secondary", disabled=not st.session_state.running):
            st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    
    if st.session_state.running or st.session_state.round > 0:
        st.metric("Round", f"{st.session_state.round}/{n_rounds}")
        st.metric("Messages", len([m for m in st.session_state.conversation_history if m.get('type') == 'message']))

# Main layout
if st.session_state.orchestrator is None and not st.session_state.running:
    st.info("👈 Configure and start the simulation from the sidebar")
else:
    # Create persistent layout
    col_left, col_right = st.columns([2, 1])
    
    # Left side - Conversation stream
    with col_left:
        st.markdown("### 💬 Live Conversation Stream")
        conversation_container = st.container(height=600)
        
    # Right side - Market info (persistent display)
    with col_right:
        st.markdown("### 📈 Market Dynamics")
        
        # Always show the chart if we have data
        if st.session_state.price_history:
            df = pd.DataFrame(st.session_state.price_history)
            
            fig = go.Figure()
            
            # Get unique traders
            traders = set()
            for prices in df['prices']:
                traders.update(prices.keys())
            
            # Define colors for traders
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Add line for each trader
            for i, trader in enumerate(sorted(traders)):
                prices = [p['prices'].get(trader, None) for p in st.session_state.price_history]
                rounds = list(range(1, len(prices) + 1))
                
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=prices,
                    mode='lines+markers',
                    name=trader,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    connectgaps=False
                ))
            
            # Add average price line
            avg_prices = []
            for price_data in st.session_state.price_history:
                prices_list = list(price_data['prices'].values())
                if prices_list:
                    avg_prices.append(sum(prices_list) / len(prices_list))
            
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
                legend=dict(x=0, y=1),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart will appear after first round...")
        
        st.markdown("### 🎯 Current Prices")
        if st.session_state.current_prices:
            price_cols = st.columns(len(st.session_state.current_prices))
            for i, (trader, price) in enumerate(sorted(st.session_state.current_prices.items())):
                with price_cols[i]:
                    st.metric(trader, f"${price:.2f}")
        else:
            st.info("Prices will appear after first round...")
        
        st.markdown("### ⚡ Recent Events")
        events_container = st.container(height=200)
        with events_container:
            if st.session_state.recent_events:
                for event in st.session_state.recent_events[-5:]:
                    if event['type'] == 'decision':
                        st.caption(f"🎯 {event['agent']}: {event['content'][:50]}...")
                    elif event['type'] == 'message':
                        st.caption(f"💬 {event['agent']}: {event['content'][:50]}...")
            else:
                st.info("Events will appear as simulation runs...")
    
    # Display current conversation history
    if st.session_state.conversation_history:
        with conversation_container:
            for msg in reversed(list(st.session_state.conversation_history)):
                time_str = msg.get('time', '')
                agent = msg.get('agent', 'Unknown')
                phase = msg.get('phase', '')
                content = msg.get('content', '')
                msg_type = msg.get('type', 'thinking')
                
                # Style based on agent role
                if 'trader' in agent:
                    color = "🔵"
                    bg_color = "#e6f2ff"
                elif 'referee' in agent:
                    color = "🟡"
                    bg_color = "#fff9e6"
                elif 'governor' in agent:
                    color = "🔴"
                    bg_color = "#ffe6e6"
                else:
                    color = "⚪"
                    bg_color = "#f0f0f0"
                
                # Format the message
                if msg_type == 'thinking':
                    st.markdown(
                        f"""<div style='background-color: {bg_color}; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <strong>{color} {agent}</strong> <span style='color: #666; font-size: 0.9em;'>[{time_str}] {phase}</span><br>
                        <em style='color: #555;'>Thinking:</em> <span style='color: #333;'>{content[:300]}...</span>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                elif msg_type == 'decision':
                    st.markdown(
                        f"""<div style='background-color: {bg_color}; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>
                        <strong>{color} {agent}</strong> <span style='color: #666; font-size: 0.9em;'>[{time_str}]</span><br>
                        <strong style='color: #2e7d32;'>Decision:</strong> <span style='color: #333;'>{content}</span>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                elif msg_type == 'message':
                    st.markdown(
                        f"""<div style='background-color: #f0f8ff; color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <strong>{color} {agent}</strong> → <em style='color: #1976d2;'>"{content}"</em>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                elif msg_type == 'system':
                    st.markdown(
                        f"""<div style='background-color: #333; color: white; padding: 10px; margin: 5px 0; border-radius: 5px; text-align: center;'>
                        <strong>📢 {content}</strong>
                        </div>""", 
                        unsafe_allow_html=True
                    )
    
    # Run simulation logic
    if st.session_state.running and st.session_state.round < n_rounds:
        orchestrator = st.session_state.orchestrator
        
        # Add round start message
        st.session_state.conversation_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'trading',
            'content': f'Starting Round {st.session_state.round + 1} - Trading Phase',
            'type': 'system'
        })
        
        # Phase 1: Trading
        for trader in orchestrator.traders:
            # Show thinking start
            st.session_state.conversation_history.append({
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
            
            with st.spinner(f"{trader.id} is thinking..."):
                decision = trader.think(context)
            
            # Show thinking content if available
            if trader.private_thoughts:
                latest_thought = trader.private_thoughts[-1]
                thinking = latest_thought.get('thinking', '')
                if thinking:
                    st.session_state.conversation_history.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': trader.id,
                        'phase': 'trading',
                        'content': thinking,
                        'type': 'thinking'
                    })
            
            # Show decision
            if decision.get('action') == 'set_price':
                st.session_state.conversation_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': trader.id,
                    'phase': 'trading',
                    'content': f"Set price to ${decision['price']:.2f} - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
                
                # Update recent events
                st.session_state.recent_events.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': trader.id,
                    'type': 'decision',
                    'content': f"Set price to ${decision['price']:.2f}"
                })
                
                # Apply to market
                orchestrator.market.update_price(trader.id, decision['price'])
                
                # Show message if any
                if decision.get('message'):
                    st.session_state.conversation_history.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': trader.id,
                        'phase': 'trading',
                        'content': decision['message'],
                        'type': 'message'
                    })
                    
                    st.session_state.recent_events.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': trader.id,
                        'type': 'message',
                        'content': decision['message']
                    })
            
            time.sleep(0.5)  # Small delay for readability
        
        # Update market state after all traders
        market_state = orchestrator.market.get_state()
        st.session_state.current_prices = market_state['prices'].copy()
        
        # Update price history
        if market_state['prices']:
            st.session_state.price_history.append({
                'round': st.session_state.round + 1,
                'prices': market_state['prices'].copy()
            })
        
        # Phase 2: Assessment
        st.session_state.conversation_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'assessment',
            'content': 'Starting Assessment Phase',
            'type': 'system'
        })
        
        alerts = []
        for referee in orchestrator.referees:
            context = {
                'market_state': orchestrator.market.get_state(),
                'phase': 'assessment'
            }
            
            with st.spinner(f"{referee.id} is assessing..."):
                assessment = referee.think(context)
            
            # Show assessment
            st.session_state.conversation_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': referee.id,
                'phase': 'assessment',
                'content': f"Assessment: {assessment['assessment']} (confidence: {assessment['confidence']})",
                'type': 'decision'
            })
            
            if assessment.get('alert'):
                alerts.append(assessment)
            
            time.sleep(0.5)
        
        # Phase 3: Governance (if needed)
        if orchestrator.governor and alerts:
            st.session_state.conversation_history.append({
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
            
            with st.spinner("Governor is deciding..."):
                decision = orchestrator.governor.think(context)
            
            # Show decision
            if decision.get('decision') == 'intervene':
                st.session_state.conversation_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': orchestrator.governor.id,
                    'phase': 'governance',
                    'content': f"INTERVENTION: {decision['intervention_type']} - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
        
        # Complete round
        orchestrator.market.complete_round()
        st.session_state.round += 1
        
        # Trigger rerun for next round
        if st.session_state.running and st.session_state.round < n_rounds:
            time.sleep(2)  # Brief pause between rounds
            st.rerun()
    
    # Show completion
    elif st.session_state.round >= n_rounds and st.session_state.orchestrator:
        st.success("✅ Simulation Complete!")
        st.session_state.running = False