"""
Live Streaming Dashboard for Pure LLM Market Simulation
Shows real-time agent thinking and maintains conversation history
"""

import streamlit as st
import json
import os
import time
import asyncio
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from orchestrator import MarketOrchestrator
from dotenv import load_dotenv
from collections import deque
import threading
from queue import Queue

# Load environment
load_dotenv()

st.set_page_config(page_title="Live Market Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🔴 LIVE: Pure LLM Market Simulation")
st.markdown("Real-time agent conversations and market dynamics")

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.running = False
    st.session_state.round = 0
    st.session_state.conversation_history = deque(maxlen=100)  # Keep last 100 messages
    st.session_state.price_history = []
    st.session_state.message_queue = Queue()

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
            
    with col2:
        if st.button("⏸️ Stop", type="secondary", disabled=not st.session_state.running):
            st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    stats_container = st.container()

# Main layout
if st.session_state.orchestrator is None:
    st.info("👈 Configure and start the simulation from the sidebar")
else:
    # Create main layout columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 💬 Live Conversation Stream")
        
        # Create a scrollable container for conversations
        conversation_container = st.container(height=600)
        
        # Placeholder for live updates
        with conversation_container:
            conversation_placeholder = st.empty()
    
    with col_right:
        # Create static containers that won't fade
        right_container = st.container()
        
        with right_container:
            st.markdown("### 📈 Market Dynamics")
            chart_container = st.container()
            chart_placeholder = chart_container.empty()
            
            st.markdown("### 🎯 Current Prices")
            price_placeholder = st.empty()
            
            st.markdown("### ⚡ Recent Events")
            events_placeholder = st.empty()
    
    # Show initial states
    price_placeholder.info("Prices will appear after first round...")
    events_placeholder.info("Events will appear as simulation runs...")
    
    # Function to display conversation history
    def display_conversations():
        with conversation_placeholder.container():
            for msg in reversed(list(st.session_state.conversation_history)):
                time_str = msg['time']
                agent = msg['agent']
                phase = msg['phase']
                content = msg['content']
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
                
                # Format the message with dark text
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
    
    # Update price chart
    def update_chart():
        if st.session_state.price_history:
            # Create chart and display directly in placeholder
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
                    rounds = list(range(1, len(prices) + 1))  # Start from round 1
                    
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
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        linecolor='black'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        linecolor='black'
                    )
                )
                
                # Display chart in the placeholder
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"price_chart_{st.session_state.round}")
    
    # Custom run function that captures agent thinking
    def run_round_with_streaming(orchestrator, round_num):
        """Run a round while streaming agent thoughts"""
        round_data = {'actions': [], 'events': []}
        
        # Phase 1: Trading
        st.session_state.conversation_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'trading',
            'content': f'Starting Round {round_num + 1} - Trading Phase',
            'type': 'system'
        })
        display_conversations()
        
        for trader in orchestrator.traders:
            # Show thinking start
            st.session_state.conversation_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': trader.id,
                'phase': 'trading',
                'content': 'Analyzing market conditions...',
                'type': 'thinking'
            })
            display_conversations()
            
            # Get decision
            context = {
                'market_state': orchestrator.market.get_state(),
                'phase': 'trading'
            }
            
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
            
            display_conversations()
            time.sleep(0.5)  # Small delay to make it readable
        
        # Update price display and history
        market_state = orchestrator.market.get_state()
        
        # Update price metrics using HTML for stability
        if market_state['prices']:
            price_html = '<div style="display: flex; justify-content: space-around; padding: 10px;">'
            for trader, price in sorted(market_state['prices'].items()):
                price_html += f'''
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 0.9em; color: #666;">{trader}</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">${price:.2f}</div>
                </div>
                '''
            price_html += '</div>'
            price_placeholder.markdown(price_html, unsafe_allow_html=True)
        
        # Update price history immediately after trading phase
        if market_state['prices']:
            st.session_state.price_history.append({
                'round': round_num + 1,
                'prices': market_state['prices'].copy()
            })
            update_chart()
            
        # Update recent events after each phase
        def update_recent_events():
            recent_items = list(st.session_state.conversation_history)[-15:]
            display_items = []
            
            # Get recent decisions and messages
            for item in reversed(recent_items):
                if item['type'] in ['decision', 'message'] and len(display_items) < 5:
                    display_items.append(item)
            
            # Build HTML for events to avoid flickering
            events_html = '<div style="padding: 10px;">'
            if display_items:
                for event in display_items:
                    if event['type'] == 'decision':
                        events_html += f'<p style="margin: 5px 0;">🎯 <strong>{event["agent"]}</strong>: {event["content"][:50]}...</p>'
                    elif event['type'] == 'message':
                        events_html += f'<p style="margin: 5px 0;">💬 <strong>{event["agent"]}</strong>: <em>{event["content"][:50]}...</em></p>'
            else:
                events_html += '<p style="color: gray;">No recent events yet...</p>'
            events_html += '</div>'
            
            events_placeholder.markdown(events_html, unsafe_allow_html=True)
        
        # Call update after trading
        update_recent_events()
        
        # Phase 2: Assessment
        st.session_state.conversation_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': 'System',
            'phase': 'assessment',
            'content': 'Starting Assessment Phase',
            'type': 'system'
        })
        display_conversations()
        
        alerts = []
        for referee in orchestrator.referees:
            context = {
                'market_state': orchestrator.market.get_state(),
                'phase': 'assessment'
            }
            
            assessment = referee.think(context)
            
            # Show referee thinking
            if referee.private_thoughts:
                latest_thought = referee.private_thoughts[-1]
                thinking = latest_thought.get('thinking', '')
                if thinking:
                    st.session_state.conversation_history.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': referee.id,
                        'phase': 'assessment',
                        'content': thinking[:400],
                        'type': 'thinking'
                    })
            
            # Show assessment
            st.session_state.conversation_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': referee.id,
                'phase': 'assessment',
                'content': f"Assessment: {assessment['assessment']} (confidence: {assessment['confidence']}) - {assessment.get('evidence', '')}",
                'type': 'decision'
            })
            
            if assessment.get('alert'):
                alerts.append(assessment)
            
            display_conversations()
            time.sleep(0.5)
        
        # Update events after assessment
        update_recent_events()
        
        # Phase 3: Governance
        if orchestrator.governor and alerts:
            st.session_state.conversation_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'agent': 'System',
                'phase': 'governance',
                'content': 'Starting Governance Phase - Alerts detected',
                'type': 'system'
            })
            display_conversations()
            
            context = {
                'market_state': orchestrator.market.get_state(),
                'alerts': alerts,
                'phase': 'governance'
            }
            
            decision = orchestrator.governor.think(context)
            
            # Show governor thinking
            if orchestrator.governor.private_thoughts:
                latest_thought = orchestrator.governor.private_thoughts[-1]
                thinking = latest_thought.get('thinking', '')
                if thinking:
                    st.session_state.conversation_history.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'agent': orchestrator.governor.id,
                        'phase': 'governance',
                        'content': thinking[:400],
                        'type': 'thinking'
                    })
            
            # Show decision
            if decision.get('decision') == 'intervene':
                st.session_state.conversation_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': orchestrator.governor.id,
                    'phase': 'governance',
                    'content': f"INTERVENTION: {decision['intervention_type']} - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
            else:
                st.session_state.conversation_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'agent': orchestrator.governor.id,
                    'phase': 'governance',
                    'content': f"No intervention needed - {decision.get('reasoning', '')}",
                    'type': 'decision'
                })
            
            display_conversations()
            
            # Update events after governance
            update_recent_events()
        
        # Complete round
        orchestrator.market.complete_round()
        
        return round_data
    
    # Run simulation
    if st.session_state.running and st.session_state.round < n_rounds:
        # Update sidebar stats
        with stats_container:
            st.metric("Round", f"{st.session_state.round + 1}/{n_rounds}")
            st.metric("Messages", len([m for m in st.session_state.conversation_history if m['type'] == 'message']))
            
            # Show recent events in sidebar
            recent_events = st.session_state.conversation_history
            if recent_events:
                st.markdown("#### Recent Activity")
                for event in list(recent_events)[-5:]:
                    if event['type'] == 'decision':
                        st.caption(f"{event['agent']}: {event['content'][:50]}...")
            
        # Run round with streaming
        try:
            round_data = run_round_with_streaming(
                st.session_state.orchestrator, 
                st.session_state.round
            )
            
            st.session_state.round += 1
            
            # Auto-rerun for next round
            if st.session_state.running and st.session_state.round < n_rounds:
                time.sleep(2)  # Pause between rounds
                st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.running = False
    
    elif st.session_state.round >= n_rounds:
        st.success("✅ Simulation Complete!")
        st.session_state.running = False
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rounds", n_rounds)
        with col2:
            st.metric("Total Messages", len([m for m in st.session_state.conversation_history if m['type'] == 'message']))
        with col3:
            st.metric("Conversation Items", len(st.session_state.conversation_history))

# Add CSS for better styling and prevent fading
st.markdown("""
<style>
    /* Prevent fading on right column */
    section[data-testid="stSidebar"] + div > div > div > div > div:last-child {
        opacity: 1 !important;
        transition: none !important;
    }
    
    /* Ensure containers stay visible */
    .element-container {
        opacity: 1 !important;
    }
    
    /* Better contrast for dark mode */
    .stContainer > div {
        background-color: white;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.2em;
    }
    
    /* Keep plotly charts visible */
    .js-plotly-plot {
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)