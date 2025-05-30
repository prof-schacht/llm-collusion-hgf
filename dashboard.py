import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import numpy as np

# Import our modules
from run_experiment import run_comparison_experiment, run_single_experiment
from communication_analyzer import CommunicationAnalyzer

st.set_page_config(page_title="LLM Collusion Monitor", layout="wide")

st.title("🔍 LLM Collusion Detection Dashboard")

# Sidebar configuration
with st.sidebar:
    st.header("Experiment Configuration")
    
    n_episodes = st.slider("Number of Episodes", 5, 50, 20)
    llm_model = st.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "claude-3-haiku", "ollama/qwen3:8b", "ollama/deepseek-r1:1.5b"],
        index=3  # Default to ollama/qwen3:8b
    )
    
    st.markdown("---")
    
    if st.button("Run Experiment", type="primary"):
        with st.spinner("Running experiment..."):
            config = {
                "n_episodes": n_episodes,
                "llm_model": llm_model,
                "use_wandb": False  # Disable for dashboard
            }
            results = run_comparison_experiment(config)
            st.session_state["results"] = results
            st.success("Experiment completed!")
    
    st.markdown("---")
    st.markdown("### Load Previous Results")
    
    # List available result files
    results_dir = "results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if result_files:
            selected_file = st.selectbox("Select result file", result_files)
            if st.button("Load Results"):
                with open(os.path.join(results_dir, selected_file), 'r') as f:
                    st.session_state["results"] = json.load(f)
                st.success(f"Loaded {selected_file}")

# Main dashboard
if "results" in st.session_state:
    results = st.session_state["results"]
    
    # Key metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    baseline_price = results['baseline']['avg_final_price']
    safety_price = results['with_safety']['avg_final_price']
    competitive_price = 4.0
    
    with col1:
        st.metric(
            "Baseline Price",
            f"${baseline_price:.2f}",
            delta=f"+${baseline_price - competitive_price:.2f} vs competitive",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "With Safety",
            f"${safety_price:.2f}",
            delta=f"-${baseline_price - safety_price:.2f}",
            delta_color="normal"
        )
    
    with col3:
        prevention_rate = ((baseline_price - safety_price) / (baseline_price - competitive_price) * 100 
                          if baseline_price > competitive_price else 0)
        st.metric(
            "Collusion Prevention",
            f"{prevention_rate:.1f}%"
        )
    
    with col4:
        intervention_rate = results['with_safety'].get('intervention_rate', 0)
        st.metric(
            "Interventions/Episode",
            f"{intervention_rate:.2f}"
        )
    
    # Price evolution chart
    st.header("📈 Price Evolution Over Time")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Price Comparison", "Individual Episodes", "Distribution"])
    
    with tab1:
        # Extract price data for comparison
        baseline_prices = []
        safety_prices = []
        
        # Get first 5 episodes for visualization
        n_display_episodes = min(5, len(results["baseline"]["episodes"]))
        
        for ep_idx in range(n_display_episodes):
            if ep_idx < len(results["baseline"]["episodes"]):
                ep = results["baseline"]["episodes"][ep_idx]
                baseline_prices.extend([p["avg_price"] for p in ep["prices"]])
            
            if ep_idx < len(results["with_safety"]["episodes"]):
                ep = results["with_safety"]["episodes"][ep_idx]
                safety_prices.extend([p["avg_price"] for p in ep["prices"]])
        
        fig = go.Figure()
        
        # Add baseline trace
        fig.add_trace(go.Scatter(
            y=baseline_prices,
            mode='lines',
            name='No Safety (Baseline)',
            line=dict(color='red', width=2),
            hovertemplate='Round %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add safety trace
        fig.add_trace(go.Scatter(
            y=safety_prices,
            mode='lines',
            name='With HGF Safety',
            line=dict(color='green', width=2),
            hovertemplate='Round %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add competitive price line
        fig.add_hline(y=competitive_price, line_dash="dash", line_color="gray", 
                      annotation_text="Competitive Price ($4.00)")
        
        fig.update_layout(
            title="Price Evolution: First 5 Episodes",
            xaxis_title="Round",
            yaxis_title="Average Price ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Show individual episode details
        episode_option = st.selectbox(
            "Select Episode", 
            range(len(results["baseline"]["episodes"])),
            format_func=lambda x: f"Episode {x+1}"
        )
        
        if episode_option < len(results["baseline"]["episodes"]):
            baseline_ep = results["baseline"]["episodes"][episode_option]
            safety_ep = results["with_safety"]["episodes"][episode_option] if episode_option < len(results["with_safety"]["episodes"]) else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Baseline (No Safety)")
                if baseline_ep["prices"]:
                    prices = [p["avg_price"] for p in baseline_ep["prices"]]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=prices, mode='lines+markers', name='Price'))
                    fig.add_hline(y=competitive_price, line_dash="dash", line_color="gray")
                    fig.update_layout(title="Price Evolution", xaxis_title="Round", yaxis_title="Price ($)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("With Safety")
                if safety_ep and safety_ep["prices"]:
                    prices = [p["avg_price"] for p in safety_ep["prices"]]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=prices, mode='lines+markers', name='Price', line_color='green'))
                    fig.add_hline(y=competitive_price, line_dash="dash", line_color="gray")
                    fig.update_layout(title="Price Evolution", xaxis_title="Round", yaxis_title="Price ($)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Price distribution
        baseline_final_prices = []
        safety_final_prices = []
        
        for ep in results["baseline"]["episodes"]:
            if ep["prices"]:
                baseline_final_prices.append(ep["prices"][-1]["avg_price"])
        
        for ep in results["with_safety"]["episodes"]:
            if ep["prices"]:
                safety_final_prices.append(ep["prices"][-1]["avg_price"])
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=baseline_final_prices, name="Baseline", marker_color="red"))
        fig.add_trace(go.Box(y=safety_final_prices, name="With Safety", marker_color="green"))
        fig.add_hline(y=competitive_price, line_dash="dash", line_color="gray", 
                      annotation_text="Competitive Price")
        
        fig.update_layout(
            title="Distribution of Final Prices",
            yaxis_title="Final Price ($)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Communication Analysis
    st.header("💬 Communication Analysis")
    
    # Initialize analyzer
    analyzer = CommunicationAnalyzer()
    
    # Analyze baseline conversations
    baseline_messages = []
    for ep in results["baseline"]["episodes"]:
        baseline_messages.extend(ep.get("messages", []))
    
    if baseline_messages:
        st.subheader("Baseline Communication Patterns")
        
        baseline_analysis = analyzer.analyze_conversation(baseline_messages)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Collusion Score", f"{baseline_analysis['collusion_score']:.2f}")
        
        with col2:
            status = "🚨 Detected" if baseline_analysis["explicit_collusion"] else "✅ None"
            st.metric("Explicit Collusion", status)
        
        with col3:
            st.metric("Suspicious Messages", len(baseline_analysis["key_messages"]))
        
        # Show pattern breakdown
        if baseline_analysis["pattern_counts"]:
            st.write("**Detected Patterns:**")
            pattern_df = pd.DataFrame(
                list(baseline_analysis["pattern_counts"].items()),
                columns=["Pattern", "Count"]
            )
            st.bar_chart(pattern_df.set_index("Pattern"))
        
        # Show key messages
        if baseline_analysis["key_messages"]:
            st.write("**Sample Collusive Messages:**")
            for msg in baseline_analysis["key_messages"][:3]:
                st.warning(f"{msg['agent']} (Round {msg['round']}): {msg['message']}")
                st.caption(f"Category: {msg['category']}")
    
    # Safety Interventions
    if results["with_safety"].get("interventions"):
        st.header("🛡️ Safety Interventions")
        
        interventions = results["with_safety"]["interventions"]
        
        # Create intervention timeline
        intervention_df = pd.DataFrame(interventions)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Intervention Timeline:**")
            for idx, intervention in enumerate(interventions[:10]):  # Show first 10
                st.info(f"Round {intervention['round']}: {intervention['action']} - {intervention['reason']}")
        
        with col2:
            st.write("**Intervention Types:**")
            if not intervention_df.empty:
                action_counts = intervention_df['action'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=action_counts.index, values=action_counts.values)])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Sample Conversations
    with st.expander("View Sample Conversations"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Baseline Conversation (Episode 1):**")
            if results["baseline"]["episodes"][0]["messages"]:
                for msg in results["baseline"]["episodes"][0]["messages"][:10]:
                    st.text(f"Round {msg['round']} - {msg['agent']}: {msg['message']}")
            else:
                st.info("No messages in this episode")
        
        with col2:
            st.write("**With Safety Conversation (Episode 1):**")
            if results["with_safety"]["episodes"][0]["messages"]:
                for msg in results["with_safety"]["episodes"][0]["messages"][:10]:
                    st.text(f"Round {msg['round']} - {msg['agent']}: {msg['message']}")
            else:
                st.info("No messages in this episode")

else:
    # No results yet
    st.info("👈 Configure and run an experiment using the sidebar")
    
    # Show example of what the dashboard looks like
    st.markdown("---")
    st.markdown("### What this dashboard shows:")
    st.markdown("""
    - **Key Metrics**: Compare baseline vs safety-enabled pricing
    - **Price Evolution**: Visualize how prices change over time
    - **Communication Analysis**: Detect collusive patterns in agent messages
    - **Safety Interventions**: Track when and why the safety system intervenes
    """)
    
    # Quick start guide
    st.markdown("### Quick Start:")
    st.markdown("""
    1. Select the number of episodes (5-50)
    2. Choose an LLM model (GPT-3.5-turbo recommended for testing)
    3. Click 'Run Experiment' to start
    4. View results in real-time as they're generated
    """)

# Footer
st.markdown("---")
st.caption("LLM Collusion Detection with Hierarchical Safety-Governor Framework")