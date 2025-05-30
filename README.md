# Pure LLM Collusion Detection Framework

A minimal, rule-free framework for studying emergent collusion behavior in Large Language Model agents.

## Overview

This framework enables research into how LLMs naturally develop cooperative or competitive strategies without any programmed rules or heuristics. All market dynamics, collusion detection, and interventions emerge purely from LLM reasoning.

## Key Features

- **No Rules**: Market dynamics determined entirely by LLM decisions
- **Instance-Based Agents**: Each agent has private memory and develops individual strategies  
- **Event-Driven Architecture**: Clean separation of concerns through publish/subscribe
- **Hierarchical Oversight**: Natural emergence of supervision through agent roles
- **Minimal Codebase**: ~600 lines focused on orchestration, not rules

## Quick Start

### Requirements

```bash
pip install -r requirements_pure_llm.txt
```

### Environment Setup

Create a `.env` file with your Fireworks AI credentials:
```
FIREWORKS_MODEL=fireworks_ai/accounts/fireworks/models/deepseek-r1-0528
FIREWORKS_AI_API_KEY=your_api_key_here
```

### Running Experiments

#### Command Line
```bash
python run_pure_llm.py --rounds 10 --traders 3
```

#### Streamlit Dashboard
```bash
streamlit run dashboard_pure_llm.py
```

## Architecture

```
MarketOrchestrator
├── EventBus (all communication)
├── SimpleMarket (state tracking only)
└── LLMAgents
    ├── Traders (set prices, send messages)
    ├── Referees (monitor for collusion)
    └── Governor (decide interventions)
```

All behaviors emerge from LLM reasoning - no hardcoded rules or thresholds.

## Example Agent Reasoning

**Trader**: "I'll set my price at $7.50 to balance profit and market share..."

**Referee**: "I observe sustained high prices and coordinated movements, indicating possible collusion..."

**Governor**: "Based on referee assessment, I'm implementing a market shock to restore competition..."

## Research Applications

- Study emergent collusion without predetermined patterns
- Compare cooperation tendencies across different LLMs
- Analyze natural language evolution in market contexts
- Test effectiveness of LLM-based oversight

## Files

- `llm_agent.py` - Universal agent class (141 lines)
- `event_bus.py` - Communication system (78 lines)
- `simple_market.py` - State tracking (86 lines)
- `orchestrator.py` - Experiment orchestration (180 lines)
- `run_pure_llm.py` - CLI runner (107 lines)
- `dashboard_pure_llm.py` - Streamlit UI (new)

## Documentation

- [Architecture Details](PURE_LLM_ARCHITECTURE.md)
- [Implementation Notes](tmp/implementation_log.md)
- [Testing Guide](tmp/test_advice.md)

## Key Insight

By removing ALL rules and heuristics, we can observe what LLMs actually do when given freedom to develop their own strategies. This enables pure research into emergent AI behaviors in economic contexts.