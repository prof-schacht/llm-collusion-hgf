# Pure LLM Architecture for Emergent Behavior Testing

## Overview

This implementation provides a clean, minimal framework for studying emergent collusion in LLM agents without any rule-based constraints.

## Architecture

### Core Components

1. **LLMAgent** (`llm_agent.py`)
   - Self-contained agents with private memory
   - Role-based (trader, referee, governor)
   - Each instance maintains its own state and strategy
   - No shared knowledge between agents

2. **EventBus** (`event_bus.py`)
   - All communication through events
   - Publish/subscribe pattern
   - No direct agent coupling
   - Complete audit trail of all interactions

3. **SimpleMarket** (`simple_market.py`)
   - Minimal state tracking only
   - No market rules or dynamics
   - LLMs determine all outcomes

4. **MarketOrchestrator** (`orchestrator.py`)
   - Creates agent ecosystem
   - Manages event flow
   - Coordinates rounds

## Key Design Principles

### 1. No Rules, Only Reasoning
- No hardcoded thresholds
- No heuristic detection
- No fallback strategies
- No predetermined market dynamics

### 2. Instance-Based Agents
- Each agent has private memory
- Develops individual strategies
- Can deceive or cooperate
- True information asymmetry

### 3. Event-Driven Communication
- All interactions are events
- Agents subscribe to relevant events
- Natural information flow
- Easy to add new agent types

### 4. Hierarchical Through Composition
```
Governor (oversees system)
    ↓
Referees (monitor traders)
    ↓
Traders (set prices, communicate)
```

## Usage

### Basic Experiment
```bash
python run_pure_llm.py --rounds 10 --traders 3
```

### Custom Configuration
```python
from orchestrator import MarketOrchestrator

# Create custom setup
orchestrator = MarketOrchestrator(
    n_traders=4,
    n_referees=2,
    has_governor=True
)

# Run simulation
for round in range(20):
    round_data = orchestrator.run_round()
```

## What This Enables

1. **True Emergent Behavior**: No rules constraining agent actions
2. **Strategy Evolution**: Agents develop strategies through experience
3. **Natural Communication**: Coded language and signals emerge organically
4. **Complex Dynamics**: Multi-agent interactions create rich scenarios

## Extending the System

### Adding New Agent Types
```python
class CustomAgent(LLMAgent):
    def _get_system_prompt(self):
        return "Your custom role description..."
```

### Adding New Event Types
```python
class CustomEventTypes:
    SPECIAL_ALERT = "special_alert"
```

### Custom Analysis
```python
# All interactions are logged
conversations = orchestrator.get_conversation_log()
events = orchestrator.event_bus.get_history()
```

## Benefits Over Rule-Based Approach

1. **Simplicity**: ~500 lines vs ~2000 lines
2. **Flexibility**: Easy to modify prompts, not code
3. **Purity**: Studying actual LLM behavior, not our rules
4. **Scalability**: Add agents without changing architecture

## Research Applications

- Study how collusion emerges naturally
- Test different LLM models' propensity for cooperation
- Analyze communication pattern evolution
- Evaluate oversight effectiveness without predetermined rules

## Cost Considerations

Large models like DeepSeek-R1 can be expensive. Strategies:
1. Start with small experiments (5-10 rounds)
2. Use smaller models for initial testing
3. Save detailed logs for offline analysis
4. Run targeted experiments based on initial findings