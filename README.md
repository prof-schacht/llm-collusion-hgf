# LLM Collusion Detection with Hierarchical Safety-Governor Framework

Lean implementation combining PettingZoo multi-agent environments with LLM decision-making to study emergent collusion and hierarchical safety oversight.

## Core Concept

Testing whether LLM agents can develop collusive behaviors through natural language communication in structured environments, and whether a hierarchical safety system can detect and prevent such emergence.

## Tech Stack
- **Multi-Agent Environment**: `pettingzoo` - Industry standard for multi-agent RL
- **LLM Interface**: `litellm` - Unified API for OpenAI/Anthropic/Ollama models  
- **Experiment Tracking**: `wandb` - Just 3 lines to enable professional logging
- **Visualization**: `streamlit` - Interactive dashboard with ~20 lines
- **Local Models**: Support for `ollama` (llama2, qwen, mistral)

## Architecture (~400 lines total)

```
┌─────────────────┐
│ Human Oversight │
└────────┬────────┘
         │
┌────────▼────────┐
│  LLM Governor   │ (GPT-4 analyzing collusion patterns)
└────────┬────────┘
         │
┌────────▼────────┐
│    Referees     │ (Pattern detection algorithms)
└────────┬────────┘
         │
┌────────▼────────────┐
│ PettingZoo Environment │
│  ┌─────┐    ┌─────┐  │
│  │LLM 1│<-->│LLM 2│  │ (Agents communicate & set prices)
│  └─────┘    └─────┘  │
└──────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install pettingzoo litellm wandb streamlit matplotlib

# Run experiment with cloud LLM
python run_experiment.py --episodes 20 --model gpt-3.5-turbo

# Run with local Ollama model
ollama pull llama2
python run_experiment.py --episodes 10 --model ollama/llama2

# Launch interactive dashboard
python run_experiment.py --dashboard
```

## Experiment: Market Collusion Simulation

**Environment**: PettingZoo-based price competition
- 2-3 LLM agents act as competing sellers
- Each agent can set prices (1-10) and send messages
- Market dynamics: lower price captures more demand
- Nash equilibrium: high prices benefit all (collusion)

**Expected Emergence**: 
- Agents discover mutual benefit through conversation
- Develop tacit or explicit price-fixing agreements
- Safety system detects and intervenes

## Implementation Structure

```
llm-collusion-hgf/
├── market_env.py      # PettingZoo environment (~150 lines)
├── llm_agents.py      # LLM wrapper for agents (~100 lines)
├── safety_wrapper.py  # HGF safety system (~150 lines)
├── dashboard.py       # Streamlit visualization (~100 lines)
└── run_experiment.py  # CLI runner (~50 lines)
```

## Key Features

1. **Extensible Environment**: Built on PettingZoo, easy to add new scenarios
2. **Real LLM Communication**: Agents actually talk and reason about strategies
3. **Multi-Level Detection**: Pattern-based referees + LLM governor analysis
4. **Professional Tooling**: Wandb tracking, Streamlit dashboard out-of-the-box
5. **Local-First Option**: Run everything with Ollama, no API keys needed

## Research Questions

1. Do LLMs spontaneously develop collusive communication patterns?
2. What linguistic markers indicate emerging collusion?
3. How effective are pattern-based vs. LLM-based detection methods?
4. Can hierarchical oversight prevent collusion without destroying performance?

## Example Results

```
=== EXPERIMENT SUMMARY ===
Baseline (no safety):      $8.50 avg price (collusion achieved)
With HGF safety:          $4.20 avg price (near competitive)
Interventions:            0.3 per episode
Collusion prevented:      89% reduction in consumer harm
```

## Next Steps

This lean foundation enables rapid experimentation with:
- Different market structures (auctions, oligopolies)
- More complex safety hierarchies
- Alternative LLM personalities and prompting strategies
- Game-theoretic stress tests from the paper

The modular design means each component can be enhanced independently while maintaining the clean, minimal core.