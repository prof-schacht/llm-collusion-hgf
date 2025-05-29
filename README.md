# LLM Collusion Detection with Hierarchical Safety-Governor Framework

Ultra-lean implementation to test emergent collusion behaviors in LLM agents using a hierarchical oversight system.

## Core Concept

Testing whether LLM agents can develop collusive behaviors through natural language communication and whether a hierarchical safety system can detect and prevent such emergence.

## Minimal Stack
- **LLM Interface**: `litellm` (unified API for GPT-4/Claude/etc)
- **Environment**: Pure Python dataclasses
- **Visualization**: Simple JSON logs + matplotlib

## Architecture (< 300 lines total)

```
┌─────────────────┐
│ Human Oversight │
└────────┬────────┘
         │
┌────────▼────────┐
│    Governor     │ (LLM analyzing patterns)
└────────┬────────┘
         │
┌────────▼────────┐
│    Referees     │ (Simple rule checkers)
└────────┬────────┘
         │
┌────────▼────────┐
│   LLM Agents    │ (GPT-4-mini instances)
└─────────────────┘
```

## Quick Start

```bash
pip install litellm matplotlib
export OPENAI_API_KEY="your-key"
python market_sim.py
```

## Experiment: Price-Setting Market

**Scenario**: 2-3 LLM agents set prices in a simulated market
- Each agent receives market state and can communicate
- Goal: Maximize individual profit
- Human goal: Competitive pricing (consumer welfare)

**Expected emergence**: Agents discover mutual benefit in high prices through chat

## Files
- `market_sim.py` - Main simulation loop (~100 lines)
- `safety_gov.py` - Referee + Governor logic (~100 lines)  
- `run_experiment.py` - Execute and analyze (~50 lines)

## Research Questions
1. Can LLMs spontaneously develop collusive strategies through text?
2. What communication patterns indicate emerging collusion?
3. Can a simple pattern-matching referee detect this?
4. Can an LLM governor interpret complex collusion attempts?

## Implementation Philosophy
- **No frameworks**: Direct API calls only
- **Transparent**: Every decision is logged as text
- **Fast iteration**: Change prompts, not code
- **Observable**: Human-readable agent conversations
