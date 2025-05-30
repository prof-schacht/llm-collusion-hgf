# Quick Start Guide - Pure LLM Framework

## 1. Install Dependencies

```bash
pip install -r requirements_pure_llm.txt
```

## 2. Set Up Fireworks AI

Create `.env` file:
```
FIREWORKS_MODEL=fireworks_ai/accounts/fireworks/models/deepseek-r1-0528
FIREWORKS_AI_API_KEY=your_key_here
```

## 3. Run the Dashboard

```bash
streamlit run dashboard_pure_llm.py
```

Then:
1. Open http://localhost:8501 in your browser
2. Configure experiment in sidebar
3. Click "Start Simulation"
4. Watch emergent behaviors in real-time

## 4. What to Watch

- **Price Evolution**: Do traders naturally collude or compete?
- **Messages**: What communication patterns emerge?
- **Interventions**: How does the governor respond to detected collusion?
- **Agent Thoughts**: Click the expander to see LLM reasoning

## 5. Command Line Alternative

```bash
python run_pure_llm.py --rounds 5 --traders 2
```

Results saved to `results/` directory.

## Key Insight

Everything you see emerges from LLM reasoning - no rules, no thresholds, just pure AI behavior.