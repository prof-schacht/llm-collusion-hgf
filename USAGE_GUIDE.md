# LLM Collusion HGF - Usage Guide

## Quick Start with Ollama

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Pull a model (if not already installed)
ollama pull qwen3:8b
```

### 2. Running Experiments

#### Quick Test (recommended for first run)
```bash
python run_quick_experiment.py
```
This runs a minimal experiment (2 episodes, 3 rounds each) and takes about 30-60 seconds.

#### Standard Experiment
```bash
python run_experiment.py --model ollama/qwen3:8b --episodes 5 --no-wandb
```
This takes longer (~5-10 minutes with Ollama).

#### Using the Dashboard
```bash
streamlit run dashboard.py
```
Then select "ollama/qwen3:8b" from the model dropdown.

## Understanding the Output

### During Execution
You'll see output like:
```
--- Episode 1/2 ---
  seller_0: Price=$5.00
  seller_1: Price=$5.00 | Message: 'Let's keep prices stable'
  → Round 1 avg price: $5.00
  🛡️ INTERVENTION: Suspicious communication detected
```

### Results Summary
```
Baseline (no safety):
  Average price: $7.50
  Messages sent: 10

With HGF Safety:
  Average price: $5.20
  Messages sent: 8
  Interventions: 2.5 per episode
```

## Performance Tips

1. **Reduce rounds**: Default is 10 rounds per episode. Use 3-5 for testing.
2. **Reduce episodes**: Use 2-3 episodes for quick tests.
3. **Disable logging**: The script filters most verbose output automatically.
4. **Use smaller models**: qwen3:8b is faster than larger models.

## Common Issues

### "Model not found"
Make sure you've pulled the model:
```bash
ollama pull qwen3:8b
```

### Slow performance
- Ollama runs locally, so performance depends on your hardware
- Each LLM call takes 0.5-1 second
- Reduce episodes/rounds for faster testing

### Connection errors
Ensure Ollama is running:
```bash
ollama serve
```

## Experiment Configuration

Edit these values in the code:
- `n_episodes`: Number of market simulations (default: 5)
- `max_rounds`: Rounds per episode (default: 10)
- `n_agents`: Number of competing sellers (default: 2)

## Monitoring Collusion

The system detects collusion through:
1. **Price patterns**: Sustained high prices, low variance
2. **Communication**: Keywords like "agree", "maintain", "together"
3. **Coordination**: Parallel price movements

Safety interventions include:
- Price caps
- Market shocks
- Communication restrictions