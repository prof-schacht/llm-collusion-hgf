# Using LLM Collusion HGF with Ollama

This guide explains how to run the LLM Collusion detection framework with Ollama models.

## Prerequisites

1. Install Ollama from https://ollama.ai
2. Pull a supported model (e.g., qwen3:8b):
   ```bash
   ollama pull qwen3:8b
   ```

## Configuration

The framework automatically configures Ollama to use `http://localhost:11434` as the base URL.

## Running Experiments

### Command Line

```bash
# Run with qwen3:8b model
python run_experiment.py --model ollama/qwen3:8b --episodes 10

# Run without wandb logging
python run_experiment.py --model ollama/qwen3:8b --episodes 10 --no-wandb
```

### Dashboard

```bash
# Launch the Streamlit dashboard
python run_experiment.py --dashboard

# Or directly:
streamlit run dashboard.py
```

In the dashboard, select "ollama/qwen3:8b" from the model dropdown.

## Supported Ollama Models

- `ollama/qwen3:8b` (recommended)
- `ollama/qwen3:14b`
- `ollama/deepseek-r1:1.5b`
- `ollama/llama3.1:8b`
- `ollama/mistral-nemo:latest`

## Troubleshooting

1. **Model not found error**: Make sure you've pulled the model with `ollama pull <model-name>`
2. **Connection error**: Ensure Ollama is running (`ollama serve`)
3. **Slow performance**: Smaller models like qwen3:8b work best for this use case

## Notes

- The framework automatically updates references from `qwen2.5:3b` to `qwen3:8b` if needed
- Ollama models run locally, so no API keys are required
- Performance depends on your hardware; GPU acceleration is recommended