# MCI Digital Desert Dashboard

## Project structure

```
mci_project/
├── config.py                         # DB path, weights, model names, schema
├── requirements.txt
├── pipeline/
│   ├── mci_pipeline.py               # Step 1 — compute all scores (timeseries-aware)
│   └── store_dashboard_tables.py     # Step 2 — build 4 dashboard tables
├── policy_engine/
│   └── rules.py                      # Rule-based policy suggestion engine
├── agents/
│   ├── data_query_agent.py           # Agent 1 — SQL generation + data retrieval
│   ├── policy_agent.py               # Agent 2 — impact analysis + recommendations
│   └── orchestrator.py              # Routes user messages to correct agent
└── dashboard/
    ├── app.py                        # Main Streamlit entry point
    ├── filters.py                    # Sidebar filters + data loading
    ├── charts.py                     # All Plotly chart builders
    ├── area_detail.py                # Drill-down panel + policy suggestions
    └── chatbot.py                    # Chatbot UI component
```

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 2. Install Ollama (for AI chatbot)
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull mistral      # Data Query Agent (~4GB)
ollama pull llama3.1     # Policy Agent (~4.7GB)

# Start Ollama server
ollama serve
```

> **Without Ollama:** The dashboard and rule-based suggestions work fully.
> The chatbot falls back to keyword-based SQL and rule-based policy text.
> No AI features are broken — they just use deterministic fallbacks.

### 3. Run the pipeline
```bash
# Step 1: Compute all factor scores and MCI (all years)
python -m pipeline.mci_pipeline

# Step 2: Build dashboard tables (clusters, RF importance, subcomponents)
python -m pipeline.store_dashboard_tables

# Step 3: Launch the dashboard
streamlit run dashboard/app.py
```

## Database tables

| Table | Purpose |
|-------|---------|
| `mci_scores` | Latest year scores per area — primary dashboard table |
| `mci_timeseries_scores` | All years — trend charts and policy agent context |
| `factor_subcomponent_scores` | Sub-component scores — radar charts |
| `rf_feature_importance` | RF feature importance — intervention levers |
| `cluster_profiles` | K-Means cluster centroids and labels |

## Chatbot agents

### Agent 1 — Data Query Agent
- **Model:** Mistral 7B (via Ollama)
- **Fallback:** Keyword-based SQL builder (no Ollama needed)
- **Does:** Converts questions to SQL, executes against DuckDB, suggests chart type
- **Example queries:** "Which areas have lowest MCI?", "Show trend for Mumbai", "Compare Tier 2 cities"

### Agent 2 — Policy Suggestion Agent  
- **Model:** LLaMA 3.1 8B (via Ollama)
- **Fallback:** Rule-based engine (`policy_engine/rules.py`)
- **Does:** Root cause analysis, scheme-specific recommendations, expected impact
- **Example queries:** "Why is Dharavi low?", "How can we improve women employment here?"

### Orchestrator
Classifies user intent (data vs policy) using keyword matching.
Falls back to policy if an area is selected in the dashboard.

## Config

Edit `config.py` to change:
- `DB_PATH` — path to your DuckDB database
- `BASELINE_YEAR` — normalisation anchor year (default 2021)
- `LATEST_YEAR` — default display year (default 2024)
- `MCI_WEIGHTS` — factor weights in the geometric mean
- `DATA_AGENT_MODEL` / `POLICY_AGENT_MODEL` — Ollama model names
- `OLLAMA_BASE_URL` — if Ollama runs on a different host/port

## Extending the policy engine

Add new rules in `policy_engine/rules.py` by appending to the `RULES` list:
```python
{
    "condition":   lambda s: s.get("YOUR_SCORE_KEY", 100) < YOUR_THRESHOLD,
    "priority":    1,    # 1=urgent, 2=moderate, 3=advisory
    "category":    "IFS",  # factor this targets
    "short_title": "Your intervention title",
    "action":      "What specifically to do",
    "rationale":   "Why this score triggers this rule",
    "scheme":      "Government scheme name",
}
```