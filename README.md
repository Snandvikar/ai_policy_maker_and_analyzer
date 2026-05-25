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

### 2. Configure AI provider
This app supports two AI backends:
- `GROQ_API_KEY` — use Groq hosted API for deployment or free cloud hosting.
- `OLLAMA_BASE_URL` — use local Ollama only for development and offline testing.

If `GROQ_API_KEY` is set, the app will call Groq first. If it is not available, the app will fall back to a local Ollama instance if `OLLAMA_BASE_URL` is reachable.

```bash
# Optional: Install Ollama for local testing
curl -fsSL https://ollama.com/install.sh | sh

# Pull local models if testing with Ollama
ollama pull mistral      # Data Query Agent (~4GB)
ollama pull llama3.1     # Policy Agent (~4.7GB)

# Start Ollama server locally
ollama serve
```

> **Without any AI backend:** The dashboard works and a deterministic fallback is used for chatbot features.
> Data queries use keyword SQL, and policy suggestions use the rule-based fallback.

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
- `GROQ_API_KEY` — Groq API key for hosted LLM inference
- `DATA_AGENT_MODEL` — Groq model name used for data-query generation (default: `llama-3.1-8b-instant`)
- `POLICY_AGENT_MODEL` — Groq model name used for policy/copilot output (default: `llama-3.3-70b-versatile`)

## Deployment (for Hackathon or Public Demo)

**Quick version:** Deploy a frozen dashboard snapshot to a public URL in ~**5 minutes** using **Streamlit Community Cloud** (free tier).

### Why This Approach?
- **Models are offline-only**: Data is pre-computed in `database.duckdb`. No ML inference at runtime.
- **No real-time updates needed**: Hackathon demo showcases static analysis.
- **Minimal infrastructure**: Streamlit Cloud handles everything; zero Docker knowledge needed.

### 🏆 Recommended: Streamlit Community Cloud (Fastest)
- See [`DEPLOY_TO_STREAMLIT_CLOUD.md`](DEPLOY_TO_STREAMLIT_CLOUD.md) for step-by-step guide.
- **Public URL in ~5 minutes** — just `git push`
- **Zero config files** — no Dockerfile, no setup
- **Auto-redeploys on GitHub push**
- Free tier: unlimited apps, fast cold starts (~5–10 seconds)
- Cost: **$0** forever (free tier sufficient for hackathon)

**How to deploy:**
```bash
# 1. Push to GitHub
git add . && git commit -m "Deploy" && git push origin main

# 2. Go to https://streamlit.io/cloud → Sign up with GitHub
# 3. Click "New app" → Select repo → Deploy

# 4. Done! URL: https://your-app.streamlit.app
```

### Alternative: Docker + Render (More Control)
- See [`DEPLOY_TO_RENDER.md`](DEPLOY_TO_RENDER.md) for step-by-step guide.
- Public URL in ~15 minutes; requires Dockerfile, .dockerignore, requirements-runtime.txt
- Free tier auto-sleeps (cold starts ~30s); auto-redeploys on GitHub push
- Cost: **$0** for demo; $7/month for always-on
- Better if: you want to learn Docker or plan to deploy non-Streamlit apps later

### Alternative: Cloud Run (Google Cloud)
- Similar to Docker+Render, but via Google Cloud
- Cost: **$0–5/month** (pay per request)
- Better cold-start performance than Render free tier

### 📚 Comparison & Decision Guide
- See [`COMPARISON_STREAMLIT_VS_DOCKER.md`](COMPARISON_STREAMLIT_VS_DOCKER.md) for detailed comparison

### Files Provided (for Docker+Render option):
- `Dockerfile` — Production-ready container spec
- `requirements-runtime.txt` — Minimal deps (no PyTorch/models)
- `docker-compose.yml` — Local testing
- `.dockerignore` — Speeds up build
- `DEPLOYMENT_DB_GUIDE.md` — Deep dive: how DuckDB access changes between local & deployed
- `DEPLOYMENT_SUMMARY.md` — High-level overview + architecture
- `DEPLOYMENT_VISUAL_GUIDE.md` — Architecture diagrams & complete walkthrough

### How DuckDB Access Works (Local vs. Deployed)

**Locally** (your machine):
```python
# config.py
DB_PATH = "database.duckdb"  # Relative to project folder
# Resolves to: d:\UMC_HACK\...\database.duckdb
```

**Deployed** (Streamlit Cloud):
```python
# config.py (same code)
DB_PATH = "database.duckdb"  # Relative to app folder
# Resolves to: /app/database.duckdb (cloned from GitHub)
```

**Result:** Same code, same data, just hosted. See [`DEPLOYMENT_DB_GUIDE.md`](DEPLOYMENT_DB_GUIDE.md) for full technical details.

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