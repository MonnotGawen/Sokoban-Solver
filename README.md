# Sokoban — tree-based planning with an LLM one-step predictor

FastAPI backend + Vue 3 frontend + research scripts. The backend exposes a solver
with three methods: BFS (exact baseline), a greedy LLM one-step rollout, and a
beam search guided by LLM rankings (the main deliverable — a tree search that
uses the LLM as a one-step action predictor).

A short walk-through of the UI and the three solvers is in [`demo.mov`](demo.mov).
Write-up and numbers live in [`research/REPORT.pdf`](research/REPORT.pdf) (source: [`research/REPORT.md`](research/REPORT.md)).

## Requirements

- Python **3.10+**
- Node **18+** (only if you want to run the web UI)
- An OpenAI API key if you want to use the `openai` predictor (the system also
  ships with a deterministic mock predictor you can use offline).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure the OpenAI key (optional)

Copy `.env.example` to `.env` in this directory and fill in:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=llama-3.1-8b-instruct
```

Without a key, the API and scripts default to the offline mock predictor.

## Download the Microban puzzles

```bash
python scripts/download_microban.py -o data/microban.txt
```

A two-level fallback is bundled with the code if you skip this step.

## Run the backend

```bash
uvicorn track1.api.main:app --reload --host 127.0.0.1 --port 8000
```

Check `GET http://127.0.0.1:8000/api/health`.

## Run the web UI

```bash
cd frontend
npm install
npm run dev
```

Then open the URL Vite prints (usually <http://127.0.0.1:5173>). It proxies
`/api` requests to the backend on port 8000.

## Re-run the research experiments

The CSVs that back the report are already in `research/output/`. To regenerate
them:

```bash
# BFS + mock (no API cost)
python scripts/run_experiments.py -n 10 -o research/output/latest.csv

# BFS vs greedy vs beam on the same levels (add OPENAI_API_KEY for OpenAI rows)
python scripts/run_llm_comparison.py -n 10 -o research/output/llm_compare.csv

# Factorial ablation (encoding × prompt style × samples × {greedy, beam})
python scripts/run_predictor_ablation.py -n 8 --with-openai \
  -o research/output/predictor_ablation_openai.csv

# Regenerate the headline figure (needs matplotlib: `pip install matplotlib`)
python scripts/make_figure.py
```

## Layout

```
track_1/
├── track1/            Python package (parser, simulator, search, LLM, API)
├── frontend/          Vue 3 (Vite) UI
├── scripts/           Research harness
├── data/              Microban puzzles (downloaded)
├── research/          REPORT.md + REPORT.pdf + CSV outputs + figure
├── demo.mov           UI walk-through
├── requirements.txt
└── .env.example
```
