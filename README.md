# Refract

**Regime-Aware Mixture-of-Experts Routing for Heterophilic Graph Neural Networks**

Refract adapts per-node inductive bias at inference time by routing predictions through attribute-dominant and topology-dominant expert networks, guided by a learned 9D regime vector computed from graph structure and node features.

---

## Quickstart

```bash
# 1. Create environment (GPU cluster)
conda env create -f environment.yml
conda activate refract

# 2. Install (local dev)
uv pip install -e ".[dev,umap]" --python .venv/bin/python3

# 3. Download all benchmark datasets
make download

# 4. Tune hyperparameters
python scripts/tune.py chameleon --device auto --n-configs 30

# 5. Full benchmark evaluation
python scripts/eval.py chameleon --device auto

# 6. Regenerate paper figures
make figures
```

## Benchmark pipeline

| Step | Command | Output |
|------|---------|--------|
| Download | `python scripts/download.py --datasets all` | `data/` |
| Tune | `python scripts/tune.py <dataset> --device auto` | `configs/hparams/<dataset>.yaml` |
| Eval | `python scripts/eval.py <dataset> --device auto` | `results/benchmarks/<dataset>.csv` |
| Figures | `python scripts/make_figures.py --legacy-import` | `paper/figures/`, `paper/tables/` |

## Cluster (SLURM)

Fill in `<YOUR_PARTITION>` and `module load cuda/...` in `cluster/`:

```bash
# Tune all datasets in parallel
sbatch cluster/tune_array.sbatch

# Evaluate all datasets in parallel
sbatch cluster/eval_array.sbatch

# Single dataset
sbatch cluster/eval.sbatch chameleon --n-splits 10 --n-seeds 3
```

## Paper figures

`make figures` (after running eval) produces:
- `paper/figures/training_curves.pdf` — val-acc curves for all models × datasets
- `paper/figures/ablation.pdf` — loss-component ablation grouped bar chart
- `paper/figures/routing_analysis.pdf` — gate weight violin plots by regime type
- `paper/tables/main_results.tex` — booktabs LaTeX table, ready to paste

## Supported datasets

**Heterophilic (Platonov 2023):** Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions

**Classic:** Cora, CiteSeer, PubMed, Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor

## Algorithm

For each node, a 9D regime vector encodes: attribute evidence, topology evidence, concordance, log-degree, clustering coefficient, 2-hop topology evidence, hop consistency, and 1-hop / 2-hop feature deviation.

A 3-layer MLP gate reads this vector → 2D softmax `[g_attr, g_topo]`.

Two experts: **AttributeExpert** (2-layer MLP, features only) and **TopologyExpert** (3-channel: low-pass GIN+SAGE×2, ego-bypass H2GCN, high-pass ACM-GNN).

Final prediction: `logits = g_attr × logits_attr + g_topo × logits_topo`

Loss: cross-entropy + alignment (JSD gate↔evidence) + smoothness + per-expert auxiliary + optional gate entropy.

## Tests

```bash
pytest tests -v
```

## Visual debugging interface (optional)

The `app/` directory contains a FastAPI + React interface for inspecting individual runs.

```bash
# Generate artifacts for one run
python scripts/train.py --config configs/hparams/chameleon.yaml --save-artifacts

# Terminal 1 — API
cd app/backend && PYTHONPATH=../.. uvicorn server.main:app --reload --port 8000

# Terminal 2 — UI
cd app/frontend && npm install && npm run dev
```

## Stack

| Layer | Tech |
|-------|------|
| Algorithm | Python 3.12 · PyTorch · PyG |
| Cluster | SLURM sbatch array jobs |
| Backend (optional) | FastAPI · Uvicorn · Pydantic v2 |
| Frontend (optional) | React 19 · TypeScript · Vite |
