# Refract — Developer Context

## Architecture

```
src/refract/      Core algorithm — pure Python, no HTTP concerns
  datasets/       GraphDataset dataclass + loaders
  regimes/        Regime vector computation and taxonomy
  models/         Expert networks, gate, routed GNN, baselines
  training/       Trainer, evaluator, checkpoints, seed, device
  counterfactuals/  Perturbations and precomputation
  projection/     Diagnostic projection (UMAP/PCA) + distortion
  artifacts/      Parquet/JSON writer and reader

schemas/          Pydantic v2 schemas — shared by server/ and refract/

scripts/
  download.py     Download all benchmark datasets
  tune.py         Hyperparameter search → configs/hparams/{dataset}.yaml
  train.py        Single training run from a YAML config
  eval.py         Full multi-split benchmark → results/benchmarks/
  make_figures.py Emit all paper figures and LaTeX table

configs/
  datasets/       Default per-dataset training configs
  hparams/        Tuned configs (output of tune.py)

cluster/          SLURM sbatch templates (fill in partition/module)

results/          Gitignored — eval outputs, sweep logs
  benchmarks/     CSVs + parquet from eval.py
  hparams/        Sweep logs from tune.py

paper/
  figures/        Generated PDFs (gitignored)
  tables/         LaTeX tables (gitignored)
  notes/          Research notes

app/              Optional visual debugging interface
  backend/
    server/       FastAPI — routers, dependencies, settings
  frontend/       React 19, Vite 5, TypeScript strict

tests/            pytest unit tests
data/             Downloaded datasets (gitignored)
```

## Dependency direction

```
server/ → schemas/ ← refract/
server/ → refract/
```

`src/refract/` never imports from `server/`. `schemas/` is framework-agnostic Pydantic.

## ML quickstart

```bash
# Install (from repo root)
uv pip install -e ".[dev,umap]" --python .venv/bin/python3

# Download all datasets
python scripts/download.py --datasets all

# Tune hyperparameters (writes configs/hparams/{dataset}.yaml)
python scripts/tune.py chameleon --device auto --n-configs 30

# Full benchmark (writes results/benchmarks/)
python scripts/eval.py chameleon --device auto

# Generate paper figures
python scripts/make_figures.py --legacy-import
```

## Running locally (web app)

```bash
cd app/frontend && npm install

# Terminal 1
cd app/backend && PYTHONPATH=../.. uvicorn server.main:app --reload --port 8000

# Terminal 2
cd app/frontend && npm run dev

# Generate debug artifacts for the web app
python scripts/train.py --config configs/hparams/chameleon.yaml --save-artifacts
```

## Running tests

```bash
pytest tests -v
cd app/frontend && npm test
```

## Key invariants

- Regime metrics never access test labels — only `train_mask` indices are used
- All ML random ops accept a `seed` parameter
- Artifact files are read-only at API time — no live ML in the HTTP layer
- Gate weights always sum to 1 (Softmax in RoutingGate)
- `GraphDataset.__post_init__` validates all tensor shapes
- Device selection order: `cuda > mps > cpu` (use `select_device("auto")` from `refract.training.device`)

## Adding a new dataset

1. Create a loader in `src/refract/datasets/` returning `GraphDataset`
2. Register the name in `src/refract/datasets/real_world.py`
3. Add a default config in `configs/datasets/{name}.yaml`

## Algorithm

For each node, Refract computes a 9D regime vector (attribute evidence, topology evidence, concordance, log-degree, clustering coefficient, 2-hop topology evidence, hop consistency, feat deviation 1-hop, feat deviation 2-hop). A 3-layer MLP gate reads this vector and outputs a 2D softmax over [attribute, topology]. Two expert networks — AttributeExpert (2-layer MLP) and TopologyExpert (3-channel: low-pass GIN+SAGE×2, ego bypass H2GCN-style, high-pass ACM-GNN). Final prediction: `logits = g_attr * logits_attr + g_topo * logits_topo`.

Loss: cross-entropy + gate alignment (JSD) + gate smoothness + per-expert auxiliary loss (with floor) + optional gate entropy.

## Adding a new expert

1. Add a new `nn.Module` in `src/refract/models/experts.py`
2. Update `RoutedGNN` in `src/refract/models/routed_gnn.py`
3. Update the gate output dimension
4. Update `ExpertKey` in `app/frontend/src/api/types.ts`

## Code style

- No comments
- Python: ruff + mypy strict, line length 100
- TypeScript: strict mode, no `any`
- No in-function imports
- No `sys.path` manipulation
