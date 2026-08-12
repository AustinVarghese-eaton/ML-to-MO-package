# How to Use — ML-to-MO Surrogate Package Generator

This guide explains how to turn a dataset into a Modelica `.mo` surrogate package using
this project — both locally and through the automated GitHub Actions pipeline.

For *why* the project exists and its requirements, see [docs/PRD.md](docs/PRD.md).
For implementation status, see [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

---

## What you get

From one dataset (CSV/XLSX) + one config file, the tool produces a Modelica 4.0.0 package:

- `Networks.SurrogateMLP` — a **function** with the trained network math (the source of truth).
- `Networks.SurrogateBlock` — a **model** with input/output connectors for the diagram canvas.
- `Examples.RunSurrogate` — a **model** you can simulate immediately with editable test inputs.

Deliverables per build: `<PackageName>.zip`, the expanded `<PackageName>/` folder, and
`<PackageName>.predictions.json` (used to verify the Modelica outputs match Python).

---

## One-time setup (for local builds)

You only need this if you want to build/test on your machine. CI needs none of it.

```powershell
# From the repository root
python -m pip install -e ".[dev]"
# Optional primary training backend; if missing, scikit-learn is used automatically
python -m pip install tensorflow
```

Requirements: Python 3.10+ (3.11 recommended). OpenModelica is only needed for the
validation step and is provided automatically in CI.

---

## How to add a new dataset and get its `.mo` package

### Step 1 — Create the dataset folder
Put your data file and a config side-by-side under a new folder in `datasets/`:

```
datasets/<your_name>/data.csv        (or data.xlsx)
datasets/<your_name>/config.yaml
```

### Step 2 — Write `config.yaml`

```yaml
dataset: data.csv          # file name, relative to this config file
sheet: null                # for .xlsx put the sheet name (e.g. Sheet1); null for CSV
package_name: MySurrogate  # must match ^[A-Za-z][A-Za-z0-9_]*$ (letter, then letters/digits/_)
inputs:  [colA, colB, colC]   # exact dataset column headers
outputs: [colX, colY]         # exact dataset column headers (no overlap with inputs)
connectors:                # optional; leave empty to use the column names as-is
  inputs:  {}
  outputs: {}
training:                  # optional overrides
  epochs: 300
  batch_size: 32
  learning_rate: 0.001
  patience: 20
tolerance:                 # parity tolerance used by OpenModelica validation
  rtol: 1.0e-4
  atol: 1.0e-6
```

Notes:
- `dataset` must match the **actual file name** (including `.xlsx` vs `.csv`).
- For Excel, set `sheet` to the sheet that holds the data.
- Only add a `connectors` override if a column name is **not** a valid Modelica identifier
  (e.g. starts with a digit or contains a space). Example: `inputs: {"1x": current}`.

### Step 3 — (Recommended) test locally first

```powershell
python -m surrogategen build datasets/<your_name>/config.yaml --out out
```

This produces `out/MySurrogate.zip` and `out/MySurrogate.predictions.json`. If the
self-check fails, it prints exactly what's wrong before you push.

### Step 4 — Commit and push

```powershell
git add datasets/<your_name>
git commit -m "Add <your_name> dataset"
git push
```

### Step 5 — Watch the pipeline
On GitHub → **Actions** tab → open the latest `build-surrogates` run. Three jobs run:

1. **detect** — sees your new dataset changed, builds a matrix for just it.
2. **build** — trains + exports, uploads the package.
3. **validate** — OpenModelica `checkModel` + numeric parity + simulate. Green check = the
   package parses, compiles, and matches the Python model.

### Step 6 — Download the `.mo` package
In the completed run's summary page, scroll to **Artifacts** → download
`surrogate-<your_name>`. Inside:

- `MySurrogate/` — the Modelica package folder
- `MySurrogate.zip` — the deliverable
- `MySurrogate.predictions.json` — parity reference

### Step 7 — Use it

- **OpenModelica:** unzip, open `MySurrogate/package.mo` in OMEdit → expand `Examples` →
  `RunSurrogate` → edit `uTest` → Simulate.
- **Dymola:** `File ▸ Open` the `package.mo`, then check/simulate.
- **In code:** `y = MySurrogate.Networks.SurrogateMLP(u);`

---

## Running the pipeline without committing a change

GitHub → **Actions** → `build-surrogates` → **Run workflow** (manual dispatch):

- Leave the `config` box **empty** to rebuild the datasets changed in the branch.
- Or type a single path like `datasets/example/config.yaml` to build just that one.

---

## What triggers a build

- Editing a file under `datasets/**` → rebuilds **only that dataset**.
- Editing anything under `src/**` or `scripts/**` → rebuilds **all datasets** (the
  generator changed, so every package is regenerated).

---

## Config field reference

| Field | Required | Description |
|-------|----------|-------------|
| `dataset` | yes | Data file name, relative to the config file. |
| `sheet` | no | Excel sheet name; `null` for CSV or first sheet. |
| `package_name` | yes | Modelica package name; `^[A-Za-z][A-Za-z0-9_]*$`. |
| `inputs` | yes | List of input column headers (network inputs). |
| `outputs` | yes | List of output column headers (network outputs). |
| `connectors.inputs` | no | Map `column -> connector name` overrides. |
| `connectors.outputs` | no | Map `column -> connector name` overrides. |
| `training.epochs` | no | Max training epochs (default 300). |
| `training.batch_size` | no | Batch size (default 32). |
| `training.learning_rate` | no | Adam learning rate (default 0.001). |
| `training.patience` | no | Early-stopping patience (default 20). |
| `tolerance.rtol` | no | Relative parity tolerance (default 1e-4). |
| `tolerance.atol` | no | Absolute parity tolerance (default 1e-6). |

Rules enforced at build time: inputs and outputs must exist in the data, must not overlap,
and all package/connector names must be valid Modelica identifiers.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `Dataset not found` | `dataset:` name doesn't match the real file (check `.xlsx` vs `.csv`). |
| `Columns not found in dataset` | An `inputs`/`outputs` name doesn't match a header exactly (case-sensitive). |
| `not a valid Modelica identifier` | Rename `package_name`, or add a `connectors` override for that column. |
| `Columns cannot be both input and output` | Remove the shared column from one list. |
| `Too few usable rows` | Need ≥ 10 rows after dropping duplicates/NaNs. |
| Self-check failure | The tool lists each problem; fix and rebuild. |
| TensorFlow errors in CI | Ignored — the build automatically falls back to scikit-learn. |
| Validate job fails | OpenModelica `checkModel` or parity mismatch; open the job log for the exact class/row. |

---

## Notes

- Each dataset can set its own `package_name`. If you plan to load several packages into
  one Modelica library at once, give them **distinct** package names to avoid clashes.
- The generated math is written once (in `SurrogateMLP`); the block and example both call
  it, so there is no duplicated/weight drift.
- The Modelica network uses standardized inputs internally and returns values in the
  original units — no manual scaling needed by the caller.
