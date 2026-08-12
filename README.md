# ML-to-MO Surrogate Package Generator

Convert tabular datasets (CSV/XLSX) into **Modelica 4.0.0** neural-network surrogate
packages (`.mo`) that drop into **OpenModelica/OMEdit** and **Dymola**. Training and
export are deterministic and automated via **GitHub Actions**.

The generated package exposes the network math once, as a function
(`Networks.SurrogateMLP`), with a thin connector model (`Networks.SurrogateBlock`) for
the diagram canvas and a runnable example (`Examples.RunSurrogate`).

## How it works

```
datasets/<name>/config.yaml + data.csv
        │
        ▼
 surrogategen build  ──►  <PKG>/            (Modelica package tree)
                          <PKG>.zip         (deliverable)
                          <PKG>.predictions.json  (parity reference)
        │
        ▼
 OpenModelica (CI)  ──►  checkModel + function parity + simulate
```

## Install

```powershell
python -m pip install -e .
# optional primary training backend (falls back to scikit-learn if absent):
python -m pip install tensorflow
```

## Build a package locally

```powershell
python -m surrogategen build datasets/example/config.yaml --out out
```

Outputs in `out/`:
- `IGBTSurrogate/` — the Modelica package folder
- `IGBTSurrogate.zip` — the deliverable (single root folder inside)
- `IGBTSurrogate.predictions.json` — Python predictions used for numeric parity

## Config schema

```yaml
dataset: data.csv          # path relative to this config file
sheet: null                # xlsx sheet name, or null for CSV / first sheet
package_name: IGBTSurrogate
inputs: [Vge, Ic, Tj]
outputs: [Vce, Eon]
connectors:                # optional; default connector name = column name
  inputs: {Vge: vge}
  outputs: {Vce: vce}
training:                  # optional overrides
  epochs: 300
  batch_size: 32
  learning_rate: 0.001
  patience: 20
tolerance:                 # parity tolerance used by OpenModelica validation
  rtol: 1.0e-4
  atol: 1.0e-6
```

Package and connector names must match the Modelica identifier regex
`^[A-Za-z][A-Za-z0-9_]*$`. Add a `connectors` override if a column name is not a valid
identifier.

## Test

```powershell
python -m pip install -e ".[dev]"
pytest
```

## GitHub Actions

`.github/workflows/build-surrogates.yml` runs on every push touching `datasets/**` (or
generator code) and on manual dispatch:

1. **detect** — builds a matrix of only the dataset configs that changed (rebuilds all if
   generator code changed).
2. **build** — installs the package, trains, exports, and uploads the ZIP +
   `predictions.json` as a run artifact per dataset.
3. **validate** — inside the `openmodelica/openmodelica` container, runs `checkModel` on
   every class, compares `SurrogateMLP(u)` to the Python predictions within tolerance, and
   simulates `RunSurrogate`.

Download the built package from the workflow run's **Artifacts** section.

## Using the package

- **Simulate:** open the package in OpenModelica, expand `Examples`, open `RunSurrogate`,
  edit `uTest`, and click Simulate.
- **Canvas:** drag `Networks.SurrogateBlock` into a diagram and wire its ports.
- **In code:** `y = <PKG>.Networks.SurrogateMLP(u);`

### Dymola

Dymola is not run in CI (it requires a licensed install + license server). The generated
code targets standard Modelica 4.0.0 and imports into Dymola manually: `File ▸ Open` the
`<PKG>/package.mo`, then check/simulate. Verify one package by hand after major changes.

## Repository layout

```
docs/                PRD, BUILD_PLAN (implementation tracker), master prompt
datasets/<name>/     config.yaml + data file
src/surrogategen/    config, data, train (TF + sklearn fallback), export/, selfcheck, cli
scripts/om_validate.py   OpenModelica compile + numeric parity
tests/               unit tests
.github/workflows/   CI pipeline
```

See [docs/PRD.md](docs/PRD.md) for requirements and [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)
for the feature-by-feature implementation status.
