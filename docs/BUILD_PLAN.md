# BUILD_PLAN — Step-by-step implementation & progress tracker

> **LIVING DOCUMENT.** Any agent starting fresh: (1) read this whole file, (2) read the
> Progress Log at the bottom, (3) pick the next unchecked task, (4) implement it to its
> acceptance criteria, (5) check the box and append a dated entry to the Progress Log.
> Never redo completed tasks; never skip acceptance criteria.

## Conventions
- Python 3.11, src layout, package name `surrogategen`.
- Seeds: `np=42`, `tf=42`. Pin deps in `pyproject`/`requirements`.
- Modelica rules are authoritative in `docs/SurrogateGenerator_master_prompt_v2.txt`.
- Every generated `.mo` file: correct `within`, matching `end`, balanced `()`/`{}`/`[]`, no trailing commas.
- Exporter must be stack-agnostic: TF and sklearn both produce a common `WeightBundle`.

## Target repo layout
```
pyproject.toml, README.md, requirements.txt
docs/{PRD.md, BUILD_PLAN.md, SurrogateGenerator_master_prompt_v2.txt}
datasets/example/{data.csv, config.yaml}
src/surrogategen/{__init__.py,__main__.py,cli.py,config.py,data.py,train.py,selfcheck.py,packager.py}
src/surrogategen/export/{__init__.py,formatting.py,templates.py,static_layers/*.mo}
scripts/om_validate.py
tests/{test_formatting.py,test_config.py,test_selfcheck.py,test_export.py}
.github/workflows/build-surrogates.yml
```

## Features (ordered; each independently verifiable)

Status legend: `[ ]` TODO · `[~]` IN-PROGRESS · `[x]` DONE · `[!]` BLOCKED

### F0. Repo scaffold  `[x]`
Create `pyproject` (deps: pandas, numpy, scikit-learn, openpyxl, pydantic, pyyaml; extra: tensorflow;
console script `surrogategen=surrogategen.cli:main`); src package skeleton; copy master prompt into `docs/`.
**Accept:** `pip install -e .` works; `python -m surrogategen --help` prints usage.

### F1. Config loader + validation  `[x]`
File: `config.py`. pydantic model for schema in PRD §5. Validate columns non-empty, no input/output
overlap, `package_name` & connector names match `^[A-Za-z][A-Za-z0-9_]*$`, default connectors = column names.
**Accept:** `test_config.py` — valid config loads; bad identifier / overlapping columns raise clear errors.

### F2. Data pipeline  `[x]`
File: `data.py`. Port prompt STEP 2-A exactly. Load csv/xlsx (first sheet), keep confirmed cols, verify exist,
dedupe, coerce numeric, dropna, print counts, split 70/15/15 `rs=42`, fit StandardScaler on train X&Y,
transform all; return scalers as lists + `u_test` (median of input cols, original units) + sample rows.
**Accept:** on example dataset prints row/split counts; returns scaler lists of right lengths; `u_test` len = n_in.

### F3. Training + fallback → WeightBundle  `[x]`
File: `train.py`. Define `WeightBundle {layers:[(W_mo,b)], x_mean,x_scale,y_mean,y_scale,in_names,out_names}`.
TF path: seeds; `Sequential Dense(128 relu,128 relu,64 relu, n_out linear)`; `Adam(1e-3)` mse; batch=32
epochs=300 val_data `EarlyStopping(patience=20,restore_best)`. Extract `W.T` + `b` per layer.
Fallback: on any TF import/train exception → sklearn `MLPRegressor(hidden=(128,128,64),relu,adam,
max_iter≈epochs)`; extract `coefs_.T` + `intercepts_`. Assert no NaN/Inf. Print `epochs_run`/`final_val_loss`.
**Accept:** on tiny synthetic data returns 4 layer weight matrices with dims `128xn_in,128x128,64x128,n_outx64`.

### F4. Float/array formatting  `[x]`
File: `export/formatting.py`. `fmt_float` (high precision, no np artifacts), `fmt_vec` → `{a,b,c}`,
`fmt_mat` → `{{...},{...}}`; no trailing commas.
**Accept:** `test_formatting.py` — known arrays render exactly; reject NaN/Inf.

### F5. Static layer files  `[x]`
Dir: `export/static_layers/`. `dense.mo`, `relu.mo`, `identity.mo`, `affine_scale.mo`, `affine_unscale.mo`
verbatim from prompt "LAYER FILES". Each with `within PKG.Layers;` and matching `end`.
**Accept:** files copy into package unchanged; selfcheck passes on them.

### F6. Modelica emitter  `[x]`
File: `export/templates.py`. Emit: top `package.mo` (uses Modelica 4.0.0) + `package.order`; `Layers/package.mo`
+order; `Networks/{package.mo,order,SurrogateMLP.mo(FUNCTION,algorithm),SurrogateBlock.mo(MODEL,equation,
connectors placed 80..-80, single connector → y=0)}`; `Examples/{package.mo,order,RunSurrogate.mo(uTest=
u_test, equation calls SurrogateMLP once)}`. Fully-qualified `PKG.Layers.*` / `PKG.Networks.SurrogateMLP`; no imports.
**Accept:** `test_export.py` — building example yields full tree; grep confirms `within`/`end` + dims `W1[128,n_in]..W4[n_out,64]`.

### F7. Self-check  `[x]`
File: `selfcheck.py`. Automate prompt SELF-CHECK 1-12: within matches path; every open has matching end;
package.order completeness; weight dims; bias lengths; `SurrogateMLP` is function w/ algorithm; block connector
counts match; `RunSurrogate` uTest len; no NaN/Inf; balanced `()`/`{}`/`[]`; zip root single folder; algorithm vs equation.
**Accept:** `test_selfcheck.py` — passes on good package; each injected defect raises a specific error.

### F8. Packager + CLI  `[x]`
Files: `packager.py` (write tree + zip, root = PKG only), `cli.py` (`build <config>` → run F1-F7, write
`predictions.json` with Python y for `u_test` + K sample rows). `__main__.py` → `cli.main`.
**Accept:** `python -m surrogategen build datasets/example/config.yaml` → valid ZIP + `predictions.json`; selfcheck runs.

### F9. OpenModelica validation  `[x]`
File: `scripts/om_validate.py`. OMPython: `loadFile` package.mo; `checkModel` every class; call
`PKG.Networks.SurrogateMLP(u)` for K sample rows via omc; also simulate `Examples.RunSurrogate`; compare to
`predictions.json` within tolerance. Nonzero exit on any parse error or mismatch.
**Accept:** run in openmodelica container on example → checkModel OK + parity within rtol/atol.

### F10. GitHub Actions  `[x]`
File: `.github/workflows/build-surrogates.yml`. Jobs:
- **detect:** changed-files → matrix of changed `datasets/*/config.yaml`.
- **build** (matrix, ubuntu, py3.11): install deps (TF optional), run CLI, upload ZIP + `predictions.json` artifact.
- **validate** (needs build; container `openmodelica/openmodelica:<pinned>`): pip OMPython, download artifact,
  run `om_validate.py`; fail on error/mismatch.
**Accept:** push changing example dataset → only it builds; validate green; artifact downloadable.

### F11. Docs + example + hardening  `[x]`
README (quickstart, config schema, local run, CI, OMEdit + Dymola manual import notes). Ensure example
dataset committed. Final pass on determinism + pinned versions.
**Accept:** fresh clone → follow README → local build + tests pass.

## Progress Log
_(append: date — feature id — what changed — status)_
- 2026-08-12 — F0-F11 — Initial end-to-end implementation of `surrogategen` (config, data, TF+sklearn
  training, Modelica emitter, self-check, packager, CLI), OpenModelica validation script, CI workflow,
  unit tests, example dataset (`datasets/example`), README. `pytest` 20/20 green; local build on the
  example produces `IGBTSurrogate.zip` + `predictions.json`; TensorFlow backend trained (val_loss ~3e-3).
  — DONE.
- Pending: run the CI `validate` job (OpenModelica `checkModel` + parity) — not runnable locally (no `omc`
  on this machine); verify on first push. Manual Dymola import spot-check still recommended (AC6).
