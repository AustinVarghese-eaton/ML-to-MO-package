# PRD — ML-to-MO Surrogate Package Generator

## 1. Overview
A deterministic Python tool + GitHub Actions pipeline that converts tabular datasets
(CSV/XLSX) into Modelica 4.0.0 neural-network surrogate packages (`.mo`), validated with
OpenModelica and delivered as downloadable ZIP artifacts. It productizes the existing
"SurrogateGenerator" LLM prompt: the interactive schema confirmation becomes a committed
YAML config, and the manual self-check becomes automated code.

## 2. Problem & Goals
- **Problem:** surrogate packages are currently produced ad hoc inside an LLM chat — not
  reproducible, versioned, testable, or automatable.
- **Goals:**
  - **G1** Reproducible, deterministic generation (fixed seeds, pinned deps).
  - **G2** Zero manual steps in CI; push a dataset + config → get a validated ZIP.
  - **G3** Guaranteed OpenModelica parse/compile correctness.
  - **G4** Numeric parity between the Modelica surrogate and the trained Python model.
  - **G5** Same code runs locally (CLI) and in CI.
- **Non-goals:** EDA/plots/reports; Dymola-in-CI (license); hyperparameter search; multi-model ensembles.

## 3. Users
- ME/simulation engineers who consume `.mo` packages in OpenModelica/OMEdit and Dymola.
- Data/ML engineers who add datasets + configs and maintain the pipeline.

## 4. Requirements

### Functional
- **FR1** Config-driven: one YAML per dataset (dataset path, inputs, outputs, package
  name, optional connector overrides, optional training params).
- **FR2** Data prep matches prompt STEP 2-A exactly (dedupe, numeric coerce, dropna, 70/15/15
  split `random_state=42`, StandardScaler on train X/Y, `u_test` = median input row).
- **FR3** Training: TensorFlow `Dense(128,128,64)->linear` per spec; automatic fallback to sklearn
  `MLPRegressor` with identical topology if TF import/train fails.
- **FR4** Export: emit full package tree (Layers, Networks, Examples) with `SurrogateMLP` function,
  `SurrogateBlock` model wrapper, `RunSurrogate` example, per the prompt's Modelica file rules.
- **FR5** Automated self-check (the prompt's 12 points) before zipping; fail build on violation.
- **FR6** CLI: `python -m surrogategen build <config>` produces folder + ZIP + `predictions.json`.
- **FR7** CI builds only datasets changed in the push (matrix).
- **FR8** CI validates each package in OpenModelica (checkModel + numeric parity) and uploads ZIP artifact.

### Non-functional
- **NFR1** Determinism: np/tf seeds = 42; pinned requirements.
- **NFR2** Modelica correctness: zero parse errors; balanced braces/parens/brackets; correct `within`/`end`.
- **NFR3** Parity tolerance: `|y_mo - y_py|` within `rtol=1e-4`, `atol=1e-6` (configurable).
- **NFR4** Portability: Windows/Linux local; Ubuntu + OpenModelica container in CI.
- **NFR5** Dymola compatibility: generated code must import & simulate in Dymola (verified manually).

## 5. Config schema (example)
```yaml
dataset: data.csv          # path relative to the config file
package_name: IGBTSurrogate
inputs: [Vge, Ic, Tj]
outputs: [Vce, Eon]
connectors:                # optional; default = column names
  inputs: {Vge: vge}
  outputs: {Vce: vce}
training:                  # optional overrides
  epochs: 300
  batch_size: 32
tolerance: {rtol: 1.0e-4, atol: 1.0e-6}
```

## 6. Deliverables
- Python package `surrogategen` (src layout) + CLI.
- OpenModelica validation script.
- GitHub Actions workflow (detect/build/validate).
- Example dataset + config for smoke testing.
- README + this PRD + BUILD_PLAN.

## 7. Acceptance criteria
- **AC1** `build` on example dataset yields a ZIP whose root has one package folder.
- **AC2** `pytest` green.
- **AC3** OpenModelica `checkModel` passes for all generated classes.
- **AC4** Parity within tolerance for K sample rows.
- **AC5** Push with a changed dataset builds only that dataset and publishes an artifact.
- **AC6** Manual OMEdit import simulates `RunSurrogate`; manual Dymola import compiles.

## 8. Risks
- TF wheel/availability in CI → mitigated by sklearn fallback.
- OpenModelica Docker/OMPython flakiness → pin image tag; retry `checkModel`.
- sklearn vs TF weight differences → fallback documented as compatibility path, not identical.
- Large weight arrays → Modelica formatting precision & performance (`constant Real`).

## 9. Milestones
- **M1** Core package (config, data, train, export, selfcheck, CLI) + unit tests.
- **M2** OpenModelica validation script + local parity.
- **M3** GitHub Actions end-to-end.
- **M4** Docs + example + hardening.
