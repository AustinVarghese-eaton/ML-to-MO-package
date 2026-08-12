"""Configuration schema and validation for a surrogate build.

One YAML file per dataset describes what to build. This replaces the interactive
"confirm the schema" step of the original SurrogateGenerator prompt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _is_valid_identifier(name: str) -> bool:
    return bool(IDENTIFIER_RE.match(name))


class ConnectorOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)


class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 20


class Tolerance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rtol: float = 1e-4
    atol: float = 1e-6


class SurrogateConfig(BaseModel):
    """Validated build configuration.

    Fields mirror PRD section 5. Column and package/connector names are validated
    against the Modelica identifier regex ``^[A-Za-z][A-Za-z0-9_]*$``.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    sheet: Optional[str] = None
    package_name: str
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    connectors: ConnectorOverrides = Field(default_factory=ConnectorOverrides)
    training: TrainingParams = Field(default_factory=TrainingParams)
    tolerance: Tolerance = Field(default_factory=Tolerance)

    # Resolved absolute path to the dataset file (populated by ``load``).
    config_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("package_name")
    @classmethod
    def _valid_package_name(cls, v: str) -> str:
        if not _is_valid_identifier(v):
            raise ValueError(
                f"package_name '{v}' is not a valid Modelica identifier "
                f"(must match ^[A-Za-z][A-Za-z0-9_]*$)"
            )
        return v

    @model_validator(mode="after")
    def _validate_schema(self) -> "SurrogateConfig":
        # No duplicate columns within inputs or outputs.
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Duplicate column names found in 'inputs'.")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Duplicate column names found in 'outputs'.")

        # No overlap between inputs and outputs.
        overlap = set(self.inputs) & set(self.outputs)
        if overlap:
            raise ValueError(
                f"Columns cannot be both input and output: {sorted(overlap)}"
            )

        # Connector overrides must reference known columns and be valid identifiers.
        for col in self.connectors.inputs:
            if col not in self.inputs:
                raise ValueError(
                    f"connectors.inputs references unknown input column '{col}'."
                )
        for col in self.connectors.outputs:
            if col not in self.outputs:
                raise ValueError(
                    f"connectors.outputs references unknown output column '{col}'."
                )

        # Every resolved connector name (override or default = column name) must be
        # a valid Modelica identifier and unique across all connectors.
        resolved = list(self.input_connectors().values()) + list(
            self.output_connectors().values()
        )
        for name in resolved:
            if not _is_valid_identifier(name):
                raise ValueError(
                    f"Connector name '{name}' is not a valid Modelica identifier "
                    f"(must match ^[A-Za-z][A-Za-z0-9_]*$). Add a 'connectors' override."
                )
        if len(set(resolved)) != len(resolved):
            raise ValueError("Connector names must be unique across inputs and outputs.")

        return self

    def input_connectors(self) -> dict[str, str]:
        """Map input column -> connector name (override or column name)."""
        return {c: self.connectors.inputs.get(c, c) for c in self.inputs}

    def output_connectors(self) -> dict[str, str]:
        """Map output column -> connector name (override or column name)."""
        return {c: self.connectors.outputs.get(c, c) for c in self.outputs}

    def dataset_path(self) -> Path:
        """Absolute path to the dataset, resolved relative to the config file."""
        base = self.config_dir or Path.cwd()
        return (base / self.dataset).resolve()


def load(config_path: str | Path) -> SurrogateConfig:
    """Load and validate a YAML config file."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    cfg = SurrogateConfig(**raw)
    cfg.config_dir = path.parent
    return cfg
