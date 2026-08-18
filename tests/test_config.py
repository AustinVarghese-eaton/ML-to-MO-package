import textwrap

import pytest

from surrogategen import config as config_mod


def _write(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_valid_config_loads(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: IGBTSurrogate
        inputs: [Vge, Ic, Tj]
        outputs: [Vce, Eon]
        """,
    )
    cfg = config_mod.load(p)
    assert cfg.package_name == "IGBTSurrogate"
    assert cfg.input_connectors() == {"Vge": "Vge", "Ic": "Ic", "Tj": "Tj"}
    assert cfg.output_connectors() == {"Vce": "Vce", "Eon": "Eon"}


def test_connector_override(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a, b]
        outputs: [c]
        connectors:
          inputs: {a: alpha}
        """,
    )
    cfg = config_mod.load(p)
    assert cfg.input_connectors()["a"] == "alpha"


def test_invalid_package_name(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: 9bad
        inputs: [a]
        outputs: [b]
        """,
    )
    with pytest.raises(ValueError):
        config_mod.load(p)


def test_overlapping_columns(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a, b]
        outputs: [b]
        """,
    )
    with pytest.raises(ValueError):
        config_mod.load(p)


def test_invalid_connector_name(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: ["1x"]
        outputs: [y]
        """,
    )
    with pytest.raises(ValueError):
        config_mod.load(p)


def test_hidden_layers_default(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a]
        outputs: [b]
        """,
    )
    cfg = config_mod.load(p)
    assert cfg.training.hidden_layers == [128, 128, 64]


def test_hidden_layers_custom(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a]
        outputs: [b]
        training:
          hidden_layers: [64, 32]
        """,
    )
    cfg = config_mod.load(p)
    assert cfg.training.hidden_layers == [64, 32]


def test_hidden_layers_empty_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a]
        outputs: [b]
        training:
          hidden_layers: []
        """,
    )
    with pytest.raises(ValueError):
        config_mod.load(p)


def test_hidden_layers_zero_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        dataset: data.csv
        package_name: Pkg
        inputs: [a]
        outputs: [b]
        training:
          hidden_layers: [64, 0, 32]
        """,
    )
    with pytest.raises(ValueError):
        config_mod.load(p)
