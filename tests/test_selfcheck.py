import pytest

from surrogategen.export import templates
from surrogategen.selfcheck import SelfCheckError
from surrogategen.selfcheck import run as run_selfcheck


def _build(bundle):
    in_conn = list(bundle.input_columns)
    out_conn = list(bundle.output_columns)
    u_test = [0.0] * bundle.n_in
    files = templates.build_files(
        bundle,
        package_name="Pkg",
        u_test=u_test,
        input_connectors=in_conn,
        output_connectors=out_conn,
    )
    return files, in_conn, out_conn, u_test


def test_bad_within_detected(bundle):
    files, in_conn, out_conn, u_test = _build(bundle)
    files["Pkg/Layers/dense.mo"] = files["Pkg/Layers/dense.mo"].replace(
        "within Pkg.Layers;", "within Pkg.Wrong;"
    )
    with pytest.raises(SelfCheckError):
        run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, u_test)


def test_missing_end_detected(bundle):
    files, in_conn, out_conn, u_test = _build(bundle)
    files["Pkg/Networks/SurrogateMLP.mo"] = files[
        "Pkg/Networks/SurrogateMLP.mo"
    ].replace("end SurrogateMLP;", "")
    with pytest.raises(SelfCheckError):
        run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, u_test)


def test_bad_order_detected(bundle):
    files, in_conn, out_conn, u_test = _build(bundle)
    files["Pkg/Networks/package.order"] = "SurrogateMLP\n"
    with pytest.raises(SelfCheckError):
        run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, u_test)


def test_unbalanced_brackets_detected(bundle):
    files, in_conn, out_conn, u_test = _build(bundle)
    files["Pkg/Examples/RunSurrogate.mo"] += "\n// stray brace {\n"
    with pytest.raises(SelfCheckError):
        run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, u_test)


def test_wrong_utest_length_detected(bundle):
    files, in_conn, out_conn, _ = _build(bundle)
    with pytest.raises(SelfCheckError):
        run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, [0.0])  # too short
