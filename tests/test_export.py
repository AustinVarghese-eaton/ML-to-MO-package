from surrogategen.export import templates
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


def test_build_files_tree(bundle):
    files, *_ = _build(bundle)
    expected = {
        "Pkg/package.mo",
        "Pkg/package.order",
        "Pkg/Layers/package.mo",
        "Pkg/Layers/package.order",
        "Pkg/Layers/dense.mo",
        "Pkg/Layers/relu.mo",
        "Pkg/Layers/identity.mo",
        "Pkg/Layers/affine_scale.mo",
        "Pkg/Layers/affine_unscale.mo",
        "Pkg/Layers/exp_mask.mo",
        "Pkg/Networks/package.mo",
        "Pkg/Networks/package.order",
        "Pkg/Networks/SurrogateMLP.mo",
        "Pkg/Networks/SurrogateBlock.mo",
        "Pkg/Examples/package.mo",
        "Pkg/Examples/package.order",
        "Pkg/Examples/RunSurrogate.mo",
    }
    assert set(files) == expected


def test_within_and_dims(bundle):
    files, *_ = _build(bundle)
    assert files["Pkg/package.mo"].startswith("within ;")
    assert files["Pkg/Layers/dense.mo"].startswith("within Pkg.Layers;")
    assert files["Pkg/Networks/SurrogateMLP.mo"].startswith("within Pkg.Networks;")
    mlp = files["Pkg/Networks/SurrogateMLP.mo"]
    # Verify all layer weight matrices are present with correct dimensions (dynamic check)
    for i, (W, _b) in enumerate(bundle.layers, start=1):
        rows, cols = len(W), len(W[0])
        assert f"W{i}[{rows}, {cols}]" in mlp


def test_selfcheck_passes(bundle):
    files, in_conn, out_conn, u_test = _build(bundle)
    run_selfcheck(files, "Pkg", bundle, in_conn, out_conn, u_test)  # should not raise

def test_single_connector_placement():
    from tests.conftest import make_bundle

    b = make_bundle(n_in=1, n_out=1)
    files, in_conn, out_conn, u_test = _build(b)
    run_selfcheck(files, "Pkg", b, in_conn, out_conn, u_test)


def test_no_exp_mask_call_without_log(bundle):
    files, *_ = _build(bundle)
    mlp = files["Pkg/Networks/SurrogateMLP.mo"]
    assert "exp_mask(" not in mlp
    assert "affine_unscale(y_s, y_mean, y_scale)" in mlp


def test_exp_mask_emitted_with_log_mask():
    from tests.conftest import make_bundle

    b = make_bundle(n_in=3, n_out=2)
    b.y_log_mask = [False, True]
    files, in_conn, out_conn, u_test = _build(b)
    mlp = files["Pkg/Networks/SurrogateMLP.mo"]
    assert "y_log_mask[2]" in mlp
    assert "exp_mask(" in mlp
    run_selfcheck(files, "Pkg", b, in_conn, out_conn, u_test)  # should not raise


def test_custom_hidden_layers_selfcheck():
    from tests.conftest import make_bundle_arch

    b = make_bundle_arch(hidden=[64, 32], n_in=3, n_out=2)
    files, in_conn, out_conn, u_test = _build(b)
    run_selfcheck(files, "Pkg", b, in_conn, out_conn, u_test)  # should not raise
    mlp = files["Pkg/Networks/SurrogateMLP.mo"]
    assert "W1[64, 3]" in mlp
    assert "W2[32, 64]" in mlp
    assert "W3[2, 32]" in mlp

