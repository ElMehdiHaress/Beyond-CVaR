def test_import_package():
    import beyond_cvar  # noqa: F401

def test_import_modules():
    import importlib
    for m in [
        "prediction_error_cpt",
        "prediction_error_srm",
        "excess_risk_over_t_srm",
        "excess_risk_over_n_srm",
    ]:
        importlib.import_module(f"beyond_cvar.{m}")
