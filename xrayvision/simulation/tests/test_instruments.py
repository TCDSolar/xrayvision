from xrayvision.simulation.instruments import rhessi_like_uv_coverage, stix_like_uv_coverage


def test_rhessi_like_uv_coverage():
    out = rhessi_like_uv_coverage()
    assert out["u"].shape == out["v"].shape == out["isc"].shape == (9, 32)


def test_stix_like_uv_coverage():
    out = stix_like_uv_coverage()
    assert out["u"].shape == out["v"].shape == out["label"].shape == (3, 10)
