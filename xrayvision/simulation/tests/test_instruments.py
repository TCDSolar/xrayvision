from xrayvision.simulation.instruments import rhessi_like_uv_coverage


def test_rhessi_like_uv_coverage():
    out = rhessi_like_uv_coverage()
    assert out["u"].shape == out["v"].shape == out["det"].shape == (9, 32)
