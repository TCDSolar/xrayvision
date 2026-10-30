import pytest

# Force MPL to use non-gui backends for testing.
try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use("Agg")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--idl", action="store_true", default=False, help="run tests that need a local SSW IDL install")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--idl"):
        return
    skip_idl = pytest.mark.skip(reason="needs SSW IDL, use --idl to run")
    for item in items:
        if "idl" in item.keywords:
            item.add_marker(skip_idl)
