from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    path = Path(__file__).parent.parent / 'data'
    return path
