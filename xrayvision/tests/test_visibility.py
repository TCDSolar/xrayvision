from pathlib import Path

import numpy as np
import pytest

import astropy.units as unit
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

from sunpy.map import Map

from xrayvision.transform import generate_uv
from xrayvision.visibility import Visibility


@pytest.fixture
def test_data_dir():
    path = Path(__file__).parent.parent / 'data'
    return path
