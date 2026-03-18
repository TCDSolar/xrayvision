import astropy.units as u
import numpy as np


def rhessi_like_uv_coverage():
    """
    Creates RHESSI-like u, v coverage coverage distribution.
    """
    # Approximate RHESSI spatial frequencies (arcsec^-1) for subcollimators 1-9,
    # based on typical RHESSI grid pitches of 2.26, 3.92, 6.79, 11.76, 20.36, 35.27, 61.08, 105.8, 183.2 arcsec.
    isc = np.arange(0, 9)  # index subcollimator (isc) detectors 1-9
    resolutions = [2.26, 3.92, 6.79, 11.76, 20.36, 35.27, 61.08, 105.8, 183.2] * u.arcsec
    radii = 1 / resolutions

    # 32 evenly-spaced rotation angles per detector (simulates spacecraft rotation)
    n_angles = 32
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    u_comp = np.outer(radii, np.cos(angles))
    v_comp = np.outer(radii, np.sin(angles))

    return {"u": u_comp, "v": v_comp, "det": isc[:, None] * np.ones(u_comp.shape)}
