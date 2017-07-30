"""HSI_VIS object from RHESSI Software

Each visibility structure corresponds to a single subcollimator, harmonic,
u, v, energy and time combination.
Reference: http://sprg.ssl.berkeley.edu/~ghurford/VisibilityGuide.pdf
"""
#
# TODO
#
__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "xxx"

import numpy as np


class hsi_vis:
    """
    hsi_vis()

    Visibility struct based on
    http://sprg.ssl.berkeley.edu/~ghurford/VisibilityGuide.pdf

    Parameters
    ----------

    Attributes
    ----------
    isc : int
        Subcollimator index, range is in [0;8]
    harm : int
        Harmonic number, range is in [1;3]
    erange : np.array
        Energy range (keV) - e.g.: np.array([E_start, E_end])
    trange : np.array
        Time range - e.g.: np.array([t_start, t_end])
    u : float
        East-west spatial frequency component (arcsec^{-1})
    v : float
        North-south spatial frequency component (arcsec^{-1})
    obsvis : complex
        Observed (semicalibrated) visibility (photon / cm^2 /s)
    totflux : float
        “Total flux” or semicalibrated ‘DC’ term (photon / cm^2 /s)
    sigamp : float
        Average statistical error in obsvis components (photon / cm^2 /s)
    chi2 : float
        Reduced CHI^2 in fitting visibility to stacked event list
    xyoffset : np.array
        West, north heliocentric offset of phase center (arcsec)

    Examples
    --------

    See Also
    --------

    References
    ----------
    | http://sprg.ssl.berkeley.edu/~ghurford/VisibilityGuide.pdf

    """
    def __init__(self):
        """
        Initializes the hsi_vis class with dummy values

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        self.isc = 0
        self.harm = 1
        self.erange = np.zeros(2)
        self.tarnge = np.zeros(2)
        self.u = 0.0
        self.v = 0.0
        self.obsvis = 0+0j
        self.totflux = 0.0
        self.sigamp = 0.0
        self.chi2 = 0.0
        self.xyoffset = np.zeros(2)
