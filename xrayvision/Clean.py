"""CLEAN algorithms"""

from scipy.ndimage.interpolation import shift
from scipy import signal
import numpy as np
from astropy.convolution import Gaussian2DKernel


class Hogbom(object):
    """
    Simplest Hogböm's CLEAN method (in case of a dirty beam with just 1s
    where there is a visibility value.

    1. Perform a back projection or map_idft on the visibilities to obtain the dirty map (DM)
    2. Perform using the same u, v locations back projection of a delta function to obtain
       the dirty beam (DB)
    3. Find the brightest point in the in the DM add to a point source map (position and intensity)
    4. Subtract at the position of the bright point the DB * intensity of brightest point * gain
    5. Go to 3 if more source or iteration threshold reached
    6. Convolve the point source map with an idealised BEAM (small gaussian)
    7. Add the residuals from the DM to the step above an return

    Parameters
    ----------
    visibility: Visibility
        The processed visibility bag
    psf:
        The dirty beam
    threshold: float
        If the highest value is smaller, it stops
    image_dimensions: tuple
        The dimensions of the image what is obtained from the visibilities
    gain: The gain what is used for the iterations

    Examples
    --------

    Notes
    -----

    """
    def __init__(self, visibility, psf, threshold: float, image_dimensions: tuple,
                 gain=0.01, niter: int = 1000):
        self.vis = visibility
        self.thres = threshold
        self.gain = gain
        self.niter = niter
        # Check the validity of the image_dimensions input
        if not len(image_dimensions) == 2:
            raise ValueError("image_dimensions: incorrect tuple length! "
                             "Example: (100x100)")
        if image_dimensions[0] == 0 or image_dimensions[1] == 0:
            raise ValueError("image_dimensions: 0 is not a valid size!")
        self.dim = image_dimensions
        self.point_source_map = np.zeros(image_dimensions)
        # #1
        self.dirty_map = np.zeros(self.dim)
        temp = self.vis.to_map(self.dirty_map)
        self.dirty_map = temp

        # Padding the psf to fit with the size
        padding_distance_x_before = int((self.dim[0] - psf.shape[0])/2)
        padding_distance_y_before = int((self.dim[1] - psf.shape[1])/2)
        padding_distance_x_after = self.dim[0] - psf.shape[0] - padding_distance_x_before
        padding_distance_y_after = self.dim[1] - psf.shape[1] - padding_distance_y_before
        self.dirty_beam = np.lib.pad(psf,
                                     ((padding_distance_x_before,
                                       padding_distance_x_after),
                                     (padding_distance_y_before,
                                      padding_distance_y_after)),
                                     'constant', constant_values=(0, 0))

    def iterate(self, gain=False):
        """
        The count of iterations depends on, how many times it has been called.
        It will not iterate if the threshold is reached.

        Parameters
        ----------
        gain: float
            If not provided it will use the one what was given at init

        Examples
        --------

        Notes
        -----

        """
        # #3
        if self.niter < 1:
            return True

        max_intesity = 0.0
        pos = (0, 0)
        summarized = 0.0
        for i in range(self.dirty_map.shape[0]):
            for j in range(self.dirty_map.shape[1]):
                intensity = self.dirty_map[i, j]
                if intensity > max_intesity:
                    max_intesity = intensity
                    pos = (i, j)
                    summarized += intensity
        # Checking if we reached the threshold or not
        abs_avg_value = summarized / self.dirty_map[pos[0], pos[1]]
        if abs_avg_value < self.thres:
            return True
        # If gain is not set, using the one that was give during init
        if not gain:
            gain = self.gain

        # Updating the point source map
        self.point_source_map[pos[0], pos[1]] += max_intesity
        # #4
        # Centering the the dirty_beam on the position of the max intensity
        centered_dirty_beam = shift(self.dirty_beam, (pos[0]-int(self.dim[0]/2),
                                                      pos[1]-int(self.dim[1]/2)))
        self.dirty_map = np.subtract(self.dirty_map,
                                      centered_dirty_beam * max_intesity * gain)
        self.niter -= 1
        print(self.niter)
        return False

    def finish(self, stddev: float):
        """
        Returns with the cleaned map

        Parameters
        ----------
        stddev: float
            The standard deviation of the convolved Gaussian

        Examples
        --------

        Notes
        -----

        """
        # #6
        gaussian_map = Gaussian2DKernel(stddev=stddev, x_size=self.dim[0],
                                        y_size=self.dim[1])
        result = signal.convolve2d(self.point_source_map, gaussian_map, mode='same')

        # 7
        return np.add(result, self.dirty_map)
