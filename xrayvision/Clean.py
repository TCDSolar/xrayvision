"""CLEAN algorithms"""


from scipy.ndimage.interpolation import shift
from scipy import signal
import numpy as np
from astropy.convolution import Gaussian2DKernel
from enum import Enum


class ReasonOfStop(Enum):
    """
    Enum values to describe the state of the CLEAN algorithm

    Parameters
    ----------

    Examples
    --------

    Notes
    -----

    """
    NOT_FINISHED = 0
    REACHED_NITER = 1
    REACHED_THRESHOLD = 2


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
    psf: np.array
        The dirty beam. It can not have a bigger size than the image
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
    def __init__(self, visibility, psf: np.array, threshold: float,
                 image_dimensions: tuple, gain=0.01, niter: int = 1000):
        self.vis = visibility
        self.thres = threshold
        self.gain = gain
        self.niter = niter
        self.iterated = 0
        self.reason_of_stop = ReasonOfStop.NOT_FINISHED
        # Check the validity of the image_dimensions input
        if not len(image_dimensions) == 2:
            raise ValueError("image_dimensions: incorrect tuple length! "
                             "Example: (100x100)")
        if image_dimensions[0] == 0 or image_dimensions[1] == 0:
            raise ValueError("image_dimensions: 0 is not a valid size!")
        if psf.shape[0] > image_dimensions[0]:
            raise ValueError("The x dimension size of the psf is greater "
                             "than the image size!")
        if psf.shape[1] > image_dimensions[1]:
            raise ValueError("The y dimension size of the psf is greater "
                             "than the image size!")
        self.dim = image_dimensions
        self.point_source_map = np.zeros(image_dimensions)
        # #1
        self.dirty_map = np.zeros(self.dim)
        temp = self.vis.to_map_v2(self.dirty_map)
        self.dirty_map = temp

        # #2 Creating the dirty beam
        # Padding the psf to fit with the size
        # Padding distance before data - x
        pdxb = int((self.dim[0] - psf.shape[0])/2)
        # Padding distance before data - y
        pdyb = int((self.dim[1] - psf.shape[1])/2)
        # Padding distance after data - x
        pdxa = self.dim[0] - psf.shape[0] - pdxb
        # Padding distance after data - y
        pdya = self.dim[1] - psf.shape[1] - pdyb
        self.dirty_beam = np.lib.pad(psf, ((pdxb, pdxa), (pdyb, pdya)),
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
        max_intesity = np.max(self.dirty_map)
        if self.niter < 1:
            self.reason_of_stop = ReasonOfStop.REACHED_NITER
            return self.reason_of_stop
        if max_intesity < self.thres:
            self.reason_of_stop = ReasonOfStop.REACHED_THRESHOLD
            return self.reason_of_stop

        pos = np.unravel_index(np.argmax(self.dirty_map),
                               self.dirty_map.shape)

        # If gain is not set, using the one that was give during init
        if not gain:
            gain = self.gain

        # Updating the point source map
        self.point_source_map[pos[0], pos[1]] += max_intesity * gain
        # #4
        # Centering the the dirty_beam on the position of the max intensity
        centered_dirty_beam = shift(self.dirty_beam, (pos[0]-int(self.dim[0]/2),
                                                      pos[1]-int(self.dim[1]/2)))
        self.dirty_map = np.subtract(self.dirty_map,
                                     centered_dirty_beam * max_intesity * gain)
        self.niter -= 1
        self.iterated += 1

        return ReasonOfStop.NOT_FINISHED

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

    @staticmethod
    def clean(dirty_map, dirty_beam, clean_beam_width=4.0, gain=0.1, thres=0.01, niter=1000):
        # Assume bear center is in middle
        beam_center = (dirty_beam.shape[0] - 1)/2.0, (dirty_beam.shape[1] - 1)/2.0

        # Model for sources
        model = np.zeros(dirty_map.shape)
        for i in range(niter):
            # Find max in dirty map and save to point source
            mx, my = np.unravel_index(dirty_map.argmax(), dirty_map.shape)
            Imax = dirty_map[mx, my]
            model[mx, my] += gain*Imax

            comp = Imax * gain * shift(dirty_beam,
                                       (mx - beam_center[0],
                                        my - beam_center[1]),
                                       order=0)

            dirty_map = np.subtract(dirty_map, comp)

            if dirty_map.max() <= thres:
                print("Break")
                break

        # Clean Beam
        clean_beam = Gaussian2DKernel(stddev=clean_beam_width, x_size=dirty_beam.shape[0], y_size=dirty_beam.shape[1]).array
        if clean_beam_width != 0.0:
            model = signal.convolve2d(model, clean_beam, mode='same')  # noqa
        clean_beam = clean_beam * (1/clean_beam.max())
        dirty_map = dirty_map / clean_beam.sum()

        # For testing turned off
        # signal.convolve2d(model, Gaussian2DKernel(stddev=1), mode='same')  # noqa
        return model + dirty_map
