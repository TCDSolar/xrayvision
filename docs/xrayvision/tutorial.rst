Tutorial
========

First a very short example that demonstrates this libraries capabilities.
Begin by importing the necessary libraries

.. code:: python

    import numpy as np

    from astropy import units as u
    from xrayvision.visibility import Visibility
    from xrayvision import clean, SAMPLE_RHESSI_VISIBILITIES

Next we will load in some sample data from the RHESSI spacecraft which has been stored
as visibilities in a fits file using the RHESSI IDL software.

.. code:: python

    rhessi_vis = Visibilty.from_fits_file(SAMPLE_RHESSI_VISIBILITIES)

At this stage we can create a map with a back projection or inverse transform also known as the dirty map
of the visibility data. The size of the map and the physical pixel size can be specified as
to the `to_map` function of the visibility.

.. code:: python

    rhessi_map = rhess_vis.to_map(shape=(65, 65), pixel_size=[4., 4.] * u.arcsec)
    rhessi_map.peek()

The artifacts due to the under sampling of the u, v plane are clear. The main goal
of this library is to provide a number image reconstruction methods such as CLEAN.
The clean algorithm needs the dirty map, the dirty beam or point spread function
inputs.

.. code:: python

    rhessi_vis.vis = np.ones(rhessi_vis.shape)
    dirty_beam = rhess_vis.to_image(shape=(65*3, 65*3), pixel_size=[4., 4.] * u.arcsec)

    clean_map, residuals = clean.clean(dirty_map = rhessi_map.data, dirty_beam = dirty_beam,
                                       gain=0.05, niter=1000, clean_beam_width = 1.0)

    rhessi_map.data[:] = clean_map
    rhessi_map.peek()


