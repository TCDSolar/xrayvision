Tutorial
========

Lets begin by importing the necessary libraries::

    import numpy as np
    from matplotlib import pyplot as plt
    from xrayvision import, visibility, transform, clean

For the purposes of this tutorial we are going to create synthetic data, the first step is to create
`x, y` array these array correspond to pysical location and as such have units.::


    transform.generate_xy()