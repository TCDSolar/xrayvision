import numpy as np
import random
import sys  
from xrayvision import *


def test_mapvis():

    random.seed()
    # Creating map data for the test
    size = 30
    map_data = np.ones((size, size))
    # Adding random values
    for i in range(size):
        map_data[random.randint(0, size-1), random.randint(0, size-1)] += 1.
    # Creating an xy array
    xy = [[], []]
    for i in range(int(-size/2), int(-size/2) + size):
        xy[0].append(i)
        xy[1].append(i)
    xy = np.array(xy)
    # We want to get the whole UV map, so we have to provide the list of all
    # the UV coordinates
    uv = [[], []]
    for i in range(size):
        for j in range(size):
            uv[0].append(i)
            uv[1].append(j)
    uv = np.array(uv)
    visout = hsi_vis_map2vis(map_data, xy, uv)
    """
    for i, vis in enumerate(visout):
        if i % size == 0:
            sys.stdout.write("\n")
        sys.stdout.write("{} ".format(vis.obsvis.item(1)))
    """
    return vis_bpmap(visout, uniform_weighting=True), map_data
