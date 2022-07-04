
import pytest
import os
import pandas as pd
import numpy as np

from glidar_parse.sci_track import SciTrack
from glidar_parse.kml_parser import KmlParser

from glidar_parse import legacy


from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

KLM_FILE = 'src/tests/fixtures/2019-Bavallen628175.kml'

class TestSciTrack:

    @classmethod
    def setup_class(self):
        """setup any state specific to the execution of the given class 
        (which usually contains tests).
        """
        self.filename = os.path.abspath(KLM_FILE)
        self.parser = KmlParser(self.filename)

        self.sci_track = SciTrack(self.parser, tau=1)

    def test_derivative(self):

        subtrack = pd.DataFrame(self.sci_track.track.iloc[1000:1100])

        plt.plot(self.sci_track.track.x, self.sci_track.track.y)
        plt.plot(subtrack.x, subtrack.y)
        plt.show()

        subtrack['x'] -= subtrack.x.mean()
        subtrack['y'] -= subtrack.y.mean()

        # subtrack['dxx'] = np.pad(subtrack.x.values[2:] - subtrack.x.values[:-2], 1, mode='edge') / subtrack.dt
        # subtrack['dyy'] = np.pad(subtrack.y.values[2:] - subtrack.y.values[:-2], 1, mode='edge') / subtrack.dt

        k = 2 

        a = subtrack.iloc[::1]
        plt.figure(figsize=(15,5))

        for index, e in a.iterrows():    
            plt.arrow(e.x, e.y, k * e.dxx, k *  e.dyy, head_width=3)
            plt.arrow(e.x, e.y, k *  e.dx, k *   e.dy, head_width=3)

            
        plt.plot(subtrack.x, subtrack.y, 'x-')
        plt.gca().set_aspect('equal')
        plt.show()

        plt.plot(subtrack.time, subtrack.dx)
        plt.plot(subtrack.time, subtrack.dxx)
        plt.plot(subtrack.time, gaussian_filter1d(subtrack.dxx, 3))
        plt.show()

    def test_order_of_diff_blur(self):
        """
        An experiment comparing the order of gaussian filtering and 
        numerical differentiation. 

        The result is that the difference is only in the treatment 
        of the boundary.
        """

        sigma = 10
        n = 5 * sigma

        dx = gaussian_filter1d(
            np.pad(self.sci_track.track.x.values[2:] - self.sci_track.track.x.values[:-2], 
            1, mode='edge'), sigma) / self.sci_track.track.dt.values

        x = gaussian_filter1d(self.sci_track.track.x.values, sigma)

        xd = np.pad(x[2:] - x[:-2], 1, mode='edge') / self.sci_track.track.dt

        print(xd - dx)

        print('max diff', np.max(np.abs(dx[n:-n] - xd[n:-n])))
        print('rel diff', np.max(np.abs(dx[n:-n] - xd[n:-n]) / np.abs(xd[n:-n]) ))

        # import matplotlib.pyplot as plt
        # plt.plot(xd[:2*sigma] - dx[:2*sigma], '.')
        # plt.plot(dx[:2*sigma], '.')
        # plt.plot(xd[:2*sigma], '.')
        # plt.plot(dx[n:-n], xd[n:-n], ',')
        # plt.show()

        assert np.allclose(dx[n:-n], xd[n:-n])


