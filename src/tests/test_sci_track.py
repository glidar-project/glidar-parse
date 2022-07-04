
import pytest
import os
import pandas as pd
import numpy as np

from glidar_parse.sci_track import SciTrack
from glidar_parse.kml_parser import KmlParser

from glidar_parse import legacy

import fixtures

from scipy.ndimage import gaussian_filter1d


@pytest.fixture
def legacy_track():

    return legacy.KmlParser(fixtures.KLM_FILE)    


class TestSciTrack:

    @classmethod
    def setup_class(self):
        """setup any state specific to the execution of the given class 
        (which usually contains tests).
        """
        self.filename = os.path.abspath(fixtures.KLM_FILE)
        self.parser = KmlParser(self.filename)

        self.sci_track = SciTrack(self.parser)

    def test_track_utm(self):

        assert self.sci_track.UTM_zone == 32

    # def test_course(self):

    #     import matplotlib.pyplot as plt
    #     plt.plot(self.sci_track.track.course, '.')
    #     plt.show()
        
    #     assert False

    def test_integration(self, legacy_track):

        for key in ['x', 'y', 'altitude', 'dt', 'dx', 'dy', 'vario']:
            try: 
                assert np.array_equal(
                    self.sci_track.track[key].values, 
                    legacy_track.__dict__[key],
                    )
            except AssertionError as e:
                e.args = (key, *e.args)
                raise e

    def test_new_stuff(self):

        assert self.sci_track.track['course'].values is not None 

    def test_wind_speed(self):

        self.sci_track.compute_wind()