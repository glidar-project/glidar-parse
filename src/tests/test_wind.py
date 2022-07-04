
import pytest

from glidar_parse.kml_parser import KmlParser
from glidar_parse.sci_track import SciTrack
from glidar_parse.wind import compute_wind

import fixtures


@pytest.fixture
def track():

    return SciTrack(KmlParser(fixtures.KLM_FILE)).track

def test_wind(track):

    result = compute_wind(track.dx, track.dy, 30)

    import matplotlib.pyplot as plt
    plt.plot(result['speed'], '.')
    plt.show()
    
    assert False
