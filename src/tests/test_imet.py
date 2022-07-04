
import pytest
import os
import pandas as pd
import numpy as np

from glidar_parse.sci_track import SciTrack
from glidar_parse.imet_parser import iMetParser


import fixtures


@pytest.fixture
def imet_parser():

    print(fixtures.IMET_FILE)
    return iMetParser(fixtures.IMET_FILE)

def test_sci_imet(imet_parser):

    print(imet_parser.track)
    print('lon', imet_parser.track.longitude.mean())
    print('lat', imet_parser.track.latitude.mean())
    track = SciTrack(imet_parser)

    print(track.track)

    assert False
    # assert track is not None