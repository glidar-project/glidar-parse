


import pytest
import os
import pandas as pd
import numpy as np
from glidar_parse.imet_parser import iMetParser
from glidar_parse.sci_track import SciTrack
from glidar_parse.kml_parser import KmlParser
from glidar_parse.segment_thermals import segment_thermals

import fixtures


@pytest.fixture
def track():

    print(fixtures.IMET_FILE)

    # parser = iMetParser(fixtures.IMET_FILE)
    parser = KmlParser(fixtures.KLM_FILE)

    sci_track = SciTrack(parser)
    return sci_track

def test_segment_thermals(track):

    N = len(track.track.time)

    thermals = segment_thermals(track.track)
    
    print(thermals.labels)

    # Making sure original data weren't modified
    assert N == len(track.track.time)
    assert not hasattr(track.track, 'labels')
    assert not hasattr(track.track, 'time_sec')

    # check if we got labels
    assert hasattr(thermals, 'labels')


    

