import pytest

from glidar_parse import legacy

import fixtures

@pytest.fixture
def legacy_track():

    return legacy.KmlParser(fixtures.KLM_FILE)    


def test_circle_fit(legacy_track):

    print(legacy_track)
    print(legacy_track.cc)
    assert legacy_track is not None