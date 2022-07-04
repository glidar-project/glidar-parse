
from glidar_parse.util import get_utm_zone
import pytest


def test_get_utm_zone():

    assert get_utm_zone(5.999) == 31

    assert get_utm_zone(6) == 32

    assert get_utm_zone(60) == 41