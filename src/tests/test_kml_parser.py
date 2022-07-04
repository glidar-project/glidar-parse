
import pytest
import os 
import fixtures

from glidar_parse.kml_parser import KmlParser


class TestKmlParser:

    @classmethod
    def setup_class(self):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.filename = os.path.abspath(fixtures.KLM_FILE)
        self.parser = KmlParser(self.filename)


    def test_parser_name(self):

        assert self.parser.filename == self.filename

    def test_parser_track(self):

        print(self.parser.track.shape)
        assert self.parser.track.shape == (4171, 4)