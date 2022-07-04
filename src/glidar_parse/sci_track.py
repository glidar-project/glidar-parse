
from glidar_parse.kml_parser import KmlParser
from glidar_parse.igc_parser import IgcParser


from glidar_parse.track import Track
from glidar_parse.util import get_utm_zone

from pyproj import Proj

from pandas import DataFrame

import numpy as np
from scipy.ndimage import gaussian_filter1d


class SciTrack:

    def __init__(self, track: Track, sigma=5, tau=3) -> None:
        
        self.id = track.filename[track.filename.rfind('/')+1:]
        self.track = track.track.copy()    # make a copy of the track

        self.sigma = sigma
        self.tau = tau
        self.turning_time = 30
        self.turning_angle = 0.05

        # Projection to local coordinates
        self.UTM_zone = get_utm_zone(self.track.longitude.mean())
        self.proj = Proj(
            proj='utm', ellps='WGS84', zone=self.UTM_zone, units='m')

        # Compute everything
        self.proccess_track(self.track)


    def proccess_track(self, track, data_filter=id):

        x, y = self.proj(track.longitude.values, track.latitude.values)
        track['x'] = x
        track['y'] = y

        track['dt'] = np.pad(
            (track.time.values[2:] - track.time.values[:-2]).astype(float) * 1e-9,
             1, mode='edge')

        track['vario'] = np.pad(
            gaussian_filter1d(track.altitude.values[2:] - track.altitude.values[:-2], self.sigma), 
            1, mode='edge') / track.dt

        track['dx'] = gaussian_filter1d(np.pad(
                track.x.values[2:] - track.x.values[:-2], 1, mode='edge'), 
            self.tau) / track.dt

        track['dy'] = gaussian_filter1d(np.pad(
                track.y.values[2:] - track.y.values[:-2], 1, mode='edge'), 
            self.tau) / track.dt

        track['course'] = np.arctan2(track.dy.values, track.dx.values)

    def check_turning(self, track):

        dc = np.gradient(track.course)
        dc = [ e if abs(e) < abs(e + np.pi) else e + np.pi for e in dc  ]
        dc = [ e if abs(e) < abs(e - np.pi) else e - np.pi for e in dc  ]

        track['dc'] = dc

        C = np.cumsum(np.abs(gaussian_filter1d(track.dc, 10)))

        i = self.turning_time
        assert i % 2 == 0
        mask = np.pad((C[i:] - C[:-i]) / i,  i//2, mode='edge') > self.turning_angle
        
        track['turning'] = mask

    def compute_wind(self, x, y, w):

        from glidar_parse.wind import compute_wind
        return compute_wind(x, y, w)