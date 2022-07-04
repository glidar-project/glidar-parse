
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
import xml.etree.ElementTree as ET
# import plotly.graph_objects as go

from tqdm import tqdm

from scipy      import optimize
from scipy.ndimage import gaussian_filter1d

from pyproj import Proj

# plt.rcParams['figure.figsize'] = (16.0, 9.0)

test_filename = "/home/juraj/Work/ConvectionAnalysis/data/Voss-2018-04-29/2018-04-29_Erik_Hamran_Nilsen.kml"


class KmlParser:
    
    myProj = Proj(proj='utm', ellps='WGS84', zone=32, units='m')

    def __init__(self, filename):
        self.parse(filename)
        self.filename = filename
        
        self.dt = np.pad(self.time_sec[2:] - self.time_sec[:-2], 1, mode='edge')
        self.vario = np.pad(gaussian_filter1d(self.track[2:,2] - self.track[:-2,2], 5), 1, mode='edge') / self.dt
        
        self.vertical_acceleration = np.pad(gaussian_filter1d(self.vario[2:] - self.vario[:-2], 5), 1, mode='edge') / self.dt
        
        self.x, self.y = self.myProj(self.track[:,0], self.track[:,1])        
        self.altitude = self.track[:,2]
        
        self.dx = gaussian_filter1d(np.pad(self.x[2:] - self.x[:-2], 1, mode='edge'), 3) / self.dt
        self.dy = gaussian_filter1d(np.pad(self.y[2:] - self.y[:-2], 1, mode='edge'), 3) / self.dt
        
        circle_fit_points = int(240 / self.time_step)
        self.circle_fit = self._circle_fit(self.dx, self.dy, circle_fit_points)
        self.circle_fit_60 = self._circle_fit(self.dx, self.dy, 60)
        self.circle_fit_20 = self._circle_fit(self.dx, self.dy, 20)
        
        self.curvature = self.curvature_hack(np.stack((self.dx, self.dy)) - self.circle_fit['center'].T)
        self.curvature_fit = self._curvature_fit(self.x, self.y, 20)
        
        self.cc = gaussian_filter1d(self.curvature_fit['radius'], 15) 
        self.dcc = np.pad((self.cc[2:] - self.cc[:-2]), 1, mode='edge') / self.dt
        
        
    def _curvature_fit(self, x, y, w):
        
        alg_cic = np.array([fit_circle_alg(x[i:i + w], y[i:i + w]) for i in np.arange(x.size - w)])
        #alg_cic = [FitCircle((x[i:i + w], y[i:i + w])) for i in np.arange(x.size - w)]
        #alg_cic = np.array([(f.R, *f.center) for f in alg_cic])
        
        npad = ((w//2, w - w//2), (0, 0))      
        alg_cic = np.pad(alg_cic, pad_width=npad, mode='edge')
        
        circle_fit_radius = alg_cic[:,0]
        circle_fit_center = alg_cic[:,1:]
        
        return {'radius': circle_fit_radius, 'center': circle_fit_center}     
        
    def _circle_fit(self, x, y, w):
        
#        print(self.filename, x, y, w)       
        alg_cic = np.array([fit_circle_alg(x[i:i + w], y[i:i + w]) for i in np.arange(x.size - w)])
        #alg_cic = [FitCircle((x[i:i + w], y[i:i + w])) for i in np.arange(x.size - w)]
        #alg_cic = np.array([(f.R, *f.center) for f in alg_cic])
        
        npad = ((w//2, w - w//2), (0, 0))      
        alg_cic = np.pad(alg_cic, pad_width=npad, mode='edge')
        
        circle_fit_radius = alg_cic[:,0]
        circle_fit_center = alg_cic[:,1:]

        v = gaussian_filter1d(np.stack((self.dx, self.dy)), 3)
        
        #npad = ((1, 1), (0, 0))      
        #circ_speed = 0.5 * np.pad(circle_fit_center[2:, :] - circle_fit_center[:-2, :], pad_width=npad, mode='edge')
              
        wind_speed = circle_fit_center        
        
        estimate_wind_speed =  np.sqrt(np.sum(wind_speed **2, axis=1))
        return {'radius': circle_fit_radius, 'center': circle_fit_center, 'speed': estimate_wind_speed}

    def curvature_hack(self, D):
        """
        Computes the curvature of a 2D line
        The shape of the array should be (2, n),
        where n is the number of points

        """

        D2 = gaussian_filter1d(D[:,2:] - D[:,:-2], 3, mode='nearest') / np.stack([self.dt[1:-1], self.dt[1:-1]])

        d2 = np.sum(np.power(D, 2), axis=0)
        d = np.power(d2, 1.5)

        k = np.pad(np.abs(D.T[1:-1, 0] * D2.T[:, 1] - D.T[1:-1, 1] * D2.T[:, 0]) / d[1:-1], 1, mode='edge')
        return k

    def parse(self, file):

        tree = ET.parse(file)
        root = tree.getroot()

        takeoff_time_text = root.find('Folder')[-1].find('Metadata').find('FsInfo').attrib['time_of_first_point']
        self.takeoff_time = datetime.datetime.strptime(takeoff_time_text[:18], "%Y-%m-%dT%H:%M:%S")
        
        self.time_sec = np.array(
            root.find('Folder')[-1]
                .find('Metadata')
                .find('FsInfo')
                .find('SecondsFromTimeOfFirstPoint')
                .text.split(),
            dtype=int)

        self.time_step = np.median(self.time_sec[1:] - self.time_sec[:-1])
        
        self.track = np.array(
            [line.split(',') for line in root.find('Folder')[-1]
                                             .find('LineString')
                                             .find('coordinates')
                                             .text.split()],
            dtype=float)

        
    def segment_thermals(self, sink_thr=0, radius_threshold=200, radius_smooth=15, min_thermal_gap=10, min_thermal_length=30):
 
        idx_vario = np.where((self.vario > sink_thr) & 
                             (np.abs(self.dcc) < 5) & 
                             (self.cc < radius_threshold))[0]

        dlabels = (idx_vario[1:] - idx_vario[:-1]) 
        labels = np.cumsum(np.where(dlabels < (min_thermal_gap / self.time_step), 0, 1))

        stuff = np.unique(labels, return_counts=True, return_index=True)

        thermals = dict()
        for i in stuff[0][:-1]:
            if stuff[2][i] > (min_thermal_length / self.time_step):
                thermals[repr(i)] = (idx_vario[stuff[1][i]+1], idx_vario[stuff[1][i+1]])
        try:
            if stuff[2][-1] > (min_thermal_length / self.time_step):
                thermals[repr(len(stuff[0]))] = (idx_vario[stuff[1][-1]+1], idx_vario[-1])
        except IndexError:
            pass 

        return thermals

class FitCircle:
    """
    Credit: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html#Using-scipy.optimize.leastsq
    """
    def __init__(self, points):
        self.x = points[0]
        self.y = points[1]
        center_estimate = np.mean(self.x), np.mean(self.y)
        self.center, self.ier = optimize.leastsq(self.f_2, center_estimate)
        
        self.Ri        = self.calc_R(*self.center)
        self.R         = self.Ri.mean()
        self.residuals = np.sum((self.Ri - self.R)**2)

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.x-xc)**2 + (self.y-yc)**2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()


def fit_circle_alg(x, y):
    """
    Credit: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html#Using-scipy.optimize.leastsq
    """
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    
    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError as e:
        #print (e)
        #print (x)
        #print (y)
        return -1,0,0
        
    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    # residu_1 = np.sum((Ri_1 - R_1) ** 2)

    return R_1, xc_1, yc_1
 
    

