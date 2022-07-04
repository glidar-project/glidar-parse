
import numpy as np

def compute_wind(x, y, w):

        from glidar_parse.legacy import fit_circle_alg

        alg_cic = np.array([fit_circle_alg(x[i:i + w], y[i:i + w]) for i in np.arange(x.size - w)])

        npad = ((w//2, w - w//2), (0, 0))      
        alg_cic = np.pad(alg_cic, pad_width=npad, mode='edge')
        
        circle_fit_radius = alg_cic[:,0]
        circle_fit_center = alg_cic[:,1:]

        # v = gaussian_filter1d(np.stack((self.dx, self.dy)), 3)
   
        wind_speed = circle_fit_center        
        
        estimate_wind_speed =  np.sqrt(np.sum(wind_speed **2, axis=1))
        return {
            'radius': circle_fit_radius, 
            'center': circle_fit_center, 
            'speed': estimate_wind_speed
            }