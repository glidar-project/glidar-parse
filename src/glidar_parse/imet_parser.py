
import pandas as pd
import numpy as np

import metpy
import metpy.calc as mpcalc
from metpy.units import units

from scipy.ndimage import gaussian_filter1d

class iMetParser:

    def __init__(self, filename) -> None:
        
        self.filename = filename
        frame = self.read_imet_file(filename)
        self.track = self.process_raw_data(frame)


    def read_imet_file(self, file_name):
        # dataFrame = pd.read_csv(file_name, names=[
        #     'code',
        #     'pressure',
        #     'temperature',
        #     'RH',
        #     'RH_temp',
        #     'd',
        #     't',
        #     'longitude',
        #     'latitude',
        #     'altitude',
        #     'gps_count'
        # ], parse_dates={'time':['d', 't']})

        dataFrame = pd.read_csv(file_name, names=[
            'id',
            'dd',
            'tt',
            'code',
            'pressure',
            'temperature',
            'RH',
            'RH_temp',
            'd',
            't',
            'longitude',
            'latitude',
            'altitude',
            'gps_count'
        ], parse_dates={'time':['d', 't']})

        dataFrame.time = pd.to_datetime(dataFrame.time, errors='coerce')
        dataFrame = dataFrame.dropna(subset=['time'])
        dataFrame = dataFrame[dataFrame.time > '2020-01-01 00:00:00']
        
        return dataFrame


    def process_raw_data(self, frame, sigma = 10):
        
        day_to_parse = pd.to_datetime(frame.time.max().date())

        dataFrame = frame

        # Units adjustement
        dataFrame['temperature'] = dataFrame.temperature * 0.01
        dataFrame['RH'] = dataFrame.RH * 0.1
        dataFrame['RH_temp'] = dataFrame.RH_temp * 0.01
        dataFrame['longitude'] = dataFrame.longitude * 1e-7
        dataFrame['latitude'] = dataFrame.latitude * 1e-7
        dataFrame['altitude'] = dataFrame.altitude * 1e-3

        # Derived quantities
        dataFrame['theta'] = mpcalc.potential_temperature(dataFrame.pressure.values * units.pascal,
                                                        dataFrame.temperature.values * units.celsius)
        dataFrame['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(
            dataFrame.temperature.values * units.celsius, dataFrame.RH.values * units.percent
        )
        dataFrame['specific_humidity'] = mpcalc.specific_humidity_from_dewpoint(
            dataFrame.pressure.values * units.pascal,
            dataFrame.dewpoint.values * units.celsius
        )
        dataFrame['mixing_ratio'] = mpcalc.mixing_ratio_from_relative_humidity(
            dataFrame.pressure.values * units.pascal,
            dataFrame.temperature.values * units.celsius,
            dataFrame.RH.values * units.percent
        )
        dataFrame['theta_virtual'] = mpcalc.virtual_potential_temperature(
            dataFrame.pressure.values * units.pascal,
            dataFrame.temperature.values * units.celsius,
            dataFrame.mixing_ratio.values 
        )
        dataFrame['equivalent_potential_temperature'] = mpcalc.equivalent_potential_temperature(
            dataFrame.pressure.values * units.pascal,
            dataFrame.temperature.values * units.celsius,
            dataFrame.dewpoint.values * units.celsius
        )


        df = dataFrame[dataFrame.time > day_to_parse]
        # df = dataFrame
        df = df[df.gps_count > 1]
        df = df[df.latitude > 1]
        df = df[df.longitude > 1]

        if df.size < 1:
            raise RuntimeError('No valid data')
        
        df = df.sort_values('time')
        df = df.set_index('time')
        df = df.reset_index()

        """
        dt = (df.time.values[1:] - df.time.values[:-1]).astype(float)
        dt *= 1e-9
        z = gaussian_filter1d(df.altitude.values, sigma)
        dz = z[1:] - z[:-1]
        vario = dz / dt

        df['vario'] = np.concatenate([vario, [vario[-1]]])
        df['dt'] = np.concatenate([dt, [dt[-1]]])

        zone = get_utm_zone(df.longitude.median())
        print('Using UTM zone:', zone)
        myProj = Proj(proj='utm', ellps='WGS84', zone=zone, units='m')

        x, y = myProj(df.longitude.values, df.latitude.values)

        df['x'] = x
        df['y'] = y
        
        dx = (df.x.values[1:] - df.x.values[:-1]) / dt
        dy = df.y.values[1:] - df.y.values[:-1] / dt
        
        df['vx'] = np.concatenate([dx, [dx[-1]]])
        df['vy'] = np.concatenate([dy, [dy[-1]]])

        df['speed'] = np.sqrt(df.vx.values **2 + df.vy.values **2)
        df = df.set_index('time', drop=False)

        
        # takeoff detection
    #     df.speed > 3.0 # m/s ... about 10.8 kmh
    #     print(df.time.iloc[0])
        first_index = (df.speed.rolling('10s').median() > 3).idxmax()
        print(first_index)
        df = pd.DataFrame(df.loc[first_index:])
        """
        
        return df
