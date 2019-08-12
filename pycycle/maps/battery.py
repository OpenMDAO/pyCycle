import numpy as np

from pycycle.maps.map_data import MapData

battery = MapData()

# Map design point values
battery.defaults = {}
battery.defaults['T_bp'] = 20.0
battery.defaults['SOC_bp'] = 0.5
battery.defaults['tU_oc'] = 3.75
battery.defaults['tC_Th'] = 20000.
battery.defaults['tR_Th'] = 0.002
battery.defaults['tR_0'] = 0.009

# Temperature break points (degC)
battery.T_bp = np.array([5., 20., 40])
battery.SOC_bp = np.array(
    [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.])  # SOC break points

battery.tU_oc = np.array([[3.5, 3.55, 3.65, 3.75, 3.9, 4.1, 4.25],
                          [3.5, 3.55, 3.65, 3.75, 3.9, 4.1, 4.25],
                          [3.5, 3.55, 3.65, 3.75, 3.9, 4.1, 4.25]])  # volts, Open Circuit Voltage

# Thevenin components for Polarization Losses
battery.tC_Th = np.array([[2000.,   2500., 22200.,  1200., 20000., 14000., 10000.],
                          [1200.,  20000., 40000., 20000., 30000., 20000., 25000.],
                          [30000., 35000., 49000., 27500., 50000., 29000., 30000.]])  # farads

battery.tR_Th = np.array([[0.0110, 0.0070, 0.0050, 0.0040, 0.0040, 0.0040, 0.0038],
                          [0.0030, 0.0029, 0.0029, 0.0020, 0.0029, 0.0025, 0.0024],
                          [0.0017, 0.0017, 0.0017, 0.0017, 0.0018, 0.0017, 0.0017]])  # ohm  Ohmic Losses

battery.tR_0 = np.array([[0.0118, 0.0110, 0.0115, 0.0109, 0.0109, 0.0115, 0.0120],
                         [0.0090, 0.0090, 0.0091, 0.0089, 0.0092, 0.0089, 0.0089],
                         [0.0085, 0.0086, 0.0082, 0.0083, 0.0086, 0.0085, 0.0086]])  # ohm

battery.units = {}
battery.units['T_bp'] = 'degC'
battery.units['tU_oc'] = 'V'
battery.units['tC_Th'] = 'F'
battery.units['tR_Th'] = 'ohm'
battery.units['tR_0'] = 'ohm'

# format for new regular grid interpolator:

battery.param_data = []
battery.output_data = []

battery.param_data.append({'name': 'T_bp', 'values': battery.T_bp,
                           'default': 20.0, 'units': 'degC'})
battery.param_data.append({'name': 'SOC_bp', 'values': battery.SOC_bp,
                           'default': 0.5, 'units': None})

battery.output_data.append({'name': 'tC_Th', 'values': battery.tC_Th,
                            'default': 20000, 'units': 'F'})
battery.output_data.append({'name': 'tR_Th', 'values': battery.tR_Th,
                            'default': 0.002, 'units': 'ohm'})
battery.output_data.append({'name': 'tR_0', 'values': battery.tR_0,
                            'default': 0.009, 'units': 'ohm'})
battery.output_data.append({'name': 'tU_oc', 'values': battery.tU_oc,
                            'default': 3.75, 'units': 'V'})
