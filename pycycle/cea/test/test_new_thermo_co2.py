import unittest

import numpy as np

from openmdao.api import Problem

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.new_thermo import Thermo
from pycycle.cea import species_data
from pycycle import constants


class SetTotalTestCase(unittest.TestCase):

    # 4000k
    p = Problem()
    p.model = Thermo(mode='total_TP', 
                     thermo_dict={'method':'CEA', 
                                  'elements': constants.CO2_CO_O2_MIX, 
                                  'thermo_data': species_data.co2_co_o2 }) 

    p.set_solver_print(level=2)
    p.setup(check=False)

    p.set_val('b0', [0.02272211, 0.04544422])
    p.set_val('T', 4000, units='degK')
    p.set_val('P', 1.034210, units='bar')
    
    p.run_model()

    expected_concentrations = np.array([0.62003271, 0.06995092, 0.31001638])

    n = p['n']
    n_moles = p['n_moles']
    concentrations = n / n_moles

    assert_near_equal(concentrations, expected_concentrations, 1e-4)

    expected_n_moles = 0.03293137

    assert_near_equal(n_moles, expected_n_moles, 1e-4)
    assert_near_equal(p['gamma'], 1.19054697, 1e-4)

    # 1500K
    p['T'] = 1500  # degK
    p['P'] = 1.034210  # bar
    p.run_model()

    expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
    n = p['n']
    n_moles = p['n_moles']
    concentrations = n / n_moles

    expected_n_moles = 0.0227262
    assert_near_equal(n_moles, expected_n_moles, 1e-4)
    assert_near_equal(p['gamma'], 1.16379233, 1e-4)


if __name__ == "__main__": 
    unittest.main()