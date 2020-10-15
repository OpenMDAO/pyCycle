import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.new_thermo import Thermo
from pycycle.cea import species_data
from pycycle import constants


class SetTotalTestCase(unittest.TestCase):

    def test_set_total_tp(self):
        p = om.Problem()
        p.model = Thermo(mode='total_TP', 
                         thermo_dict={'method':'CEA', 
                                      'elements': constants.CO2_CO_O2_MIX, 
                                      'thermo_data': species_data.co2_co_o2 }) 

        p.set_solver_print(level=2)
        p.setup(check=False)

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

        p.set_val('T', 1500, units='degK')
        p.set_val('P', 1.034210, units='bar')
        p.run_model()

        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        expected_n_moles = 0.0227262
        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.16379233, 1e-4)


    def test_set_total_hp(self):

        p = om.Problem()
        p.model = Thermo(mode='total_hP', 
                         thermo_dict={'method':'CEA', 
                                      'elements': constants.CO2_CO_O2_MIX, 
                                      'thermo_data': species_data.co2_co_o2 }) 

        p.set_solver_print(level=2)

        p.setup(check=False)

        p.set_val('h', 340, units='cal/g')
        p.set_val('P', 1.034210, units='bar')

        p.run_model()

        expected_concentrations = np.array([0.61989858, 0.07015213, 0.30994929])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329281722301

        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.19039688581, 1e-4)

        # 1500K
        p['h'] = -1801.35537381
        p.run_model()

        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.022726185333
        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.16379012007, 1e-4)

    def test_set_total_sp(self):

        p = om.Problem()
        p.model = Thermo(mode='total_SP', 
                         thermo_dict={'method':'CEA', 
                                      'elements': constants.CO2_CO_O2_MIX, 
                                      'thermo_data': species_data.co2_co_o2 }) 
       
        p.model.suppress_solver_output = True
        r = p.model

        p.set_solver_print(level=2)

        p.setup(check=False)
        p.final_setup()

        # p.model.nonlinear_solver.options['maxiter'] = 0

        p.set_val('S', 2.35711010759, units="Btu/(lbm*degR)")
        p.set_val('P', 1.034210, units="bar")

        p.run_model()

        expected_concentrations = np.array([0.62003271, 0.06995092, 0.31001638])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles
        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329313730421

        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.19054696779, 1e-4)

        # 1500K
        p['T'] = 4000. 

        p['S'] = 1.5852424435
        p.run_model()

        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.022726185333
        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.16396871, 1e-4)


if __name__ == "__main__": 
    unittest.main()