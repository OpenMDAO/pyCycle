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


    def test_set_total_Sp(self):

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


class TestSetTotalJanaf(unittest.TestCase): 

    def test_set_total_equivalence(self): 

        p_TP = om.Problem()
        p_TP.model = Thermo(mode='total_TP', 
                            thermo_dict={'method':'CEA', 
                                         'elements': constants.AIR_MIX, 
                                         'thermo_data': species_data.janaf }) 
        p_TP.setup()

        p_hP = om.Problem()
        p_hP.model = Thermo(mode='total_hP', 
                            thermo_dict={'method':'CEA', 
                                         'elements': constants.AIR_MIX, 
                                         'thermo_data': species_data.janaf }) 
        p_hP.setup()
        p_hP.final_setup()

        p_SP = om.Problem()
        p_SP.model = Thermo(mode='total_SP', 
                            thermo_dict={'method':'CEA', 
                                         'elements': constants.AIR_MIX, 
                                         'thermo_data': species_data.janaf }) 
        p_SP.setup()


        def check(T, P): 

            p_TP.set_val('T', T, units='degR')
            p_TP.set_val('P', P, units='psi')

            # print('TP check')
            p_TP.run_model()
            h_from_TP = p_TP.get_val('flow:h', units='cal/g')
            S_from_TP = p_TP.get_val('flow:S', units='cal/(g*degK)')


            p_hP.set_val('h', h_from_TP, units='cal/g')
            p_hP.set_val('P', P, units='psi')

            # print('hp check')
            p_hP.run_model()

            assert_near_equal(p_hP['flow:T'], p_TP['flow:T'], 1e-4)

            p_SP.set_val('S', S_from_TP, units='cal/(g*degK)')
            p_SP.set_val('P', P, units='psi')

            # print('SP check')
            p_SP.run_model()

            assert_near_equal(p_SP['flow:T'], p_TP['flow:T'], 1e-4)

        check(518., 14.7)
        check(3000., 30.)
        check(1500., 80.)


if __name__ == "__main__": 
    unittest.main()