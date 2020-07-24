import unittest

import numpy as np

from openmdao.api import Problem

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.set_total import SetTotal
from pycycle.cea import species_data
from pycycle import constants


class SetTotalTestCase(unittest.TestCase):

    def test_set_total_tp(self):

        thermo = species_data.Thermo(species_data.co2_co_o2, constants.co2_co_o2_init_prod_amounts)
        init_reacts = {'CO':1, 'CO2':1, 'O2':1}

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, init_reacts=init_reacts, mode="T")
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('T', 4000., units='degK')
        p.model.set_input_defaults('P', 1.034210, units="bar")
        r = p.model

        p.set_solver_print(level=2)
        p.setup(check=False)

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
        p['T'] = 1500  # degK
        p['P'] = 1.034210  # bar
        p.run_model()

        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        expected_n_moles = 0.022726185333
        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.16380, 1e-4)

    def test_set_total_hp(self):

        thermo = species_data.Thermo(species_data.co2_co_o2, init_reacts=constants.co2_co_o2_init_prod_amounts)
        init_reacts = {'CO':1, 'CO2':1, 'O2':1}

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, init_reacts=init_reacts, mode="h")
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('h', 340, units='cal/g')
        p.model.set_input_defaults('P', 1.034210, units='bar')
        p.model.suppress_solver_output = True

        r = p.model

        p.set_solver_print(level=2)

        p.setup(check=False)

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
        # seems to want to start from a different guess than the last converged point
        p['n'] = np.array([.33333, .33333, .33333])
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

        thermo = species_data.Thermo(species_data.co2_co_o2, init_reacts=constants.co2_co_o2_init_prod_amounts)
        init_reacts = {'CO':1, 'CO2':1, 'O2':1}

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, init_reacts=init_reacts, mode="S")
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('S', 2.35711010759, units="Btu/(lbm*degR)")
        p.model.set_input_defaults('P', 1.034210, units="bar")
        p.model.suppress_solver_output = True
        r = p.model

        p.set_solver_print(level=2)

        p.setup(check=False)

        p.run_model()

        np.seterr(all='raise')

        expected_concentrations = np.array([0.62003271, 0.06995092, 0.31001638])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles
        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329313730421

        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.19054696779, 1e-4)

        # 1500K
        p['S'] = 1.5852424435
        p.run_model()

        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_near_equal(concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.022726185333
        assert_near_equal(n_moles, expected_n_moles, 1e-4)
        assert_near_equal(p['gamma'], 1.16381209181, 1e-4)

if __name__ == "__main__":

    import numpy as np
    import scipy as sp

    np.seterr(all='raise')

    unittest.main()
