import unittest

import numpy as np

from openmdao.api import Problem, Group

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea.props_rhs import PropsRHS
from pycycle.thermo.cea.props_calcs import PropsCalcs
from pycycle.thermo.cea import species_data
from pycycle import constants


class PropsRHSTestCase(unittest.TestCase):

    def setUp(self):

        self.thermo = species_data.Properties(species_data.co2_co_o2, init_elements=constants.CO2_CO_O2_ELEMENTS)

        p = self.prob = Problem()
        p.model = Group()
        p.model.suppress_solver_output = True

        p.model.add_subsystem('props_rhs', PropsRHS(thermo=self.thermo),
                              promotes=['T', 'n', 'composition', 'rhs_T', 'rhs_P', 'lhs_TP', 'n_moles'])
        p.model.set_input_defaults('T', 4000., units='degK')

        n = np.array([0.02040741, 0.0023147, 0.0102037])
        p.model.set_input_defaults('n', n)

        b = np.array([0.02272211, 0.04544422])
        p.model.set_input_defaults('composition', b)
        p.model.set_input_defaults('n_moles', 0.03292581)

        p.setup(check=False)
        p['n_moles'] = 0.03292581

    def test_total_rhs(self):

        p = self.prob
        p.run_model()

        goal_rhs_T = np.array([0.00016837, 0.07307206, 0.04281455])
        goal_rhs_P = np.array([0.02272211, 0.04544422, 0.03292581])
        goal_lhs_TP = np.array([[0.02272211, 0.02503681, 0.02272211],
                                [0.02503681, 0.07048103, 0.04544422],
                                [0.02272211, 0.04544422, 0.]])

        tol = 1e-5
        assert_near_equal(p['rhs_T'], goal_rhs_T, tol)
        assert_near_equal(p['rhs_P'], goal_rhs_P, tol)
        assert_near_equal(p['lhs_TP'], goal_lhs_TP, tol)

class PropsCalcsTestCase(unittest.TestCase):

    def setUp(self):

        self.thermo = species_data.Properties(species_data.co2_co_o2, init_elements=constants.CO2_CO_O2_ELEMENTS)

        p = self.prob = Problem()
        p.model = Group()
        p.model.suppress_solver_output = True

        p.model.add_subsystem('props', PropsCalcs(thermo=self.thermo), promotes=['*'])

        p.model.set_input_defaults('T', 4000., units='degK')
        p.model.set_input_defaults('P', 1.034210, units='bar')

        n = np.array([0.02040741, 0.0023147, 0.0102037])
        p.model.set_input_defaults('n', n)

        result_T = np.array([-1.74791977, 1.81604241, -0.24571810])
        p.model.set_input_defaults('result_T', result_T)

        result_P = np.array([0.48300853, 0.48301125, -0.01522548])
        p.model.set_input_defaults('result_P', result_P)

        p.setup(check=False)
        p['n_moles'] = 0.03292581

    def test_total_calcs(self):

        p = self.prob
        p.run_model()

        tol = 1e-4
        assert_near_equal(p['Cp'], 0.579647062532, tol)
        assert_near_equal(p['gamma'], 1.19039, tol)
        assert_near_equal(p['h'], 340.324938088, tol)
        assert_near_equal(p['S'], 2.35850919305, tol)
        assert_near_equal(p['rho'], 9.44448e-5, tol)

        p['T'] = 1500.
        p['n'] = np.array([8.15344274e-06, 2.27139552e-02, 4.07672137e-06])
        p['result_T'] = np.array([-1.48684061e+01, -5.86384040e+00, -2.68839475e-03])
        p['result_P'] = np.array([3.33393139e-01, 3.33393139e-01, -5.97840423e-05])
        p['n_moles'] = 0.022726185333
        p.run_model()

        assert_near_equal(p['Cp'], 0.322460071411, tol)
        assert_near_equal(p['gamma'], 1.16380, tol)
        assert_near_equal(p['h'], -1801.35777129, tol)
        assert_near_equal(p['S'], 1.58630171846, tol)
        assert_near_equal(p['rho'], 0.0003648856, tol)


if __name__ == "__main__":
    unittest.main()
