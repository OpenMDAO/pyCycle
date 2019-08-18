import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp

from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.props_rhs import PropsRHS
from pycycle.cea.props_calcs import PropsCalcs
from pycycle.cea import species_data


class PropsRHSTestCase(unittest.TestCase):

    def setUp(self):

        self.thermo = species_data.Thermo(species_data.co2_co_o2)

        p = self.prob = Problem()
        p.model = Group()
        p.model.suppress_solver_output = True

        p.model.add_subsystem('props_rhs', PropsRHS(thermo=self.thermo),
                              promotes=['T', 'n', 'b0', 'rhs_T', 'rhs_P', 'lhs_TP', 'n_moles'])
        p.model.add_subsystem('T', IndepVarComp('T', 4000., units='degK'), promotes=['*'])
        p.model.add_subsystem('P', IndepVarComp('P', 1.034210, units='bar'), promotes=['*'])

        n = np.array([0.02040741, 0.0023147, 0.0102037])
        p.model.add_subsystem('n', IndepVarComp('n', n), promotes=['*'])

        b = np.array([0.02272211, 0.04544422])
        p.model.add_subsystem('b0', IndepVarComp('b0', b), promotes=['*'])
        p.model.add_subsystem('n_moles_desvar', IndepVarComp('n_moles', 0.03292581), promotes=['*'])

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
        assert_rel_error(self, p['rhs_T'], goal_rhs_T, tol)
        assert_rel_error(self, p['rhs_P'], goal_rhs_P, tol)
        assert_rel_error(self, p['lhs_TP'], goal_lhs_TP, tol)

        # p.check_partial_derivatives()


class PropsCalcsTestCase(unittest.TestCase):

    def setUp(self):

        self.thermo = species_data.Thermo(species_data.co2_co_o2)

        p = self.prob = Problem()
        p.model = Group()
        p.model.suppress_solver_output = True

        p.model.add_subsystem('props', PropsCalcs(thermo=self.thermo), promotes=['*'])
        p.model.add_subsystem('T', IndepVarComp('T', 4000., units='degK'), promotes=['*'])
        p.model.add_subsystem('P', IndepVarComp('P', 1.034210, units='bar'), promotes=['*'])

        n = np.array([0.02040741, 0.0023147, 0.0102037])
        p.model.add_subsystem('n', IndepVarComp('n', n), promotes=['*'])

        result_T = np.array([-1.74791977, 1.81604241, -0.24571810])
        p.model.add_subsystem('r_T', IndepVarComp('result_T', result_T), promotes=['*'])

        result_P = np.array([0.48300853, 0.48301125, -0.01522548])
        p.model.add_subsystem('r_P', IndepVarComp('result_P', result_P), promotes=['*'])

        p.setup(check=False)
        p['n_moles'] = 0.03292581

    def test_total_calcs(self):

        p = self.prob
        p.run_model()

        tol = 1e-4
        assert_rel_error(self, p['Cp'], 0.579647062532, tol)
        assert_rel_error(self, p['gamma'], 1.19039, tol)
        assert_rel_error(self, p['h'], 340.324938088, tol)
        assert_rel_error(self, p['S'], 2.35850919305, tol)
        assert_rel_error(self, p['rho'], 9.44448e-5, tol)

        p['T'] = 1500.
        p['n'] = np.array([8.15344274e-06, 2.27139552e-02, 4.07672137e-06])
        p['result_T'] = np.array([-1.48684061e+01, -5.86384040e+00, -2.68839475e-03])
        p['result_P'] = np.array([3.33393139e-01, 3.33393139e-01, -5.97840423e-05])
        p['n_moles'] = 0.022726185333
        p.run_model()

        assert_rel_error(self, p['Cp'], 0.322460071411, tol)
        assert_rel_error(self, p['gamma'], 1.16380, tol)
        assert_rel_error(self, p['h'], -1801.35777129, tol)
        assert_rel_error(self, p['S'], 1.58630171846, tol)
        assert_rel_error(self, p['rho'], 0.0003648856, tol)


if __name__ == "__main__":
    unittest.main()
