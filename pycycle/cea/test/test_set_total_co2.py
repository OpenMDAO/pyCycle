import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp

from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.set_total import SetTotal
from pycycle.cea import species_data


class SetTotalTestCase(unittest.TestCase):

    def test_set_total_tp(self):

        thermo = species_data.Thermo(species_data.co2_co_o2)

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, mode="T")
        r = p.model
        r.add_subsystem(
            'n_init',
            IndepVarComp(
                'init_prod_amounts',
                thermo.init_prod_amounts),
            promotes=["*"])
        r.add_subsystem('T_init', IndepVarComp('T', 4000., units='degK'), promotes=["*"])
        r.add_subsystem('P_init', IndepVarComp('P', 1.034210, units="bar"), promotes=["*"])

        p.set_solver_print(level=2)
        p.setup(check=False)

        # from openmdao.api import view_tree
        # view_tree(p)
        p.run_model()

        # p.check_partial_derivatives()

        # goal_concentrations = np.array([0.61976,0.07037,0.30988]) # original
        # cea mole fractions
        expected_concentrations = np.array([0.62003271, 0.06995092, 0.31001638])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        # print(expected_concentrations)
        # print(concentrations)
        assert_rel_error(self, concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329313730421

        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.19054696779, 1e-4)

        # 1500K
        p['T'] = 1500  # degK
        p['P'] = 1.034210  # bar
        p.run_model()

        # [0.00036, 0.99946, 0.00018])
        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        expected_n_moles = 0.022726185333
        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.16380, 1e-4)

    def test_set_total_hp(self):

        thermo = species_data.Thermo(species_data.co2_co_o2)

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, mode="h")
        p.model.suppress_solver_output = True

        r = p.model
        r.add_subsystem(
            'n_init',
            IndepVarComp(
                'init_prod_amounts',
                thermo.init_prod_amounts),
            promotes=["*"])
        r.add_subsystem('h_init', IndepVarComp('h', 340, units='cal/g'), promotes=["*"])
        r.add_subsystem('P_init', IndepVarComp('P', 1.034210, units='bar'), promotes=["*"])

        p.set_solver_print(level=2)

        p.setup(check=False)

        p.run_model()

        # goal_concentrations = np.array([0.61976,0.07037,0.30988]) # original
        # cea mole fractions

        expected_concentrations = np.array([0.61989858, 0.07015213, 0.30994929])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_rel_error(self, concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329281722301

        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.19039688581, 1e-4)

        # 1500K
        p['h'] = -1801.35537381
        # seems to want to start from a different guess than the last converged point
        p['n'] = np.array([.33333, .33333, .33333])
        p.run_model()

        # [0.00036, 0.99946, 0.00018])
        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        # print(expected_concentrations)
        # print(concentrations)
        # print(p['T'])
        assert_rel_error(self, concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.022726185333
        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.16379012007, 1e-4)

    def test_set_total_sp(self):

        thermo = species_data.Thermo(species_data.co2_co_o2)

        # 4000k
        p = Problem()
        p.model = SetTotal(thermo_data=species_data.co2_co_o2, mode="S")
        p.model.suppress_solver_output = True
        r = p.model
        r.add_subsystem(
            'n_init',
            IndepVarComp(
                'init_prod_amounts',
                thermo.init_prod_amounts),
            promotes=["*"])
        r.add_subsystem(
            'S_init',
            IndepVarComp(
                'S',
                2.35711010759,
                units="Btu/(lbm*degR)"),
            promotes=["*"])
        r.add_subsystem('P_init', IndepVarComp('P', 1.034210, units="bar"), promotes=["*"])

        p.set_solver_print(level=2)

        p.setup(check=False)

        p.run_model()

        np.seterr(all='raise')

        # goal_concentrations = np.array([0.61976,0.07037,0.30988]) # original
        # cea mole fractions
        expected_concentrations = np.array([0.62003271, 0.06995092, 0.31001638])

        # [  2.35337787e-01   1.16205327e+03   1.17668894e-01]

        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles
        assert_rel_error(self, concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.0329313730421

        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.19054696779, 1e-4)

        # 1500K
        p['S'] = 1.5852424435
        # p.model.chem_eq.DEBUG=True
        p.run_model()

        # [0.00036, 0.99946, 0.00018])
        expected_concentrations = np.array([3.58768646e-04, 9.99461847e-01, 1.79384323e-04])
        n = p['n']
        n_moles = p['n_moles']
        concentrations = n / n_moles

        assert_rel_error(self, concentrations, expected_concentrations, 1e-4)

        expected_n_moles = 0.022726185333
        assert_rel_error(self, n_moles, expected_n_moles, 1e-4)
        assert_rel_error(self, p['gamma'], 1.16381209181, 1e-4)

        # check = p.check_partial_derivatives()


if __name__ == "__main__":

    import numpy as np
    import scipy as sp

    np.seterr(all='raise')
    sp.seterr(all='raise')

    unittest.main()
