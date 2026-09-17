"""
Tests for RDECombustor element in pyCycle.

Verifies:
1. Clean integration into pyCycle's Cycle class and flow port network.
2. Pressure gain (Pt_out > Pt_in) with physical fuel addition and chemical equilibrium.
3. Compatibility with both CEA and TABULAR thermodynamics packages.
4. On-design (MN specified) and Off-design (area specified) static property solves.
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc
from pycycle.elements.rde_combustor import RDECombustor, RDEPressureGainComp


class TestRDECombustor(unittest.TestCase):

    def test_rde_combustor_cea(self):
        """Test RDECombustor within a Cycle using CEA thermodynamics."""
        prob = om.Problem()
        model = prob.model = pyc.Cycle()
        model.options['thermo_method'] = 'CEA'
        model.options['thermo_data'] = pyc.species_data.janaf

        model.add_subsystem('flow_start', pyc.FlowStart())
        model.add_subsystem('rde', RDECombustor(fuel_type="JP-7"))

        model.pyc_connect_flow('flow_start.Fl_O', 'rde.Fl_I')

        model.set_input_defaults('rde.Fl_I:FAR', 0.02)
        model.set_input_defaults('rde.MN', 0.4)
        model.set_input_defaults('rde.dPqP_inj', 0.10)
        model.set_input_defaults('rde.eta_rde', 0.85)

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        # Inflow conditions: Station 3 representative
        prob.set_val('flow_start.P', 150.0, units='psi')
        prob.set_val('flow_start.T', 1100.0, units='degR')
        prob.set_val('flow_start.W', 100.0, units='lbm/s')

        prob.run_model()

        pt_in = prob.get_val('flow_start.Fl_O:tot:P', units='psi')[0]
        pt_out = prob.get_val('rde.Fl_O:tot:P', units='psi')[0]
        pr_rde = prob.get_val('rde.PR_RDE')[0]
        w_fuel = prob.get_val('rde.Wfuel', units='lbm/s')[0]
        d_cj = prob.get_val('rde.D_cj', units='ft/s')[0]
        f_rde = prob.get_val('rde.f_rde', units='Hz')[0]

        # 1. Verify net pressure gain
        self.assertGreater(pt_out, pt_in, "RDE exit total pressure must be greater than inlet total pressure")
        self.assertGreater(pr_rde, 1.0, "Net pressure ratio must exceed 1.0")

        # 2. Verify fuel mass balance: Wfuel = W_air * FAR = 100 * 0.02 = 2.0
        assert_near_equal(w_fuel, 2.0, tolerance=1e-4)

        # 3. Verify wave kinematics are in physical ranges
        self.assertGreater(d_cj, 4000.0, "Detonation velocity must be > 4000 ft/s (~1200 m/s)")
        self.assertLess(d_cj, 8000.0, "Detonation velocity must be < 8000 ft/s (~2400 m/s)")
        self.assertGreater(f_rde, 1000.0, "RDE frequency must be > 1 kHz")

    def test_rde_combustor_tabular(self):
        """Test RDECombustor within a Cycle using fast TABULAR thermodynamics."""
        prob = om.Problem()
        model = prob.model = pyc.Cycle()
        model.options['thermo_method'] = 'TABULAR'
        model.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        model.add_subsystem('flow_start', pyc.FlowStart())
        model.add_subsystem('rde', RDECombustor(fuel_type="FAR"))

        model.pyc_connect_flow('flow_start.Fl_O', 'rde.Fl_I')

        model.set_input_defaults('rde.Fl_I:FAR', 0.025)
        model.set_input_defaults('rde.MN', 0.35)
        model.set_input_defaults('rde.dPqP_inj', 0.12)
        model.set_input_defaults('rde.eta_rde', 0.85)

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        prob.set_val('flow_start.P', 200.0, units='psi')
        prob.set_val('flow_start.T', 1200.0, units='degR')
        prob.set_val('flow_start.W', 150.0, units='lbm/s')

        prob.run_model()

        pt_in = prob.get_val('flow_start.Fl_O:tot:P', units='psi')[0]
        pt_out = prob.get_val('rde.Fl_O:tot:P', units='psi')[0]
        pr_rde = prob.get_val('rde.PR_RDE')[0]

        self.assertGreater(pt_out, pt_in, "RDE exit pressure must exceed inlet pressure in tabular mode")
        self.assertGreater(pr_rde, 1.0)

    def test_rde_combustor_off_design(self):
        """Test RDECombustor in off-design mode with area specified."""
        prob = om.Problem()
        model = prob.model = pyc.Cycle(design=False)
        model.options['thermo_method'] = 'TABULAR'
        model.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        model.add_subsystem('flow_start', pyc.FlowStart())
        model.add_subsystem('rde', RDECombustor(fuel_type="FAR"))

        model.pyc_connect_flow('flow_start.Fl_O', 'rde.Fl_I')

        model.set_input_defaults('rde.Fl_I:FAR', 0.02)
        model.set_input_defaults('rde.area', 50.0, units='inch**2')
        model.set_input_defaults('rde.dPqP_inj', 0.12)
        model.set_input_defaults('rde.eta_rde', 0.85)

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        prob.set_val('flow_start.P', 180.0, units='psi')
        prob.set_val('flow_start.T', 1150.0, units='degR')
        prob.set_val('flow_start.W', 120.0, units='lbm/s')

        prob.run_model()

        # Check that exit area matches specified input
        area_out = prob.get_val('rde.Fl_O:stat:area', units='inch**2')[0]
        assert_near_equal(area_out, 50.0, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
