import numpy as np
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc
from example_cycles.rde_turbojet import MPRDETurbojet


class RDETurbojetTestCase(unittest.TestCase):

    def test_benchmark(self):
        self.benchmark_case1()

    def benchmark_case1(self):
        prob = om.Problem()
        mp_turbojet = prob.model = MPRDETurbojet(use_rde=True)

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        # Initial Conditions for DESIGN
        prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)

        # Balance initial guesses
        prob['DESIGN.balance.FAR'] = 0.01755
        prob['DESIGN.balance.W'] = 125.0
        prob['DESIGN.balance.turb_PR'] = 4.20
        prob['DESIGN.fc.balance.Pt'] = 14.696
        prob['DESIGN.fc.balance.Tt'] = 518.67

        for i, pt in enumerate(mp_turbojet.od_pts):
            prob[pt + '.balance.W'] = 120.0
            prob[pt + '.balance.FAR'] = 0.01680
            prob[pt + '.balance.Nmech'] = 8197.38
            prob[pt + '.fc.balance.Pt'] = 15.703
            prob[pt + '.fc.balance.Tt'] = 558.31
            prob[pt + '.turb.PR'] = 4.6690

        old = np.seterr(divide='raise')
        try:
            prob.run_model()
            tol = 1e-4

            # --- DESIGN Point Regressions ---
            # Airflow W
            assert_near_equal(prob['DESIGN.inlet.Fl_O:stat:W'][0], 123.6359, tol)

            # Overall Pressure Ratio
            assert_near_equal(prob['DESIGN.perf.OPR'][0], 13.5000, tol)

            # Fuel to Air Ratio
            assert_near_equal(prob['DESIGN.balance.FAR'][0], 0.017765, tol)

            # Turbine Expansion Ratio
            assert_near_equal(prob['DESIGN.balance.turb_PR'][0], 3.8546, tol)

            # Gross Thrust
            assert_near_equal(prob['DESIGN.perf.Fg'][0], 11799.99, tol)

            # TSFC (16% reduction vs conventional Brayton 0.7985)
            assert_near_equal(prob['DESIGN.perf.TSFC'][0], 0.67008, tol)

            # Combustor Inlet Temperature
            assert_near_equal(prob['DESIGN.comp.Fl_O:tot:T'][0], 1187.76, tol)

            # RDE Combustor Pressure Ratio (Pressure Gain Combustion: Pt4/Pt3 = 1.863)
            assert_near_equal(prob['DESIGN.burner.PR_RDE'][0], 1.8633, tol)

            # Injector plenum pressure
            assert_near_equal(prob['DESIGN.burner.Pt_inj'][0], 174.587, tol)

            # Detonation wave speed
            assert_near_equal(prob['DESIGN.burner.D_cj'][0], 4184.15, tol)

            # Detonation frequency
            assert_near_equal(prob['DESIGN.burner.f_rde'][0], 1331.86, tol)

            # --- OD0 Off-Design Point Regressions ---
            assert_near_equal(prob['OD0.inlet.Fl_O:stat:W'][0], 97.1832, tol)
            assert_near_equal(prob['OD0.perf.OPR'][0], 12.2789, tol)
            assert_near_equal(prob['OD0.balance.FAR'][0], 0.015692, tol)
            assert_near_equal(prob['OD0.balance.Nmech'][0], 7621.79, tol)
            assert_near_equal(prob['OD0.perf.Fg'][0], 8662.85, tol)
            assert_near_equal(prob['OD0.perf.TSFC'][0], 0.68625, tol)
            assert_near_equal(prob['OD0.comp.Fl_O:tot:T'][0], 1119.53, tol)
            assert_near_equal(prob['OD0.burner.PR_RDE'][0], 1.8033, tol)

        finally:
            np.seterr(**old)


if __name__ == "__main__":
    unittest.main()
