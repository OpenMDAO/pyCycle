import numpy as np
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc
from example_cycles.rde_ramjet import MPRDERamjet


class RDERamjetTestCase(unittest.TestCase):

    def test_benchmark(self):
        self.benchmark_case1()

    def benchmark_case1(self):
        prob = om.Problem()
        mp_ramjet = prob.model = MPRDERamjet()

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        # Design flight condition: Mach 2.50 at 40,000 ft
        prob.set_val('DESIGN.fc.alt', 40000.0, units='ft')
        prob.set_val('DESIGN.fc.MN', 2.50)

        old = np.seterr(divide='raise')
        try:
            prob.run_model()
            tol = 1e-4

            # Inlet airflow
            assert_near_equal(prob['DESIGN.inlet.Fl_O:stat:W'][0], 100.0, tol)

            # Overall pressure ratio (due to RDE detonation pressure rise)
            assert_near_equal(prob['DESIGN.perf.OPR'][0], 2.9628, tol)

            # Net thrust Fn = Fg - Fram
            assert_near_equal(prob['DESIGN.perf.Fn'][0], 7660.7, tol)

            # Gross thrust Fg
            assert_near_equal(prob['DESIGN.perf.Fg'][0], 15185.85, tol)

            # Ram drag
            assert_near_equal(prob['DESIGN.inlet.F_ram'][0], 7525.2, tol)

            # TSFC
            assert_near_equal(prob['DESIGN.perf.TSFC'][0], 1.3158, tol)

            # Combustor pressure ratio (Pt4 / Pt2)
            assert_near_equal(prob['DESIGN.burner.PR_RDE'][0], 2.9628, tol)

            # Injector plenum pressure
            assert_near_equal(prob['DESIGN.burner.Pt_inj'][0], 37.186, tol)

            # Detonation wave frequency
            assert_near_equal(prob['DESIGN.burner.f_rde'][0], 1317.9, tol)

            # Nozzle throat area
            assert_near_equal(prob['DESIGN.nozz.Throat:stat:area'][0], 82.733, tol)

        finally:
            np.seterr(**old)


if __name__ == "__main__":
    unittest.main()
