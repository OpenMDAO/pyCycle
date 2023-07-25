import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.single_spool_turboshaft import MPSingleSpool

class SingleSpoolTestCase(unittest.TestCase):

    def benchmark_case1(self):

        prob = om.Problem()

        prob.model = mp_single_spool = MPSingleSpool()

        prob.setup(check=False)

        #Define the initial design point
        prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
        prob.set_val('DESIGN.balance.pwr_target', 4000.0, units='hp')
        prob.set_val('DESIGN.balance.nozz_PR_target', 1.2)
        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)
        prob.set_val('DESIGN.pt.eff', 0.9)

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 27.265
        prob['DESIGN.balance.turb_PR'] = 3.8768
        prob['DESIGN.balance.pt_PR'] = 2.8148
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        for i,pt in enumerate(mp_single_spool.od_pts):

            # initial guesses
            prob[pt+'.balance.W'] = 27.265
            prob[pt+'.balance.FAR'] = 0.0175506829934
            prob[pt+'.balance.HP_Nmech'] = 8070.0
            prob[pt+'.fc.balance.Pt'] = 15.703
            prob[pt+'.fc.balance.Tt'] = 558.31
            prob[pt+'.turb.PR'] = 3.8768
            prob[pt+'.pt.PR'] = 2.8148


        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

        old = np.seterr(divide='raise')

        try:
            prob.run_model()
            tol = 1e-5
            print()

            reg_data = 27.265344349
            ans = prob['DESIGN.inlet.Fl_O:stat:W'][0]
            print('W:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 13.5
            ans = prob['DESIGN.perf.OPR'][0]
            print('OPR:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.01755865988
            ans = prob['DESIGN.balance.FAR'][0]
            print('Main FAR:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 3.876811443516
            ans = prob['DESIGN.balance.turb_PR'][0]
            print('HPT PR:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 800.85349568
            ans = prob['DESIGN.perf.Fg'][0]
            print('Fg:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 2.15204967
            ans = prob['DESIGN.perf.TSFC'][0]
            print('TSFC:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 1190.1777648
            ans = prob['DESIGN.comp.Fl_O:tot:T'][0]
            print('Tt3:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            print('#'*10)
            print('# OD')
            print('#'*10)
            reg_data = 25.8972423
            ans = prob['OD.inlet.Fl_O:stat:W'][0]
            print('W:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 12.4297249
            ans = prob['OD.perf.OPR'][0]
            print('OPR:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.0163117040
            ans = prob['OD.balance.FAR'][0]
            print('Main FAR:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 7853.754354
            ans = prob['OD.balance.HP_Nmech'][0]
            print('HP Nmech:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 696.62110739
            ans = prob['OD.perf.Fg'][0]
            print('Fg:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 2.5063687
            ans = prob['OD.perf.TSFC'][0]
            print('TSFC:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 1158.519646
            ans = prob['OD.comp.Fl_O:tot:T'][0]
            print('Tt3:', ans, reg_data)
            assert_near_equal(ans, reg_data, tol)
        finally:
            np.seterr(**old)


if __name__ == "__main__":
    unittest.main()