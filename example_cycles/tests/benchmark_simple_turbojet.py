import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.simple_turbojet import MPTurbojet

class SimpleTurbojetTestCase(unittest.TestCase):

    def benchmark_case1(self):

        prob = om.Problem()

        prob.model = mp_turbojet = MPTurbojet()

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

        prob.setup(check=False)

        ##Initial Conditions
        prob.set_val('DESIGN.fc.alt', 0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')

        ##Values that will go away when set_input_defaults is fixed
        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 168.453135137
        prob['DESIGN.balance.turb_PR'] = 4.46138725662
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        for i,pt in enumerate(mp_turbojet.od_pts):

            # initial guesses
            prob[pt+'.balance.W'] = 166.073
            prob[pt+'.balance.FAR'] = 0.01680
            prob[pt+'.balance.Nmech'] = 8197.38
            prob[pt+'.fc.balance.Pt'] = 15.703
            prob[pt+'.fc.balance.Tt'] = 558.31
            prob[pt+'.turb.PR'] = 4.6690

        old = np.seterr(divide='raise')

        try:
            prob.run_model()
            tol = 1e-5
            print()

            reg_data = 147.333
            ans = prob['DESIGN.inlet.Fl_O:stat:W'][0]
            print('W:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 13.500
            ans = prob['DESIGN.perf.OPR'][0]
            print('OPR:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.01776487
            ans = prob['DESIGN.balance.FAR'][0]
            print('Main FAR:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 3.8591364
            ans = prob['DESIGN.balance.turb_PR'][0]
            print('HPT PR:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 11800.00455497
            ans = prob['DESIGN.perf.Fg'][0]
            print('Fg:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.7985165
            ans = prob['DESIGN.perf.TSFC'][0]
            print('TSFC:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 1187.7610184
            ans = prob['DESIGN.comp.Fl_O:tot:T'][0]
            print('Tt3:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 142.786698
            ans = prob['OD0.inlet.Fl_O:stat:W'][0]
            print('W:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 12.8588424
            ans = prob['OD0.perf.OPR'][0]
            print('OPR:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.01676938946
            ans = prob['OD0.balance.FAR'][0]
            print('Main FAR:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 7943.9331210
            ans = prob['OD0.balance.Nmech'][0]
            print('HP Nmech:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 11000.00488519
            ans = prob['OD0.perf.Fg'][0]
            print('Fg:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 0.783636794
            ans = prob['OD0.perf.TSFC'][0]
            print('TSFC:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            reg_data = 1168.067147267
            ans = prob['OD0.comp.Fl_O:tot:T'][0]
            print('Tt3:', reg_data, ans)
            assert_near_equal(ans, reg_data, tol)

            print()
        finally:
            np.seterr(**old)

if __name__ == "__main__":
    unittest.main()

