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

        prob.model = MPSingleSpool()

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

        prob.setup(check=False)

        ##Intial contitions
        prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
        prob.set_val('DESIGN.balance.pwr_target', 4000.0, units='hp')
        prob.set_val('DESIGN.balance.nozz_PR_target', 1.2)

        ##Values will go away after set_input_defaults is fixed
        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)
        prob.set_val('DESIGN.pt.eff', 0.9)

        ##Initial conditions and initial balance guesses for OD points
        od_pts = ['OD', 'OD2']
        od_MNs = [0.1, 0.000001]
        od_alts =[0.0, 0.0]
        od_pwrs =[3500.0, 3500.0]
        od_nmechs =[5000., 5000.]

        for i,pt in enumerate(od_pts):

            prob.set_val(pt+'.fc.alt', od_alts[i], units='ft')
            prob.set_val(pt+'.fc.MN', od_MNs[i])
            prob.set_val(pt+'.LP_Nmech', od_nmechs[i], units='rpm')
            prob.set_val(pt+'.balance.pwr_target', od_pwrs[i], units='hp')

            prob[pt+'.balance.W'] = 27.265
            prob[pt+'.balance.FAR'] = 0.0175506829934
            prob[pt+'.balance.HP_Nmech'] = 8070.0
            prob[pt+'.fc.balance.Pt'] = 15.703
            prob[pt+'.fc.balance.Tt'] = 558.31
            prob[pt+'.turb.PR'] = 3.8768
            prob[pt+'.pt.PR'] = 2.8148

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 27.265
        prob['DESIGN.balance.turb_PR'] = 3.8768
        prob['DESIGN.balance.pt_PR'] = 2.8148
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        np.seterr(divide='raise')

        prob.run_model()
        tol = 1e-3
        print()

        reg_data = 27.265342457866705
        ans = prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 13.5
        ans = prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.01755077946196377
        ans = prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 3.876811443569159
        ans = prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 800.8503668285215
        ans = prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 2.151092078410839
        ans = prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1190.1777648503974
        ans = prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 25.897231212494944
        ans = prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 12.42972778706185
        ans = prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.016304387482120156
        ans = prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 7853.753342243985
        ans = prob['OD.balance.HP_Nmech'][0]
        print('HP Nmech:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 696.618372248896
        ans = prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 2.5052545862974696
        ans = prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1158.5197002795887
        ans = prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()