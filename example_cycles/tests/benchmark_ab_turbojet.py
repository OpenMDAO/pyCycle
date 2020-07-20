import numpy as np
import unittest
import os

import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.afterburning_turbojet import MPABTurbojet


class DesignTestCase(unittest.TestCase):

    def setUp(self):

        #Set up problem:

        self.prob = om.Problem()

        self.prob.model = MPABTurbojet()

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)
        self.prob.final_setup()

        #These values will go away when set_input_defaults is fixed:
        self.prob.set_val('DESIGN.comp.PR', 13.5),
        self.prob.set_val('DESIGN.comp.eff', 0.83),
        self.prob.set_val('DESIGN.turb.eff', 0.86),

        #Set initial conditions and initial guesses:

        self.prob.set_val('DESIGN.fc.alt', 0.0, units='ft'),
        self.prob.set_val('DESIGN.fc.MN', 0.000001),
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2370.0, units='degR'),
        self.prob.set_val('DESIGN.balance.rhs:W', 11800.0, units='lbf'),

        self.prob['DESIGN.balance.FAR'] = 0.0175506829934
        self.prob['DESIGN.balance.W'] = 168.453135137
        self.prob['DESIGN.balance.turb_PR'] = 4.46138725662
        self.prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        self.prob['DESIGN.fc.balance.Tt'] = 518.665288153

        pts = ['OD1','OD2','OD1dry','OD2dry','OD3dry','OD4dry','OD5dry','OD6dry','OD7dry','OD8dry'] 

        MNs = [0.000001, 0.8, 0.000001, 0.8, 1.00001, 1.2, 0.6, 1.6, 1.6, 1.8]
        alts = [0.0, 0.0, 0.0, 0.0, 15000.0, 25000.0, 35000.0, 35000.0, 50000.0, 70000.0]
        T4s = [2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0]
        ab_FARs = [0.031523391, 0.022759941, 0, 0, 0, 0, 0, 0, 0, 0]
        Rlines = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        W_guess = [168.0, 225.917, 168.005, 225.917, 166.074, 141.2, 61.70780608, 145.635, 71.53855266, 33.347]
        FAR_guess = [.01755, .016289, .01755, .01629, .0168, .01689, 0.01872827, .016083, 0.01619524, 0.015170]
        Nmech_guess = [8070., 8288.85, 8070, 8288.85, 8197.39, 8181.03, 8902.24164717, 8326.586, 8306.00268554, 8467.2404]
        Pt_guess = [14.696, 22.403, 14.696, 22.403, 15.7034, 13.230, 4.41149502, 14.707, 7.15363767, 3.7009]
        Tt_guess = [518.67, 585.035, 518.67, 585.04, 558.310, 553.409, 422.29146617, 595.796, 589.9425019, 646.8115]
        PR_guess = [4.4613, 4.8185, 4.4613, 4.8185, 4.669, 4.6425, 4.42779036, 4.8803, 4.84652723, 5.11582]

        for i, pt in enumerate(pts):
            self.prob.set_val(pt+'.fc.alt', alts[i], units='ft'),
            self.prob.set_val(pt+'.fc.MN', MNs[i]),

            self.prob.set_val(pt+'.balance.rhs:FAR', T4s[i], units='degR'),
            self.prob.set_val(pt+'.balance.rhs:W', Rlines[i]),
            self.prob.set_val(pt+'.ab.Fl_I:FAR', ab_FARs[i]),

            self.prob[pt+'.balance.W'] = W_guess[i]
            self.prob[pt+'.balance.FAR'] = FAR_guess[i]
            self.prob[pt+'.balance.Nmech'] = Nmech_guess[i]
            self.prob[pt+'.fc.balance.Pt'] = Pt_guess[i]
            self.prob[pt+'.fc.balance.Tt'] = Tt_guess[i]
            self.prob[pt+'.turb.PR'] = PR_guess[i]


    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 168.005
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01755
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.4613
        pyc = self.prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11800.0
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.79415
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1190.18
        pyc = self.prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 168.005
        pyc = self.prob['OD1.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['OD1.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01755
        pyc = self.prob['OD1.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8070.00
        pyc = self.prob['OD1.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 17799.7
        pyc = self.prob['OD1.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.61420
        pyc = self.prob['OD1.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1190.18
        pyc = self.prob['OD1.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 225.917
        pyc = self.prob['OD2.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11.971
        pyc = self.prob['OD2.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01629
        pyc = self.prob['OD2.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8288.85
        pyc = self.prob['OD2.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 24085.2
        pyc = self.prob['OD2.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.71066
        pyc = self.prob['OD2.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1280.18
        pyc = self.prob['OD2.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 168.005
        pyc = self.prob['OD1dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['OD1dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01755
        pyc = self.prob['OD1dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8070.00
        pyc = self.prob['OD1dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11800.0
        pyc = self.prob['OD1dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.79415
        pyc = self.prob['OD1dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1190.18
        pyc = self.prob['OD1dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 225.917
        pyc = self.prob['OD2dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11.971
        pyc = self.prob['OD2dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01629
        pyc = self.prob['OD2dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8288.85
        pyc = self.prob['OD2dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 17210.0
        pyc = self.prob['OD2dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.06924
        pyc = self.prob['OD2dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1280.18
        pyc = self.prob['OD2dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 166.073
        pyc = self.prob['OD3dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 12.526
        pyc = self.prob['OD3dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01680
        pyc = self.prob['OD3dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8197.38
        pyc = self.prob['OD3dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13881.2
        pyc = self.prob['OD3dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.05281
        pyc = self.prob['OD3dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1243.86
        pyc = self.prob['OD3dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 141.201
        pyc = self.prob['OD4dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 12.636
        pyc = self.prob['OD4dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01689
        pyc = self.prob['OD4dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8181.03
        pyc = self.prob['OD4dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 12589.2
        pyc = self.prob['OD4dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.04755
        pyc = self.prob['OD4dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1237.20
        pyc = self.prob['OD4dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 61.708
        pyc = self.prob['OD5dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16.956
        pyc = self.prob['OD5dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01873
        pyc = self.prob['OD5dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8902.24
        pyc = self.prob['OD5dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5109.4
        pyc = self.prob['OD5dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.92067
        pyc = self.prob['OD5dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1105.24
        pyc = self.prob['OD5dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 145.635
        pyc = self.prob['OD6dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11.765
        pyc = self.prob['OD6dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01608
        pyc = self.prob['OD6dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8326.59
        pyc = self.prob['OD6dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14061.9
        pyc = self.prob['OD6dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.06142
        pyc = self.prob['OD6dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1294.80
        pyc = self.prob['OD6dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 71.539
        pyc = self.prob['OD7dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11.876
        pyc = self.prob['OD7dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01620
        pyc = self.prob['OD7dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8306.00
        pyc = self.prob['OD7dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 6930.5
        pyc = self.prob['OD7dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.05654
        pyc = self.prob['OD7dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1286.85
        pyc = self.prob['OD7dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 33.347
        pyc = self.prob['OD8dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 10.741
        pyc = self.prob['OD8dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01517
        pyc = self.prob['OD8dry.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8467.24
        pyc = self.prob['OD8dry.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 3290.0
        pyc = self.prob['OD8dry.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.08806
        pyc = self.prob['OD8dry.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1359.21
        pyc = self.prob['OD8dry.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()
