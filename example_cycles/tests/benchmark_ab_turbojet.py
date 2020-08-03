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

        self.prob.model = mp_abturbojet = MPABTurbojet()

        self.prob.setup()

        #Define the design point
        self.prob.set_val('DESIGN.comp.PR', 13.5),
        self.prob.set_val('DESIGN.comp.eff', 0.83),
        self.prob.set_val('DESIGN.turb.eff', 0.86),
        self.prob.set_val('DESIGN.fc.alt', 0.0, units='ft'),
        self.prob.set_val('DESIGN.fc.MN', 0.000001),
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2370.0, units='degR'),
        self.prob.set_val('DESIGN.balance.rhs:W', 11800.0, units='lbf'),

        # Set initial guesses for balances
        self.prob['DESIGN.balance.FAR'] = 0.0175506829934
        self.prob['DESIGN.balance.W'] = 168.453135137
        self.prob['DESIGN.balance.turb_PR'] = 4.46138725662
        self.prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        self.prob['DESIGN.fc.balance.Tt'] = 518.665288153

        W_guess = [168.0, 225.917, 168.005, 225.917, 166.074, 141.2, 61.70780608, 145.635, 71.53855266, 33.347]
        FAR_guess = [.01755, .016289, .01755, .01629, .0168, .01689, 0.01872827, .016083, 0.01619524, 0.015170]
        Nmech_guess = [8070., 8288.85, 8070, 8288.85, 8197.39, 8181.03, 8902.24164717, 8326.586, 8306.00268554, 8467.2404]
        Pt_guess = [14.696, 22.403, 14.696, 22.403, 15.7034, 13.230, 4.41149502, 14.707, 7.15363767, 3.7009]
        Tt_guess = [518.67, 585.035, 518.67, 585.04, 558.310, 553.409, 422.29146617, 595.796, 589.9425019, 646.8115]
        PR_guess = [4.4613, 4.8185, 4.4613, 4.8185, 4.669, 4.6425, 4.42779036, 4.8803, 4.84652723, 5.11582]

        for i, pt in enumerate(mp_abturbojet.od_pts):

            # initial guesses
            self.prob[pt+'.balance.W'] = W_guess[i]
            self.prob[pt+'.balance.FAR'] = FAR_guess[i]
            self.prob[pt+'.balance.Nmech'] = Nmech_guess[i]
            self.prob[pt+'.fc.balance.Pt'] = Pt_guess[i]
            self.prob[pt+'.fc.balance.Tt'] = Tt_guess[i]
            self.prob[pt+'.turb.PR'] = PR_guess[i]

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)


    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-5
        print()

        reg_data = 168.005
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.017550779
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

        reg_data = 0.017550779
        pyc = self.prob['OD1.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 8070.00
        pyc = self.prob['OD1.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 17799.4778
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

        reg_data = 11.97055297
        pyc = self.prob['OD2.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01628932
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

        reg_data = 0.017550779
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

        reg_data = 11.970553
        pyc = self.prob['OD2dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01628932
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

        reg_data = 12.5263207
        pyc = self.prob['OD3dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01680008
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

        reg_data = 12.636232
        pyc = self.prob['OD4dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01689342
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

        reg_data = 16.956252
        pyc = self.prob['OD5dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.018728274
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

        reg_data = 0.92065667
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

        reg_data = 11.765481
        pyc = self.prob['OD6dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01608294
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

        reg_data = 11.8758
        pyc = self.prob['OD7dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.01619524
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

        reg_data = 33.347375
        pyc = self.prob['OD8dry.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 10.740919
        pyc = self.prob['OD8dry.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.015169875
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
