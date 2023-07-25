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
        old = np.seterr(divide='raise')

        try:
            self.prob.run_model()
            tol = 1e-5
            print()

            reg_data = 167.78120192079388
            pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 13.500
            pyc = self.prob['DESIGN.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0177588
            pyc = self.prob['DESIGN.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 4.44344812
            pyc = self.prob['DESIGN.balance.turb_PR'][0]
            print('HPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11800.00602
            pyc = self.prob['DESIGN.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.80249303
            pyc = self.prob['DESIGN.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1188.187733639949
            pyc = self.prob['DESIGN.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)


            print('#'*10)
            print('# OD1')
            print('#'*10)
            reg_data = 167.7812019326619
            pyc = self.prob['OD1.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 13.500
            pyc = self.prob['OD1.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.01775886276199831
            pyc = self.prob['OD1.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8070.00
            pyc = self.prob['OD1.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 17673.630645166682
            pyc = self.prob['OD1.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.6131345333
            pyc = self.prob['OD1.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1188.1877337
            pyc = self.prob['OD1.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)


            print('#'*10)
            print('# OD2')
            print('#'*10)
            reg_data = 226.1582667
            pyc = self.prob['OD2.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11.97034442
            pyc = self.prob['OD2.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.016484348
            pyc = self.prob['OD2.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8288.751672
            pyc = self.prob['OD2.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 24024.738726
            pyc = self.prob['OD2.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.71183305
            pyc = self.prob['OD2.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1278.233868
            pyc = self.prob['OD2.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)


            print('#'*10)
            print('# OD1dry')
            print('#'*10)
            reg_data = 167.7812019
            pyc = self.prob['OD1dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 13.500
            pyc = self.prob['OD1dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.01775886
            pyc = self.prob['OD1dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8070.00
            pyc = self.prob['OD1dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11800.00602
            pyc = self.prob['OD1dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.80249303
            pyc = self.prob['OD1dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1188.187733
            pyc = self.prob['OD1dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD2dry')
            print('#'*10)
            reg_data = 226.158267
            pyc = self.prob['OD2dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11.9703444
            pyc = self.prob['OD2dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.016484348
            pyc = self.prob['OD2dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8288.751674
            pyc = self.prob['OD2dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 17263.807784
            pyc = self.prob['OD2dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.0785433412
            pyc = self.prob['OD2dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1278.233868
            pyc = self.prob['OD2dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD3')
            print('#'*10)
            reg_data = 167.0537804
            pyc = self.prob['OD3dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 12.5255004
            pyc = self.prob['OD3dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0169796
            pyc = self.prob['OD3dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8197.2178375
            pyc = self.prob['OD3dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 13978.106823
            pyc = self.prob['OD3dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.0621843383
            pyc = self.prob['OD3dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1243.395077
            pyc = self.prob['OD3dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD4')
            print('#'*10)
            reg_data = 141.05599880
            pyc = self.prob['OD4dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 12.635341223
            pyc = self.prob['OD4dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.017071756
            pyc = self.prob['OD4dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8180.853747
            pyc = self.prob['OD4dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 12577.7442
            pyc = self.prob['OD4dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.0583853809
            pyc = self.prob['OD4dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1236.912578
            pyc = self.prob['OD4dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD5 dry')
            print('#'*10)
            reg_data = 62.08668193
            pyc = self.prob['OD5dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 16.95492445
            pyc = self.prob['OD5dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.018923417
            pyc = self.prob['OD5dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8901.5303811
            pyc = self.prob['OD5dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 5148.015919
            pyc = self.prob['OD5dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.9285676181
            pyc = self.prob['OD5dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1105.04970532
            pyc = self.prob['OD5dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD6 dry')
            print('#'*10)
            reg_data = 146.2281497
            pyc = self.prob['OD6dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11.76520553
            pyc = self.prob['OD6dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.016271478
            pyc = self.prob['OD6dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8326.43166292
            pyc = self.prob['OD6dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14144.18906
            pyc = self.prob['OD6dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.07004953
            pyc = self.prob['OD6dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1293.191306
            pyc = self.prob['OD6dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD7dry')
            print('#'*10)
            reg_data = 71.7685995
            pyc = self.prob['OD7dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11.87578349
            pyc = self.prob['OD7dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.01639430
            pyc = self.prob['OD7dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8305.89555495
            pyc = self.prob['OD7dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 6965.1181098
            pyc = self.prob['OD7dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.065737783
            pyc = self.prob['OD7dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1284.56340514
            pyc = self.prob['OD7dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD8dry')
            print('#'*10)
            reg_data = 33.5400276475
            pyc = self.prob['OD8dry.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 10.740144458636
            pyc = self.prob['OD8dry.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0153433156
            pyc = self.prob['OD8dry.balance.FAR'][0]
            print('Main FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 8466.9052471
            pyc = self.prob['OD8dry.balance.Nmech'][0]
            print('HP Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 3314.4866922
            pyc = self.prob['OD8dry.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.0964228045193
            pyc = self.prob['OD8dry.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1357.997875
            pyc = self.prob['OD8dry.comp.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print()
        finally:
            np.seterr(**old)

if __name__ == "__main__":
    unittest.main()
