import numpy as np
import unittest
import os

import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.high_bypass_turbofan import MPhbtf


class HBTFTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model = mp_hbtf = MPhbtf()
        self.prob.setup()

        #Define the design point
        self.prob.set_val('DESIGN.fan.PR', 1.685)
        self.prob.set_val('DESIGN.fan.eff', 0.8948)
        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.lpc.eff', 0.9243)
        self.prob.set_val('DESIGN.hpc.PR', 9.369),
        self.prob.set_val('DESIGN.hpc.eff', 0.8707),
        self.prob.set_val('DESIGN.hpt.eff', 0.8888),
        self.prob.set_val('DESIGN.lpt.eff', 0.8996),
        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.8)
        self.prob.set_val('DESIGN.T4_MAX', 2857, units='degR')
        self.prob.set_val('DESIGN.Fn_DES', 5500.0, units='lbf') 

        # Set initial guesses for balances
        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 316.0
        self.prob['DESIGN.balance.lpt_PR'] = 4.4
        self.prob['DESIGN.balance.hpt_PR'] = 3.6
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0   

        W_guesses = [300, 300, 700, 700]
        for i, pt in enumerate(mp_hbtf.od_pts):
            self.prob[pt+'.balance.FAR'] = 0.02467
            self.prob[pt+'.balance.W'] = W_guesses[i]
            self.prob[pt+'.balance.BPR'] = 5.105
            self.prob[pt+'.balance.lp_Nmech'] = 5000 
            self.prob[pt+'.balance.hp_Nmech'] = 15000 
            self.prob[pt+'.hpt.PR'] = 3.
            self.prob[pt+'.lpt.PR'] = 4.
            self.prob[pt+'.fan.map.RlineMap'] = 2.0
            self.prob[pt+'.lpc.map.RlineMap'] = 2.0
            self.prob[pt+'.hpc.map.RlineMap'] = 2.0

    def benchmark_case1(self):
        np.seterr(divide='raise')

        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-5
        print()

        reg_data = 321.25763
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02491989
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 3.622786
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.368917
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.49860
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.63130225
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.47613
        pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
        print('Tt3:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)


        print('#'*10)
        print('# OD0')
        print('#'*10)
        reg_data = 321.25685
        pyc = self.prob['OD0.inlet.Fl_O:stat:W'][0]
        print('W:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.0937151
        pyc = self.prob['OD0.perf.OPR'][0]
        print('OPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.0249198
        pyc = self.prob['OD0.balance.FAR'][0]
        print('FAR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14705.7
        pyc = self.prob['OD0.balance.hp_Nmech'][0]
        print('HPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4666.1
        pyc = self.prob['OD0.balance.lp_Nmech'][0]
        print('LPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['OD0.perf.Fg'][0]
        print('Fg:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.6313022
        pyc = self.prob['OD0.perf.TSFC'][0]
        print('TSFC:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['OD0.bld3.Fl_O:tot:T'][0]
        print('Tt3:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.105
        pyc = self.prob['OD0.balance.BPR'][0]
        print('BPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)


        print('#'*10)
        print('# OD1')
        print('#'*10)
        reg_data = 327.2713
        pyc = self.prob['OD1.inlet.Fl_O:stat:W'][0]
        print('W:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 32.41462
        pyc = self.prob['OD1.perf.OPR'][0]
        print('OPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02616998
        pyc = self.prob['OD1.balance.FAR'][0]
        print('FAR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14952.3
        pyc = self.prob['OD1.balance.hp_Nmech'][0]
        print('HPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4933.5037
        pyc = self.prob['OD1.balance.lp_Nmech'][0]
        print('LPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13890.0598
        pyc = self.prob['OD1.perf.Fg'][0]
        print('Fg:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.645675
        pyc = self.prob['OD1.perf.TSFC'][0]
        print('TSFC:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1317.31
        pyc = self.prob['OD1.bld3.Fl_O:tot:T'][0]
        print('Tt3:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.898435
        pyc = self.prob['OD1.balance.BPR'][0]
        print('BPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        print('#'*10)
        print('# OD2')
        print('#'*10)

        reg_data = 825.0649
        pyc = self.prob['OD2.inlet.Fl_O:stat:W'][0]
        print('W:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.998212
        pyc = self.prob['OD2.perf.OPR'][0]
        print('OPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data =  0.0297665
        pyc = self.prob['OD2.balance.FAR'][0]
        print('FAR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16222.1
        pyc = self.prob['OD2.balance.hp_Nmech'][0]
        print('HPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5050.0624
        pyc = self.prob['OD2.balance.lp_Nmech'][0]
        print('LPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 29930.9196
        pyc = self.prob['OD2.perf.Fg'][0]
        print('Fg:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.4750893
        pyc = self.prob['OD2.perf.TSFC'][0]
        print('TSFC:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1536.94
        pyc = self.prob['OD2.bld3.Fl_O:tot:T'][0]
        print('Tt3:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.2427278
        pyc = self.prob['OD2.balance.BPR'][0]
        print('BPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)


        print('#'*10)
        print('# OD3')
        print('#'*10)
        reg_data = 786.75457
        pyc = self.prob['OD3.inlet.Fl_O:stat:W'][0]
        print('W:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.41786
        pyc = self.prob['OD3.perf.OPR'][0]
        print('OPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data =  0.0291345
        pyc = self.prob['OD3.balance.FAR'][0]
        print('FAR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16065.1
        pyc = self.prob['OD3.balance.hp_Nmech'][0]
        print('HPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4949.15079
        pyc = self.prob['OD3.balance.lp_Nmech'][0]
        print('LPT Nmech:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 27113.2800
        pyc = self.prob['OD3.perf.Fg'][0]
        print('Fg:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.366778752
        pyc = self.prob['OD3.perf.TSFC'][0]
        print('TSFC:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1509.41
        pyc = self.prob['OD3.bld3.Fl_O:tot:T'][0]
        print('Tt3:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.28251347
        pyc = self.prob['OD3.balance.BPR'][0]
        print('BPR:', pyc, reg_data)
        assert_near_equal(pyc, reg_data, tol)


if __name__ == "__main__":
    unittest.main()