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

        self.prob.set_val('DESIGN.fan.PR', 1.685)
        self.prob.set_val('DESIGN.fan.eff', 0.8948)

        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.lpc.eff', 0.9243)

        self.prob.set_val('DESIGN.hpc.PR', 9.369)
        self.prob.set_val('DESIGN.hpc.eff', 0.8707)

        self.prob.set_val('DESIGN.hpt.eff', 0.8888)
        self.prob.set_val('DESIGN.lpt.eff', 0.8996)

        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.8)

        self.prob.set_val('DESIGN.T4_MAX', 2857, units='degR')
        self.prob.set_val('DESIGN.Fn_DES', 5900.0, units='lbf')

        self.prob.set_val('OD_full_pwr.T4_MAX', 2857, units='degR')
        self.prob.set_val('OD_part_pwr.PC', 0.8)


        # Set initial guesses for balances
        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 100.
        self.prob['DESIGN.balance.lpt_PR'] = 4.0
        self.prob['DESIGN.balance.hpt_PR'] = 3.0
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0


        for pt in ['OD_full_pwr', 'OD_part_pwr']:

            # initial guesses
            self.prob[pt+'.balance.FAR'] = 0.02467
            self.prob[pt+'.balance.W'] = 300
            self.prob[pt+'.balance.BPR'] = 5.105
            self.prob[pt+'.balance.lp_Nmech'] = 5000
            self.prob[pt+'.balance.hp_Nmech'] = 15000
            self.prob[pt+'.hpt.PR'] = 3.
            self.prob[pt+'.lpt.PR'] = 4.
            self.prob[pt+'.fan.map.RlineMap'] = 2.0
            self.prob[pt+'.lpc.map.RlineMap'] = 2.0
            self.prob[pt+'.hpc.map.RlineMap'] = 2.0


    def benchmark_case1(self):
        old = np.seterr(divide='raise')

        try:
            self.prob.set_solver_print(level=-1)
            self.prob.set_solver_print(level=2, depth=1)
            self.prob.run_model()
            tol = 1e-5
            print()

            reg_data = 344.303
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

            reg_data = 3.61475
            pyc = self.prob['DESIGN.balance.hpt_PR'][0]
            print('HPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 4.3656305
            pyc = self.prob['DESIGN.balance.lpt_PR'][0]
            print('LPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14232.2
            pyc = self.prob['DESIGN.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.63071997
            pyc = self.prob['DESIGN.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1276.47613
            pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)


            print('#'*10)
            print('# OD_full_pwr')
            print('#'*10)
            reg_data = 344.303
            pyc = self.prob['OD_full_pwr.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 30.0937151
            pyc = self.prob['OD_full_pwr.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0249198
            pyc = self.prob['OD_full_pwr.balance.FAR'][0]
            print('FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14705.7
            pyc = self.prob['OD_full_pwr.balance.hp_Nmech'][0]
            print('HPT Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 4666.1
            pyc = self.prob['OD_full_pwr.balance.lp_Nmech'][0]
            print('LPT Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14232.2
            pyc = self.prob['OD_full_pwr.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.63071997
            pyc = self.prob['OD_full_pwr.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1276.48
            pyc = self.prob['OD_full_pwr.bld3.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 5.105
            pyc = self.prob['OD_full_pwr.balance.BPR'][0]
            print('BPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)


            print('#'*10)
            print('# OD_part_pwr')
            print('#'*10)
            reg_data = 324.633
            pyc = self.prob['OD_part_pwr.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 25.21321086
            pyc = self.prob['OD_part_pwr.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0224776
            pyc = self.prob['OD_part_pwr.balance.FAR'][0]
            print('FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14251.973
            pyc = self.prob['OD_part_pwr.balance.hp_Nmech'][0]
            print('HPT Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data =  4301.935
            pyc = self.prob['OD_part_pwr.balance.lp_Nmech'][0]
            print('LPT Nmech:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 12576.2
            pyc = self.prob['OD_part_pwr.perf.Fg'][0]
            print('Fg:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.61883
            pyc = self.prob['OD_part_pwr.perf.TSFC'][0]
            print('TSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1205.007
            pyc = self.prob['OD_part_pwr.bld3.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 5.614823
            pyc = self.prob['OD_part_pwr.balance.BPR'][0]
            print('BPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)
        finally:
            np.seterr(**old)

if __name__ == "__main__":
    unittest.main()