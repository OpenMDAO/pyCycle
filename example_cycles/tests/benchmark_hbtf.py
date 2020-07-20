import numpy as np
import unittest
import os

import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.high_bypass_turbofan import MPhbtf


class CFM56TestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model = MPhbtf()
        self.prob.setup(check=False)


        ##Values that should be gotten rid of when set_input_defaults is fixed:
        self.prob.set_val('DESIGN.fan.PR', 1.685)
        self.prob.set_val('DESIGN.fan.eff', 0.8948)
        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.lpc.eff', 0.9243)
        self.prob.set_val('DESIGN.hpc.PR', 9.369),
        self.prob.set_val('DESIGN.hpc.eff', 0.8707),
        self.prob.set_val('DESIGN.hpt.eff', 0.8888),
        self.prob.set_val('DESIGN.lpt.eff', 0.8996),

        ##Flight conditions and initial values
        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.8)
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2857, units='degR')
        self.prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf') 
        self.prob.set_val('OD.fc.MN', 0.8)
        self.prob.set_val('OD.fc.alt', 35000.0, units='ft')
        self.prob.set_val('OD.balance.rhs:FAR', 5500.0, units='lbf')
        self.prob.set_val('OD.fc.dTs', 0.0, units='degR')
        self.prob.set_val('OD.hpc.cust:frac_W', 0.0445)

        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 316.0
        self.prob['DESIGN.balance.lpt_PR'] = 4.4
        self.prob['DESIGN.balance.hpt_PR'] = 3.6
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0    

    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 321.253
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 3.6228
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.3687
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 321.251
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14705.7
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4666.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.105
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

    def benchmark_case2(self):
        # TOC Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.8
        self.prob['OD.fc.alt'] = 35000.0
        self.prob['OD.balance.rhs:FAR'] = 5970.0
        self.prob['OD.fc.dTs'] = 0.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0422
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 327.265
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 32.415
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02616
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14952.3
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4933.4
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13889.9
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.64539
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1317.31
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.898
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

    def benchmark_case3(self):
        # RTO Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03165
        self.prob['OD.balance.W'] = 810.83
        self.prob['OD.balance.BPR'] = 5.1053
        self.prob['OD.balance.lp_Nmech'] = 4975.9
        self.prob['OD.balance.hp_Nmech'] = 16230.1
        self.prob['OD.fc.balance.Pt'] = 15.349
        self.prob['OD.fc.balance.Tt'] = 552.49
        self.prob['OD.hpt.PR'] = 3.591
        self.prob['OD.lpt.PR'] = 4.173
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.25
        self.prob['OD.fc.alt'] = 0.0
        self.prob['OD.balance.rhs:FAR'] = 22590.0
        self.prob['OD.fc.dTs'] = 27.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0177
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 825.049
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.998
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02975
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16222.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5050
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 29930.8
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.47488
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1536.94
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.243
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

    def benchmark_case4(self):
        # SLS Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03114
        self.prob['OD.balance.W'] = 771.34
        self.prob['OD.balance.BPR'] = 5.0805
        self.prob['OD.balance.lp_Nmech'] = 4912.7
        self.prob['OD.balance.hp_Nmech'] = 16106.9
        self.prob['OD.fc.balance.Pt'] = 14.696
        self.prob['OD.fc.balance.Tt'] = 545.67
        self.prob['OD.hpt.PR'] = 3.595
        self.prob['OD.lpt.PR'] = 4.147
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.00001
        self.prob['OD.fc.alt'] = 0.0
        self.prob['OD.balance.rhs:FAR'] = 27113.0
        self.prob['OD.fc.dTs'] = 27.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0185
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 786.741
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.418
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02912
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16065.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4949.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 27113.3
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.36662
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1509.41
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.282
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()