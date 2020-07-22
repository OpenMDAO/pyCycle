import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
import pycycle.api as pyc
from openmdao.utils.units import convert_units as cu
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.multi_spool_turboshaft import MPMultiSpool


class MultiSpoolTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        self.prob.model = MPMultiSpool()

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        ##Values will go away once set_input_defaults is fixed
        self.prob.set_val('DESIGN.lpc.PR', 5.000),
        self.prob.set_val('DESIGN.lpc.eff', 0.8900),
        self.prob.set_val('DESIGN.hpc_axi.PR', 3.0),
        self.prob.set_val('DESIGN.hpc_axi.eff', 0.8900),
        self.prob.set_val('DESIGN.hpc_centri.PR', 2.7),
        self.prob.set_val('DESIGN.hpc_centri.eff', 0.8800),
        self.prob.set_val('DESIGN.hpt.eff', 0.89),
        self.prob.set_val('DESIGN.lpt.eff', 0.9),
        self.prob.set_val('DESIGN.pt.eff', 0.85),

        ##Initial conditions
        self.prob.set_val('DESIGN.fc.alt', 28000., units='ft'),
        self.prob.set_val('DESIGN.fc.MN', 0.5),
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2740.0, units='degR'),
        self.prob.set_val('DESIGN.balance.rhs:W', 1.1)
        self.prob.set_val('OD.balance.rhs:FAR', 1600.0, units='hp')
        self.prob.set_val('OD.LP_Nmech', 12750.0, units='rpm')
        self.prob.set_val('OD.fc.alt', 28000, units='ft')
        self.prob.set_val('OD.fc.MN', .5)

        ##Initial guesses
        self.prob['DESIGN.balance.FAR'] = 0.02261
        self.prob['DESIGN.balance.W'] = 10.76
        self.prob['DESIGN.balance.pt_PR'] = 4.939
        self.prob['DESIGN.balance.lpt_PR'] = 1.979
        self.prob['DESIGN.balance.hpt_PR'] = 4.236
        self.prob['DESIGN.fc.balance.Pt'] = 5.666
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

        self.prob['OD.balance.FAR'] = 0.02135
        self.prob['OD.balance.W'] = 10.775
        self.prob['OD.balance.HP_Nmech'] = 14800.000
        self.prob['OD.balance.IP_Nmech'] = 12000.000
        self.prob['OD.hpt.PR'] = 4.233
        self.prob['OD.lpt.PR'] = 1.979
        self.prob['OD.pt.PR'] = 4.919
        self.prob['OD.fc.balance.Pt'] = 5.666
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.nozzle.PR'] = 1.1

    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 10.774
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 40.419
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02135
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.2325
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1.9782
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.921
        pyc = self.prob['DESIGN.balance.pt_PR'][0]
        print('PT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.3758
        pyc = self.prob['DESIGN.nozzle.Fl_O:stat:MN'][0]
        print('Nozz MN:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.31342
        pyc = self.prob['DESIGN.perf.PSFC'][0]
        print('PSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1377.8
        pyc = self.prob['DESIGN.duct6.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 10.235
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 37.711
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.020230
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 10.235
        pyc = self.prob['OD.balance.W'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 11557.
        pyc = self.prob['OD.balance.IP_Nmech'][0]
        print('LPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14620.
        pyc = self.prob['OD.balance.HP_Nmech'][0]
        print('PT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.35259
        pyc = self.prob['OD.nozzle.Fl_O:stat:MN'][0]
        print('Nozz MN:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.31738
        pyc = self.prob['OD.perf.PSFC'][0]
        print('PSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1346.0
        pyc = self.prob['OD.duct6.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
