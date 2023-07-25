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

        self.prob.model = mp_multispool = MPMultiSpool()

        self.prob.setup()

        #Define the design point
        self.prob.set_val('DESIGN.lpc.PR', 5.000),
        self.prob.set_val('DESIGN.lpc.eff', 0.8900),
        self.prob.set_val('DESIGN.hpc_axi.PR', 3.0),
        self.prob.set_val('DESIGN.hpc_axi.eff', 0.8900),
        self.prob.set_val('DESIGN.hpc_centri.PR', 2.7),
        self.prob.set_val('DESIGN.hpc_centri.eff', 0.8800),
        self.prob.set_val('DESIGN.hpt.eff', 0.89),
        self.prob.set_val('DESIGN.lpt.eff', 0.9),
        self.prob.set_val('DESIGN.pt.eff', 0.85),
        self.prob.set_val('DESIGN.fc.alt', 28000., units='ft'),
        self.prob.set_val('DESIGN.fc.MN', 0.5),
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2740.0, units='degR'),
        self.prob.set_val('DESIGN.balance.rhs:W', 1.1)

        # Set initial guesses for balances
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

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)

    def benchmark_case1(self):
        old = np.seterr(divide='raise')

        try:
            self.prob.run_model()
            tol = 1e-5
            print()

            reg_data = 10.774726815
            pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 40.419000000000004
            pyc = self.prob['DESIGN.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0213592428
            pyc = self.prob['DESIGN.balance.FAR'][0]
            print('FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 4.23253914
            pyc = self.prob['DESIGN.balance.hpt_PR'][0]
            print('HPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1.978929924
            pyc = self.prob['DESIGN.balance.lpt_PR'][0]
            print('LPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 4.919002289
            pyc = self.prob['DESIGN.balance.pt_PR'][0]
            print('PT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.37581601
            pyc = self.prob['DESIGN.nozzle.Fl_O:stat:MN'][0]
            print('Nozz MN:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.313564855
            pyc = self.prob['DESIGN.perf.PSFC'][0]
            print('PSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1377.8012797
            pyc = self.prob['DESIGN.duct6.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print('#'*10)
            print('# OD')
            print('#'*10)
            reg_data = 10.23511401
            pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
            print('W:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 37.71143287
            pyc = self.prob['OD.perf.OPR'][0]
            print('OPR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.0202391534
            pyc = self.prob['OD.balance.FAR'][0]
            print('FAR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 10.235114018
            pyc = self.prob['OD.balance.W'][0]
            print('HPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 11557.417714
            pyc = self.prob['OD.balance.IP_Nmech'][0]
            print('LPT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 14620.2646120
            pyc = self.prob['OD.balance.HP_Nmech'][0]
            print('PT PR:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.3525941
            pyc = self.prob['OD.nozzle.Fl_O:stat:MN'][0]
            print('Nozz MN:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 0.3175212620
            pyc = self.prob['OD.perf.PSFC'][0]
            print('PSFC:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            reg_data = 1345.9668282
            pyc = self.prob['OD.duct6.Fl_O:tot:T'][0]
            print('Tt3:', pyc, reg_data)
            assert_near_equal(pyc, reg_data, tol)

            print()
        finally:
            np.seterr(**old)

if __name__ == "__main__":
    unittest.main()
