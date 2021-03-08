import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.mixedflow_turbofan import MPMixedFlowTurbofan

class MixedFlowTurbofanTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = prob = Problem()

        self.prob.model = mp_mixedflow = MPMixedFlowTurbofan()

        prob.setup(check=False)

        #design variables
        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft') #DV
        self.prob.set_val('DESIGN.fc.MN', 0.8) #DV
        self.prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')
        self.prob.set_val('DESIGN.balance.rhs:FAR_core', 3200, units='degR')
        self.prob.set_val('OD.balance.rhs:FAR_core', 3200, units='degR')

        #Values that should be removed when set_input_defaults is fixed
        self.prob.set_val('DESIGN.fan.PR', 3.3) #ADV
        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.hpc.PR', 4.9)
        self.prob.set_val('DESIGN.fan.eff', 0.8948)
        self.prob.set_val('DESIGN.lpc.eff', 0.9243)
        self.prob.set_val('DESIGN.hpc.eff', 0.8707)
        self.prob.set_val('DESIGN.hpt.eff', 0.8888)
        self.prob.set_val('DESIGN.lpt.eff', 0.8996)

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)


    def benchmark_case1(self):
        ''' Runs the design point and an off design point to make sure they match perfectly '''
        prob = self.prob

        # initial guesses
        self.prob['DESIGN.balance.FAR_core'] = 0.025
        self.prob['DESIGN.balance.FAR_ab'] = 0.025
        self.prob['DESIGN.balance.BPR'] = 1.0
        self.prob['DESIGN.balance.W'] = 100.
        self.prob['DESIGN.balance.lpt_PR'] = 3.5
        self.prob['DESIGN.balance.hpt_PR'] = 2.5
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0
        self.prob['DESIGN.mixer.balance.P_tot']=100

        self.prob['OD.balance.FAR_core'] = 0.031
        self.prob['OD.balance.FAR_ab'] = 0.038
        self.prob['OD.balance.BPR'] = 2.2
        self.prob['OD.balance.W'] = 60

        # really sensitive to these initial guesses
        self.prob['OD.balance.HP_Nmech'] = 15000
        self.prob['OD.balance.LP_Nmech'] = 5000

        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.mixer.balance.P_tot']= 100
        self.prob['OD.hpt.PR'] = 2.5
        self.prob['OD.lpt.PR'] = 3.5
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob.run_model()

        tol = 1e-5

        reg_data = 53.833997155
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.0311248
        pyc = self.prob['DESIGN.balance.FAR_core'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.balance.FAR_core'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data =  0.0387335612
        pyc = self.prob['DESIGN.balance.FAR_ab'][0]
        print('Main FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.balance.FAR_ab'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 2.0430265465465354
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.hpt.PR'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.098132533864145
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.lpt.PR'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 6802.79655491
        pyc = self.prob['DESIGN.mixed_nozz.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.mixed_nozz.Fg'][0]
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1287.084732
        pyc = self.prob['DESIGN.hpc.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)
        pyc = self.prob['OD.hpc.Fl_O:tot:T'][0]
        assert_near_equal(pyc, reg_data, tol)

if __name__ == "__main__":
    unittest.main()

