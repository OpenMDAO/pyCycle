import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.electric_propulsor import MPpropulsor


class ElectricPropulsorTestCase(unittest.TestCase): 


    def benchmark_case1(self): 

        prob = om.Problem()

        prob.model = mp_propulsor = MPpropulsor()

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=2)

        prob.setup()

        #Define the design point
        prob.set_val('design.fc.alt', 10000, units='m')
        prob.set_val('design.fc.MN', 0.8)
        prob.set_val('design.inlet.MN', 0.6)
        prob.set_val('design.fan.PR', 1.2)
        prob.set_val('pwr_target', -3486.657, units='hp')
        prob.set_val('design.fan.eff', 0.96)
        prob.set_val('off_design.fc.alt', 12000, units='m')

        # Set initial guesses for balances
        prob['design.balance.W'] = 200.
        
        # initial guesses
        prob['off_design.fan.PR'] = 1.2
        prob['off_design.balance.W'] = 406.790
        prob['off_design.balance.Nmech'] = 1. # normalized value

        
        prob.model.design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

        prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['maxiter'] = 10

        self.prob = prob


        prob.run_model()

        tol = 3e-5
        assert_near_equal(prob['design.fc.Fl_O:stat:W'], 409.636, tol)
        assert_near_equal(prob['design.nozz.Fg'], 12139.282, tol)
        assert_near_equal(prob['design.fan.SMN'], 36.64057531, tol)
        assert_near_equal(prob['design.fan.SMW'], 29.886, tol)


        assert_near_equal(prob['off_design.fc.Fl_O:stat:W'], 317.36096893 , tol)
        assert_near_equal(prob['off_design.nozz.Fg'], 9696.45125337, tol)
        assert_near_equal(prob['off_design.fan.SMN'], 22.98592129, tol)
        assert_near_equal(prob['off_design.fan.SMW'], 19.53411898, 5e-4) # this one is a little noisy

if __name__ == "__main__":
    unittest.main()