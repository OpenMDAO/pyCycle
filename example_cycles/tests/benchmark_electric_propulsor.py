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

        prob.model = MPpropulsor()

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=2)

        prob.setup(check=False)
        prob.final_setup()

        prob.set_val('design.fc.alt', 10000, units='m')
        prob.set_val('off_design.fc.alt', 12000, units='m') #12000

        prob['design.fc.MN'] = 0.8
        prob['design.inlet.MN'] = 0.6
        prob['design.fan.PR'] = 1.2
        prob['pwr_target'] = -3486.657
        prob['design.fan.eff'] = 0.96

        prob['off_design.fc.MN'] = 0.8


        prob.model.design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

        prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['maxiter'] = 10

        ########################
        # initial guesses
        ########################
        
        prob['design.balance.W'] = 200.

        prob['off_design.balance.W'] = 406.790
        prob['off_design.balance.Nmech'] = 1. # normalized value
        prob['off_design.fan.PR'] = 1.2
        prob['off_design.fan.map.RlineMap'] = 2.2

        
        self.prob = prob


        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['design.fc.Fl_O:stat:W'], 406.790, tol)
        assert_near_equal(prob['design.nozz.Fg'], 12070.380, tol)
        assert_near_equal(prob['design.fan.SMN'], 36.64057531, tol)
        assert_near_equal(prob['design.fan.SMW'], 29.886, tol)


        assert_near_equal(prob['off_design.fc.Fl_O:stat:W'], 315.344 , tol)
        assert_near_equal(prob['off_design.nozz.Fg'], 9653.170, tol)
        assert_near_equal(prob['off_design.fan.SMN'], 22.13770023, tol)
        assert_near_equal(prob['off_design.fan.SMW'], 18.95649282, tol)

if __name__ == "__main__":
    unittest.main()