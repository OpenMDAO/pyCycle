import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.wet_propulsor import MPWetPropulsor

class WetPropulsorTestCase(unittest.TestCase): 


    def benchmark_case1(self): 

        prob = om.Problem()

        prob.model = MPWetPropulsor()

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=2)

        prob.setup(check=False)

        prob.model.design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

        prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6
        prob.model.off_design.nonlinear_solver.options['maxiter'] = 10

    		

    	# parameters
        prob['des:MN'] = .8
        prob['OD:MN'] = .8

    	# initial guess
        prob['design.balance.W'] = 200.

        prob['off_design.balance.W'] = 406.790
        prob['off_design.balance.Nmech'] = 1. # normalized value
        prob['off_design.fan.PR'] = 1.2
        prob['off_design.fan.map.RlineMap'] = 2.2

        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['design.fc.Fl_O:stat:W'], 406.56272, tol)
        assert_near_equal(prob['design.nozz.Fg'], 12066.67634, tol)
        assert_near_equal(prob['design.fan.SMN'], 36.64058, tol)
        assert_near_equal(prob['design.fan.SMW'], 29.88607, tol)


        assert_near_equal(prob['off_design.fc.Fl_O:stat:W'], 406.56272, tol)
        assert_near_equal(prob['off_design.nozz.Fg'], 12066.67634, tol)
        assert_near_equal(prob['off_design.fan.SMN'], 36.64058, tol)
        assert_near_equal(prob['off_design.fan.SMW'], 29.88607, tol)

if __name__ == "__main__":
    unittest.main()