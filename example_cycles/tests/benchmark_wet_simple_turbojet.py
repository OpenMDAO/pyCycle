import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.wet_simple_turbojet import MPWetTurbojet

class WetSimpleTurbojetTestCase(unittest.TestCase): 


    def benchmark_case1(self): 

        prob = om.Problem()

        prob.model = mp_wet_turbojet = MPWetTurbojet()
        

        prob.setup()

        #Define the design point
        prob.set_val('DESIGN.comp.PR', 13.5),
        prob.set_val('DESIGN.comp.eff', 0.83),
        prob.set_val('DESIGN.turb.eff', 0.86),

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 168.453135137
        prob['DESIGN.balance.turb_PR'] = 4.46138725662
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        prob['OD1.balance.W'] = 166.073
        prob['OD1.balance.FAR'] = 0.01680
        prob['OD1.balance.Nmech'] = 8197.38
        prob['OD1.fc.balance.Pt'] = 15.703
        prob['OD1.fc.balance.Tt'] = 558.31
        prob['OD1.turb.PR'] = 4.6690

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        tol = 1e-5
        print()
        assert_near_equal(prob['DESIGN.inlet.Fl_O:stat:W'][0], 147.47225811, tol)
        assert_near_equal(prob['DESIGN.perf.OPR'][0], 13.5, tol)
        assert_near_equal(prob['DESIGN.balance.FAR'][0], 0.01757989383, tol)
        assert_near_equal(prob['DESIGN.balance.turb_PR'][0], 3.87564568, tol)
        assert_near_equal(prob['DESIGN.perf.Fg'][0], 11800.004971016797, tol)
        assert_near_equal(prob['DESIGN.perf.TSFC'][0], 0.79094647, tol)
        assert_near_equal(prob['DESIGN.comp.Fl_O:tot:T'][0], 1189.923682, tol)
        assert_near_equal(prob['OD1.inlet.Fl_O:stat:W'][0], 142.615366, tol)
        assert_near_equal(prob['OD1.perf.OPR'][0], 12.840813, tol)
        assert_near_equal(prob['OD1.balance.FAR'][0], 0.016678628, tol)
        assert_near_equal(prob['OD1.balance.Nmech'][0], 7936.357395342184, tol)
        assert_near_equal(prob['OD1.perf.Fg'][0], 11000.004883208685, tol)
        assert_near_equal(prob['OD1.perf.TSFC'][0], 0.778460327, tol)
        assert_near_equal(prob['OD1.comp.Fl_O:tot:T'][0], 1169.2669863777, tol)

        print()

if __name__ == "__main__":
    unittest.main()