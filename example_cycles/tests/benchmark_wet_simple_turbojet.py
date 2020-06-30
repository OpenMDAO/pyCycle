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

        prob.model = MPWetTurbojet()

        pts = ['OD1']

        prob.setup(check=False)

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

        tol = 1e-3
        print()
        assert_near_equal(prob['DESIGN.inlet.Fl_O:stat:W'][0], 146.47849316571225, tol)
        assert_near_equal(prob['DESIGN.perf.OPR'][0], 13.500, tol)
        assert_near_equal(prob['DESIGN.balance.FAR'][0], 0.017841354941527617, tol)
        assert_near_equal(prob['DESIGN.balance.turb_PR'][0], 3.8594130386640213, tol)
        assert_near_equal(prob['DESIGN.perf.Fg'][0], 11800.00494559822, tol)
        assert_near_equal(prob['DESIGN.perf.TSFC'][0], 0.7973007930679725, tol)
        assert_near_equal(prob['DESIGN.comp.Fl_O:tot:T'][0], 1186.2347155867712, tol)
        assert_near_equal(prob['OD1.inlet.Fl_O:stat:W'][0], 141.6490041824415, tol)
        assert_near_equal(prob['OD1.perf.OPR'][0], 12.840225396836292, tol)
        assert_near_equal(prob['OD1.balance.FAR'][0], 0.016926951873953676, tol)
        assert_near_equal(prob['OD1.balance.Nmech'][0], 7936.216037878517, tol)
        assert_near_equal(prob['OD1.perf.Fg'][0], 11000.004855898664, tol)
        assert_near_equal(prob['OD1.perf.TSFC'][0], 0.7846972013516105, tol)
        assert_near_equal(prob['OD1.comp.Fl_O:tot:T'][0], 1165.6950986328175, tol)

        print()

if __name__ == "__main__":
    unittest.main()