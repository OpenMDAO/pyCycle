""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.mp_cycle import Cycle
from pycycle.elements.bleed_out import BleedOut
from pycycle.elements.flow_start import FlowStart
from pycycle import constants
from pycycle.thermo.cea import species_data


class BleedOutTestCase(unittest.TestCase):

    def test_case1(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = species_data.janaf

        cycle.add_subsystem('flow_start', FlowStart(), promotes=['MN', 'P', 'T'])
        cycle.add_subsystem('bleed', BleedOut(bleed_names=['bld1', 'bld2']), promotes=['MN'])

        cycle.pyc_connect_flow('flow_start.Fl_O', 'bleed.Fl_I')

        cycle.set_input_defaults('MN', 0.5)
        cycle.set_input_defaults('bleed.bld1:frac_W', 0.1)
        cycle.set_input_defaults('bleed.bld2:frac_W', 0.1)
        cycle.set_input_defaults('P', 17., units='psi')
        cycle.set_input_defaults('T', 500., units='degR')
        cycle.set_input_defaults('flow_start.W', 500., units='lbm/s')

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.set_solver_print(level=-1)

        self.prob.run_model()

        tol = 2.0e-5

        Tt_in = self.prob.get_val('bleed.Fl_I:tot:T', units='degR')
        Pt_in = self.prob.get_val('bleed.Fl_I:tot:P', units='psi')
        W_in = self.prob['bleed.Fl_I:stat:W']
        
        assert_near_equal(self.prob['bleed.Fl_O:tot:T'], Tt_in, tol)
        assert_near_equal(self.prob['bleed.bld1:tot:T'], Tt_in, tol)
        assert_near_equal(self.prob['bleed.bld2:tot:T'], Tt_in, tol)

        assert_near_equal(self.prob['bleed.Fl_O:tot:P'], Pt_in, tol)
        assert_near_equal(self.prob['bleed.bld1:tot:P'], Pt_in, tol)
        assert_near_equal(self.prob['bleed.bld2:tot:P'], Pt_in, tol)

        assert_near_equal(self.prob['bleed.Fl_O:stat:W'], W_in*0.8, tol)
        assert_near_equal(self.prob['bleed.bld1:stat:W'], W_in*0.1, tol)
        assert_near_equal(self.prob['bleed.bld2:stat:W'], W_in*0.1, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                includes=['bleed.*'], excludes=['*.base_thermo.*',])
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
