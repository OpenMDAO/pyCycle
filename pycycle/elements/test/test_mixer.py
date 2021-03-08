""" Tests the duct component. """

import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.api import Problem, Group

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.constants import CEA_AIR_COMPOSITION, CEA_AIR_FUEL_COMPOSITION
from pycycle.mp_cycle import Cycle
from pycycle.elements.mixer import Mixer
from pycycle.elements.flow_start import FlowStart
from pycycle.connect_flow import connect_flow
from pycycle.thermo.cea.species_data import janaf


class MixerTestcase(unittest.TestCase):

    def test_mix_same(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        p = Problem()

        cycle = p.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf

        cycle.set_input_defaults('P', 17., units='psi')
        cycle.set_input_defaults('T', 500., units='degR')
        cycle.set_input_defaults('MN', 0.5)
        cycle.set_input_defaults('W', 100., units='lbm/s')

        cycle.add_subsystem('start1', FlowStart(), promotes=['P', 'T', 'MN', 'W'])
        cycle.add_subsystem('start2', FlowStart(), promotes=['P', 'T', 'MN', 'W'])

        cycle.add_subsystem('mixer', Mixer(design=True))

        cycle.pyc_connect_flow('start1.Fl_O', 'mixer.Fl_I1')
        cycle.pyc_connect_flow('start2.Fl_O', 'mixer.Fl_I2')

        p.set_solver_print(level=-1)

        p.setup()

        p.run_model()

        tol = 1e-6
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2*p['start1.Fl_O:stat:area'], tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], p['P'], tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1, tolerance=tol)

    def test_mix_diff(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        p = Problem()
        cycle = p.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf

        cycle.set_input_defaults('start1.P', 17., units='psi')
        cycle.set_input_defaults('start2.P', 15., units='psi')
        cycle.set_input_defaults('T', 500., units='degR')
        cycle.set_input_defaults('MN', 0.5)
        cycle.set_input_defaults('W', 100., units='lbm/s')

        cycle.add_subsystem('start1', FlowStart(), promotes=['MN', 'T', 'W'])
        cycle.add_subsystem('start2', FlowStart(), promotes=['MN', 'T', 'W'])

        cycle.add_subsystem('mixer', Mixer(design=True))

        cycle.pyc_connect_flow('start1.Fl_O', 'mixer.Fl_I1')
        cycle.pyc_connect_flow('start2.Fl_O', 'mixer.Fl_I2')

        p.set_solver_print(level=-1)

        p.setup()
        p.run_model()
        tol = 1e-6
        assert_near_equal(p['mixer.Fl_O:stat:area'], 653.26524074, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 15.89206597, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.1333333333, tolerance=tol)

    def test_mix_air_with_airfuel(self):

        p = Problem()

        cycle = p.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf
        
        cycle.set_input_defaults('start1.P', 9.218, units='psi')
        cycle.set_input_defaults('start1.T', 1524.32, units='degR')
        cycle.set_input_defaults('start1.MN', 0.4463)
        cycle.set_input_defaults('start1.W', 161.49, units='lbm/s')

        cycle.set_input_defaults('start2.P', 8.68, units='psi')
        cycle.set_input_defaults('start2.T', 524., units='degR')
        cycle.set_input_defaults('start2.MN', 0.4463)
        cycle.set_input_defaults('start2.W', 158., units='lbm/s')

        cycle.add_subsystem('start1', FlowStart(composition=CEA_AIR_FUEL_COMPOSITION))
        cycle.add_subsystem('start2', FlowStart(composition=CEA_AIR_COMPOSITION))

        cycle.add_subsystem('mixer', Mixer(design=True, designed_stream=1))

        cycle.pyc_connect_flow('start1.Fl_O', 'mixer.Fl_I1')
        cycle.pyc_connect_flow('start2.Fl_O', 'mixer.Fl_I2')

        p.setup(force_alloc_complex=True)

        p.set_solver_print(level=-1)

        p.run_model()

        tol = 1e-6
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2786.86877031, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 8.87520497, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.06198157, tolerance=tol)

        partials = p.check_partials(includes=['mixer.area_calc*', 'mixer.mix_flow*', 'mixer.imp_out*'], out_stream=None, method='cs')
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()