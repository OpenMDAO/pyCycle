""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
from pycycle.elements.mixer import Mixer
from pycycle.elements.flow_start import FlowStart
from pycycle.connect_flow import connect_flow
from pycycle.elements.test.util import check_element_partials
from pycycle.cea.species_data import Thermo, janaf
# from pycycle.cea.thermo_data import janaf2


class MixerTestcase(unittest.TestCase):

    def test_mix_same(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        thermo = Thermo(janaf, AIR_MIX)

        p = Problem()

        p.model.set_input_defaults('P', 17., units='psi')
        p.model.set_input_defaults('T', 500., units='degR')
        p.model.set_input_defaults('MN', 0.5)
        p.model.set_input_defaults('W', 100., units='lbm/s')

        p.model.add_subsystem('start1', FlowStart(), promotes=['P', 'T', 'MN', 'W'])
        p.model.add_subsystem('start2', FlowStart(), promotes=['P', 'T', 'MN', 'W'])

        p.model.add_subsystem('mixer', Mixer(design=True, Fl_I1_elements=AIR_MIX, Fl_I2_elements=AIR_MIX))

        connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
        connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')
        p.set_solver_print(level=-1)

        p.setup()
        p['mixer.balance.P_tot'] = 17
        p.run_model()
        tol = 2e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2*p['start1.Fl_O:stat:area'], tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], p['P'], tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1, tolerance=tol)

    def test_mix_diff(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        thermo = Thermo(janaf, AIR_MIX)

        p = Problem()

        p.model.set_input_defaults('start1.P', 17., units='psi')
        p.model.set_input_defaults('start2.P', 15., units='psi')
        p.model.set_input_defaults('T', 500., units='degR')
        p.model.set_input_defaults('MN', 0.5)
        p.model.set_input_defaults('W', 100., units='lbm/s')

        p.model.add_subsystem('start1', FlowStart(), promotes=['MN', 'T', 'W'])
        p.model.add_subsystem('start2', FlowStart(), promotes=['MN', 'T', 'W'])

        p.model.add_subsystem('mixer', Mixer(design=True, Fl_I1_elements=AIR_MIX, Fl_I2_elements=AIR_MIX))

        connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
        connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')

        p.set_solver_print(level=-1)

        p.setup()
        p.run_model()
        tol = 2e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 653.2652635, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 15.94216641, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.1333333333, tolerance=tol)

    def _build_problem(self, designed_stream=1, complex=False):

            p = Problem()
            
            p.model.set_input_defaults('start1.P', 9.218, units='psi')
            p.model.set_input_defaults('start1.T', 1524.32, units='degR')
            p.model.set_input_defaults('start1.MN', 0.4463)
            p.model.set_input_defaults('start1.W', 161.49, units='lbm/s')

            p.model.set_input_defaults('start2.P', 8.68, units='psi')
            p.model.set_input_defaults('start2.T', 524., units='degR')
            p.model.set_input_defaults('start2.MN', 0.4463)
            p.model.set_input_defaults('start2.W', 158., units='lbm/s')

            p.model.add_subsystem('start1', FlowStart(elements=AIR_FUEL_MIX))
            p.model.add_subsystem('start2', FlowStart(elements=AIR_MIX))

            p.model.add_subsystem('mixer', Mixer(design=True, designed_stream=designed_stream,
                                                 Fl_I1_elements=AIR_FUEL_MIX, Fl_I2_elements=AIR_MIX))

            connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
            connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')

            p.setup(force_alloc_complex=complex)

            p.set_solver_print(level=-1)

            return p

    def test_mix_air_with_airfuel(self):

        p = self._build_problem(designed_stream=1)
        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2636.54161119, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 8.8823286, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.06198157, tolerance=tol)

        p = self._build_problem(designed_stream=2)

        p.model.mixer.impulse_converge.nonlinear_solver.options['maxiter'] = 10

        p.run_model()

    def test_mixer_partials(self):

        p = self._build_problem(designed_stream=1, complex=True)
        p.run_model()
        partials = p.check_partials(includes=['mixer.area_calc*', 'mixer.mix_flow*', 'mixer.imp_out*'], out_stream=None)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()