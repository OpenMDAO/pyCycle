""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
from pycycle.elements.mixer import Mixer
from pycycle.elements.flow_start import FlowStart
from pycycle.connect_flow import connect_flow
from pycycle.elements.test.util import check_element_partials
from pycycle.cea.species_data import Thermo, janaf


class MixerTestcase(unittest.TestCase):

    def test_mix_same(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        thermo = Thermo(janaf)

        p = Problem()

        des_vars = p.model.add_subsystem('des_vars', IndepVarComp())
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 100., units='lbm/s')
        des_vars.add_output('MN', 0.5)

        p.model.add_subsystem('start1', FlowStart())
        p.model.add_subsystem('start2', FlowStart())

        p.model.connect('des_vars.P', ['start1.P', 'start2.P'])
        p.model.connect('des_vars.T', ['start1.T', 'start2.T'])
        p.model.connect('des_vars.W', ['start1.W', 'start2.W'])
        p.model.connect('des_vars.MN', ['start1.MN', 'start2.MN'])

        p.model.add_subsystem('mixer', Mixer(design=True, Fl_I1_elements=AIR_MIX, Fl_I2_elements=AIR_MIX))

        connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
        connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')
        p.set_solver_print(level=-1)

        p.model.mixer.set_input_defaults('Fl_I1:stat:b0', thermo.b0)

        p.setup()
        p['mixer.balance.P_tot'] = 17
        p.run_model()
        tol = 2e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2*p['start1.Fl_O:stat:area'], tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], p['des_vars.P'], tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1, tolerance=tol)

    def test_mix_diff(self):
        # mix two identical streams and make sure you get twice the area and the same total pressure

        thermo = Thermo(janaf)

        p = Problem()

        des_vars = p.model.add_subsystem('des_vars', IndepVarComp())
        des_vars.add_output('P1', 17., units='psi')
        des_vars.add_output('P2', 15., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 100., units='lbm/s')
        des_vars.add_output('MN', 0.5)

        p.model.add_subsystem('start1', FlowStart())
        p.model.add_subsystem('start2', FlowStart())

        p.model.connect('des_vars.P1', 'start1.P')
        p.model.connect('des_vars.P2', 'start2.P')
        p.model.connect('des_vars.T', ['start1.T', 'start2.T'])
        p.model.connect('des_vars.W', ['start1.W', 'start2.W'])
        p.model.connect('des_vars.MN', ['start1.MN', 'start2.MN'])

        p.model.add_subsystem('mixer', Mixer(design=True, Fl_I1_elements=AIR_MIX, Fl_I2_elements=AIR_MIX))

        connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
        connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')

        p.set_solver_print(level=-1)

        p.model.mixer.set_input_defaults('Fl_I1:stat:b0', thermo.b0)

        p.setup()
        p.run_model()
        tol = 2e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 653.2652635, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 15.94216641, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.1333333333, tolerance=tol)

    def _build_problem(self, designed_stream=1, complex=False):

            p = Problem()

            des_vars = p.model.add_subsystem('des_vars', IndepVarComp())
            des_vars.add_output('P1', 9.218, units='psi')
            des_vars.add_output('T1', 1524.32, units='degR')
            des_vars.add_output('W1', 161.49, units='lbm/s')
            des_vars.add_output('MN1', 0.4463)

            des_vars.add_output('P2', 8.68, units='psi')
            des_vars.add_output('T2', 524., units='degR')
            des_vars.add_output('W2', 158., units='lbm/s')
            des_vars.add_output('MN2', 0.4463)

            p.model.add_subsystem('start1', FlowStart(elements=AIR_FUEL_MIX))
            p.model.add_subsystem('start2', FlowStart(elements=AIR_MIX))

            p.model.connect('des_vars.P1', 'start1.P' )
            p.model.connect('des_vars.T1', 'start1.T' )
            p.model.connect('des_vars.W1',  'start1.W' )
            p.model.connect('des_vars.MN1', 'start1.MN' )


            p.model.connect('des_vars.P2', 'start2.P' )
            p.model.connect('des_vars.T2', 'start2.T' )
            p.model.connect('des_vars.W2',  'start2.W' )
            p.model.connect('des_vars.MN2', 'start2.MN' )

            p.model.add_subsystem('mixer', Mixer(design=True, designed_stream=designed_stream,
                                                 Fl_I1_elements=AIR_FUEL_MIX, Fl_I2_elements=AIR_MIX))

            connect_flow(p.model, 'start1.Fl_O', 'mixer.Fl_I1')
            connect_flow(p.model, 'start2.Fl_O', 'mixer.Fl_I2')

            if designed_stream == 1:
                thermo = Thermo(janaf, AIR_FUEL_MIX)
                p.model.mixer.set_input_defaults('Fl_I1:stat:b0', thermo.b0)
            else:
                thermo = Thermo(janaf, AIR_MIX)
                p.model.mixer.set_input_defaults('Fl_I2:tot:b0', thermo.b0)

            p.setup(force_alloc_complex=complex)

            p.set_solver_print(level=-1)

            return p

    def test_mix_air_with_airfuel(self):

        p = self._build_problem(designed_stream=1)
        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mixer.Fl_O:stat:area'], 2636.58258193, tolerance=tol)
        assert_near_equal(p['mixer.Fl_O:tot:P'], 8.88271201, tolerance=tol)
        assert_near_equal(p['mixer.ER'], 1.06198157, tolerance=tol)

        p = self._build_problem(designed_stream=2)

        p.model.mixer.impulse_converge.nonlinear_solver.options['maxiter'] = 10

        p.run_model()

        # assert_near_equal(p['mixer.Fl_O:stat:area'], 3290.1586448, tolerance=tol)
        # assert_near_equal(p['mixer.Fl_O:tot:P'], 8.91898798, tolerance=tol)
        # assert_near_equal(p['mixer.ER'], 1.06198157, tolerance=tol)

    def test_mixer_partials(self):

        p = self._build_problem(designed_stream=1, complex=True)
        p.run_model()
        partials = p.check_partials(includes=['mixer.area_calc*', 'mixer.mix_flow*', 'mixer.imp_out*'], out_stream=None)
        # print(partials)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()