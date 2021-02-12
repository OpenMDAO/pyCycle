import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from pycycle.mp_cycle import Cycle
from pycycle.connect_flow import connect_flow

from pycycle.elements.compressor import Compressor
from pycycle.elements.flow_start import FlowStart
from pycycle.maps.axi5 import AXI5
from pycycle import constants
from pycycle.thermo.cea import species_data


class CompressorODTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['design'] = False
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = species_data.janaf

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('compressor', Compressor(
                map_data=AXI5, design=False, map_extrap=False))

        cycle.set_input_defaults('compressor.s_PR', val=1.)
        cycle.set_input_defaults('compressor.s_eff', val=1.)
        cycle.set_input_defaults('compressor.s_Wc', val=1.)
        cycle.set_input_defaults('compressor.s_Nc', val=1.)
        cycle.set_input_defaults('compressor.map.alphaMap', val=0.)
        cycle.set_input_defaults('compressor.Nmech', 0., units='rpm')
        cycle.set_input_defaults('flow_start.P', 17., units='psi')
        cycle.set_input_defaults('flow_start.T', 500., units='degR')
        cycle.set_input_defaults('flow_start.W', 0., units='lbm/s')
        cycle.set_input_defaults('compressor.area', 50., units='inch**2')

        cycle.pyc_connect_flow("flow_start.Fl_O", "compressor.Fl_I")

        newton = self.prob.model.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.prob.model.linear_solver = DirectSolver(assemble_jac=True)


        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        np.seterr(divide='raise')

        fpath = os.path.dirname(os.path.realpath(__file__))
        ref_data = np.loadtxt(fpath + "/reg_data/compressorOD1.csv",
                              delimiter=",", skiprows=1)

        header = [
            'comp.PRdes',
            'comp.effDes',
            'shaft.Nmech',
            'comp.Fl_I.W',
            'comp.Fl_I.Pt',
            'comp.Fl_I.Tt',
            'comp.Fl_I.ht',
            'comp.Fl_I.s',
            'comp.Fl_I.MN',
            'comp.Fl_I.V',
            'comp.Fl_I.A',
            'comp.Fl_I.Ps',
            'comp.Fl_I.Ts',
            'comp.Fl_I.hs',
            'comp.Fl_O.W',
            'comp.Fl_O.Pt',
            'comp.Fl_O.Tt',
            'comp.Fl_O.ht',
            'comp.Fl_O.s',
            'comp.Fl_O.MN',
            'comp.Fl_O.V',
            'comp.Fl_O.A',
            'comp.Fl_O.Ps',
            'comp.Fl_O.Ts',
            'comp.Fl_O.hs',
            'comp.PR',
            'comp.eff',
            'comp.Nc',
            'comp.Wc',
            'comp.pwr',
            'comp.RlineMap',
            'comp.PRmap',
            'comp.effMap',
            'comp.NcMap',
            'comp.WcMap',
            'comp.s_WcDes',
            'comp.s_PRdes',
            'comp.s_effDes',
            'comp.s_NcDes',
            'comp.SMW',
            'comp.SMN']

        h_map = dict(((v_name, i) for i, v_name in enumerate(header)))
        # 6 cases to check against
        for i, data in enumerate(ref_data):
                self.prob['compressor.s_PR'] = data[h_map['comp.s_PRdes']]
                self.prob['compressor.s_Wc'] = data[h_map['comp.s_WcDes']]
                self.prob['compressor.s_eff'] = data[h_map['comp.s_effDes']]
                self.prob['compressor.s_Nc'] = data[h_map['comp.s_NcDes']]
                self.prob['compressor.map.RlineMap'] = data[h_map['comp.RlineMap']]
                self.prob['compressor.Nmech'] = data[h_map['shaft.Nmech']]

                # input flowstation
                self.prob['flow_start.P'] = data[h_map['comp.Fl_I.Pt']]
                self.prob['flow_start.T'] = data[h_map['comp.Fl_I.Tt']]
                self.prob['flow_start.W'] = data[h_map['comp.Fl_I.W']]
                self.prob['compressor.area'] = data[h_map['comp.Fl_O.A']]

                self.prob.run_model()
                tol = 3.0e-3  # seems a little generous,
                # FL_O.Ps is off by 4% or less, everything else is <1% tol

                print('----- Test Case', i, '-----')
                npss = data[h_map['comp.Fl_I.Pt']]
                pyc = self.prob['flow_start.Fl_O:tot:P'][0]
                print('Pt in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.s']]
                pyc = self.prob['flow_start.Fl_O:tot:S'][0]
                print('S in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.W']]
                pyc = self.prob['compressor.Fl_O:stat:W'][0]
                print('W in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.s']]
                pyc = self.prob['flow_start.Fl_O:tot:S'][0]
                print('S in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.RlineMap']]
                pyc = self.prob['compressor.map.RlineMap'][0]
                print('RlineMap:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.PR']]
                pyc = self.prob['compressor.PR'][0]
                print('PR:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.eff']]
                pyc = self.prob['compressor.eff'][0]
                print('eff:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.ht']] - data[h_map['comp.Fl_I.ht']]
                pyc = self.prob['compressor.Fl_O:tot:h'][0] - self.prob['flow_start.Fl_O:tot:h'][0]
                print('delta h:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.s']]
                pyc = self.prob['compressor.Fl_O:tot:S'][0]
                print('S out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.pwr']]
                pyc = self.prob['compressor.power'][0]
                print('Power:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.Ps']]
                pyc = self.prob['compressor.Fl_O:stat:P'][0]
                print('Ps out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.Ts']]
                pyc = self.prob['compressor.Fl_O:stat:T'][0]
                print('Ts out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.SMW']]
                pyc = self.prob['compressor.SMW'][0]
                print('SMW:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.SMN']]
                pyc = self.prob['compressor.SMN'][0]
                print('SMN:', npss, pyc)
                assert_near_equal(pyc, npss, tol)
                print()

                partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['compressor.*'], excludes=['*.base_thermo.*',])
                assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)
if __name__ == "__main__":
    unittest.main()
