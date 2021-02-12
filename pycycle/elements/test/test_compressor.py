import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.compressor import Compressor
from pycycle.elements.flow_start import FlowStart
from pycycle import constants

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/compressor.csv",
                      delimiter=",", skiprows=1)

header = [
    'start.W',
    'start.Pt',
    'start.Tt',
    'start.Fl_O.ht',
    'start.Fl_O.s',
    'start.Fl_O.MN',
    'comp.PRdes',
    'comp.effDes',
    'comp.Fl_O.MN',
    'shaft.Nmech',
    'comp.Fl_O.Pt',
    'comp.Fl_O.Tt',
    'comp.Fl_O.ht',
    'comp.Fl_O.s',
    'comp.pwr',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams',
    'comp.effPoly']


h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class CompressorTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf

        cycle.add_subsystem('flow_start', FlowStart(thermo_data=janaf))
        cycle.add_subsystem('compressor', Compressor(design=True))

        cycle.set_input_defaults('flow_start.P', 17., units='psi')
        cycle.set_input_defaults('flow_start.T', 500., units='degR')
        cycle.set_input_defaults('compressor.MN', 0.5)
        cycle.set_input_defaults('flow_start.W', 10., units='lbm/s')
        cycle.set_input_defaults('compressor.PR', 6.)
        cycle.set_input_defaults('compressor.eff', 0.9)

        cycle.pyc_connect_flow("flow_start.Fl_O", "compressor.Fl_I")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['compressor.PR'] = data[h_map['comp.PRdes']]
            self.prob['compressor.eff'] = data[h_map['comp.effDes']]
            self.prob['compressor.MN'] = data[h_map['comp.Fl_O.MN']]

            # input flowstation
            self.prob['flow_start.P'] = data[h_map['start.Pt']]
            self.prob['flow_start.T'] = data[h_map['start.Tt']]
            self.prob['flow_start.W'] = data[h_map['start.W']]
            self.prob.run_model()

            tol = 1e-3

            npss = data[h_map['comp.Fl_O.Pt']]
            pyc = self.prob['compressor.Fl_O:tot:P'][0]
            print('Pt out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.Fl_O.Tt']]
            pyc = self.prob['compressor.Fl_O:tot:T'][0]
            print('Tt out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.Fl_O.ht']] - data[h_map['start.Fl_O.ht']]
            pyc = self.prob['compressor.Fl_O:tot:h'] - self.prob['flow_start.Fl_O:tot:h']
            print('delta h:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['start.Fl_O.s']]
            pyc = self.prob['flow_start.Fl_O:tot:S'][0]
            print('S in:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.Fl_O.s']]
            pyc = self.prob['compressor.Fl_O:tot:S'][0]
            print('S out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.pwr']]
            pyc = self.prob['compressor.power'][0]
            print('Power:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['Fl_O.Ps']]
            pyc = self.prob['compressor.Fl_O:stat:P'][0]
            print('Ps out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['Fl_O.Ts']]
            pyc = self.prob['compressor.Fl_O:stat:T'][0]
            print('Ts out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.effPoly']]
            pyc = self.prob['compressor.eff_poly'][0]
            print('effPoly:', npss, pyc)
            assert_near_equal(pyc, npss, tol)
            print("")

            partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['compressor.*'], excludes=['*.base_thermo.*',])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
