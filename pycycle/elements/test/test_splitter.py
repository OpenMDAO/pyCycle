import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf

from pycycle.elements.splitter import Splitter
from pycycle.elements.flow_start import FlowStart


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/splitter.csv", delimiter=",", skiprows=1)

header = [
    'BPR',
    'Fl_I.W',
    'Fl_I.V',
    'Fl_I.MN',
    'Fl_I.s',
    'Fl_I.Pt',
    'Fl_I.Tt',
    'Fl_I.ht',
    'Fl_I.rhot',
    'Fl_I.gamt',
    'Fl_O1.MN',
    'Fl_O1.s',
    'Fl_O1.Pt',
    'Fl_O1.Tt',
    'Fl_O1.ht',
    'Fl_O1.rhot',
    'Fl_O1.gamt',
    'Fl_O1.Ps',
    'Fl_O1.Ts',
    'Fl_O1.hs',
    'Fl_O1.rhos',
    'Fl_O1.gams',
    'Fl_O1.W',
    'Fl_O2.MN',
    'Fl_O2.s',
    'Fl_O2.Pt',
    'Fl_O2.Tt',
    'Fl_O2.ht',
    'Fl_O2.rhot',
    'Fl_O2.gamt',
    'Fl_O2.Ps',
    'Fl_O2.Ts',
    'Fl_O2.hs',
    'Fl_O2.rhos',
    'Fl_O2.gams',
    'Fl_O2.W']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class splitterTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('splitter', Splitter())

        cycle.set_input_defaults('flow_start.P', 17., units='psi')
        cycle.set_input_defaults('flow_start.T', 500., units='degR')
        cycle.set_input_defaults('splitter.MN1', 0.5)
        cycle.set_input_defaults('splitter.MN2', 0.5)
        cycle.set_input_defaults('flow_start.W', 10., units='lbm/s')

        cycle.pyc_connect_flow('flow_start.Fl_O', 'splitter.Fl_I')

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['splitter.BPR'] = data[h_map['BPR']]

            # input flowstation
            self.prob['flow_start.P'] = data[h_map['Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['Fl_I.Tt']]
            self.prob['flow_start.W'] = data[h_map['Fl_I.W']]
            self.prob['splitter.MN1'] = data[h_map['Fl_O1.MN']]
            self.prob['splitter.MN2'] = data[h_map['Fl_O2.MN']]
            self.prob['splitter.Fl_I:stat:V'] = data[h_map['Fl_I.V']]
            self.prob.run_model()

            # check flow1 outputs
            pt1, ht1, ps1, ts1 = data[
                h_map['Fl_O2.Pt']], data[
                h_map['Fl_O1.ht']], data[
                h_map['Fl_O1.Ps']], data[
                h_map['Fl_O1.Ts']]
            pt1_computed = self.prob['splitter.Fl_O1:tot:P']
            ht1_computed = self.prob['splitter.Fl_O1:tot:h']
            ps1_computed = self.prob['splitter.Fl_O1:stat:P']
            ts1_computed = self.prob['splitter.Fl_O1:stat:T']

            tol = 1e-4
            assert_near_equal(pt1_computed, pt1, tol)
            assert_near_equal(ht1_computed, ht1, tol)
            assert_near_equal(ps1_computed, ps1, tol)
            assert_near_equal(ts1_computed, ts1, tol)

            # check flow2 outputs
            pt2, ht2, ps2, ts2 = data[
                h_map['Fl_O2.Pt']], data[
                h_map['Fl_O2.ht']], data[
                h_map['Fl_O2.Ps']], data[
                h_map['Fl_O2.Ts']]
            pt2_computed = self.prob['splitter.Fl_O2:tot:P']
            ht2_computed = self.prob['splitter.Fl_O2:tot:h']
            ps2_computed = self.prob['splitter.Fl_O2:stat:P']
            ts2_computed = self.prob['splitter.Fl_O2:stat:T']

            assert_near_equal(pt2_computed, pt2, tol)
            assert_near_equal(ht2_computed, ht2, tol)
            assert_near_equal(ps2_computed, ps2, tol)
            assert_near_equal(ts2_computed, ts2, tol)

            partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['splitter.*'], excludes=['*.base_thermo.*',])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
