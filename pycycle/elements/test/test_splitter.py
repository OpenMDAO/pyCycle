import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.constants import AIR_MIX
from pycycle.cea.species_data import janaf

from pycycle.elements.splitter import Splitter
from pycycle.elements.flow_start import FlowStart

from pycycle.elements.test.util import check_element_partials


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

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 10., units='lbm/s')
        des_vars.add_output('MN1', 0.5)
        des_vars.add_output('MN2', 0.5)
        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect('MN1', 'splitter.MN1')
        self.prob.model.connect('MN2', 'splitter.MN2')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('splitter', Splitter(elements=AIR_MIX))

        #total and static
        fl_src = "flow_start.Fl_O"
        fl_target = "splitter.Fl_I"
        for v_name in ('h', 'T', 'P', 'S', 'rho', 'gamma', 'Cp', 'Cv', 'n'):
            self.prob.model.connect(
                '%s:tot:%s' %
                (fl_src, v_name), '%s:tot:%s' %
                (fl_target, v_name))
        # no prefix
        for v_name in ('W', ):  # ('Wc', 'W', 'FAR'):
            self.prob.model.connect(
                '%s:stat:%s' %
                (fl_src, v_name), '%s:stat:%s' %
                (fl_target, v_name))

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['splitter.BPR'] = data[h_map['BPR']]

            # input flowstation
            self.prob['P'] = data[h_map['Fl_I.Pt']]
            self.prob['T'] = data[h_map['Fl_I.Tt']]
            self.prob['W'] = data[h_map['Fl_I.W']]
            self.prob['MN1'] = data[h_map['Fl_O1.MN']]
            self.prob['MN2'] = data[h_map['Fl_O2.MN']]
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
            assert_rel_error(self, pt1_computed, pt1, tol)
            assert_rel_error(self, ht1_computed, ht1, tol)
            assert_rel_error(self, ps1_computed, ps1, tol)
            assert_rel_error(self, ts1_computed, ts1, tol)

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

            assert_rel_error(self, pt2_computed, pt2, tol)
            assert_rel_error(self, ht2_computed, ht2, tol)
            assert_rel_error(self, ps2_computed, ps2, tol)
            assert_rel_error(self, ts2_computed, ts2, tol)

            check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
