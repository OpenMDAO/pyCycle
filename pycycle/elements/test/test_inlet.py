""" Tests the inlet component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf, Thermo
from pycycle.elements.inlet import Inlet
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_MIX, janaf_init_prod_amounts

from pycycle.elements.test.util import check_element_partials

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/inlet.csv", delimiter=",", skiprows=1)

header = [
    'eRamBase',
    'Fl_I.W',
    'Fl_I.V',
    'Fl_I.MN',
    'Fl_I.s',
    'Fl_I.Pt',
    'Fl_I.Tt',
    'Fl_I.ht',
    'Fl_I.rhot',
    'Fl_I.gamt',
    'Fl_O.MN',
    'Fl_O.s',
    'Fl_O.Pt',
    'Fl_O.Tt',
    'Fl_O.ht',
    'Fl_O.rhot',
    'Fl_O.gamt',
    'Fram',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class InletTestCase(unittest.TestCase):

    def setUp(self):

        thermo = Thermo(janaf, janaf_init_prod_amounts)

        self.prob = Problem()
        self.prob.model = Group()

        self.prob.model.set_input_defaults('flow_start.P', 17, units='psi')
        self.prob.model.set_input_defaults('flow_start.T', 500.0, units='degR')
        self.prob.model.set_input_defaults('inlet.MN', 0.5)
        self.prob.model.set_input_defaults('inlet.Fl_I:stat:V', 1., units='ft/s')

        # Remaining des_vars will be removed when set_input_defaults is fixed
        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('W', 1., units='lbm/s')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('inlet', Inlet(elements=AIR_MIX))

        # total and static
        fl_src = "flow_start.Fl_O"
        fl_target = "inlet.Fl_I"
        for v_name in ('h', 'T', 'P', 'S', 'rho', 'gamma', 'Cp', 'Cv', 'n'):
            self.prob.model.connect('%s:tot:%s' % (
                fl_src, v_name), '%s:tot:%s' % (fl_target, v_name))

        # no prefix
        for v_name in ('W', ):  # ('Wc', 'W', 'FAR'):
            self.prob.model.connect('%s:stat:%s' % (
                fl_src, v_name), '%s:stat:%s' % (fl_target, v_name))

        self.prob.model.connect("W", "flow_start.W")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['inlet.ram_recovery'] = data[h_map['eRamBase']]

            # input flowstation
            self.prob['flow_start.P'] = data[h_map['Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['Fl_I.Tt']]
            self.prob['inlet.MN'] = data[h_map['Fl_O.MN']]
            self.prob['W'] = data[h_map['Fl_I.W']]
            self.prob['inlet.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

            self.prob.run_model()

            # check outputs

            pt, ht, Fram, ps, ts = data[h_map['Fl_O.Pt']], data[h_map['Fl_O.ht']], data[
                h_map['Fram']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
            pt_computed = self.prob['inlet.Fl_O:tot:P']
            ht_computed = self.prob['inlet.Fl_O:tot:h']
            Fram_computed = self.prob['inlet.F_ram']
            ps_computed = self.prob['inlet.Fl_O:stat:P']
            ts_computed = self.prob['inlet.Fl_O:stat:T']

            tol = 1e-4
            assert_near_equal(pt_computed, pt, tol)
            assert_near_equal(ht_computed, ht, tol)
            assert_near_equal(Fram_computed, Fram, tol)
            assert_near_equal(ps_computed, ps, tol)
            assert_near_equal(ts_computed, ts, tol)

            check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
