""" Tests the duct component. """

from __future__ import print_function

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.species_data import janaf
from pycycle.elements.duct import Duct
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_MIX

from pycycle.elements.test.util import check_element_partials
from pycycle.connect_flow import connect_flow


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/duct.csv", delimiter=",", skiprows=1)

header = [
    'dPqP',
    'Qin',
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
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))

np.seterr(all="raise")


class DuctTestCase(unittest.TestCase):

    def test_case1(self):

        self.prob = Problem()
        self.prob.model = Group()
        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX))
        self.prob.model.add_subsystem('duct', Duct(elements=AIR_MIX))

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'duct.Fl_I')

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 500., units='lbm/s')
        des_vars.add_output('MN', 0.5)
        des_vars.add_output('dPqP_des', 0.0)

        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect("MN", ["duct.MN", "flow_start.MN"])
        self.prob.model.connect("dPqP_des", "duct.dPqP")

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

        # 6 cases to check against
        for i, data in enumerate(ref_data):

            self.prob['dPqP_des'] = data[h_map['dPqP']]

            # input flowstation
            self.prob['P'] = data[h_map['Fl_I.Pt']]
            self.prob['T'] = data[h_map['Fl_I.Tt']]
            self.prob['MN'] = data[h_map['Fl_O.MN']]
            self.prob['W'] = data[h_map['Fl_I.W']]
            self.prob['duct.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

            # give a decent initial guess for Ps

            print(i, self.prob['P'], self.prob['T'], self.prob['MN'])

            self.prob.run_model()

            # check outputs
            pt, ht, ps, ts = data[h_map['Fl_O.Pt']], data[
                h_map['Fl_O.ht']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
            pt_computed = self.prob['duct.Fl_O:tot:P']
            ht_computed = self.prob['duct.Fl_O:tot:h']
            ps_computed = self.prob['duct.Fl_O:stat:P']
            ts_computed = self.prob['duct.Fl_O:stat:T']

            tol = 2.0e-2
            assert_rel_error(self, pt_computed, pt, tol)
            assert_rel_error(self, ht_computed, ht, tol)
            assert_rel_error(self, ps_computed, ps, tol)
            assert_rel_error(self, ts_computed, ts, tol)

            check_element_partials(self, self.prob)


    def test_case_with_dPqP_MN(self):

        self.prob = Problem()
        self.prob.model = Group()
        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX))
        self.prob.model.add_subsystem('flow_start_OD', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX))

        expMN = 1.0
        self.prob.model.add_subsystem('duct_des', Duct(elements=AIR_MIX, expMN=expMN))
        self.prob.model.add_subsystem('duct_OD', Duct(elements=AIR_MIX, expMN=expMN, design=False))

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'duct_des.Fl_I')
        connect_flow(self.prob.model, 'flow_start_OD.Fl_O', 'duct_OD.Fl_I')

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 500., units='lbm/s')
        des_vars.add_output('MN', 0.5)
        des_vars.add_output('MN_OD', 0.25)
        des_vars.add_output('dPqP_des', 0.0)

        self.prob.model.connect("P", ["flow_start.P", 'flow_start_OD.P'])
        self.prob.model.connect("T", ["flow_start.T", 'flow_start_OD.T'])
        self.prob.model.connect("W", ["flow_start.W", 'flow_start_OD.W'])
        self.prob.model.connect("MN", ["duct_des.MN", "flow_start.MN"])
        self.prob.model.connect("MN_OD", "flow_start_OD.MN")
        self.prob.model.connect("duct_des.s_dPqP", "duct_OD.s_dPqP")
        self.prob.model.connect("duct_des.Fl_O:stat:area", "duct_OD.area")
        self.prob.model.connect("dPqP_des", "duct_des.dPqP")


        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

    
        data = ref_data[0]
        self.prob['dPqP_des'] = data[h_map['dPqP']]

        # input flowstation
        self.prob['P'] = data[h_map['Fl_I.Pt']]
        self.prob['T'] = data[h_map['Fl_I.Tt']]
        self.prob['MN'] = data[h_map['Fl_O.MN']]
        self.prob['W'] = data[h_map['Fl_I.W']]
        self.prob['duct_des.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

        # give a decent initial guess for Ps

        print(self.prob['P'], self.prob['T'], self.prob['MN'])

        self.prob.run_model()

        # check outputs
        pt, ht, ps, ts = data[h_map['Fl_O.Pt']], data[
            h_map['Fl_O.ht']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
        pt_computed = self.prob['duct_OD.Fl_O:tot:P']
        ht_computed = self.prob['duct_OD.Fl_O:tot:h']
        ps_computed = self.prob['duct_OD.Fl_O:stat:P']
        ts_computed = self.prob['duct_OD.Fl_O:stat:T']

        tol = 1.0e-4
        assert_rel_error(self, pt_computed, 8.84073152, tol)
        assert_rel_error(self, ht_computed, ht, tol)
        assert_rel_error(self, ps_computed, 8.26348914, tol)
        assert_rel_error(self, ts_computed, ts, tol)

        check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
