""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.duct import Duct
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_MIX
from pycycle import constants

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
                                                              elements=AIR_MIX), promotes=['MN', 'P', 'T'])
        self.prob.model.add_subsystem('duct', Duct(elements=AIR_MIX), promotes=['MN'])

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'duct.Fl_I')

        self.prob.model.set_input_defaults('MN', 0.5)
        self.prob.model.set_input_defaults('duct.dPqP', 0.0)
        self.prob.model.set_input_defaults('P', 17., units='psi')
        self.prob.model.set_input_defaults('T', 500., units='degR')
        self.prob.model.set_input_defaults('flow_start.W', 500., units='lbm/s')

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

        # 6 cases to check against
        for i, data in enumerate(ref_data):

            self.prob['duct.dPqP'] = data[h_map['dPqP']]

            # input flowstation
            self.prob['P'] = data[h_map['Fl_I.Pt']]
            self.prob['T'] = data[h_map['Fl_I.Tt']]
            self.prob['MN'] = data[h_map['Fl_O.MN']]
            self.prob['flow_start.W'] = data[h_map['Fl_I.W']]
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
            assert_near_equal(pt_computed, pt, tol)
            assert_near_equal(ht_computed, ht, tol)
            assert_near_equal(ps_computed, ps, tol)
            assert_near_equal(ts_computed, ts, tol)

            check_element_partials(self, self.prob)


    def test_case_with_dPqP_MN(self):

        self.prob = Problem()
        self.prob.model = Group()
        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX), promotes=['P', 'T', 'MN', 'W'])
        self.prob.model.add_subsystem('flow_start_OD', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX), promotes=['P', 'T', 'W'])

        expMN = 1.0
        self.prob.model.add_subsystem('duct_des', Duct(elements=AIR_MIX, expMN=expMN), promotes=['MN'])
        self.prob.model.add_subsystem('duct_OD', Duct(elements=AIR_MIX, expMN=expMN, design=False))

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'duct_des.Fl_I')
        connect_flow(self.prob.model, 'flow_start_OD.Fl_O', 'duct_OD.Fl_I')

        self.prob.model.set_input_defaults('P', 17., units='psi')
        self.prob.model.set_input_defaults('T', 500., units='degR')
        self.prob.model.set_input_defaults('MN', 0.5)
        self.prob.model.set_input_defaults('flow_start_OD.MN', 0.25)
        self.prob.model.set_input_defaults('duct_des.dPqP', 0.0)
        self.prob.model.set_input_defaults('W', 500., units='lbm/s')

        self.prob.model.connect("duct_des.s_dPqP", "duct_OD.s_dPqP")
        self.prob.model.connect("duct_des.Fl_O:stat:area", "duct_OD.area")

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)


        data = ref_data[0]
        self.prob['duct_des.dPqP'] = data[h_map['dPqP']]

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
        assert_near_equal(pt_computed, 8.84073152, tol)
        assert_near_equal(ht_computed, ht, tol)
        assert_near_equal(ps_computed, 8.26348914, tol)
        assert_near_equal(ts_computed, ts, tol)

        check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
