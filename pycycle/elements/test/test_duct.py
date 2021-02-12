""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.mp_cycle import Cycle
from pycycle.elements.duct import Duct
from pycycle.elements.flow_start import FlowStart
from pycycle import constants
from pycycle.thermo.cea import species_data


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
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = species_data.janaf

        cycle.add_subsystem('flow_start', FlowStart(), promotes=['MN', 'P', 'T'])
        cycle.add_subsystem('duct', Duct(), promotes=['MN'])

        cycle.pyc_connect_flow('flow_start.Fl_O', 'duct.Fl_I')

        cycle.set_input_defaults('MN', 0.5)
        cycle.set_input_defaults('duct.dPqP', 0.0)
        cycle.set_input_defaults('P', 17., units='psi')
        cycle.set_input_defaults('T', 500., units='degR')
        cycle.set_input_defaults('flow_start.W', 500., units='lbm/s')

        self.prob.setup(check=False, force_alloc_complex=True)
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

            partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['duct.*'], excludes=['*.base_thermo.*',])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)



    def test_case_with_dPqP_MN(self):

        self.prob = Problem()

        # need two cycles, because we can't mix design and off-design
        cycle_DES = self.prob.model.add_subsystem('DESIGN', Cycle())
        cycle_DES.options['thermo_method'] = 'CEA'
        cycle_DES.options['thermo_data'] = species_data.janaf
        
        cycle_OD = self.prob.model.add_subsystem('OFF_DESIGN', Cycle())
        cycle_OD.options['design'] = False
        cycle_OD.options['thermo_method'] = 'CEA'
        cycle_OD.options['thermo_data'] = species_data.janaf


        cycle_DES.add_subsystem('flow_start', FlowStart(), promotes=['P', 'T', 'MN', 'W'])
        
        cycle_OD.add_subsystem('flow_start_OD', FlowStart(), promotes=['P', 'T', 'W', 'MN'])

        expMN = 1.0
        cycle_DES.add_subsystem('duct', Duct(expMN=expMN), promotes=['MN'])
        cycle_OD.add_subsystem('duct', Duct(expMN=expMN, design=False))

        cycle_DES.pyc_connect_flow('flow_start.Fl_O', 'duct.Fl_I')
        cycle_OD.pyc_connect_flow('flow_start_OD.Fl_O', 'duct.Fl_I')

        cycle_DES.set_input_defaults('P', 17., units='psi')
        cycle_DES.set_input_defaults('T', 500., units='degR')
        cycle_DES.set_input_defaults('MN', 0.5)
        cycle_DES.set_input_defaults('duct.dPqP', 0.0)
        cycle_DES.set_input_defaults('W', 500., units='lbm/s')

        cycle_OD.set_input_defaults('P', 17., units='psi')
        cycle_OD.set_input_defaults('T', 500., units='degR')
        cycle_OD.set_input_defaults('MN', 0.25)
        cycle_OD.set_input_defaults('duct.dPqP', 0.0)
        cycle_OD.set_input_defaults('W', 500., units='lbm/s')

        self.prob.model.connect("DESIGN.duct.s_dPqP", "OFF_DESIGN.duct.s_dPqP")
        self.prob.model.connect("DESIGN.duct.Fl_O:stat:area", "OFF_DESIGN.duct.area")

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.set_solver_print(level=-1)


        data = ref_data[0]
        self.prob['DESIGN.duct.dPqP'] = data[h_map['dPqP']]

        # input flowstation
        self.prob['DESIGN.P'] = data[h_map['Fl_I.Pt']]
        self.prob['DESIGN.T'] = data[h_map['Fl_I.Tt']]
        self.prob['DESIGN.MN'] = data[h_map['Fl_O.MN']]
        self.prob['DESIGN.W'] = data[h_map['Fl_I.W']]
        # self.prob['DESIGN.duct.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

        # input flowstation
        self.prob['OFF_DESIGN.P'] = data[h_map['Fl_I.Pt']]
        self.prob['OFF_DESIGN.T'] = data[h_map['Fl_I.Tt']]
        self.prob['OFF_DESIGN.W'] = data[h_map['Fl_I.W']]
        # self.prob['OFF_DESIGN.duct.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

        # give a decent initial guess for Ps

        print(self.prob['DESIGN.P'], self.prob['DESIGN.T'], self.prob['DESIGN.MN'])

        self.prob.run_model()

        # check outputs
        pt, ht, ps, ts = data[h_map['Fl_O.Pt']], data[
            h_map['Fl_O.ht']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
        pt_computed = self.prob['OFF_DESIGN.duct.Fl_O:tot:P']
        ht_computed = self.prob['OFF_DESIGN.duct.Fl_O:tot:h']
        ps_computed = self.prob['OFF_DESIGN.duct.Fl_O:stat:P']
        ts_computed = self.prob['OFF_DESIGN.duct.Fl_O:stat:T']

        tol = 1.0e-4
        assert_near_equal(pt_computed, 8.84073152, tol)
        assert_near_equal(ht_computed, ht, tol)
        assert_near_equal(ps_computed, 8.26348914, tol)
        assert_near_equal(ts_computed, ts, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['duct.*'], excludes=['*.base_thermo.*',])
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
