""" Tests the turbine component. """

import os
import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf, Thermo
from pycycle.connect_flow import connect_flow
from pycycle.constants import AIR_MIX, janaf_init_prod_amounts
from pycycle.elements.turbine import Turbine
from pycycle.elements.flow_start import FlowStart
from pycycle.maps.lpt2269 import LPT2269

from pycycle.elements.test.util import check_element_partials

# AIR_MIX = {'O':1, 'C':1, 'N':1, 'Ar':1}

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/turbine.csv", delimiter=",", skiprows=1)

header = [
    'Fl_I.W',
    'Fl_I.Pt',
    'Fl_I.Tt',
    'Fl_I.ht',
    'Fl_I.s',
    'Fl_I.MN',
    'Fl_O.Pt',
    'Fl_O.Tt',
    'Fl_O.ht',
    'Fl_O.s',
    'Fl_O.MN',
    'PRdes',
    'EffDes',
    'Power',
    'Nmech',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams',
    'effPoly']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TurbineTestCase(unittest.TestCase):

    def setUp(self):

        thermo = Thermo(janaf, janaf_init_prod_amounts)
        self.prob = Problem()
        self.prob.model = Group()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17.0, units='psi')
        des_vars.add_output('T', 500.0, units='degR')
        des_vars.add_output('W', 100.0, units='lbm/s')
        des_vars.add_output('PR', 4.0)
        des_vars.add_output('MN', 0.0)
        des_vars.add_output('Nmech', 1000.0, units='rpm')
        des_vars.add_output('flow_start_MN', 0.0)
        des_vars.add_output('eff', 0.9)


        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('turbine', Turbine(map_data=LPT2269, design=True,
                                                         elements=AIR_MIX))

        connect_flow(self.prob.model, 'flow_start.Fl_O', 'turbine.Fl_I')

        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect("flow_start_MN", "flow_start.MN")

        self.prob.model.connect("PR", "turbine.PR")
        self.prob.model.connect("MN", "turbine.MN")
        self.prob.model.connect("Nmech", "turbine.Nmech")
        self.prob.model.connect('eff','turbine.eff')

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):

            print()
            print('---- Test Case,', i, '----')

            self.prob['PR'] = data[h_map['PRdes']]
            self.prob['Nmech'] = data[h_map['Nmech']]
            self.prob['eff'] = data[h_map['EffDes']]
            self.prob['MN'] = data[h_map['Fl_O.MN']]

            # input flowstation
            self.prob['P'] = data[h_map['Fl_I.Pt']]
            self.prob['T'] = data[h_map['Fl_I.Tt']]
            self.prob['W'] = data[h_map['Fl_I.W']]
            self.prob['flow_start_MN'] = data[h_map['Fl_I.MN']]

            self.prob.run_model()

            # check outputs
            tol = 5e-4

            # print("effDes", self.prob['turbine.map.effDes'])
            # print("S_eff", self.prob['turbine.s_effDes'])
            # print("effMap", self.prob['turbine.map.desMap.effMap'])


            npss = data[h_map['EffDes']]
            pyc  = self.prob['eff'][0]
            print('EffDes:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.s']]
            pyc  = self.prob['flow_start.Fl_O:tot:S'][0]
            print('Fl_I.s:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.Tt']]
            pyc  = self.prob['flow_start.Fl_O:tot:T'][0]
            print('Fl_I.Tt:', npss, pyc)
            assert_near_equal(npss, pyc , tol)

            npss = data[h_map['Fl_I.ht']]
            pyc  = self.prob['flow_start.Fl_O:tot:h'][0]
            print('Fl_I.ht:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_I.W']]
            pyc  = self.prob['flow_start.Fl_O:stat:W'][0]
            print('Fl_I.W:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.MN']]
            pyc  = self.prob['MN'][0]
            print('Fl_O.MN:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.Pt']]
            pyc  = self.prob['turbine.Fl_O:tot:P'][0]
            print('Fl_O.Pt:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.ht']]
            pyc  = self.prob['turbine.Fl_O:tot:h'][0]
            print('Fl_O.ht:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['Fl_O.Tt']]
            pyc  = self.prob['turbine.Fl_O:tot:T'][0]
            print('Fl_O.Tt:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.ht']] - data[h_map['Fl_I.ht']]
            pyc  = self.prob['turbine.Fl_O:tot:h'][0] -self.prob['flow_start.Fl_O:tot:h'][0]
            print('Fl_O.ht - Fl_I.ht:', npss, pyc)
            assert_near_equal(npss , pyc ,tol)

            npss = data[h_map['Fl_O.s']]
            pyc  = self.prob['turbine.Fl_O:tot:S'][0]
            print('Fl_O.s:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['Power']]
            pyc  = self.prob['turbine.power'][0]
            print('Power:', npss, pyc)
            assert_near_equal(npss, pyc , tol)

            npss = data[h_map['Fl_O.Ps']]
            pyc  = self.prob['turbine.Fl_O:stat:P'][0]
            print('Fl_O.Ps:', npss, pyc)
            assert_near_equal(npss , pyc , tol)

            npss = data[h_map['Fl_O.Ts']]
            pyc  = self.prob['turbine.Fl_O:stat:T'][0]
            print('Fl_O.Ts:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            npss = data[h_map['effPoly']]
            pyc  = self.prob['turbine.eff_poly'][0]
            print('effPoly:', npss, pyc)
            assert_near_equal(npss, pyc, tol)

            check_element_partials(self, self.prob, tol=1e-4)

if __name__ == "__main__":
    np.seterr(divide='warn')
    unittest.main()
