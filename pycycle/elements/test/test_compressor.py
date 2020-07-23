import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.constants import AIR_MIX
from pycycle.connect_flow import connect_flow
from pycycle.cea.species_data import janaf, Thermo
from pycycle.elements.compressor import Compressor
from pycycle.elements.flow_start import FlowStart
from pycycle import constants

from pycycle.elements.test.util import check_element_partials

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

        thermo = Thermo(janaf, constants.janaf_init_prod_amounts)

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('MN', 0.5)
        des_vars.add_output('W', 10., units='lbm/s')
        des_vars.add_output('PR', 6.)
        des_vars.add_output('eff', 0.9)
        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect("MN", "compressor.MN")
        self.prob.model.connect('PR', 'compressor.PR')
        self.prob.model.connect('eff', 'compressor.eff')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('compressor', Compressor(design=True, elements=AIR_MIX))

        connect_flow(self.prob.model, "flow_start.Fl_O", "compressor.Fl_I")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def test_case1(self):
        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['PR'] = data[h_map['comp.PRdes']]
            self.prob['eff'] = data[h_map['comp.effDes']]
            self.prob['MN'] = data[h_map['comp.Fl_O.MN']]

            # input flowstation
            self.prob['P'] = data[h_map['start.Pt']]
            self.prob['T'] = data[h_map['start.Tt']]
            self.prob['W'] = data[h_map['start.W']]
            self.prob.run_model()

            # print("    mapPRdes     :         PRdes       :        PR       :      scalarsPRmapDes : scaledOutput.PR")
            # print(self.prob['PR'], data[h_map['comp.PRdes']], self.prob[
            #       'PR'], self.prob['compressor.map.PRmap'])
            # print("s_PR", self.prob['compressor.s_PR'])

            # print("    mapeffDes     :         effDes       :        eff       :      scalars_effMapDes : scaledOutput.eff")
            # print(self.prob['compressor.map.effDes'], data[h_map['comp.effDes']], self.prob[
            #       'compressor.eff'], self.prob['compressor.map.scalars.effMapDes'])
            # print("Rline", self.prob['compressor.map.RlineMap'])

            # print(self.prob.model.resids._dat.keys())

            #print("errWc: ", self.prob.model.resids['compressor.map.RlineMap'])

            # quit()
            # check outputs
            tol = 1e-3

            npss = data[h_map['comp.Fl_O.Pt']]
            pyc = self.prob['compressor.Fl_O:tot:P'][0]
            print('Pt out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.Fl_O.Tt']]
            pyc = self.prob['compressor.Fl_O:tot:T'][0]
            # print('foo test:', self.prob['compressor.enth_rise.ht_out'][0], data[h_map['start.Fl_O.ht']])
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

            check_element_partials(self, self.prob,tol = 5e-5)

if __name__ == "__main__":
    unittest.main()
