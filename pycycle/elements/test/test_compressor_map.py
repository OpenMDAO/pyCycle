import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.elements.compressor_map import CompressorMap
from pycycle.maps.axi5 import AXI5


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/compressorOD1.csv", delimiter=",", skiprows=1)

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


class CompressorMapTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        self.prob.model.add_subsystem(
            'map',
            CompressorMap(
                map_data=AXI5,
                design=False),
            promotes=['*'])

        self.prob.model.set_input_defaults('Wc', 322.60579101811692, units='lbm/s')
        self.prob.model.set_input_defaults('Nc', 172.11870165984794, units='rpm')
        self.prob.model.set_input_defaults('alphaMap', 1.0)
        self.prob.model.set_input_defaults('s_Nc', 1.721074624)
        self.prob.model.set_input_defaults('s_PR', 0.147473296)
        self.prob.model.set_input_defaults('s_Wc', 2.152309293)
        self.prob.model.set_input_defaults('s_eff', 0.9950409659)

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


        self.prob.setup(check=False)

    def test_caseOD1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['Wc'] = data[h_map['comp.Wc']]
            self.prob['Nc'] = data[h_map['comp.Nc']]
            self.prob['alphaMap'] = 0.
            self.prob['s_Nc'] = data[h_map['comp.s_NcDes']]
            self.prob['s_PR'] = data[h_map['comp.s_PRdes']]
            self.prob['s_Wc'] = data[h_map['comp.s_WcDes']]
            self.prob['s_eff'] = data[h_map['comp.s_effDes']]

            self.prob['RlineMap'] = data[h_map['comp.RlineMap']]
            self.prob['NcMap'] = data[h_map['comp.NcMap']]

            # input flowstation
            self.prob.run_model()

            # check outputs
            tol = 0.3e-2  # a little high... need to fix exit static pressure

            print('-------------- Test Case ', i, '-------------')
            # check output of shaftNc
            npss = data[h_map['comp.NcMap']]
            pyc = self.prob['NcMap'][0]
            print('NcMap:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.RlineMap']]
            pyc = self.prob['RlineMap'][0]
            print('RlineMap:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            # check outputs of readMap
            npss = data[h_map['comp.effMap']]
            pyc = self.prob['effMap'][0]
            print('effMap:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.PRmap']]
            pyc = self.prob['PRmap'][0]
            print('PRmap:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.WcMap']]
            pyc = self.prob['WcMap'][0]
            print('WcMap:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            # check outputs of scaledOutput
            npss = data[h_map['comp.eff']]
            pyc = self.prob['eff'][0]
            print('eff:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            # check top level outputs
            npss = data[h_map['comp.PR']]
            pyc = self.prob['PR'][0]
            print('PR:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.SMW']]
            pyc = self.prob['SMW'][0]
            print('SMW:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['comp.SMN']]
            pyc = self.prob['SMN'][0]
            print('SMN:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            print('Wc:', data[h_map['comp.Wc']], self.prob['Wc'][0], self.prob['scaledOutput.Wc'][0])
            print()

if __name__ == "__main__":
    np.seterr(divide='warn')
    unittest.main()
