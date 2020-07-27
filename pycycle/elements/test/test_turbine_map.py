import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.elements.turbine_map import TurbineMap
from pycycle.maps.lpt2269 import LPT2269


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/turbineOD1.csv", delimiter=",", skiprows=1)

header = [
    'turb.PRdes',
    'turb.effDes',
    'shaft.Nmech',
    'burn.FAR',
    'burn.Fl_I.W',
    'burn.Fl_I.Pt',
    'burn.Fl_I.Tt',
    'burn.Fl_I.ht',
    'burn.Fl_I.s',
    'burn.Fl_I.MN',
    'burn.Fl_I.V',
    'burn.Fl_I.A',
    'burn.Fl_I.Ps',
    'burn.Fl_I.Ts',
    'burn.Fl_I.hs',
    'turb.Fl_I.W',
    'turb.Fl_I.Pt',
    'turb.Fl_I.Tt',
    'turb.Fl_I.ht',
    'turb.Fl_I.s',
    'turb.Fl_I.MN',
    'turb.Fl_I.V',
    'turb.Fl_I.A',
    'turb.Fl_I.Ps',
    'turb.Fl_I.Ts',
    'turb.Fl_I.hs',
    'turb.Fl_O.W',
    'turb.Fl_O.Pt',
    'turb.Fl_O.Tt',
    'turb.Fl_O.ht',
    'turb.Fl_O.s',
    'turb.Fl_O.MN',
    'turb.Fl_O.V',
    'turb.Fl_O.A',
    'turb.Fl_O.Ps',
    'turb.Fl_O.Ts',
    'turb.Fl_O.hs',
    'turb.PR',
    'turb.eff',
    'turb.Np',
    'turb.Wp',
    'turb.pwr',
    'turb.PRmap',
    'turb.effMap',
    'turb.NpMap',
    'turb.WpMap',
    'turb.s_WpDes',
    'turb.s_PRdes',
    'turb.s_effDes',
    'turb.s_NpDes']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))

class TurbineMapTestCase(unittest.TestCase):

    def setUp(self):
        print('\n')
        self.prob = Problem()
        
        self.prob.model.add_subsystem(
            'map',
            TurbineMap(
                map_data=LPT2269,
                design=True),
            promotes=['*'])

        self.prob.model.set_input_defaults('PR', 5.0)
        self.prob.model.set_input_defaults('alphaMap', 1.0)
        self.prob.model.set_input_defaults('Wp', 322.60579101811692, units='lbm/s')
        self.prob.model.set_input_defaults('Np', 172.11870165984794, units='rpm')

        self.prob.setup(check=False)

    def test_case1(self):
        # 4 cases to check against
        data = ref_data[0]  # only the first case is a design case

        self.prob['Wp'] = data[h_map['turb.Wp']]
        self.prob['Np'] = data[h_map['turb.Np']]
        self.prob['alphaMap'] = 1.
        self.prob['PR'] = data[h_map['turb.PR']]

        self.prob['eff'] = data[h_map['turb.effDes']]

        # input flowstation
        self.prob.run_model()
        # self.prob.run()

        # check outputs
        tol = 1e-4  # a little high... need to fix exit static pressure

        print('-------------- Design Test Case -------------')

        # check output of scalars
        npss = data[h_map['turb.s_NpDes']]
        pyc = self.prob['s_Np'][0]
        print('s_NpDes:', pyc, npss)
        assert_near_equal(pyc, npss, tol)

        npss = data[h_map['turb.NpMap']]
        pyc = self.prob['NpMap'][0]
        print('NpMap:', pyc, npss)
        assert_near_equal(pyc, npss, tol)

        npss = data[h_map['turb.s_PRdes']]
        pyc = self.prob['s_PR'][0]
        print('s_PRdes:', pyc, npss)
        assert_near_equal(pyc, npss, tol)

        # check outputs of readMap
        npss = data[h_map['turb.effMap']]
        pyc = self.prob['effMap'][0]
        print('effMap:', pyc, npss)
        assert_near_equal(pyc, npss, tol)

        npss = data[h_map['turb.WpMap']]
        pyc = self.prob['WpMap'][0]
        print('WpMap:', pyc, npss)
        assert_near_equal(pyc, npss, tol)

        # check outputs of scaledOutput
        npss = data[h_map['turb.eff']]
        pyc = self.prob['eff']
        rel_err = abs(npss - pyc)/npss
        print('eff:', npss, pyc, rel_err)
        self.assertLessEqual(rel_err, tol)

        # check top level outputs
        npss = data[h_map['turb.PR']]
        pyc = self.prob['PR']
        rel_err = abs(npss - pyc)/npss
        print('PR:', npss, pyc, rel_err)
        self.assertLessEqual(rel_err, tol)


class TurbineMapODTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        self.prob.model.add_subsystem(
            'map',
            TurbineMap(
                map_data=LPT2269,
                design=False),
            promotes=['*'])


        self.prob.model.set_input_defaults('s_Wp', 2.152309293)
        self.prob.model.set_input_defaults('s_eff', 0.9950409659)
        self.prob.model.set_input_defaults('s_Np', 1.721074624)
        self.prob.model.set_input_defaults('s_PR', 0.147473296)
        self.prob.model.set_input_defaults('Wp', 322.60579101811692, units='lbm/s')
        self.prob.model.set_input_defaults('Np', 172.11870165984794, units='rpm')
        self.prob.model.set_input_defaults('alphaMap', 1.0)

        self.prob.setup(check=False)

    def test_caseOD1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['Wp'] = data[h_map['turb.Wp']]
            self.prob['Np'] = data[h_map['turb.Np']]
            self.prob['alphaMap'] = 1.
            self.prob['s_Np'] = data[h_map['turb.s_NpDes']]
            self.prob['s_PR'] = data[h_map['turb.s_PRdes']]
            self.prob['s_Wp'] = data[h_map['turb.s_WpDes']]
            self.prob['s_eff'] = data[h_map['turb.s_effDes']]
            self.prob['PR'] = data[h_map['turb.PR']]
            self.prob['PRmap'] = data[h_map['turb.PRmap']]
            self.prob['NpMap'] = data[h_map['turb.NpMap']]

            self.prob.run_model()

            # check outputs
            tol = 3e-3  # a little high... need to fix exit static pressure

            print('-------------- Test Case ', i, '-------------')
            # check output of mapInputs
            npss = data[h_map['turb.PRmap']]
            pyc = self.prob['PRmap'][0]
            print('PRmap:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.NpMap']]
            pyc = self.prob['NpMap'][0]
            print('NpMap:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            # check outputs of readMap
            npss = data[h_map['turb.effMap']]
            pyc = self.prob['effMap'][0]
            print('effMap:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.WpMap']]
            pyc = self.prob['WpMap'][0]
            print('WpMap:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            # check outputs of scaledOutput
            npss = data[h_map['turb.eff']]
            pyc = self.prob['eff'][0]
            print('eff:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            # check top level outputs
            npss = data[h_map['turb.PR']]
            pyc = self.prob['PR'][0]
            print('PR:', pyc, npss)
            assert_near_equal(pyc, npss, tol)

            # check to make sure balance is converged
            assert_near_equal(self.prob['Np'], self.prob['scaledOutput.Np'], tol)
            print('Np balance:',self.prob['Np'][0], self.prob['scaledOutput.Np'][0])

            assert_near_equal(self.prob['Wp'], self.prob['scaledOutput.Wp'], tol)
            print('Wp balance:',self.prob['Wp'][0], self.prob['scaledOutput.Wp'][0])
            print()

if __name__ == "__main__":
    np.seterr(divide='warn')
    unittest.main()
