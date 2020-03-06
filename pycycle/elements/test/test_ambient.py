import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp


from pycycle.elements.ambient import Ambient

from pycycle.elements.test.util import check_element_partials

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/ambient.csv",
                      delimiter=",", skiprows=1)

header = ['alt','MN','dTs','Pt','Ps','Tt','Ts']

h_map = dict(((v_name,i) for i,v_name in enumerate(header)))

class FlowStartTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()


        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp())
        des_vars.add_output('alt', 0, units='ft')
        des_vars.add_output('dTs', 0, units='degR')

        self.prob.model.add_subsystem('amb', Ambient())
        self.prob.model.connect('des_vars.alt', 'amb.alt')
        self.prob.model.connect('des_vars.dTs', 'amb.dTs')

        self.prob.setup(check=False)

    def test_case1(self):
        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['des_vars.alt'] = data[h_map['alt']]
            # self.prob['des_vars.MN'] = data[h_map['MN']]
            self.prob['des_vars.dTs'] = data[h_map['dTs']]

            self.prob.run_model()

            # check outputs
            tol = 1.0e-2 # seems a little generous

            npss = data[h_map['Ps']]
            pyc = self.prob['amb.Ps']
            rel_err = abs(npss - pyc)/npss
            # print( 'Ps:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Ts']]
            pyc = self.prob['amb.Ts']
            rel_err = abs(npss - pyc)/npss
            # print( 'Ts:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            # print()
            check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()