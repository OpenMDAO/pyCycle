import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials


from pycycle.elements.ambient import Ambient


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/ambient.csv",
                      delimiter=",", skiprows=1)

header = ['alt','MN','dTs','Pt','Ps','Tt','Ts']

h_map = dict(((v_name,i) for i,v_name in enumerate(header)))

class FlowStartTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        self.prob.model.add_subsystem('amb', Ambient())
        self.prob.model.set_input_defaults('amb.alt', 0, units='ft')
        self.prob.model.set_input_defaults('amb.dTs', 0, units='degR')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['amb.alt'] = data[h_map['alt']]
            self.prob['amb.dTs'] = data[h_map['dTs']]

            self.prob.run_model()

            # check outputs
            tol = 1.0e-2 # seems a little generous

            npss = data[h_map['Ps']]
            pyc = self.prob['amb.Ps']
            rel_err = abs(npss - pyc)/npss
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Ts']]
            pyc = self.prob['amb.Ts']
            rel_err = abs(npss - pyc)/npss
            self.assertLessEqual(rel_err, tol)

            partial_data = self.prob.check_partials(out_stream=None, method='cs', 
                                                    includes=['amb.*'], excludes=['*.base_thermo.*', 'amb.readAtmTable'])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)



if __name__ == "__main__":
    unittest.main()