import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.elements.shaft import Shaft

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/shaft.csv",
                      delimiter=",", skiprows=1)

header = [
    'trqLoad1',
    'trqLoad2',
    'trqLoad3',
    'Nmech',
    'HPX',
    'fracLoss',
    'trqIn',
    'trqOut',
    'trqNet',
    'pwrIn',
    'pwrOut',
    'pwrNet']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class ShaftTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()
        self.prob.model = Group()
        self.prob.model.add_subsystem("shaft", Shaft(num_ports=3), promotes=["*"])

        self.prob.model.set_input_defaults('trq_0', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('trq_1', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('trq_2', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('Nmech', 17., units='rpm')
        self.prob.model.set_input_defaults('HPX', 17., units='hp')
        self.prob.model.set_input_defaults('fracLoss', 17.)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            # input torques
            self.prob['trq_0'] = data[h_map['trqLoad1']]
            self.prob['trq_1'] = data[h_map['trqLoad2']]
            self.prob['trq_2'] = data[h_map['trqLoad3']]

            # shaft inputs
            self.prob['Nmech'] = data[h_map['Nmech']]
            self.prob['HPX'] = data[h_map['HPX']]
            self.prob['fracLoss'] = data[h_map['fracLoss']]
            self.prob.run_model()

            # check outputs
            trqIn, trqOut, trqNet = data[
                h_map['trqIn']], data[
                h_map['trqOut']], data[
                h_map['trqNet']]
            pwrIn, pwrOut, pwrNet = data[
                h_map['pwrIn']], data[
                h_map['pwrOut']], data[
                h_map['pwrNet']]
            trqIn_comp = self.prob['trq_in']
            trqOut_comp = self.prob['trq_out']
            trqNet_comp = self.prob['trq_net']
            pwrIn_comp = self.prob['pwr_in']
            pwrOut_comp = self.prob['pwr_out']
            pwrNet_comp = self.prob['pwr_net']

            tol = 1.0e-4
            assert_near_equal(trqIn_comp, trqIn, tol)
            assert_near_equal(trqOut_comp, trqOut, tol)
            assert_near_equal(trqNet_comp, trqNet, tol)
            assert_near_equal(pwrIn_comp, pwrIn, tol)
            assert_near_equal(pwrOut_comp, pwrOut, tol)
            assert_near_equal(pwrNet_comp, pwrNet, tol)

            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
