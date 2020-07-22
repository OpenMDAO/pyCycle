import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver


from pycycle.constants import AIR_MIX
from pycycle.cea.species_data import janaf, Thermo
from pycycle.connect_flow import connect_flow

from pycycle.elements.compressor import Compressor
from pycycle.elements.flow_start import FlowStart
from pycycle.maps.axi5 import AXI5

from pycycle.elements.test.util import check_element_partials


class CompressorODTestCase(unittest.TestCase):

    def setUp(self):

        thermo = Thermo(janaf)

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 0., units='lbm/s')
        des_vars.add_output('Nmech', 0., units='rpm')
        des_vars.add_output('area_targ', 50., units='inch**2')

        des_vars.add_output('s_PR', val=1.)
        des_vars.add_output('s_eff', val=1.)
        des_vars.add_output('s_Wc', val=1.)
        des_vars.add_output('s_Nc', val=1.)
        des_vars.add_output('alphaMap', val=0.)

        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect("Nmech", "compressor.Nmech")
        self.prob.model.connect("area_targ", "compressor.area")

        self.prob.model.connect("s_PR", "compressor.s_PR")
        self.prob.model.connect("s_eff", "compressor.s_eff")
        self.prob.model.connect("s_Wc", "compressor.s_Wc")
        self.prob.model.connect("s_Nc", "compressor.s_Nc")
        self.prob.model.connect('alphaMap', 'compressor.map.alphaMap')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('compressor', Compressor(
                map_data=AXI5, design=False, elements=AIR_MIX, map_extrap=False))

        connect_flow(self.prob.model, "flow_start.Fl_O", "compressor.Fl_I")

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

        self.prob.model.compressor.set_input_defaults('Fl_I:tot:b0', thermo.b0)


        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):
        np.seterr(divide='raise')

        fpath = os.path.dirname(os.path.realpath(__file__))
        ref_data = np.loadtxt(fpath + "/reg_data/compressorOD1.csv",
                              delimiter=",", skiprows=1)

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
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            # if i == 0:
            #     pass  # skip design case
            # else:
                self.prob['s_PR'] = data[h_map['comp.s_PRdes']]
                self.prob['s_Wc'] = data[h_map['comp.s_WcDes']]
                self.prob['s_eff'] = data[h_map['comp.s_effDes']]
                self.prob['s_Nc'] = data[h_map['comp.s_NcDes']]
                self.prob['compressor.map.RlineMap'] = data[h_map['comp.RlineMap']]
                self.prob['Nmech'] = data[h_map['shaft.Nmech']]

                # input flowstation
                self.prob['P'] = data[h_map['comp.Fl_I.Pt']]
                self.prob['T'] = data[h_map['comp.Fl_I.Tt']]
                self.prob['W'] = data[h_map['comp.Fl_I.W']]
                # cu(data[h_map['comp.Fl_O.A']],"inch**2", "m**2")
                self.prob['area_targ'] = data[h_map['comp.Fl_O.A']]

                self.prob.run_model()
                tol = 3.0e-3  # seems a little generous,
                # FL_O.Ps is off by 4% or less, everything else is <1% tol

                print('----- Test Case', i, '-----')
                npss = data[h_map['comp.Fl_I.Pt']]
                pyc = self.prob['flow_start.Fl_O:tot:P'][0]
                print('Pt in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.s']]
                pyc = self.prob['flow_start.Fl_O:tot:S'][0]
                print('S in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.W']]
                pyc = self.prob['compressor.Fl_O:stat:W'][0]
                print('W in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_I.s']]
                pyc = self.prob['flow_start.Fl_O:tot:S'][0]
                print('S in:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.RlineMap']]
                pyc = self.prob['compressor.map.RlineMap'][0]
                print('RlineMap:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.PR']]
                pyc = self.prob['compressor.PR'][0]
                print('PR:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.eff']]
                pyc = self.prob['compressor.eff'][0]
                print('eff:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.ht']] - data[h_map['comp.Fl_I.ht']]
                pyc = self.prob['compressor.Fl_O:tot:h'][0] - self.prob['flow_start.Fl_O:tot:h'][0]
                print('delta h:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.s']]
                pyc = self.prob['compressor.Fl_O:tot:S'][0]
                print('S out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.pwr']]
                pyc = self.prob['compressor.power'][0]
                print('Power:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.Ps']]
                pyc = self.prob['compressor.Fl_O:stat:P'][0]
                print('Ps out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.Fl_O.Ts']]
                pyc = self.prob['compressor.Fl_O:stat:T'][0]
                print('Ts out:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.SMW']]
                pyc = self.prob['compressor.SMW'][0]
                print('SMW:', npss, pyc)
                assert_near_equal(pyc, npss, tol)

                npss = data[h_map['comp.SMN']]
                pyc = self.prob['compressor.SMN'][0]
                print('SMN:', npss, pyc)
                assert_near_equal(pyc, npss, tol)
                print()

                check_element_partials(self, self.prob,tol = 5e-5)

if __name__ == "__main__":
    unittest.main()
