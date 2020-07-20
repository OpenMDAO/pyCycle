""" Tests the inlet component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.species_data import janaf, Thermo
from pycycle.elements.inlet import Inlet
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_MIX

from pycycle.elements.test.util import check_element_partials

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/inlet.csv", delimiter=",", skiprows=1)

header = [
    'eRamBase',
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
    'Fram',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class InletTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        self.prob.model = Group()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('P', 17, units='psi')
        des_vars.add_output('T', 500.0, units='degR')
        des_vars.add_output('MN', 0.5)
        des_vars.add_output('W', 1., units='lbm/s')
        des_vars.add_output('V', 1., units='ft/s')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('inlet', Inlet(elements=AIR_MIX))

        # total and static
        fl_src = "flow_start.Fl_O"
        fl_target = "inlet.Fl_I"
        for v_name in ('h', 'T', 'P', 'S', 'rho', 'gamma', 'Cp', 'Cv', 'n'):
            self.prob.model.connect('%s:tot:%s' % (
                fl_src, v_name), '%s:tot:%s' % (fl_target, v_name))

        # no prefix
        for v_name in ('W', ):  # ('Wc', 'W', 'FAR'):
            self.prob.model.connect('%s:stat:%s' % (
                fl_src, v_name), '%s:stat:%s' % (fl_target, v_name))

        self.prob.model.connect("P", "flow_start.P")
        self.prob.model.connect("T", "flow_start.T")
        self.prob.model.connect("MN", "inlet.MN")
        self.prob.model.connect("W", "flow_start.W")
        self.prob.model.connect("V", "inlet.Fl_I:stat:V")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['inlet.ram_recovery'] = data[h_map['eRamBase']]

            # input flowstation
            self.prob['P'] = data[h_map['Fl_I.Pt']]
            self.prob['T'] = data[h_map['Fl_I.Tt']]
            self.prob['MN'] = data[h_map['Fl_O.MN']]
            self.prob['W'] = data[h_map['Fl_I.W']]
            self.prob['V'] = data[h_map['Fl_I.V']]

            self.prob.run_model()

            # check outputs

            pt, ht, Fram, ps, ts = data[h_map['Fl_O.Pt']], data[h_map['Fl_O.ht']], data[
                h_map['Fram']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
            pt_computed = self.prob['inlet.Fl_O:tot:P']
            ht_computed = self.prob['inlet.Fl_O:tot:h']
            Fram_computed = self.prob['inlet.F_ram']
            ps_computed = self.prob['inlet.Fl_O:stat:P']
            ts_computed = self.prob['inlet.Fl_O:stat:T']

            tol = 1e-4
            # rel_err = abs(pt_computed - pt) / pt_computed
            assert_rel_error(self, pt_computed, pt, tol)
            assert_rel_error(self, ht_computed, ht, tol)
            assert_rel_error(self, Fram_computed, Fram, tol)
            assert_rel_error(self, ps_computed, ps, tol)
            assert_rel_error(self, ts_computed, ts, tol)

            check_element_partials(self, self.prob)


class TestInletGenerated(unittest.TestCase):

    def test_case0(self):
        # captured inputs:
        prob = Problem()
        prob.model = Inlet()
        thermo = Thermo(janaf)
        prob.model.set_input_defaults('Fl_I:tot:T', 284, units='degK')
        prob.model.set_input_defaults('Fl_I:tot:P', 5.0, units='lbf/inch**2')
        prob.model.set_input_defaults('Fl_I:tot:n', thermo.init_prod_amounts)
        prob.model.set_input_defaults('Fl_I:stat:V', 0.0, units='ft/s')#keep
        prob.model.set_input_defaults('Fl_I:stat:W', 1, units='kg/s')


        prob.setup()

        # view_model(p)
        prob.run_model()
        tol = 1e-7

        prob['flow_in.Fl_I:tot:h'] = np.array([1.])
        prob['flow_in.Fl_I:tot:T'] = np.array([518.])
        prob['flow_in.Fl_I:tot:P'] = np.array([1.])
        prob['flow_in.Fl_I:tot:rho'] = np.array([1.])
        prob['flow_in.Fl_I:tot:gamma'] = np.array([1.3999999999999999])
        prob['flow_in.Fl_I:tot:Cp'] = np.array([1.])
        prob['flow_in.Fl_I:tot:Cv'] = np.array([1.])
        prob['flow_in.Fl_I:tot:S'] = np.array([1.])
        prob['flow_in.Fl_I:tot:n'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        prob['flow_in.Fl_I:stat:h'] = np.array([1.])
        prob['flow_in.Fl_I:stat:T'] = np.array([518.])
        prob['flow_in.Fl_I:stat:P'] = np.array([1.])
        prob['flow_in.Fl_I:stat:rho'] = np.array([1.])
        prob['flow_in.Fl_I:stat:gamma'] = np.array([1.3999999999999999])
        prob['flow_in.Fl_I:stat:Cp'] = np.array([1.])
        prob['flow_in.Fl_I:stat:Cv'] = np.array([1.])
        prob['flow_in.Fl_I:stat:S'] = np.array([0.])
        prob['flow_in.Fl_I:stat:n'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        prob['flow_in.Fl_I:stat:V'] = np.array([1.])
        prob['flow_in.Fl_I:stat:Vsonic'] = np.array([1.])
        prob['flow_in.Fl_I:stat:MN'] = np.array([1.])
        prob['flow_in.Fl_I:stat:area'] = np.array([1.])
        prob['flow_in.Fl_I:stat:Wc'] = np.array([1.])
        prob['flow_in.Fl_I:stat:W'] = np.array([0.])
        prob['flow_in.Fl_I:FAR'] = np.array([1.])
        prob['calcs_inlet.Pt_in'] = np.array([5.])
        prob['calcs_inlet.ram_recovery'] = np.array([1.])
        prob['calcs_inlet.V_in'] = np.array([0.])
        prob['calcs_inlet.W_in'] = np.array([100.])
        prob['real_flow.flow.T'] = np.array([284.])
        prob['real_flow.flow.P'] = np.array([5.])
        prob['real_flow.flow.h'] = np.array([-7.9733188552527334])
        prob['real_flow.flow.S'] = np.array([1.7017265964076564])
        prob['real_flow.flow.gamma'] = np.array([1.4002877621616485])
        prob['real_flow.flow.Cp'] = np.array([0.2398396648541221])
        prob['real_flow.flow.Cv'] = np.array([0.1712788399253588])
        prob['real_flow.flow.rho'] = np.array([0.0263992728679074])
        prob['real_flow.flow.n'] = np.array([3.2331925838204755e-04, 1.0000000000000000e-10,
                                             1.1013124070888753e-05, 1.0000000000000000e-10,
                                             1.0000000000000000e-10, 1.0000000000000000e-10,
                                             1.0000000000000000e-10, 2.6957886582171697e-02,
                                             1.0000000000000000e-10, 7.2319938237413537e-03])
        prob['real_flow.flow.n_moles'] = np.array([0.034524213388366])
        prob['FAR_passthru.Fl_I:FAR'] = np.array([0.])
        prob['out_stat.statics.ps_resid.Ts'] = np.array([270.4606098605202078])
        prob['out_stat.statics.ps_resid.ht'] = np.array([-18545.9396573178637482])
        prob['out_stat.statics.ps_resid.hs'] = np.array([-32137.5489635905178147])
        prob['out_stat.statics.ps_resid.n_moles'] = np.array([0.034524213388366])
        prob['out_stat.statics.ps_resid.gamma'] = np.array([1.4005512841014192])
        prob['out_stat.statics.ps_resid.W'] = np.array([1.])
        prob['out_stat.statics.ps_resid.rho'] = np.array([0.3743153852344497])
        prob['out_stat.statics.ps_resid.guess:gamt'] = np.array([1.4002877621616485])
        prob['out_stat.statics.ps_resid.guess:Pt'] = np.array([0.3447378650257302])
        prob['out_stat.statics.ps_resid.MN'] = np.array([0.5])
        prob['out_stat.flow.T'] = np.array([486.8290977489364195])
        prob['out_stat.flow.P'] = np.array([4.2148347084158546])
        prob['out_stat.flow.h'] = np.array([-13.8166590557138917])
        prob['out_stat.flow.S'] = np.array([1.701726596407656])
        prob['out_stat.flow.gamma'] = np.array([1.4005512841014192])
        prob['out_stat.flow.Cp'] = np.array([0.2397269927781015])
        prob['out_stat.flow.Cv'] = np.array([0.1711661644315738])
        prob['out_stat.flow.rho'] = np.array([0.0233677461124606])
        prob['out_stat.flow.n'] = np.array([3.2331925838204755e-04, 1.0000000000330098e-10,
                                            1.1013124070888747e-05, 1.0000000000001391e-10,
                                            1.0000000000147303e-10, 1.0000000000000000e-10,
                                            1.0000000000053357e-10, 2.6957886582171697e-02,
                                            1.0000000000021437e-10, 7.2319938237413537e-03])
        prob['out_stat.flow.n_moles'] = np.array([0.034524213388366])
        prob['out_stat.flow_static.area'] = np.array([25.1156515934009938])
        prob['out_stat.flow_static.W'] = np.array([1.])
        prob['out_stat.flow_static.V'] = np.array([540.9230351753141122])
        prob['out_stat.flow_static.Vsonic'] = np.array([1081.8460703506282243])
        prob['out_stat.flow_static.MN'] = np.array([0.5])

        # captured outputs:
        assert_rel_error(self, prob['flow_in.foo'], np.array([1.]), tol)
        assert_rel_error(self, prob['calcs_inlet.Pt_out'], np.array([5.]), tol)
        assert_rel_error(self, prob['calcs_inlet.F_ram'], np.array([0.]), tol)
        # assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:T'], np.array([284.]), tol)#######################
        assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:P'], np.array([5.]), tol)
        assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:h'],
                         np.array([-7.9733188552527361]), tol)
        assert_rel_error(
            self,
            prob['real_flow.flow.Fl_O:tot:S'],
            np.array(
                [1.701726596407656]),
            tol)
        assert_rel_error(
            self,
            prob['real_flow.flow.Fl_O:tot:gamma'],
            np.array(
                [1.4002877621616485]),
            tol)
        assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:Cp'],
                         np.array([0.2398396648541221]), tol)
        assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:Cv'],
                         np.array([0.1712788399253588]), tol)
        assert_rel_error(self, prob['real_flow.flow.Fl_O:tot:rho'],
                         np.array([0.0263992728679074]), tol)
        assert_rel_error(self,
                         prob['real_flow.flow.Fl_O:tot:n'],
                         np.array([3.2331925838204755e-04,
                                   1.0000000000000000e-10,
                                   1.1013124070888753e-05,
                                   1.0000000000000000e-10,
                                   1.0000000000000000e-10,
                                   1.0000000000000000e-10,
                                   1.0000000000000000e-10,
                                   2.6957886582171697e-02,
                                   1.0000000000000000e-10,
                                   7.2319938237413537e-03]),
                         tol)
        assert_rel_error(
            self,
            prob['real_flow.flow.Fl_O:tot:n_moles'],
            np.array(
                [0.034524213388366]),
            tol)
        assert_rel_error(self, prob['FAR_passthru.Fl_O:FAR'], np.array([0.]), tol)
        assert_rel_error(
            self,
            prob['out_stat.statics.ps_resid.Ps'],
            np.array(
                [0.2906026237631253]),
            tol)
        assert_rel_error(self, prob['out_stat.statics.ps_resid.V'],
                         np.array([164.8733411214357432]), tol)
        assert_rel_error(self, prob['out_stat.statics.ps_resid.Vsonic'],
                         np.array([329.7466822428714863]), tol)
        assert_rel_error(
            self,
            prob['out_stat.statics.ps_resid.area'],
            np.array(
                [0.0162036137819986]),
            tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:T'],
                         np.array([486.8290977489363058]), tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:P'],
                         np.array([4.2148347084158537]), tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:h'],
                         np.array([-13.8166590557138935]), tol)
        assert_rel_error(
            self,
            prob['out_stat.flow.Fl_O:stat:S'],
            np.array(
                [1.701726596407656]),
            tol)
        assert_rel_error(
            self,
            prob['out_stat.flow.Fl_O:stat:gamma'],
            np.array(
                [1.4005512841014192]),
            tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:Cp'],
                         np.array([0.2397269927781014]), tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:Cv'],
                         np.array([0.1711661644315738]), tol)
        assert_rel_error(self, prob['out_stat.flow.Fl_O:stat:rho'],
                         np.array([0.0233677461124606]), tol)
        assert_rel_error(self,
                         prob['out_stat.flow.Fl_O:stat:n'],
                         np.array([3.2331925838204755e-04,
                                   1.0000000000330098e-10,
                                   1.1013124070888747e-05,
                                   1.0000000000001391e-10,
                                   1.0000000000147303e-10,
                                   1.0000000000000000e-10,
                                   1.0000000000053357e-10,
                                   2.6957886582171697e-02,
                                   1.0000000000021437e-10,
                                   7.2319938237413537e-03]),
                         tol)
        assert_rel_error(
            self,
            prob['out_stat.flow.Fl_O:stat:n_moles'],
            np.array(
                [0.034524213388366]),
            tol)
        assert_rel_error(
            self,
            prob['out_stat.flow_static.Fl_O:stat:area'],
            np.array(
                [25.1156515934009938]),
            tol)
        assert_rel_error(self, prob['out_stat.flow_static.Fl_O:stat:W'], np.array([1.]), tol)########################
        assert_rel_error(self, prob['out_stat.flow_static.Fl_O:stat:V'],
                         np.array([540.9230351753141122]), tol)
        assert_rel_error(
            self,
            prob['out_stat.flow_static.Fl_O:stat:Vsonic'],
            np.array(
                [1081.8460703506282243]),
            tol)
        assert_rel_error(self, prob['out_stat.flow_static.Fl_O:stat:MN'], np.array([0.5]), tol)

        check_element_partials(self, prob, depth=1)
if __name__ == "__main__":
    unittest.main()
