import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.afterburning_turbojet import ABTurbojet


class ABTurbojetOffdesignWetTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
        des_vars.add_output('OD_MN', 0.000001)
        des_vars.add_output('OD_alt', 0.0, units='ft')
        des_vars.add_output('OD_T4', 2370.0, units='degR')
        des_vars.add_output('OD_ab_FAR', 0.031523391)
        des_vars.add_output('OD_Rline', 2.0)

        des_vars.add_output('duct1:dPqP', 0.02)
        des_vars.add_output('burn:dPqP', 0.03)
        des_vars.add_output('ab:dPqP', 0.06)
        des_vars.add_output('nozz:Cv', 0.99)
        des_vars.add_output('comp:cool1:frac_W', 0.0789)
        des_vars.add_output('comp:cool1:frac_P', 1.0)
        des_vars.add_output('comp:cool1:frac_work', 1.0)
        des_vars.add_output('comp:cool2:frac_W', 0.0383)
        des_vars.add_output('comp:cool2:frac_P', 1.0)
        des_vars.add_output('comp:cool2:frac_work', 1.0)
        des_vars.add_output('turb:cool1:frac_P', 1.0)
        des_vars.add_output('turb:cool2:frac_P', 0.0)

        des_vars.add_output('comp:s_PRdes', 2.97619047619)
        des_vars.add_output('comp:s_WcDes', 5.71447539197)
        des_vars.add_output('comp:s_effDes', 0.975323149236)
        des_vars.add_output('comp:s_NcDes', 8070.0)
        des_vars.add_output('turb:s_PRdes', 0.692263737296)
        des_vars.add_output('turb:s_WpDes', 0.259890960213)
        des_vars.add_output('turb:s_effDes', 0.927123760241)
        des_vars.add_output('turb:s_NpDes', 1.65767490056)

        des_vars.add_output('inlet:Fl_O:stat:area', 581.693962336, units='inch**2')
        des_vars.add_output('duct1:Fl_O:stat:area', 593.565267689, units='inch**2')
        des_vars.add_output('comp:Fl_O:stat:area', 148.263145086, units='inch**2')
        des_vars.add_output('burner:Fl_O:stat:area', 224.924279335, units='inch**2')
        des_vars.add_output('turb:Fl_O:stat:area', 504.741845228, units='inch**2')
        des_vars.add_output('ab:Fl_O:stat:area', 536.954431167, units='inch**2')

        self.prob.model.add_subsystem('OD', ABTurbojet(design=False))

        self.prob.model.connect('OD_alt', 'OD.fc.alt')
        self.prob.model.connect('OD_MN', 'OD.fc.MN')
        self.prob.model.connect('OD_T4', 'OD.balance.rhs:FAR')
        self.prob.model.connect('OD_Rline', 'OD.balance.rhs:W')
        self.prob.model.connect('OD_ab_FAR', 'OD.ab.Fl_I:FAR')

        self.prob.model.connect('duct1:dPqP', 'OD.duct1.dPqP')
        self.prob.model.connect('burn:dPqP', 'OD.burner.dPqP')
        self.prob.model.connect('ab:dPqP', 'OD.ab.dPqP')
        self.prob.model.connect('nozz:Cv', 'OD.nozz.Cv')

        self.prob.model.connect('comp:cool1:frac_W', 'OD.comp.cool1:frac_W')
        self.prob.model.connect('comp:cool1:frac_P', 'OD.comp.cool1:frac_P')
        self.prob.model.connect('comp:cool1:frac_work', 'OD.comp.cool1:frac_work')

        self.prob.model.connect('comp:cool2:frac_W', 'OD.comp.cool2:frac_W')
        self.prob.model.connect('comp:cool2:frac_P', 'OD.comp.cool2:frac_P')
        self.prob.model.connect('comp:cool2:frac_work', 'OD.comp.cool2:frac_work')

        self.prob.model.connect('turb:cool1:frac_P', 'OD.turb.cool1:frac_P')
        self.prob.model.connect('turb:cool2:frac_P', 'OD.turb.cool2:frac_P')

        self.prob.model.connect('comp:s_PRdes', 'OD.comp.s_PR')
        self.prob.model.connect('comp:s_WcDes', 'OD.comp.s_Wc')
        self.prob.model.connect('comp:s_effDes', 'OD.comp.s_eff')
        self.prob.model.connect('comp:s_NcDes', 'OD.comp.s_Nc')

        self.prob.model.connect('turb:s_PRdes', 'OD.turb.s_PR')
        self.prob.model.connect('turb:s_WpDes', 'OD.turb.s_Wp')
        self.prob.model.connect('turb:s_effDes', 'OD.turb.s_eff')
        self.prob.model.connect('turb:s_NpDes', 'OD.turb.s_Np')

        self.prob.model.connect('inlet:Fl_O:stat:area', 'OD.inlet.area')
        self.prob.model.connect('duct1:Fl_O:stat:area', 'OD.duct1.area')
        self.prob.model.connect('comp:Fl_O:stat:area', 'OD.comp.area')
        self.prob.model.connect('burner:Fl_O:stat:area', 'OD.burner.area')
        self.prob.model.connect('turb:Fl_O:stat:area', 'OD.turb.area')
        self.prob.model.connect('ab:Fl_O:stat:area', 'OD.ab.area')

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        self.prob['OD.balance.W'] = 168.453135137
        self.prob['OD.balance.FAR'] = 0.0175506829934
        self.prob['OD.balance.Nmech'] = 8070.0
        self.prob['OD.fc.balance.Pt'] = 14.6955113159
        self.prob['OD.fc.balance.Tt'] = 518.665288153
        self.prob['OD.turb.PR'] = 4.46138725662

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def benchmark_case1(self):
        # ADP Point
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 168.005
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.01755
        pyc = self.prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 8070.00
        pyc = self.prob['OD.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 17799.7
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1.61420
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1190.18
        pyc = self.prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    # @unittest.expectedFailure
    def benchmark_case2(self):
        np.seterr(divide='raise')
        self.prob['OD_MN'] = 0.8
        self.prob['OD_alt'] = 0.0
        self.prob['OD_T4'] = 2370.0
        self.prob['OD_ab_FAR'] = 0.022759941
        self.prob['OD_Rline'] = 2.0

        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 225.917
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 11.971
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.01629
        pyc = self.prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 8288.85
        pyc = self.prob['OD.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 24085.2
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1.71066
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1280.18
        pyc = self.prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()
