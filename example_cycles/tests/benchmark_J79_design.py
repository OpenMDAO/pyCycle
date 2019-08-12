import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.J79 import Turbojet


class J79DesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('alt', 0.0, units='ft'),
        des_vars.add_output('MN', 0.000001),
        des_vars.add_output('T4max', 2370.0, units='degR'),
        des_vars.add_output('Fn_des', 11800.0, units='lbf'),
        des_vars.add_output('duct1:dPqP', 0.02),
        des_vars.add_output('comp:PRdes', 13.5),
        des_vars.add_output('comp:effDes', 0.83),
        des_vars.add_output('burn:dPqP', 0.03),
        des_vars.add_output('turb:effDes', 0.86),
        des_vars.add_output('ab:dPqP', 0.06),
        des_vars.add_output('nozz:Cv', 0.99),
        des_vars.add_output('shaft:Nmech', 8070.0, units='rpm'),
        des_vars.add_output('inlet:MN_out', 0.60),
        des_vars.add_output('duct1:MN_out', 0.60),
        des_vars.add_output('comp:MN_out', 0.20),
        des_vars.add_output('burner:MN_out', 0.20),
        des_vars.add_output('turb:MN_out', 0.4),
        des_vars.add_output('ab:MN_out', 0.4),
        des_vars.add_output('ab:FAR', 0.000),
        des_vars.add_output('comp:cool1:frac_W', 0.0789),
        des_vars.add_output('comp:cool1:frac_P', 1.0),
        des_vars.add_output('comp:cool1:frac_work', 1.0),
        des_vars.add_output('comp:cool2:frac_W', 0.0383),
        des_vars.add_output('comp:cool2:frac_P', 1.0),
        des_vars.add_output('comp:cool2:frac_work', 1.0),
        des_vars.add_output('turb:cool1:frac_P', 1.0),
        des_vars.add_output('turb:cool2:frac_P', 0.0),

        self.prob.model.add_subsystem('DESIGN', Turbojet(statics=True))

        self.prob.model.connect('alt', 'DESIGN.fc.alt')
        self.prob.model.connect('MN', 'DESIGN.fc.MN')
        self.prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
        self.prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
        self.prob.model.connect('duct1:dPqP', 'DESIGN.duct1.dPqP')
        self.prob.model.connect('comp:PRdes', 'DESIGN.comp.PR')
        self.prob.model.connect('comp:effDes', 'DESIGN.comp.eff')
        self.prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
        self.prob.model.connect('turb:effDes', 'DESIGN.turb.eff')
        self.prob.model.connect('ab:dPqP', 'DESIGN.ab.dPqP')
        self.prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
        self.prob.model.connect('shaft:Nmech', 'DESIGN.Nmech')
        self.prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
        self.prob.model.connect('duct1:MN_out', 'DESIGN.duct1.MN')
        self.prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
        self.prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
        self.prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')
        self.prob.model.connect('ab:MN_out', 'DESIGN.ab.MN')
        self.prob.model.connect('ab:FAR', 'DESIGN.ab.Fl_I:FAR')
        self.prob.model.connect('comp:cool1:frac_W', 'DESIGN.comp.cool1:frac_W')
        self.prob.model.connect('comp:cool1:frac_P', 'DESIGN.comp.cool1:frac_P')
        self.prob.model.connect('comp:cool1:frac_work', 'DESIGN.comp.cool1:frac_work')
        self.prob.model.connect('comp:cool2:frac_W', 'DESIGN.comp.cool2:frac_W')
        self.prob.model.connect('comp:cool2:frac_P', 'DESIGN.comp.cool2:frac_P')
        self.prob.model.connect('comp:cool2:frac_work', 'DESIGN.comp.cool2:frac_work')
        self.prob.model.connect('turb:cool1:frac_P', 'DESIGN.turb.cool1:frac_P')
        self.prob.model.connect('turb:cool2:frac_P', 'DESIGN.turb.cool2:frac_P')

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        self.prob['DESIGN.balance.FAR'] = 0.0175506829934
        self.prob['DESIGN.balance.W'] = 168.453135137
        self.prob['DESIGN.balance.turb_PR'] = 4.46138725662
        self.prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        self.prob['DESIGN.fc.balance.Tt'] = 518.665288153

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 168.005
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.01755
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.4613
        pyc = self.prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 11800.0
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.79415
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1190.18
        pyc = self.prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()
