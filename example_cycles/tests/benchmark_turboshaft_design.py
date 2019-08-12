import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.units import convert_units as cu
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.Turboshaft import Turboshaft


class TurboshaftDesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('alt', 28000., units='ft'),
        des_vars.add_output('MN', 0.5),
        des_vars.add_output('T4max', 2740.0, units='degR'),
        des_vars.add_output('nozz_PR_des', 1.1)
        des_vars.add_output('inlet:ram_recovery', 1.0),
        des_vars.add_output('inlet:MN_out', 0.4),
        des_vars.add_output('duct1:dPqP', 0.0),
        des_vars.add_output('duct1:MN_out', 0.4),
        des_vars.add_output('lpc:PRdes', 5.000),
        des_vars.add_output('lpc:effDes', 0.8900),
        des_vars.add_output('lpc:MN_out', 0.3),
        des_vars.add_output('icduct:dPqP', 0.002),
        des_vars.add_output('icduct:MN_out', 0.3),
        des_vars.add_output('hpc_axi:PRdes', 3.0),
        des_vars.add_output('hpc_axi:effDes', 0.8900),
        des_vars.add_output('hpc_axi:MN_out', 0.25),
        des_vars.add_output('bld25:cool1:frac_W', 0.024),
        des_vars.add_output('bld25:cool2:frac_W', 0.0146),
        des_vars.add_output('bld25:MN_out', 0.3000),
        des_vars.add_output('hpc_centri:PRdes', 2.7),
        des_vars.add_output('hpc_centri:effDes', 0.8800),
        des_vars.add_output('hpc_centri:MN_out', 0.20),
        des_vars.add_output('bld3:cool3:frac_W', 0.1705),
        des_vars.add_output('bld3:cool4:frac_W', 0.1209),
        des_vars.add_output('bld3:MN_out', 0.2000),
        des_vars.add_output('duct6:dPqP', 0.00),
        des_vars.add_output('duct6:MN_out', 0.2000),
        des_vars.add_output('burner:dPqP', 0.050),
        des_vars.add_output('burner:MN_out', 0.15),
        des_vars.add_output('hpt:effDes', 0.89),
        des_vars.add_output('hpt:cool3:frac_P', 1.0),
        des_vars.add_output('hpt:cool4:frac_P', 0.0),
        des_vars.add_output('hpt:MN_out', 0.30),
        des_vars.add_output('duct43:dPqP', 0.0051),
        des_vars.add_output('duct43:MN_out', 0.30),
        des_vars.add_output('lpt:effDes', 0.9),
        des_vars.add_output('lpt:cool1:frac_P', 1.0),
        des_vars.add_output('lpt:cool2:frac_P', 0.0),
        des_vars.add_output('lpt:MN_out', 0.4),
        des_vars.add_output('itduct:dPqP', 0.00),
        des_vars.add_output('itduct:MN_out', 0.4),
        des_vars.add_output('pt:effDes', 0.85),
        des_vars.add_output('pt:MN_out', 0.4),
        des_vars.add_output('duct12:dPqP', 0.00),
        des_vars.add_output('duct12:MN_out', 0.4),
        des_vars.add_output('nozzle:Cv', 0.99),
        des_vars.add_output('lp_shaft:Nmech', 12750., units='rpm'),
        des_vars.add_output('lp_shaft:HPX', 1800.0, units='hp'),
        des_vars.add_output('ip_shaft:Nmech', 12000., units='rpm'),
        des_vars.add_output('hp_shaft:Nmech', 14800., units='rpm'),

        self.prob.model.add_subsystem('DESIGN', Turboshaft(statics=True))

        self.prob.model.connect('alt', 'DESIGN.fc.alt')
        self.prob.model.connect('MN', 'DESIGN.fc.MN')
        self.prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
        self.prob.model.connect('nozz_PR_des', 'DESIGN.balance.rhs:W')
        self.prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
        self.prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
        self.prob.model.connect('duct1:dPqP', 'DESIGN.duct1.dPqP')
        self.prob.model.connect('duct1:MN_out', 'DESIGN.duct1.MN')
        self.prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
        self.prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
        self.prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')
        self.prob.model.connect('icduct:dPqP', 'DESIGN.icduct.dPqP')
        self.prob.model.connect('icduct:MN_out', 'DESIGN.icduct.MN')
        self.prob.model.connect('hpc_axi:PRdes', 'DESIGN.hpc_axi.PR')
        self.prob.model.connect('hpc_axi:effDes', 'DESIGN.hpc_axi.eff')
        self.prob.model.connect('hpc_axi:MN_out', 'DESIGN.hpc_axi.MN')
        self.prob.model.connect('bld25:cool1:frac_W', 'DESIGN.bld25.cool1:frac_W')
        self.prob.model.connect('bld25:cool2:frac_W', 'DESIGN.bld25.cool2:frac_W')
        self.prob.model.connect('bld25:MN_out', 'DESIGN.bld25.MN')
        self.prob.model.connect('hpc_centri:PRdes', 'DESIGN.hpc_centri.PR')
        self.prob.model.connect('hpc_centri:effDes', 'DESIGN.hpc_centri.eff')
        self.prob.model.connect('hpc_centri:MN_out', 'DESIGN.hpc_centri.MN')
        self.prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')
        self.prob.model.connect('bld3:cool4:frac_W', 'DESIGN.bld3.cool4:frac_W')
        self.prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')
        self.prob.model.connect('duct6:dPqP', 'DESIGN.duct6.dPqP')
        self.prob.model.connect('duct6:MN_out', 'DESIGN.duct6.MN')
        self.prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
        self.prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
        self.prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
        self.prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
        self.prob.model.connect('hpt:cool4:frac_P', 'DESIGN.hpt.cool4:frac_P')
        self.prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')
        self.prob.model.connect('duct43:dPqP', 'DESIGN.duct43.dPqP')
        self.prob.model.connect('duct43:MN_out', 'DESIGN.duct43.MN')
        self.prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
        self.prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')
        self.prob.model.connect('lpt:cool2:frac_P', 'DESIGN.lpt.cool2:frac_P')
        self.prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')
        self.prob.model.connect('itduct:dPqP', 'DESIGN.itduct.dPqP')
        self.prob.model.connect('itduct:MN_out', 'DESIGN.itduct.MN')
        self.prob.model.connect('pt:effDes', 'DESIGN.pt.eff')
        self.prob.model.connect('pt:MN_out', 'DESIGN.pt.MN')
        self.prob.model.connect('duct12:dPqP', 'DESIGN.duct12.dPqP')
        self.prob.model.connect('duct12:MN_out', 'DESIGN.duct12.MN')
        self.prob.model.connect('nozzle:Cv', 'DESIGN.nozzle.Cv')
        self.prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
        self.prob.model.connect('lp_shaft:HPX', 'DESIGN.lp_shaft.HPX')
        self.prob.model.connect('ip_shaft:Nmech', 'DESIGN.IP_Nmech')
        self.prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        self.prob['DESIGN.balance.FAR'] = 0.02261
        self.prob['DESIGN.balance.W'] = 10.76
        self.prob['DESIGN.balance.pt_PR'] = 4.939
        self.prob['DESIGN.balance.lpt_PR'] = 1.979
        self.prob['DESIGN.balance.hpt_PR'] = 4.236
        self.prob['DESIGN.fc.balance.Pt'] = 5.666
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def test_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 10.774
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 40.419
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02135
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.2325
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1.9782
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.921
        pyc = self.prob['DESIGN.balance.pt_PR'][0]
        print('PT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.3758
        pyc = self.prob['DESIGN.nozzle.Fl_O:stat:MN'][0]
        print('Nozz MN:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.31342
        pyc = self.prob['DESIGN.perf.PSFC'][0]
        print('PSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1377.8
        pyc = self.prob['DESIGN.duct6.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
