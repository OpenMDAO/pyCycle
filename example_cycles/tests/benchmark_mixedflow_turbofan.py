import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.mixedflow_turbofan import MixedFlowTurbofan

class MixedFlowTurbofanTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        element_params = prob.model.add_subsystem('element_params', IndepVarComp(), promotes=["*"])

        des_vars.add_output('alt', 35000., units='ft') #DV
        des_vars.add_output('MN', 0.8) #DV
        des_vars.add_output('T4max', 3200, units='degR')
        des_vars.add_output('T4maxab', 3400, units='degR')
        des_vars.add_output('Fn_des', 5500.0, units='lbf')
        des_vars.add_output('Mix_ER', 1.05 ,units=None) # defined as 1 over 2
        des_vars.add_output('fan:PRdes', 3.3) #ADV
        des_vars.add_output('lpc:PRdes', 1.935)
        des_vars.add_output('hpc:PRdes', 4.9)


        element_params.add_output('inlet:ram_recovery', 0.9990)
        element_params.add_output('inlet:MN_out', 0.751)

        element_params.add_output('inlet_duct:dPqP', 0.0107)
        element_params.add_output('inlet_duct:MN_out', 0.4463)


        element_params.add_output('fan:effDes', 0.8948)
        element_params.add_output('fan:MN_out', 0.4578)

        #element_params.add_output('splitter:BPR', 5.105) not needed for mixed flow turbofan. balanced based on mixer total pressure ratio
        element_params.add_output('splitter:MN_out1', 0.3104)
        element_params.add_output('splitter:MN_out2', 0.4518)

        element_params.add_output('splitter_core_duct:dPqP', 0.0048)
        element_params.add_output('splitter_core_duct:MN_out', 0.3121)

        element_params.add_output('lpc:effDes', 0.9243)
        element_params.add_output('lpc:MN_out', 0.3059)

        element_params.add_output('lpc_duct:dPqP', 0.0101)
        element_params.add_output('lpc_duct:MN_out', 0.3563)


        element_params.add_output('hpc:effDes', 0.8707)
        element_params.add_output('hpc:MN_out', 0.2442)

        element_params.add_output('bld3:MN_out', 0.3000)

        element_params.add_output('burner:dPqP', 0.0540)
        element_params.add_output('burner:MN_out', 0.1025)

        element_params.add_output('hpt:effDes', 0.8888)
        element_params.add_output('hpt:MN_out', 0.3650)

        element_params.add_output('hpt_duct:dPqP', 0.0051)
        element_params.add_output('hpt_duct:MN_out', 0.3063)

        element_params.add_output('lpt:effDes', 0.8996)
        element_params.add_output('lpt:MN_out', 0.4127)

        element_params.add_output('lpt_duct:dPqP', 0.0107)
        element_params.add_output('lpt_duct:MN_out', 0.4463)

        element_params.add_output('bypass_duct:dPqP', 0.0107)
        element_params.add_output('bypass_duct:MN_out', 0.4463)

        # No params for mixer

        element_params.add_output('mixer_duct:dPqP', 0.0107)
        element_params.add_output('mixer_duct:MN_out', 0.4463)

        element_params.add_output('afterburner:dPqP', 0.0540)
        element_params.add_output('afterburner:MN_out', 0.1025)

        element_params.add_output('mixed_nozz:Cfg', 0.9933)

        element_params.add_output('lp_shaft:Nmech', 4666.1, units='rpm')
        element_params.add_output('hp_shaft:Nmech', 14705.7, units='rpm')
        element_params.add_output('hp_shaft:HPX', 250.0, units='hp')

        element_params.add_output('hpc:cool1:frac_W', 0.050708)
        element_params.add_output('hpc:cool1:frac_P', 0.5)
        element_params.add_output('hpc:cool1:frac_work', 0.5)

        element_params.add_output('bld3:cool3:frac_W', 0.067214)

        element_params.add_output('hpt:cool3:frac_P', 1.0)
        element_params.add_output('lpt:cool1:frac_P', 1.0)

        prob.model.add_subsystem('DESIGN', MixedFlowTurbofan(design=True))

        prob.model.connect('alt', 'DESIGN.fc.alt')
        prob.model.connect('MN', 'DESIGN.fc.MN')
        prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
        prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR_core')
        prob.model.connect('T4maxab', 'DESIGN.balance.rhs:FAR_ab')
        prob.model.connect('Mix_ER', 'DESIGN.balance.rhs:BPR')

        prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
        prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')

        prob.model.connect('inlet_duct:dPqP', 'DESIGN.inlet_duct.dPqP')
        prob.model.connect('inlet_duct:MN_out', 'DESIGN.inlet_duct.MN')

        prob.model.connect('fan:PRdes', 'DESIGN.fan.PR')
        prob.model.connect('fan:effDes', 'DESIGN.fan.eff')
        prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')

        #prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
        prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
        prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')

        prob.model.connect('splitter_core_duct:dPqP', 'DESIGN.splitter_core_duct.dPqP')
        prob.model.connect('splitter_core_duct:MN_out', 'DESIGN.splitter_core_duct.MN')

        prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
        prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
        prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')

        prob.model.connect('lpc_duct:dPqP', 'DESIGN.lpc_duct.dPqP')
        prob.model.connect('lpc_duct:MN_out', 'DESIGN.lpc_duct.MN')

        prob.model.connect('hpc:PRdes', 'DESIGN.hpc.PR')
        prob.model.connect('hpc:effDes', 'DESIGN.hpc.eff')
        prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')

        prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')

        prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
        prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')

        prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
        prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')

        prob.model.connect('hpt_duct:dPqP', 'DESIGN.hpt_duct.dPqP')
        prob.model.connect('hpt_duct:MN_out', 'DESIGN.hpt_duct.MN')

        prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
        prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')

        prob.model.connect('lpt_duct:dPqP', 'DESIGN.lpt_duct.dPqP')
        prob.model.connect('lpt_duct:MN_out', 'DESIGN.lpt_duct.MN')

        prob.model.connect('bypass_duct:dPqP', 'DESIGN.bypass_duct.dPqP')
        prob.model.connect('bypass_duct:MN_out', 'DESIGN.bypass_duct.MN')

        prob.model.connect('mixer_duct:dPqP', 'DESIGN.mixer_duct.dPqP')
        prob.model.connect('mixer_duct:MN_out', 'DESIGN.mixer_duct.MN')

        prob.model.connect('afterburner:dPqP', 'DESIGN.afterburner.dPqP')
        prob.model.connect('afterburner:MN_out', 'DESIGN.afterburner.MN')

        prob.model.connect('mixed_nozz:Cfg', 'DESIGN.mixed_nozz.Cfg')

        prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
        prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
        prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')

        prob.model.connect('hpc:cool1:frac_W', 'DESIGN.hpc.cool1:frac_W')
        prob.model.connect('hpc:cool1:frac_P', 'DESIGN.hpc.cool1:frac_P')
        prob.model.connect('hpc:cool1:frac_work', 'DESIGN.hpc.cool1:frac_work')

        prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')

        prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
        prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')


        ####################
        # OFF DESIGN CASES
        ####################
        self.od_pts = ['OD0',]
        # od_pts = []

        od_alts = [35000,]
        od_MNs = [0.8, ]

        des_vars.add_output('OD:alts', val=od_alts, units='ft')
        des_vars.add_output('OD:MNs', val=od_MNs)


        for i,pt in enumerate(self.od_pts):
            prob.model.add_subsystem(pt, MixedFlowTurbofan(design=False))

            prob.model.connect('OD:alts', pt+'.fc.alt', src_indices=[i,])
            prob.model.connect('OD:MNs', pt+'.fc.MN', src_indices=[i,])

            prob.model.connect('T4max', pt+'.balance.rhs:FAR_core')
            prob.model.connect('T4maxab', pt+'.balance.rhs:FAR_ab')

            prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
            prob.model.connect('mixed_nozz:Cfg', pt+'.mixed_nozz.Cfg')
            prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')


            # duct pressure losses
            prob.model.connect('inlet_duct:dPqP', pt+'.inlet_duct.dPqP')
            prob.model.connect('splitter_core_duct:dPqP', pt+'.splitter_core_duct.dPqP')
            prob.model.connect('bypass_duct:dPqP', pt+'.bypass_duct.dPqP')
            prob.model.connect('lpc_duct:dPqP', pt+'.lpc_duct.dPqP')
            prob.model.connect('hpt_duct:dPqP', pt+'.hpt_duct.dPqP')
            prob.model.connect('lpt_duct:dPqP', pt+'.lpt_duct.dPqP')
            prob.model.connect('mixer_duct:dPqP', pt+'.mixer_duct.dPqP')

            # burner pressure losses
            prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
            prob.model.connect('afterburner:dPqP', pt+'.afterburner.dPqP')

            # cooling flow fractions
            prob.model.connect('hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
            prob.model.connect('hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
            prob.model.connect('hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')
            prob.model.connect('bld3:cool3:frac_W', pt+'.bld3.cool3:frac_W')
            prob.model.connect('hpt:cool3:frac_P', pt+'.hpt.cool3:frac_P')
            prob.model.connect('lpt:cool1:frac_P', pt+'.lpt.cool1:frac_P')

            # map scalars
            prob.model.connect('DESIGN.fan.s_PR', pt+'.fan.s_PR')
            prob.model.connect('DESIGN.fan.s_Wc', pt+'.fan.s_Wc')
            prob.model.connect('DESIGN.fan.s_eff', pt+'.fan.s_eff')
            prob.model.connect('DESIGN.fan.s_Nc', pt+'.fan.s_Nc')
            prob.model.connect('DESIGN.lpc.s_PR', pt+'.lpc.s_PR')
            prob.model.connect('DESIGN.lpc.s_Wc', pt+'.lpc.s_Wc')
            prob.model.connect('DESIGN.lpc.s_eff', pt+'.lpc.s_eff')
            prob.model.connect('DESIGN.lpc.s_Nc', pt+'.lpc.s_Nc')
            prob.model.connect('DESIGN.hpc.s_PR', pt+'.hpc.s_PR')
            prob.model.connect('DESIGN.hpc.s_Wc', pt+'.hpc.s_Wc')
            prob.model.connect('DESIGN.hpc.s_eff', pt+'.hpc.s_eff')
            prob.model.connect('DESIGN.hpc.s_Nc', pt+'.hpc.s_Nc')
            prob.model.connect('DESIGN.hpt.s_PR', pt+'.hpt.s_PR')
            prob.model.connect('DESIGN.hpt.s_Wp', pt+'.hpt.s_Wp')
            prob.model.connect('DESIGN.hpt.s_eff', pt+'.hpt.s_eff')
            prob.model.connect('DESIGN.hpt.s_Np', pt+'.hpt.s_Np')
            prob.model.connect('DESIGN.lpt.s_PR', pt+'.lpt.s_PR')
            prob.model.connect('DESIGN.lpt.s_Wp', pt+'.lpt.s_Wp')
            prob.model.connect('DESIGN.lpt.s_eff', pt+'.lpt.s_eff')
            prob.model.connect('DESIGN.lpt.s_Np', pt+'.lpt.s_Np')

            # flow areas
            prob.model.connect('DESIGN.mixed_nozz.Throat:stat:area', pt+'.balance.rhs:W')

            prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
            prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
            prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
            prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
            prob.model.connect('DESIGN.splitter_core_duct.Fl_O:stat:area', pt+'.splitter_core_duct.area')
            prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
            prob.model.connect('DESIGN.lpc_duct.Fl_O:stat:area', pt+'.lpc_duct.area')
            prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
            prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
            prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
            prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
            prob.model.connect('DESIGN.hpt_duct.Fl_O:stat:area', pt+'.hpt_duct.area')
            prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
            prob.model.connect('DESIGN.lpt_duct.Fl_O:stat:area', pt+'.lpt_duct.area')
            prob.model.connect('DESIGN.bypass_duct.Fl_O:stat:area', pt+'.bypass_duct.area')
            prob.model.connect('DESIGN.mixer.Fl_O:stat:area', pt+'.mixer.area')
            prob.model.connect('DESIGN.mixer.Fl_I1_calc:stat:area', pt+'.mixer.Fl_I1_stat_calc.area')
            prob.model.connect('DESIGN.mixer_duct.Fl_O:stat:area', pt+'.mixer_duct.area')
            prob.model.connect('DESIGN.afterburner.Fl_O:stat:area', pt+'.afterburner.area')


    def benchmark_run_des(self):
        ''' Runs the design point and an off design point to make sure they match perfectly '''
        prob = self.prob

        # setup problem
        prob.setup(check=False)#True)

        # initial guesses
        prob['DESIGN.balance.FAR_core'] = 0.025
        prob['DESIGN.balance.FAR_ab'] = 0.025
        prob['DESIGN.balance.BPR'] = 1.0
        prob['DESIGN.balance.W'] = 100.
        prob['DESIGN.balance.lpt_PR'] = 3.5
        prob['DESIGN.balance.hpt_PR'] = 2.5
        prob['DESIGN.fc.balance.Pt'] = 5.2
        prob['DESIGN.fc.balance.Tt'] = 440.0
        prob['DESIGN.mixer.balance.P_tot']=100

        for pt in self.od_pts:
            prob[pt+'.balance.FAR_core'] = 0.031
            prob[pt+'.balance.FAR_ab'] = 0.038
            prob[pt+'.balance.BPR'] = 2.2
            prob[pt+'.balance.W'] = 60

            # really sensitive to these initial guesses
            prob[pt+'.balance.HP_Nmech'] = 15000
            prob[pt+'.balance.LP_Nmech'] = 5000

            prob[pt+'.fc.balance.Pt'] = 5.2
            prob[pt+'.fc.balance.Tt'] = 440.0
            prob[pt+'.mixer.balance.P_tot']= 100
            prob[pt+'.hpt.PR'] = 2.5
            prob[pt+'.lpt.PR'] = 3.5
            prob[pt+'.fan.map.RlineMap'] = 2.0
            prob[pt+'.lpc.map.RlineMap'] = 2.0
            prob[pt+'.hpc.map.RlineMap'] = 2.0

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

        prob.run_model()

        tol = 1e-5

        reg_data = 53.83467876114857
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.inlet.Fl_O:stat:W'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.0311108
        pyc = self.prob['DESIGN.balance.FAR_core'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.balance.FAR_core'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.038716210473225536
        pyc = self.prob['DESIGN.balance.FAR_ab'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.balance.FAR_ab'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 2.0430265465465354
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.hpt.PR'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.098132533864145
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.lpt.PR'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 6802.813118292415
        pyc = self.prob['DESIGN.mixed_nozz.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.mixed_nozz.Fg'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1287.084732
        pyc = self.prob['DESIGN.hpc.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.hpc.Fl_O:tot:T'][0]
        assert_rel_error(self, pyc, reg_data, tol)

if __name__ == "__main__":
    unittest.main()

