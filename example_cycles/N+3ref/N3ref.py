from __future__ import print_function

import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

from small_core_eff_balance import SmallCoreEffBalance

from N3_Fan_map import FanMap
from N3_LPC_map import LPCMap
from N3_HPC_map import HPCMap
from N3_HPT_map import HPTMap
from N3_LPT_map import LPTMap


class N3(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('cooling', default=False,
                              desc='If True, calculate cooling flow values.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']
        cooling = self.options['cooling']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('fan', pyc.Compressor(map_data=FanMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True,
                                                 bleed_names=[]),
                           promotes_inputs=[('Nmech','Fan_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('duct2', pyc.Duct(design=design, expMN=2.0, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True),
                            promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('bld25', pyc.BleedOut(design=design, bleed_names=['sbv']))
        self.add_subsystem('duct25', pyc.Duct(design=design, expMN=2.0, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc', pyc.Compressor(map_data=HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True,
                                        bleed_names=['bld_inlet','bld_exit','cust']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['bld_inlet','bld_exit']))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, map_extrap=True,
                                              bleed_names=['bld_inlet','bld_exit']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('duct45', pyc.Duct(design=design, expMN=2.0, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('lpt', pyc.Turbine(map_data=LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, map_extrap=True,
                                              bleed_names=['bld_inlet','bld_exit']),
                           promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct5', pyc.Duct(design=design, expMN=2.0, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        self.add_subsystem('byp_bld', pyc.BleedOut(design=design, bleed_names=['bypBld']))
        self.add_subsystem('duct17', pyc.Duct(design=design, expMN=2.0, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_MIX))

        self.add_subsystem('fan_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech','Fan_Nmech')])
        self.add_subsystem('gearbox', pyc.Gearbox(design=design), promotes_inputs=[('N_in','LP_Nmech'), ('N_out','Fan_Nmech')])
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        self.connect('fan.trq', 'fan_shaft.trq_0')
        self.connect('gearbox.trq_out', 'fan_shaft.trq_1')
        self.connect('gearbox.trq_in', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        self.add_subsystem('ext_ratio', om.ExecComp('ER = core_V_ideal * core_Cv / ( byp_V_ideal *  byp_Cv )',
                        core_V_ideal={'value':1000.0, 'units':'ft/s'},
                        core_Cv={'value':0.98, 'units':None},
                        byp_V_ideal={'value':1000.0, 'units':'ft/s'},
                        byp_Cv={'value':0.98, 'units':None},
                        ER={'value':1.4, 'units':None}))

        self.connect('core_nozz.ideal_flow.V', 'ext_ratio.core_V_ideal')
        self.connect('byp_nozz.ideal_flow.V', 'ext_ratio.byp_V_ideal')


        main_order = ['fc', 'inlet', 'fan', 'splitter', 'duct2', 'lpc', 'bld25', 'duct25', 'hpc', 'bld3', 'burner', 'hpt', 'duct45',
                            'lpt', 'duct5', 'core_nozz', 'byp_bld', 'duct17', 'byp_nozz', 'gearbox', 'fan_shaft', 'lp_shaft', 'hp_shaft',
                            'perf', 'ext_ratio']



        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('lpt_PR', val=10.937, lower=1.001, upper=20, eq_units='hp', rhs_val=0., res_ref=1e4)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=4.185, lower=1.001, upper=8, eq_units='hp', rhs_val=0., res_ref=1e4)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

            balance.add_balance('gb_trq', val=23928.0, units='ft*lbf', eq_units='hp', rhs_val=0.0)
            self.connect('balance.gb_trq', 'gearbox.trq_base')
            self.connect('fan_shaft.pwr_net', 'balance.lhs:gb_trq')

            balance.add_balance('hpc_PR', val=14.0, units=None, eq_units=None)
            self.connect('balance.hpc_PR', ['hpc.PR', 'opr_calc.HPCPR'])
            # self.connect('perf.OPR', 'balance.lhs:hpc_PR')
            self.connect('opr_calc.OPR_simple', 'balance.lhs:hpc_PR')

            balance.add_balance('fan_eff', val=0.9689, units=None, eq_units=None)
            self.connect('balance.fan_eff', 'fan.eff')
            self.connect('fan.eff_poly', 'balance.lhs:fan_eff')

            balance.add_balance('lpc_eff', val=0.8895, units=None, eq_units=None)
            self.connect('balance.lpc_eff', 'lpc.eff')
            self.connect('lpc.eff_poly', 'balance.lhs:lpc_eff')

            # balance.add_balance('hpc_eff', val=0.8470, units=None, eq_units=None)
            # self.connect('balance.hpc_eff', 'hpc.eff')
            # self.connect('hpc.eff_poly', 'balance.lhs:hpc_eff')

            balance.add_balance('hpt_eff', val=0.9226, units=None, eq_units=None)
            self.connect('balance.hpt_eff', 'hpt.eff')
            self.connect('hpt.eff_poly', 'balance.lhs:hpt_eff')

            balance.add_balance('lpt_eff', val=0.9401, units=None, eq_units=None)
            self.connect('balance.lpt_eff', 'lpt.eff')
            self.connect('lpt.eff_poly', 'balance.lhs:lpt_eff')

            self.add_subsystem('hpc_CS',
                    om.ExecComp('CS = Win *(pow(Tout/518.67,0.5)/(Pout/14.696))',
                            Win= {'value': 10.0, 'units':'lbm/s'},
                            Tout={'value': 14.696, 'units':'degR'},
                            Pout={'value': 518.67, 'units':'psi'},
                            CS={'value': 10.0, 'units':'lbm/s'}))
            self.connect('duct25.Fl_O:stat:W', 'hpc_CS.Win')
            self.connect('hpc.Fl_O:tot:T', 'hpc_CS.Tout')
            self.connect('hpc.Fl_O:tot:P', 'hpc_CS.Pout')
            self.add_subsystem('hpc_EtaBalance', SmallCoreEffBalance(eng_type = 'large', tech_level = 0))
            self.connect('hpc_CS.CS','hpc_EtaBalance.CS')
            self.connect('hpc.eff_poly', 'hpc_EtaBalance.eta_p')
            self.connect('hpc_EtaBalance.eta_a', 'hpc.eff')

            self.add_subsystem('fan_dia', om.ExecComp('FanDia = 2.0*(area/(pi*(1.0-hub_tip**2.0)))**0.5',
                            area={'value':7000.0, 'units':'inch**2'},
                            hub_tip={'value':0.3125, 'units':None},
                            FanDia={'value':100.0, 'units':'inch'}))
            self.connect('inlet.Fl_O:stat:area', 'fan_dia.area')

            self.add_subsystem('opr_calc', om.ExecComp('OPR_simple = FPR*LPCPR*HPCPR',
                            FPR={'value':1.3, 'units':None},
                            LPCPR={'value':3.0, 'units':None},
                            HPCPR={'value':14.0, 'units':None},
                            OPR_simple={'value':55.0, 'units':None}))


            # order_add = ['hpc_CS', 'fan_dia', 'opr_calc']
            order_add = ['hpc_CS', 'hpc_EtaBalance', 'fan_dia', 'opr_calc']

        else:

            balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')
            # self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', lower=10., upper=2500., eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=15., upper=40.)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('fan.map.RlineMap', 'balance.lhs:BPR')

            balance.add_balance('fan_Nmech', val=2000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.fan_Nmech', 'Fan_Nmech')
            self.connect('fan_shaft.pwr_net', 'balance.lhs:fan_Nmech')

            balance.add_balance('lp_Nmech', val=6000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=20000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hp_Nmech')

            order_add = []

        if cooling:
            self.add_subsystem('hpt_cooling', pyc.TurbineCooling(n_stages=2, thermo_data=pyc.species_data.janaf, T_metal=2460.))
            self.add_subsystem('hpt_chargable', pyc.CombineCooling(n_ins=3))

            pyc.connect_flow(self, 'bld3.bld_inlet', 'hpt_cooling.Fl_cool', connect_stat=False)
            pyc.connect_flow(self, 'burner.Fl_O', 'hpt_cooling.Fl_turb_I')
            pyc.connect_flow(self, 'hpt.Fl_O', 'hpt_cooling.Fl_turb_O')

            self.connect('hpt_cooling.row_1.W_cool', 'hpt_chargable.W_1')
            self.connect('hpt_cooling.row_2.W_cool', 'hpt_chargable.W_2')
            self.connect('hpt_cooling.row_3.W_cool', 'hpt_chargable.W_3')
            self.connect('hpt.power', 'hpt_cooling.turb_pwr')

            balance.add_balance('hpt_nochrg_cool_frac', val=0.063660111, lower=0.02, upper=.15, eq_units='lbm/s')
            self.connect('balance.hpt_nochrg_cool_frac', 'bld3.bld_inlet:frac_W')
            self.connect('bld3.bld_inlet:stat:W', 'balance.lhs:hpt_nochrg_cool_frac')
            self.connect('hpt_cooling.row_0.W_cool', 'balance.rhs:hpt_nochrg_cool_frac')

            balance.add_balance('hpt_chrg_cool_frac', val=0.07037185, lower=0.02, upper=.15, eq_units='lbm/s')
            self.connect('balance.hpt_chrg_cool_frac', 'bld3.bld_exit:frac_W')
            self.connect('bld3.bld_exit:stat:W', 'balance.lhs:hpt_chrg_cool_frac')
            self.connect('hpt_chargable.W_cool', 'balance.rhs:hpt_chrg_cool_frac')



            order_add = ['hpt_cooling', 'hpt_chargable']



        self.set_order(main_order + order_add + ['balance'])

        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        pyc.connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        pyc.connect_flow(self, 'fan.Fl_O', 'splitter.Fl_I')
        pyc.connect_flow(self, 'splitter.Fl_O1', 'duct2.Fl_I')
        pyc.connect_flow(self, 'duct2.Fl_O', 'lpc.Fl_I')
        pyc.connect_flow(self, 'lpc.Fl_O', 'bld25.Fl_I')
        pyc.connect_flow(self, 'bld25.Fl_O', 'duct25.Fl_I')
        pyc.connect_flow(self, 'duct25.Fl_O', 'hpc.Fl_I')
        pyc.connect_flow(self, 'hpc.Fl_O', 'bld3.Fl_I')
        pyc.connect_flow(self, 'bld3.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        pyc.connect_flow(self, 'hpt.Fl_O', 'duct45.Fl_I')
        pyc.connect_flow(self, 'duct45.Fl_O', 'lpt.Fl_I')
        pyc.connect_flow(self, 'lpt.Fl_O', 'duct5.Fl_I')
        pyc.connect_flow(self, 'duct5.Fl_O','core_nozz.Fl_I')
        pyc.connect_flow(self, 'splitter.Fl_O2', 'byp_bld.Fl_I')
        pyc.connect_flow(self, 'byp_bld.Fl_O', 'duct17.Fl_I')
        pyc.connect_flow(self, 'duct17.Fl_O', 'byp_nozz.Fl_I')

        pyc.connect_flow(self, 'hpc.bld_inlet', 'lpt.bld_inlet', connect_stat=False)
        pyc.connect_flow(self, 'hpc.bld_exit', 'lpt.bld_exit', connect_stat=False)
        pyc.connect_flow(self, 'bld3.bld_inlet', 'hpt.bld_inlet', connect_stat=False)
        pyc.connect_flow(self, 'bld3.bld_exit', 'hpt.bld_exit', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-4
        newton.options['rtol'] = 1e-4
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['reraise_child_analysiserror'] = False
        # newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['maxiter'] = 2
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        # if design:
        #     newton.linesearch.options['print_bound_enforce'] = True

        # newton.options['debug_print'] = True

        self.linear_solver = om.DirectSolver(assemble_jac=True)

def viewer(prob, pt, file=sys.stdout): 
    """
    print a report of all the relevant cycle properties
    """

    # if pt == 'DESIGN':
    #     MN = prob['DESIGN.fc.Fl_O:stat:MN']
    #     LPT_PR = prob['DESIGN.balance.lpt_PR']
    #     HPT_PR = prob['DESIGN.balance.hpt_PR']
    #     FAR = prob['DESIGN.balance.FAR']
    # else:
    #     MN = prob[pt+'.fc.Fl_O:stat:MN']
    #     LPT_PR = prob[pt+'.lpt.PR']
    #     HPT_PR = prob[pt+'.hpt.PR']
    #     FAR = prob[pt+'.balance.FAR']

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']), file=file, flush=True)
    
    fs_names = ['fc.Fl_O','inlet.Fl_O','fan.Fl_O','splitter.Fl_O1','duct2.Fl_O',
                'lpc.Fl_O','bld25.Fl_O','duct25.Fl_O','hpc.Fl_O','bld3.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct45.Fl_O','lpt.Fl_O','duct5.Fl_O',
                'core_nozz.Fl_O','splitter.Fl_O2','byp_bld.Fl_O','duct17.Fl_O',
                'byp_nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['fan', 'lpc', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['core_nozz', 'byp_nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft', 'lp_shaft', 'fan_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['hpc', 'bld3','bld3','bld25']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)




if __name__ == "__main__":

    import time

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    # FOR DESIGN
    des_vars.add_output('alt', 35000., units='ft'),
    des_vars.add_output('MN', 0.8),
    des_vars.add_output('T4max', 3150.0, units='degR'),
    # des_vars.add_output('FAR', 0.02833)
    des_vars.add_output('Fn_des', 6073.4, units='lbf'),

    des_vars.add_output('inlet:ram_recovery', 0.9980),
    des_vars.add_output('fan:PRdes', 1.300),
    des_vars.add_output('fan:effDes', 0.9689),
    des_vars.add_output('splitter:BPR', 23.8572), #23.9878
    des_vars.add_output('duct2:dPqP', 0.0100),
    des_vars.add_output('lpc:PRdes', 3.000),
    des_vars.add_output('lpc:effDes', 0.8894),
    des_vars.add_output('duct25:dPqP', 0.0150),
    des_vars.add_output('hpc:PRdes', 14.103),
    des_vars.add_output('hpc:effDes', 0.8469),
    des_vars.add_output('burner:dPqP', 0.0400),
    des_vars.add_output('hpt:effDes', 0.9313),
    des_vars.add_output('duct45:dPqP', 0.0050),
    des_vars.add_output('lpt:effDes', 0.9410),
    des_vars.add_output('duct5:dPqP', 0.0100),
    des_vars.add_output('core_nozz:Cv', 0.9999),
    des_vars.add_output('duct17:dPqP', 0.0150),
    des_vars.add_output('byp_nozz:Cv', 0.9975),
    des_vars.add_output('fan_shaft:Nmech', 2184.5, units='rpm'),
    des_vars.add_output('lp_shaft:Nmech', 6772.0, units='rpm'),
    des_vars.add_output('lp_shaft:fracLoss', 0.01)
    des_vars.add_output('hp_shaft:Nmech', 20871.0, units='rpm'),
    des_vars.add_output('hp_shaft:HPX', 350.0, units='hp'),


    des_vars.add_output('bld25:sbv:frac_W', 0.0),
    des_vars.add_output('hpc:bld_inlet:frac_W', 0.0),
    des_vars.add_output('hpc:bld_inlet:frac_P', 0.1465),
    des_vars.add_output('hpc:bld_inlet:frac_work', 0.5),
    des_vars.add_output('hpc:bld_exit:frac_W', 0.02),
    des_vars.add_output('hpc:bld_exit:frac_P', 0.1465),
    des_vars.add_output('hpc:bld_exit:frac_work', 0.5),
    des_vars.add_output('hpc:cust:frac_W', 0.0),
    des_vars.add_output('hpc:cust:frac_P', 0.1465),
    des_vars.add_output('hpc:cust:frac_work', 0.35),
    des_vars.add_output('bld3:bld_inlet:frac_W', 0.0625),
    des_vars.add_output('bld3:bld_exit:frac_W', 0.0693),
    des_vars.add_output('hpt:bld_inlet:frac_P', 1.0),
    des_vars.add_output('hpt:bld_exit:frac_P', 0.0),
    des_vars.add_output('lpt:bld_inlet:frac_P', 1.0),
    des_vars.add_output('lpt:bld_exit:frac_P', 0.0),
    des_vars.add_output('bypBld:frac_W', 0.0),

    des_vars.add_output('inlet:MN_out', 0.625),
    des_vars.add_output('fan:MN_out', 0.45)
    des_vars.add_output('splitter:MN_out1', 0.45)
    des_vars.add_output('splitter:MN_out2', 0.45)
    des_vars.add_output('duct2:MN_out', 0.45),
    des_vars.add_output('lpc:MN_out', 0.45),
    des_vars.add_output('bld25:MN_out', 0.45),
    des_vars.add_output('duct25:MN_out', 0.45),
    des_vars.add_output('hpc:MN_out', 0.30),
    des_vars.add_output('bld3:MN_out', 0.30)
    des_vars.add_output('burner:MN_out', 0.10),
    des_vars.add_output('hpt:MN_out', 0.30),
    des_vars.add_output('duct45:MN_out', 0.45),
    des_vars.add_output('lpt:MN_out', 0.35),
    des_vars.add_output('duct5:MN_out', 0.25),
    des_vars.add_output('bypBld:MN_out', 0.45),
    des_vars.add_output('duct17:MN_out', 0.45),


    # OFF DESIGN 1
    des_vars.add_output('OD1_MN', 0.8),
    des_vars.add_output('OD1_alt', 35000.0, units='ft'),
    des_vars.add_output('OD1_Fn_target', 6073.4, units='lbf'), #8950.0
    des_vars.add_output('OD1_dTs', 0.0, units='degR')
    # des_vars.add_output('OD1_cust_fracW', 0.0)

    # # OFF DESIGN 2
    # des_vars.add_output('OD2_MN', 0.8),
    # des_vars.add_output('OD2_alt', 35000.0, units='ft'),
    # des_vars.add_output('OD2_Fn_target', 5970.0, units='lbf'),
    # des_vars.add_output('OD2_dTs', 0.0, units='degR')
    # des_vars.add_output('OD2_cust_fracW', 0.0422)

    # # OFF DESIGN 3
    # des_vars.add_output('OD3_MN', 0.25),
    # des_vars.add_output('OD3_alt', 0.0, units='ft'),
    # des_vars.add_output('OD3_Fn_target', 22590.0, units='lbf'),
    # des_vars.add_output('OD3_dTs', 27.0, units='degR')
    # des_vars.add_output('OD3_cust_fracW', 0.0177)

    # # OFF DESIGN 4
    # des_vars.add_output('OD4_MN', 0.00001),
    # des_vars.add_output('OD4_alt', 0.0, units='ft'),
    # des_vars.add_output('OD4_Fn_target', 27113.0, units='lbf'),
    # des_vars.add_output('OD4_dTs', 27.0, units='degR')
    # des_vars.add_output('OD4_cust_fracW', 0.0185)

    # DESIGN CASE
    prob.model.add_subsystem('DESIGN', N3())

    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
    # prob.model.connect('FAR','DESIGN.burner.Fl_I:FAR')

    prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
    prob.model.connect('fan:PRdes', 'DESIGN.fan.PR')
    prob.model.connect('fan:effDes', 'DESIGN.fan.eff')
    prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
    prob.model.connect('duct2:dPqP', 'DESIGN.duct2.dPqP')
    prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
    prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
    prob.model.connect('duct25:dPqP', 'DESIGN.duct25.dPqP')
    prob.model.connect('hpc:PRdes', 'DESIGN.hpc.PR')
    prob.model.connect('hpc:effDes', 'DESIGN.hpc.eff')
    prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
    prob.model.connect('duct45:dPqP', 'DESIGN.duct45.dPqP')
    prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
    prob.model.connect('duct5:dPqP', 'DESIGN.duct5.dPqP')
    prob.model.connect('core_nozz:Cv', 'DESIGN.core_nozz.Cv')
    prob.model.connect('duct17:dPqP', 'DESIGN.duct17.dPqP')
    prob.model.connect('byp_nozz:Cv', 'DESIGN.byp_nozz.Cv')
    prob.model.connect('fan_shaft:Nmech', 'DESIGN.Fan_Nmech')
    prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
    prob.model.connect('lp_shaft:fracLoss', 'DESIGN.lp_shaft.fracLoss')
    prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
    prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')

    prob.model.connect('bld25:sbv:frac_W', 'DESIGN.bld25.sbv:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_W', 'DESIGN.hpc.bld_inlet:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_P', 'DESIGN.hpc.bld_inlet:frac_P')
    prob.model.connect('hpc:bld_inlet:frac_work', 'DESIGN.hpc.bld_inlet:frac_work')
    prob.model.connect('hpc:bld_exit:frac_W', 'DESIGN.hpc.bld_exit:frac_W')
    prob.model.connect('hpc:bld_exit:frac_P', 'DESIGN.hpc.bld_exit:frac_P')
    prob.model.connect('hpc:bld_exit:frac_work', 'DESIGN.hpc.bld_exit:frac_work')
    prob.model.connect('bld3:bld_inlet:frac_W', 'DESIGN.bld3.bld_inlet:frac_W')
    prob.model.connect('bld3:bld_exit:frac_W', 'DESIGN.bld3.bld_exit:frac_W')
    prob.model.connect('hpc:cust:frac_W', 'DESIGN.hpc.cust:frac_W')
    prob.model.connect('hpc:cust:frac_P', 'DESIGN.hpc.cust:frac_P')
    prob.model.connect('hpc:cust:frac_work', 'DESIGN.hpc.cust:frac_work')
    prob.model.connect('hpt:bld_inlet:frac_P', 'DESIGN.hpt.bld_inlet:frac_P')
    prob.model.connect('hpt:bld_exit:frac_P', 'DESIGN.hpt.bld_exit:frac_P')
    prob.model.connect('lpt:bld_inlet:frac_P', 'DESIGN.lpt.bld_inlet:frac_P')
    prob.model.connect('lpt:bld_exit:frac_P', 'DESIGN.lpt.bld_exit:frac_P')
    prob.model.connect('bypBld:frac_W', 'DESIGN.byp_bld.bypBld:frac_W')

    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
    prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')
    prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
    prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')
    prob.model.connect('duct2:MN_out', 'DESIGN.duct2.MN')
    prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')
    prob.model.connect('bld25:MN_out', 'DESIGN.bld25.MN')
    prob.model.connect('duct25:MN_out', 'DESIGN.duct25.MN')
    prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')
    prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
    prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')
    prob.model.connect('duct45:MN_out', 'DESIGN.duct45.MN')
    prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')
    prob.model.connect('duct5:MN_out', 'DESIGN.duct5.MN')
    prob.model.connect('bypBld:MN_out', 'DESIGN.byp_bld.MN')
    prob.model.connect('duct17:MN_out', 'DESIGN.duct17.MN')


    # OFF DESIGN CASES
    pts = [] # ['OD1','OD2','OD3','OD4']
    OD_statics = True

    for pt in pts:
        ODpt = prob.model.add_subsystem(pt, N3(design=False, statics=OD_statics))
        ODpt.nonlinear_solver.options['maxiter'] = 0

        prob.model.connect(pt+'_alt', pt+'.fc.alt')
        prob.model.connect(pt+'_MN', pt+'.fc.MN')
        prob.model.connect(pt+'_Fn_target', pt+'.balance.rhs:FAR')
        prob.model.connect(pt+'_dTs', pt+'.fc.dTs')
        # prob.model.connect(pt+'_cust_fracW', pt+'.hpc.cust:frac_W')

        prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
        # prob.model.connect('splitter:BPR', pt+'.splitter.BPR')
        prob.model.connect('duct2:dPqP', pt+'.duct2.dPqP')
        prob.model.connect('duct25:dPqP', pt+'.duct25.dPqP')
        prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
        prob.model.connect('duct45:dPqP', pt+'.duct45.dPqP')
        prob.model.connect('duct5:dPqP', pt+'.duct5.dPqP')
        prob.model.connect('core_nozz:Cv', pt+'.core_nozz.Cv')
        prob.model.connect('duct17:dPqP', pt+'.duct17.dPqP')
        prob.model.connect('byp_nozz:Cv', pt+'.byp_nozz.Cv')
        prob.model.connect('lp_shaft:fracLoss', pt+'.lp_shaft.fracLoss')
        prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

        prob.model.connect('bld25:sbv:frac_W', pt+'.bld25.sbv:frac_W')
        prob.model.connect('hpc:bld_inlet:frac_W', pt+'.hpc.bld_inlet:frac_W')
        prob.model.connect('hpc:bld_inlet:frac_P', pt+'.hpc.bld_inlet:frac_P')
        prob.model.connect('hpc:bld_inlet:frac_work', pt+'.hpc.bld_inlet:frac_work')
        prob.model.connect('hpc:bld_exit:frac_W', pt+'.hpc.bld_exit:frac_W')
        prob.model.connect('hpc:bld_exit:frac_P', pt+'.hpc.bld_exit:frac_P')
        prob.model.connect('hpc:bld_exit:frac_work', pt+'.hpc.bld_exit:frac_work')
        prob.model.connect('bld3:bld_inlet:frac_W', pt+'.bld3.bld_inlet:frac_W')
        prob.model.connect('bld3:bld_exit:frac_W', pt+'.bld3.bld_exit:frac_W')
        prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
        prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
        prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
        prob.model.connect('hpt:bld_inlet:frac_P', pt+'.hpt.bld_inlet:frac_P')
        prob.model.connect('hpt:bld_exit:frac_P', pt+'.hpt.bld_exit:frac_P')
        prob.model.connect('lpt:bld_inlet:frac_P', pt+'.lpt.bld_inlet:frac_P')
        prob.model.connect('lpt:bld_exit:frac_P', pt+'.lpt.bld_exit:frac_P')
        prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

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

        prob.model.connect('DESIGN.gearbox.gear_ratio', pt+'.gearbox.gear_ratio')
        prob.model.connect('DESIGN.core_nozz.Throat:stat:area',pt+'.balance.rhs:W')
        prob.model.connect('DESIGN.byp_nozz.Throat:stat:area',pt+'.balance.rhs:BPR')

        if OD_statics:
            prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
            prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
            prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
            prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
            prob.model.connect('DESIGN.duct2.Fl_O:stat:area', pt+'.duct2.area')
            prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
            prob.model.connect('DESIGN.bld25.Fl_O:stat:area', pt+'.bld25.area')
            prob.model.connect('DESIGN.duct25.Fl_O:stat:area', pt+'.duct25.area')
            prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
            prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
            prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
            prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
            prob.model.connect('DESIGN.duct45.Fl_O:stat:area', pt+'.duct45.area')
            prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
            prob.model.connect('DESIGN.duct5.Fl_O:stat:area', pt+'.duct5.area')
            prob.model.connect('DESIGN.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
            prob.model.connect('DESIGN.duct17.Fl_O:stat:area', pt+'.duct17.area')

    prob.setup(check=False)

    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.02833
    prob['DESIGN.balance.W'] = 813.5
    prob['DESIGN.balance.lpt_PR'] = 7.9474
    prob['DESIGN.balance.hpt_PR'] = 4.1006
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0

    for pt in pts:
        # ADP and TOC guesses
        prob[pt+'.balance.FAR'] = 0.02650
        prob[pt+'.balance.W'] = 813.437
        prob[pt+'.balance.BPR'] = 23.915
        prob[pt+'.balance.fan_Nmech'] = 2184.5
        prob[pt+'.balance.lp_Nmech'] = 6772.0
        prob[pt+'.balance.hp_Nmech'] = 20871.0
        prob[pt+'.fc.balance.Pt'] = 5.2
        prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.hpt.PR'] = 4.1120
        prob[pt+'.lpt.PR'] = 11.0696
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.2
        prob[pt+'.hpc.map.RlineMap'] = 2.0
        prob[pt+'.gearbox.trq_base'] = 23708.32365987

    #     # RTO guesses
    #     # prob[pt+'.thrust_balance.indep'] = 0.03165
    #     # prob[pt+'.core_flow_balance.indep'] = 810.83
    #     # prob[pt+'.byp_flow_balance.indep'] = 5.1053
    #     # prob[pt+'.lp_shaft_balance.indep'] = 4975.9
    #     # prob[pt+'.hp_shaft_balance.indep'] = 16230.1
    #     # prob[pt+'.fc.balance.Pt'] = 15.349
    #     # prob[pt+'.fc.balance.Tt'] = 552.49
    #     # prob[pt+'.hpt.PR'] = 3.591
    #     # prob[pt+'.lpt.PR'] = 4.173
    #     # prob[pt+'.fan.map.RlineMap'] = 2.0
    #     # prob[pt+'.lpc.map.RlineMap'] = 2.0
    #     # prob[pt+'.hpc.map.RlineMap'] = 2.0

    #     # SLS guesses
    #     prob[pt+'.balance.FAR'] = 0.03114
    #     prob[pt+'.balance.W'] = 771.34
    #     prob[pt+'.balance.BPR'] = 5.0805
    #     prob[pt+'.balance.lp_Nmech'] = 4912.7
    #     prob[pt+'.balance.hp_Nmech'] = 16106.9
    #     prob[pt+'.fc.balance.Pt'] = 14.696
    #     prob[pt+'.fc.balance.Tt'] = 545.67
    #     prob[pt+'.hpt.PR'] = 3.595
    #     prob[pt+'.lpt.PR'] = 4.147
    #     prob[pt+'.fan.map.RlineMap'] = 2.0
    #     prob[pt+'.lpc.map.RlineMap'] = 2.0
    #     prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    # prob.model.OD1.nonlinear_solver.options['maxiter']=1


    # from openmdao.api import view_model
    # view_model(prob)
    # exit()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()
    # prob.check_partials(comps=['OD1.gearbox'], compact_print=True)

    # prob.model.OD1.list_outputs(residuals=True)

    # prob.check_partials(compact_print=False,abs_err_tol=1e-3, rel_err_tol=1e-3)
    # exit()

    for pt in ['DESIGN']+pts:

        if pt == 'DESIGN':
            MN = prob['MN']
            LPT_PR = prob['DESIGN.balance.lpt_PR']
            HPT_PR = prob['DESIGN.balance.hpt_PR']
            FAR = prob['DESIGN.balance.FAR']
            # FAR = prob['FAR']
        else:
            MN = prob[pt+'_MN']
            LPT_PR = prob[pt+'.lpt.PR']
            HPT_PR = prob[pt+'.hpt.PR']
            FAR = prob[pt+'.balance.FAR']


        print("----------------------------------------------------------------------------")
        print("                              POINT:", pt)
        print("----------------------------------------------------------------------------")
        print("                       PERFORMANCE CHARACTERISTICS")
        print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ")
        print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']))
        print("----------------------------------------------------------------------------")
        print("                         FLOW STATION PROPERTIES")
        print("Component        Pt      Tt      ht       S       W      MN       V        A")
        if pt == 'DESIGN' or OD_statics:
            print("Start       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f      " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W'], prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.Fl_O:stat:V']))
            print("Inlet       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W'], prob[pt+'.inlet.Fl_O:stat:MN'], prob[pt+'.inlet.Fl_O:stat:V'], prob[pt+'.inlet.Fl_O:stat:area']))
            print("Fan         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.fan.Fl_O:tot:P'], prob[pt+'.fan.Fl_O:tot:T'], prob[pt+'.fan.Fl_O:tot:h'], prob[pt+'.fan.Fl_O:tot:S'], prob[pt+'.fan.Fl_O:stat:W'], prob[pt+'.fan.Fl_O:stat:MN'], prob[pt+'.fan.Fl_O:stat:V'], prob[pt+'.fan.Fl_O:stat:area']))
            print("Splitter1   %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.splitter.Fl_O1:tot:P'], prob[pt+'.splitter.Fl_O1:tot:T'], prob[pt+'.splitter.Fl_O1:tot:h'], prob[pt+'.splitter.Fl_O1:tot:S'], prob[pt+'.splitter.Fl_O1:stat:W'], prob[pt+'.splitter.Fl_O1:stat:MN'], prob[pt+'.splitter.Fl_O1:stat:V'], prob[pt+'.splitter.Fl_O1:stat:area']))
            print("Splitter2   %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.splitter.Fl_O2:tot:P'], prob[pt+'.splitter.Fl_O2:tot:T'], prob[pt+'.splitter.Fl_O2:tot:h'], prob[pt+'.splitter.Fl_O2:tot:S'], prob[pt+'.splitter.Fl_O2:stat:W'], prob[pt+'.splitter.Fl_O2:stat:MN'], prob[pt+'.splitter.Fl_O2:stat:V'], prob[pt+'.splitter.Fl_O2:stat:area']))
            print("Duct2       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct2.Fl_O:tot:P'], prob[pt+'.duct2.Fl_O:tot:T'], prob[pt+'.duct2.Fl_O:tot:h'], prob[pt+'.duct2.Fl_O:tot:S'], prob[pt+'.duct2.Fl_O:stat:W'], prob[pt+'.duct2.Fl_O:stat:MN'], prob[pt+'.duct2.Fl_O:stat:V'], prob[pt+'.duct2.Fl_O:stat:area']))
            print("LPC         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.lpc.Fl_O:tot:P'], prob[pt+'.lpc.Fl_O:tot:T'], prob[pt+'.lpc.Fl_O:tot:h'], prob[pt+'.lpc.Fl_O:tot:S'], prob[pt+'.lpc.Fl_O:stat:W'], prob[pt+'.lpc.Fl_O:stat:MN'], prob[pt+'.lpc.Fl_O:stat:V'], prob[pt+'.lpc.Fl_O:stat:area']))
            print("Duct25      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct25.Fl_O:tot:P'], prob[pt+'.duct25.Fl_O:tot:T'], prob[pt+'.duct25.Fl_O:tot:h'], prob[pt+'.duct25.Fl_O:tot:S'], prob[pt+'.duct25.Fl_O:stat:W'], prob[pt+'.duct25.Fl_O:stat:MN'], prob[pt+'.duct25.Fl_O:stat:V'], prob[pt+'.duct25.Fl_O:stat:area']))
            print("HPC         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.hpc.Fl_O:tot:P'], prob[pt+'.hpc.Fl_O:tot:T'], prob[pt+'.hpc.Fl_O:tot:h'], prob[pt+'.hpc.Fl_O:tot:S'], prob[pt+'.hpc.Fl_O:stat:W'], prob[pt+'.hpc.Fl_O:stat:MN'], prob[pt+'.hpc.Fl_O:stat:V'], prob[pt+'.hpc.Fl_O:stat:area']))
            print("Bld3        %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.bld3.Fl_O:tot:P'], prob[pt+'.bld3.Fl_O:tot:T'], prob[pt+'.bld3.Fl_O:tot:h'], prob[pt+'.bld3.Fl_O:tot:S'], prob[pt+'.bld3.Fl_O:stat:W'], prob[pt+'.bld3.Fl_O:stat:MN'], prob[pt+'.bld3.Fl_O:stat:V'], prob[pt+'.bld3.Fl_O:stat:area']))
            print("Burner      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W'], prob[pt+'.burner.Fl_O:stat:MN'], prob[pt+'.burner.Fl_O:stat:V'], prob[pt+'.burner.Fl_O:stat:area']))
            print("HPT         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.hpt.Fl_O:tot:P'], prob[pt+'.hpt.Fl_O:tot:T'], prob[pt+'.hpt.Fl_O:tot:h'], prob[pt+'.hpt.Fl_O:tot:S'], prob[pt+'.hpt.Fl_O:stat:W'], prob[pt+'.hpt.Fl_O:stat:MN'], prob[pt+'.hpt.Fl_O:stat:V'], prob[pt+'.hpt.Fl_O:stat:area']))
            print("Duct45      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct45.Fl_O:tot:P'], prob[pt+'.duct45.Fl_O:tot:T'], prob[pt+'.duct45.Fl_O:tot:h'], prob[pt+'.duct45.Fl_O:tot:S'], prob[pt+'.duct45.Fl_O:stat:W'], prob[pt+'.duct45.Fl_O:stat:MN'], prob[pt+'.duct45.Fl_O:stat:V'], prob[pt+'.duct45.Fl_O:stat:area']))
            print("LPT         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.lpt.Fl_O:tot:P'], prob[pt+'.lpt.Fl_O:tot:T'], prob[pt+'.lpt.Fl_O:tot:h'], prob[pt+'.lpt.Fl_O:tot:S'], prob[pt+'.lpt.Fl_O:stat:W'], prob[pt+'.lpt.Fl_O:stat:MN'], prob[pt+'.lpt.Fl_O:stat:V'], prob[pt+'.lpt.Fl_O:stat:area']))
            print("Duct5       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct5.Fl_O:tot:P'], prob[pt+'.duct5.Fl_O:tot:T'], prob[pt+'.duct5.Fl_O:tot:h'], prob[pt+'.duct5.Fl_O:tot:S'], prob[pt+'.duct5.Fl_O:stat:W'], prob[pt+'.duct5.Fl_O:stat:MN'], prob[pt+'.duct5.Fl_O:stat:V'], prob[pt+'.duct5.Fl_O:stat:area']))
            print("CoreNozz    %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.core_nozz.Fl_O:tot:P'], prob[pt+'.core_nozz.Fl_O:tot:T'], prob[pt+'.core_nozz.Fl_O:tot:h'], prob[pt+'.core_nozz.Fl_O:tot:S'], prob[pt+'.core_nozz.Fl_O:stat:W'], prob[pt+'.core_nozz.Fl_O:stat:MN'], prob[pt+'.core_nozz.Fl_O:stat:V'], prob[pt+'.core_nozz.Fl_O:stat:area']))
            print("BypBld      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.byp_bld.Fl_O:tot:P'], prob[pt+'.byp_bld.Fl_O:tot:T'], prob[pt+'.byp_bld.Fl_O:tot:h'], prob[pt+'.byp_bld.Fl_O:tot:S'], prob[pt+'.byp_bld.Fl_O:stat:W'], prob[pt+'.byp_bld.Fl_O:stat:MN'], prob[pt+'.byp_bld.Fl_O:stat:V'], prob[pt+'.byp_bld.Fl_O:stat:area']))
            print("Duct17      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct17.Fl_O:tot:P'], prob[pt+'.duct17.Fl_O:tot:T'], prob[pt+'.duct17.Fl_O:tot:h'], prob[pt+'.duct17.Fl_O:tot:S'], prob[pt+'.duct17.Fl_O:stat:W'], prob[pt+'.duct17.Fl_O:stat:MN'], prob[pt+'.duct17.Fl_O:stat:V'], prob[pt+'.duct17.Fl_O:stat:area']))
            print("BypNozz     %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.byp_nozz.Fl_O:tot:P'], prob[pt+'.byp_nozz.Fl_O:tot:T'], prob[pt+'.byp_nozz.Fl_O:tot:h'], prob[pt+'.byp_nozz.Fl_O:tot:S'], prob[pt+'.byp_nozz.Fl_O:stat:W'], prob[pt+'.byp_nozz.Fl_O:stat:MN'], prob[pt+'.byp_nozz.Fl_O:stat:V'], prob[pt+'.byp_nozz.Fl_O:stat:area']))
        else:
            print("Start       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W']))
            print("Inlet       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W']))
            print("Fan         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.fan.Fl_O:tot:P'], prob[pt+'.fan.Fl_O:tot:T'], prob[pt+'.fan.Fl_O:tot:h'], prob[pt+'.fan.Fl_O:tot:S'], prob[pt+'.fan.Fl_O:stat:W']))
            print("Splitter1   %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.splitter.Fl_O1:tot:P'], prob[pt+'.splitter.Fl_O1:tot:T'], prob[pt+'.splitter.Fl_O1:tot:h'], prob[pt+'.splitter.Fl_O1:tot:S'], prob[pt+'.splitter.Fl_O1:stat:W']))
            print("Splitter2   %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.splitter.Fl_O2:tot:P'], prob[pt+'.splitter.Fl_O2:tot:T'], prob[pt+'.splitter.Fl_O2:tot:h'], prob[pt+'.splitter.Fl_O2:tot:S'], prob[pt+'.splitter.Fl_O2:stat:W']))
            print("Duct2       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct2.Fl_O:tot:P'], prob[pt+'.duct2.Fl_O:tot:T'], prob[pt+'.duct2.Fl_O:tot:h'], prob[pt+'.duct2.Fl_O:tot:S'], prob[pt+'.duct2.Fl_O:stat:W']))
            print("LPC         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.lpc.Fl_O:tot:P'], prob[pt+'.lpc.Fl_O:tot:T'], prob[pt+'.lpc.Fl_O:tot:h'], prob[pt+'.lpc.Fl_O:tot:S'], prob[pt+'.lpc.Fl_O:stat:W']))
            print("Duct25      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct25.Fl_O:tot:P'], prob[pt+'.duct25.Fl_O:tot:T'], prob[pt+'.duct25.Fl_O:tot:h'], prob[pt+'.duct25.Fl_O:tot:S'], prob[pt+'.duct25.Fl_O:stat:W']))
            print("HPC         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.hpc.Fl_O:tot:P'], prob[pt+'.hpc.Fl_O:tot:T'], prob[pt+'.hpc.Fl_O:tot:h'], prob[pt+'.hpc.Fl_O:tot:S'], prob[pt+'.hpc.Fl_O:stat:W']))
            print("Bld3        %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.bld3.Fl_O:tot:P'], prob[pt+'.bld3.Fl_O:tot:T'], prob[pt+'.bld3.Fl_O:tot:h'], prob[pt+'.bld3.Fl_O:tot:S'], prob[pt+'.bld3.Fl_O:stat:W']))
            print("Burner      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W']))
            print("HPT         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.hpt.Fl_O:tot:P'], prob[pt+'.hpt.Fl_O:tot:T'], prob[pt+'.hpt.Fl_O:tot:h'], prob[pt+'.hpt.Fl_O:tot:S'], prob[pt+'.hpt.Fl_O:stat:W']))
            print("Duct45      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct45.Fl_O:tot:P'], prob[pt+'.duct45.Fl_O:tot:T'], prob[pt+'.duct45.Fl_O:tot:h'], prob[pt+'.duct45.Fl_O:tot:S'], prob[pt+'.duct45.Fl_O:stat:W']))
            print("LPT         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.lpt.Fl_O:tot:P'], prob[pt+'.lpt.Fl_O:tot:T'], prob[pt+'.lpt.Fl_O:tot:h'], prob[pt+'.lpt.Fl_O:tot:S'], prob[pt+'.lpt.Fl_O:stat:W']))
            print("Duct5       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct5.Fl_O:tot:P'], prob[pt+'.duct5.Fl_O:tot:T'], prob[pt+'.duct5.Fl_O:tot:h'], prob[pt+'.duct5.Fl_O:tot:S'], prob[pt+'.duct5.Fl_O:stat:W']))
            print("CoreNozz    %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.core_nozz.Fl_O:tot:P'], prob[pt+'.core_nozz.Fl_O:tot:T'], prob[pt+'.core_nozz.Fl_O:tot:h'], prob[pt+'.core_nozz.Fl_O:tot:S'], prob[pt+'.core_nozz.Fl_O:stat:W']))
            print("BypBld      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.byp_bld.Fl_O:tot:P'], prob[pt+'.byp_bld.Fl_O:tot:T'], prob[pt+'.byp_bld.Fl_O:tot:h'], prob[pt+'.byp_bld.Fl_O:tot:S'], prob[pt+'.byp_bld.Fl_O:stat:W']))
            print("Duct17      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct17.Fl_O:tot:P'], prob[pt+'.duct17.Fl_O:tot:T'], prob[pt+'.duct17.Fl_O:tot:h'], prob[pt+'.duct17.Fl_O:tot:S'], prob[pt+'.duct17.Fl_O:stat:W']))
            print("BypNozz     %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.byp_nozz.Fl_O:tot:P'], prob[pt+'.byp_nozz.Fl_O:tot:T'], prob[pt+'.byp_nozz.Fl_O:tot:h'], prob[pt+'.byp_nozz.Fl_O:tot:S'], prob[pt+'.byp_nozz.Fl_O:stat:W']))
        print("----------------------------------------------------------------------------")
        print("                          COMPRESSOR PROPERTIES")
        print("Component        Wc      PR     eff      Nc      pwr   Rline   NcMap s_Wc s_PR s_eff  s_Nc")
        print("Fan         %7.2f %7.4f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.fan.Wc'],prob[pt+'.fan.PR'],prob[pt+'.fan.eff'],prob[pt+'.fan.Nc'],prob[pt+'.fan.power'],prob[pt+'.fan.map.RlineMap'],prob[pt+'.fan.map.readMap.NcMap'],prob[pt+'.fan.s_Wc'],prob[pt+'.fan.s_PR'],prob[pt+'.fan.s_eff'],prob[pt+'.fan.s_Nc']))
        print("LPC         %7.2f %7.4f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.lpc.Wc'],prob[pt+'.lpc.PR'],prob[pt+'.lpc.eff'],prob[pt+'.lpc.Nc'],prob[pt+'.lpc.power'],prob[pt+'.lpc.map.RlineMap'],prob[pt+'.lpc.map.readMap.NcMap'],prob[pt+'.lpc.s_Wc'],prob[pt+'.lpc.s_PR'],prob[pt+'.lpc.s_eff'],prob[pt+'.lpc.s_Nc']))
        print("HPC         %7.2f %7.4f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.hpc.Wc'],prob[pt+'.hpc.PR'],prob[pt+'.hpc.eff'],prob[pt+'.hpc.Nc'],prob[pt+'.hpc.power'],prob[pt+'.hpc.map.RlineMap'],prob[pt+'.hpc.map.readMap.NcMap'],prob[pt+'.hpc.s_Wc'],prob[pt+'.hpc.s_PR'],prob[pt+'.hpc.s_eff'],prob[pt+'.hpc.s_Nc']))
        print("----------------------------------------------------------------------------")
        print("                            BURNER PROPERTIES")
        print("Component      dPqP   TtOut   Wfuel      FAR")
        print("Burner      %7.4f %7.2f %7.4f  %7.5f" %(prob[pt+'.burner.dPqP'], prob[pt+'.burner.Fl_O:tot:T'],prob[pt+'.burner.Wfuel'], FAR))
        print("----------------------------------------------------------------------------")
        print("                           TURBINE PROPERTIES")
        print("Component        Wp      PR     eff      Np      pwr   NpMap s_Wp s_PR s_eff s_Np")
        print("HPT         %7.3f %7.4f %7.5f %7.1f %8.1f %7.3f %7.3f %7.3f %8.3f %7.3f" %(prob[pt+'.hpt.Wp'], HPT_PR, prob[pt+'.hpt.eff'],prob[pt+'.hpt.Np'],prob[pt+'.hpt.power'],prob[pt+'.hpt.map.readMap.NpMap'],prob[pt+'.hpt.s_Wp'],prob[pt+'.hpt.s_PR'],prob[pt+'.hpt.s_eff'],prob[pt+'.hpt.s_Np']))
        print("LPT         %7.3f %7.4f %7.5f %7.1f %8.1f %7.3f %7.3f %7.3f %8.3f %7.3f" %(prob[pt+'.lpt.Wp'], LPT_PR, prob[pt+'.lpt.eff'],prob[pt+'.lpt.Np'],prob[pt+'.lpt.power'],prob[pt+'.lpt.map.readMap.NpMap'],prob[pt+'.lpt.s_Wp'],prob[pt+'.lpt.s_PR'],prob[pt+'.lpt.s_eff'],prob[pt+'.lpt.s_Np']))
        print("----------------------------------------------------------------------------")
        print("                            NOZZLE PROPERTIES")
        print("Component        PR      Cv     Ath    MNth   MNout       V      Fg")
        print("CoreNozz    %7.4f %7.4f %7.2f %7.4f %7.4f %7.1f %7.1f" %(prob[pt+'.core_nozz.PR'],prob[pt+'.core_nozz.Cv'],prob[pt+'.core_nozz.Throat:stat:area'],prob[pt+'.core_nozz.Throat:stat:MN'],prob[pt+'.core_nozz.Fl_O:stat:MN'],prob[pt+'.core_nozz.Fl_O:stat:V'],prob[pt+'.core_nozz.Fg']))
        print("BypNozz     %7.4f %7.4f %7.2f %7.4f %7.4f %7.1f %7.1f" %(prob[pt+'.byp_nozz.PR'],prob[pt+'.byp_nozz.Cv'],prob[pt+'.byp_nozz.Throat:stat:area'],prob[pt+'.byp_nozz.Throat:stat:MN'],prob[pt+'.byp_nozz.Fl_O:stat:MN'],prob[pt+'.byp_nozz.Fl_O:stat:V'],prob[pt+'.byp_nozz.Fg']))
        print("----------------------------------------------------------------------------")
        print("                            SHAFT PROPERTIES")
        print("Component     Nmech    trqin   trqout    pwrin   pwrout")
        print("HP_Shaft    %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.hp_shaft.Nmech'],prob[pt+'.hp_shaft.trq_in'],prob[pt+'.hp_shaft.trq_out'],prob[pt+'.hp_shaft.pwr_in'],prob[pt+'.hp_shaft.pwr_out']))
        print("LP_Shaft    %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.lp_shaft.Nmech'],prob[pt+'.lp_shaft.trq_in'],prob[pt+'.lp_shaft.trq_out'],prob[pt+'.lp_shaft.pwr_in'],prob[pt+'.lp_shaft.pwr_out']))
        print("Fan_Shaft   %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.fan_shaft.Nmech'],prob[pt+'.fan_shaft.trq_in'],prob[pt+'.fan_shaft.trq_out'],prob[pt+'.fan_shaft.pwr_in'],prob[pt+'.fan_shaft.pwr_out']))
        print("----------------------------------------------------------------------------")
        print("                            BLEED PROPERTIES")
        print("Bleed       Wb/Win   Pfrac Workfrac       W      Tt      ht      Pt")
        print("SBV        %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld25.sbv:frac_W'],1.0,1.0,prob[pt+'.bld25.sbv:stat:W'],prob[pt+'.bld25.sbv:tot:T'],prob[pt+'.bld25.sbv:tot:h'],prob[pt+'.bld25.sbv:tot:P']))
        print("bld_inlet      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.bld_inlet:frac_W'],prob[pt+'.hpc.bld_inlet:frac_P'],prob[pt+'.hpc.bld_inlet:frac_work'],prob[pt+'.hpc.bld_inlet:stat:W'],prob[pt+'.hpc.bld_inlet:tot:T'],prob[pt+'.hpc.bld_inlet:tot:h'],prob[pt+'.hpc.bld_inlet:tot:P']))
        print("bld_exit      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.bld_exit:frac_W'],prob[pt+'.hpc.bld_exit:frac_P'],prob[pt+'.hpc.bld_exit:frac_work'],prob[pt+'.hpc.bld_exit:stat:W'],prob[pt+'.hpc.bld_exit:tot:T'],prob[pt+'.hpc.bld_exit:tot:h'],prob[pt+'.hpc.bld_exit:tot:P']))
        print("bld_inlet      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld3.bld_inlet:frac_W'],1.0,1.0,prob[pt+'.bld3.bld_inlet:stat:W'],prob[pt+'.bld3.bld_inlet:tot:T'],prob[pt+'.bld3.bld_inlet:tot:h'],prob[pt+'.bld3.bld_inlet:tot:P']))
        print("bld_exit      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld3.bld_exit:frac_W'],1.0,1.0,prob[pt+'.bld3.bld_exit:stat:W'],prob[pt+'.bld3.bld_exit:tot:T'],prob[pt+'.bld3.bld_exit:tot:h'],prob[pt+'.bld3.bld_exit:tot:P']))
        print("Cust       %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.cust:frac_W'],prob[pt+'.hpc.cust:frac_P'],prob[pt+'.hpc.cust:frac_work'],prob[pt+'.hpc.cust:stat:W'],prob[pt+'.hpc.cust:tot:T'],prob[pt+'.hpc.cust:tot:h'],prob[pt+'.hpc.cust:tot:P']))
        print("BypBld     %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.byp_bld.bypBld:frac_W'],1.0,1.0,prob[pt+'.byp_bld.bypBld:stat:W'],prob[pt+'.byp_bld.bypBld:tot:T'],prob[pt+'.byp_bld.bypBld:tot:h'],prob[pt+'.byp_bld.bypBld:tot:P']))
        print("----------------------------------------------------------------------------")
        print()

        # print('fan.s_PR', prob['DESIGN.fan.s_PR'][0])
        # print('fan.s_Wc', prob['DESIGN.fan.s_Wc'][0])
        # print('fan.s_eff', prob['DESIGN.fan.s_eff'][0])
        # print('fan.s_Nc', prob['DESIGN.fan.s_Nc'][0])
        # print('lpc.s_PR', prob['DESIGN.lpc.s_PR'][0])
        # print('lpc.s_Wc', prob['DESIGN.lpc.s_Wc'][0])
        # print('lpc.s_eff', prob['DESIGN.lpc.s_eff'][0])
        # print('lpc.s_Nc', prob['DESIGN.lpc.s_Nc'][0])
        # print('hpc.s_PR', prob['DESIGN.hpc.s_PR'][0])
        # print('hpc.s_Wc', prob['DESIGN.hpc.s_Wc'][0])
        # print('hpc.s_eff', prob['DESIGN.hpc.s_eff'][0])
        # print('hpc.s_Nc', prob['DESIGN.hpc.s_Nc'][0])
        # print('hpt.s_PR', prob['DESIGN.hpt.s_PR'][0])
        # print('hpt.s_Wp', prob['DESIGN.hpt.s_Wp'][0])
        # print('hpt.s_eff', prob['DESIGN.hpt.s_eff'][0])
        # print('hpt.s_Np', prob['DESIGN.hpt.s_Np'][0])
        # print('lpt.s_PR', prob['DESIGN.lpt.s_PR'][0])
        # print('lpt.s_Wp', prob['DESIGN.lpt.s_Wp'][0])
        # print('lpt.s_eff', prob['DESIGN.lpt.s_eff'][0])
        # print('lpt.s_Np', prob['DESIGN.lpt.s_Np'][0])

        # print(prob['DESIGN.core_nozz.Throat:stat:area'][0])
        # print(prob['DESIGN.byp_nozz.Throat:stat:area'][0])

        # print("Inlet     area" , prob[pt+'.inlet.Fl_O:stat:area'][0])
        # print("Fan       area" , prob[pt+'.fan.Fl_O:stat:area'][0])
        # print("Splitter1 area" , prob[pt+'.splitter.Fl_O1:stat:area'][0])
        # print("Splitter2 area" , prob[pt+'.splitter.Fl_O2:stat:area'][0])
        # print("Duct2     area" , prob[pt+'.duct2.Fl_O:stat:area'][0])
        # print("LPC       area" , prob[pt+'.lpc.Fl_O:stat:area'][0])
        # print("Duct25     area" , prob[pt+'.duct25.Fl_O:stat:area'][0])
        # print("HPC       area" , prob[pt+'.hpc.Fl_O:stat:area'][0])
        # print("Bld3      area" , prob[pt+'.bld3.Fl_O:stat:area'][0])
        # print("Burner    area" , prob[pt+'.burner.Fl_O:stat:area'][0])
        # print("HPT       area" , prob[pt+'.hpt.Fl_O:stat:area'][0])
        # print("Duct45    area" , prob[pt+'.duct45.Fl_O:stat:area'][0])
        # print("LPT       area" , prob[pt+'.lpt.Fl_O:stat:area'][0])
        # print("Duct5    area" , prob[pt+'.duct5.Fl_O:stat:area'][0])
        # print("CoreNozz  area" , prob[pt+'.core_nozz.Fl_O:stat:area'][0])
        # print("BypBld    area" , prob[pt+'.byp_bld.Fl_O:stat:area'][0])
        # print("Duct17    area" , prob[pt+'.duct17.Fl_O:stat:area'][0])
        # print("BypNozz   area" , prob[pt+'.byp_nozz.Fl_O:stat:area'][0])

        # print(prob[pt+'.core_nozz.Fl_O:stat:V'][0]/prob[pt+'.byp_nozz.Fl_O:stat:V'][0])

    print(prob['DESIGN.gearbox.trq_base'], prob['OD1.gearbox.trq_base'])

    print()
    print("time", time.time() - st)

    # print(ODpt.list_residuals())
    # data = ODpt.list_residuals(out_stream=None)
    # for r in data:
    #     mag = np.linalg.norm(r[1])
    #     if mag > 1e-1:
    #         print(r[0], np.linalg.norm(r[1]))
