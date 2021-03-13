import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

class MultiSpoolTurboshaft(pyc.Cycle):

    def initialize(self):
        self.options.declare('maxiter', default=10,
                              desc='Maximum number of Newton solver iterations.')
        super().initialize()

    def setup(self):

        design = self.options['design']
        maxiter = self.options['maxiter']
        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data'] = pyc.species_data.janaf

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('duct1', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('icduct', pyc.Duct())
        self.add_subsystem('hpc_axi', pyc.Compressor(map_data=pyc.HPCMap),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld25', pyc.BleedOut(bleed_names=['cool1','cool2']))
        self.add_subsystem('hpc_centri', pyc.Compressor(map_data=pyc.HPCMap),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['cool3','cool4']))
        self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem('burner', pyc.Combustor(fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap, bleed_names=['cool3','cool4']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('duct43', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, bleed_names=['cool1','cool2']),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('itduct', pyc.Duct())
        self.add_subsystem('pt', pyc.Turbine(map_data=pyc.LPTMap),
                           promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct12', pyc.Duct())
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=1),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('ip_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        self.connect('duct1.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc_centri.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')
        self.connect('lp_shaft.pwr_in', 'perf.power')

        self.connect('pt.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'ip_shaft.trq_0')
        self.connect('lpt.trq', 'ip_shaft.trq_1')
        self.connect('hpc_axi.trq', 'hp_shaft.trq_0')
        self.connect('hpc_centri.trq', 'hp_shaft.trq_1')
        self.connect('hpt.trq', 'hp_shaft.trq_2')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.PR', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

            balance.add_balance('pt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.pt_PR', 'pt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:pt_PR')


        else:
            balance.add_balance('FAR', eq_units='hp', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('IP_Nmech', val=12000.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.IP_Nmech', 'IP_Nmech')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:IP_Nmech')

            balance.add_balance('HP_Nmech', val=14800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:HP_Nmech')

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'duct1.Fl_I')
        self.pyc_connect_flow('duct1.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'icduct.Fl_I')
        self.pyc_connect_flow('icduct.Fl_O', 'hpc_axi.Fl_I')
        self.pyc_connect_flow('hpc_axi.Fl_O', 'bld25.Fl_I')
        self.pyc_connect_flow('bld25.Fl_O', 'hpc_centri.Fl_I')
        self.pyc_connect_flow('hpc_centri.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct43.Fl_I')
        self.pyc_connect_flow('duct43.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'itduct.Fl_I')
        self.pyc_connect_flow('itduct.Fl_O', 'pt.Fl_I')
        self.pyc_connect_flow('pt.Fl_O', 'duct12.Fl_I')
        self.pyc_connect_flow('duct12.Fl_O', 'nozzle.Fl_I')

        self.pyc_connect_flow('bld25.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('bld25.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = maxiter
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver()

        super().setup()

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'], 
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                    prob[pt+'.perf.OPR'], prob[pt+'.perf.PSFC'])

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     PSFC ")
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" %summary_data)


    fs_names = ['fc.Fl_O','inlet.Fl_O','duct1.Fl_O','lpc.Fl_O',
                'icduct.Fl_O','hpc_axi.Fl_O','bld25.Fl_O',
                'hpc_centri.Fl_O','bld3.Fl_O','duct6.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct43.Fl_O','lpt.Fl_O',
                'itduct.Fl_O','pt.Fl_O','duct12.Fl_O','nozzle.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['lpc','hpc_axi','hpc_centri']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt','lpt','pt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozzle']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft','ip_shaft','lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['bld25', 'bld3']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)

class MPMultiSpool(pyc.MPCycle):

    def setup(self):

        self.pyc_add_pnt('DESIGN', MultiSpoolTurboshaft(thermo_method='CEA'))

        self.set_input_defaults('DESIGN.inlet.MN', 0.4),
        self.set_input_defaults('DESIGN.duct1.MN', 0.4),
        self.set_input_defaults('DESIGN.lpc.MN', 0.3),
        self.set_input_defaults('DESIGN.icduct.MN', 0.3),
        self.set_input_defaults('DESIGN.hpc_axi.MN', 0.25),
        self.set_input_defaults('DESIGN.bld25.MN', 0.3000),
        self.set_input_defaults('DESIGN.hpc_centri.MN', 0.20),
        self.set_input_defaults('DESIGN.bld3.MN', 0.2000),
        self.set_input_defaults('DESIGN.duct6.MN', 0.2000),
        self.set_input_defaults('DESIGN.burner.MN', 0.15),
        self.set_input_defaults('DESIGN.hpt.MN', 0.30),
        self.set_input_defaults('DESIGN.duct43.MN', 0.30),
        self.set_input_defaults('DESIGN.lpt.MN', 0.4),
        self.set_input_defaults('DESIGN.itduct.MN', 0.4),
        self.set_input_defaults('DESIGN.pt.MN', 0.4),
        self.set_input_defaults('DESIGN.duct12.MN', 0.4),
        self.set_input_defaults('DESIGN.LP_Nmech', 12750., units='rpm'),
        self.set_input_defaults('DESIGN.lp_shaft.HPX', 1800.0, units='hp'),
        self.set_input_defaults('DESIGN.IP_Nmech', 12000., units='rpm'),
        self.set_input_defaults('DESIGN.HP_Nmech', 14800., units='rpm'),

        self.pyc_add_cycle_param('inlet.ram_recovery', 1.0)
        self.pyc_add_cycle_param('duct1.dPqP', 0.0)
        self.pyc_add_cycle_param('icduct.dPqP', 0.002)
        self.pyc_add_cycle_param('bld25.cool1:frac_W', 0.024)
        self.pyc_add_cycle_param('bld25.cool2:frac_W', 0.0146)
        self.pyc_add_cycle_param('duct6.dPqP', 0.00)
        self.pyc_add_cycle_param('burner.dPqP', 0.050)
        self.pyc_add_cycle_param('bld3.cool3:frac_W', 0.1705)
        self.pyc_add_cycle_param('bld3.cool4:frac_W', 0.1209)
        self.pyc_add_cycle_param('duct43.dPqP', 0.0051)
        self.pyc_add_cycle_param('itduct.dPqP', 0.00)
        self.pyc_add_cycle_param('duct12.dPqP', 0.00)
        self.pyc_add_cycle_param('nozzle.Cv', 0.99)
        self.pyc_add_cycle_param('hpt.cool3:frac_P', 1.0)
        self.pyc_add_cycle_param('hpt.cool4:frac_P', 0.0)
        self.pyc_add_cycle_param('lpt.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.cool2:frac_P', 0.0)

        self.od_pts = ['OD'] 
        self.od_pwrs = [1600.0,]
        self.od_Nmechs = [12750.0,]
        self.od_alts = [28000,]
        self.od_MNs = [.5,]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, MultiSpoolTurboshaft(design=False, thermo_method='CEA', maxiter=10))

            self.set_input_defaults(pt+'.balance.rhs:FAR', self.od_pwrs[i], units='hp')
            self.set_input_defaults(pt+'.LP_Nmech', self.od_Nmechs[i], units='rpm')
            self.set_input_defaults(pt+'.fc.alt', self.od_alts[i], units='ft')
            self.set_input_defaults(pt+'.fc.MN', self.od_MNs[i])

        self.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        self.pyc_connect_des_od('hpc_axi.s_PR', 'hpc_axi.s_PR')
        self.pyc_connect_des_od('hpc_axi.s_Wc', 'hpc_axi.s_Wc')
        self.pyc_connect_des_od('hpc_axi.s_eff', 'hpc_axi.s_eff')
        self.pyc_connect_des_od('hpc_axi.s_Nc', 'hpc_axi.s_Nc')
        self.pyc_connect_des_od('hpc_centri.s_PR', 'hpc_centri.s_PR')
        self.pyc_connect_des_od('hpc_centri.s_Wc', 'hpc_centri.s_Wc')
        self.pyc_connect_des_od('hpc_centri.s_eff', 'hpc_centri.s_eff')
        self.pyc_connect_des_od('hpc_centri.s_Nc', 'hpc_centri.s_Nc')
        self.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')
        self.pyc_connect_des_od('pt.s_PR', 'pt.s_PR')
        self.pyc_connect_des_od('pt.s_Wp', 'pt.s_Wp')
        self.pyc_connect_des_od('pt.s_eff', 'pt.s_eff')
        self.pyc_connect_des_od('pt.s_Np', 'pt.s_Np')

        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('duct1.Fl_O:stat:area', 'duct1.area')
        self.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        self.pyc_connect_des_od('icduct.Fl_O:stat:area', 'icduct.area')
        self.pyc_connect_des_od('hpc_axi.Fl_O:stat:area', 'hpc_axi.area')
        self.pyc_connect_des_od('bld25.Fl_O:stat:area', 'bld25.area')
        self.pyc_connect_des_od('hpc_centri.Fl_O:stat:area', 'hpc_centri.area')
        self.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.pyc_connect_des_od('duct43.Fl_O:stat:area', 'duct43.area')
        self.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.pyc_connect_des_od('itduct.Fl_O:stat:area', 'itduct.area')
        self.pyc_connect_des_od('pt.Fl_O:stat:area', 'pt.area')
        self.pyc_connect_des_od('duct12.Fl_O:stat:area', 'duct12.area')
        self.pyc_connect_des_od('nozzle.Throat:stat:area','balance.rhs:W')

        super().setup()

if __name__ == "__main__":

    import time
    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    prob.model = mp_multispool = MPMultiSpool()

    prob.setup()

    #Define the design point
    prob.set_val('DESIGN.fc.alt', 28000., units='ft'),
    prob.set_val('DESIGN.fc.MN', 0.5),
    prob.set_val('DESIGN.balance.rhs:FAR', 2740.0, units='degR'),
    prob.set_val('DESIGN.balance.rhs:W', 1.1)
    prob.set_val('DESIGN.lpc.PR', 5.000),
    prob.set_val('DESIGN.lpc.eff', 0.8900),
    prob.set_val('DESIGN.hpc_axi.PR', 3.0),
    prob.set_val('DESIGN.hpc_axi.eff', 0.8900),
    prob.set_val('DESIGN.hpc_centri.PR', 2.7),
    prob.set_val('DESIGN.hpc_centri.eff', 0.8800),
    prob.set_val('DESIGN.hpt.eff', 0.89),
    prob.set_val('DESIGN.lpt.eff', 0.9),
    prob.set_val('DESIGN.pt.eff', 0.85),

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.02261
    prob['DESIGN.balance.W'] = 10.76
    prob['DESIGN.balance.hpt_PR'] = 4.233
    prob['DESIGN.balance.lpt_PR'] = 1.979
    prob['DESIGN.balance.pt_PR'] = 4.919
    prob['DESIGN.fc.balance.Pt'] = 5.666
    prob['DESIGN.fc.balance.Tt'] = 440.0

    for i, pt in enumerate(mp_multispool.od_pts):

        # initial guesses
        prob[pt+'.balance.FAR'] = 0.02135
        prob[pt+'.balance.W'] = 10.775
        prob[pt+'.balance.HP_Nmech'] = 14800.000
        prob[pt+'.balance.IP_Nmech'] = 12000.000
        prob[pt+'.hpt.PR'] = 4.233
        prob[pt+'.lpt.PR'] = 1.979
        prob[pt+'.pt.PR'] = 4.919
        prob[pt+'.fc.balance.Pt'] = 5.666
        prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.nozzle.PR'] = 1.1

    st = time.time()


    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+mp_multispool.od_pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)