from __future__ import print_function

import sys
import numpy as np

import openmdao.api as om 

import pycycle.api as pyc

class ABTurbojet(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('duct1', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='JP-7'))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=['Nmech'])
        self.add_subsystem('ab', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_FUEL_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='JP-7'))
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, internal_solver=True))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=2))

        self.connect('duct1.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('ab.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')
        # self.connect('shaft.Nmech', 'comp.Nmech')
        # self.connect('shaft.Nmech', 'turb.Nmech')
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')

            # self.set_order(['fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf', 'thrust_balance', 'temp_balance', 'shaft_balance'])
            self.set_order(['balance', 'fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf'])

        else:

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=8000., units='rpm', lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            balance.add_balance('W', val=100.0, units='lbm/s', eq_units=None, rhs_val=2.0)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('comp.map.RlineMap', 'balance.lhs:W')

            self.set_order(['balance', 'fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf'])

    
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'duct1.Fl_I', connect_stat=False)
        pyc.connect_flow(self, 'duct1.Fl_O', 'comp.Fl_I', connect_stat=False)
        pyc.connect_flow(self, 'comp.Fl_O', 'burner.Fl_I', connect_stat=False)
        pyc.connect_flow(self, 'burner.Fl_O', 'turb.Fl_I', connect_stat=False)
        pyc.connect_flow(self, 'turb.Fl_O', 'ab.Fl_I', connect_stat=False)
        pyc.connect_flow(self, 'ab.Fl_O', 'nozz.Fl_I', connect_stat=False)

        pyc.connect_flow(self, 'comp.cool1', 'turb.cool1', connect_stat=False)
        pyc.connect_flow(self, 'comp.cool2', 'turb.cool2', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        # newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = .0001
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

def viewer(prob, pt, file=sys.stdout): 
    """
    print a report of all the relevant cycle properties
    """

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC']), file=file, flush=True)
    

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'duct1.Fl_O', 'comp.Fl_O', 'burner.Fl_O', 
                'turb.Fl_O', 'ab.Fl_O','nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['turb']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['comp.cool1', 'comp.cool2']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


if __name__ == "__main__":

    import time

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    # FOR DESIGN
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
    des_vars.add_output('ab:MN_out',0.4),
    des_vars.add_output('ab:FAR', 0.000),
    des_vars.add_output('comp:cool1:frac_W', 0.0789),
    des_vars.add_output('comp:cool1:frac_P', 1.0),
    des_vars.add_output('comp:cool1:frac_work', 1.0),
    des_vars.add_output('comp:cool2:frac_W', 0.0383),
    des_vars.add_output('comp:cool2:frac_P', 1.0),
    des_vars.add_output('comp:cool2:frac_work', 1.0),
    des_vars.add_output('turb:cool1:frac_P', 1.0),
    des_vars.add_output('turb:cool2:frac_P', 0.0),
    # OFF DESIGN 1

    des_vars.add_output('OD1_MN', 0.000001),
    des_vars.add_output('OD1_alt', 0.0, units='ft'),
    des_vars.add_output('OD1_T4', 2370.0, units='degR'),
    des_vars.add_output('OD1_ab_FAR', 0.031523391),
    des_vars.add_output('OD1_Rline', 2.0),
    # OFF DESIGN 2
    des_vars.add_output('OD2_MN', 0.8),
    des_vars.add_output('OD2_alt', 0.0, units='ft'),
    des_vars.add_output('OD2_T4', 2370.0, units='degR'),
    des_vars.add_output('OD2_ab_FAR', 0.022759941),
    des_vars.add_output('OD2_Rline', 2.0),
    # OFF DESIGN 3
    des_vars.add_output('OD3_MN', 1.0),
    des_vars.add_output('OD3_alt', 15000.0, units='ft'),
    des_vars.add_output('OD3_T4', 2370.0, units='degR'),
    des_vars.add_output('OD3_ab_FAR', 0.036849745),
    des_vars.add_output('OD3_Rline', 2.0),
    # OFF DESIGN 4
    des_vars.add_output('OD4_MN', 1.2),
    des_vars.add_output('OD4_alt', 25000.0, units='ft'),
    des_vars.add_output('OD4_T4', 2370.0, units='degR'),
    des_vars.add_output('OD4_ab_FAR', 0.035266091),
    des_vars.add_output('OD4_Rline', 2.0),
    # OFF DESIGN 5
    des_vars.add_output('OD5_MN', 0.6),
    des_vars.add_output('OD5_alt', 35000.0, units='ft'),
    des_vars.add_output('OD5_T4', 2370.0, units='degR'),
    des_vars.add_output('OD5_ab_FAR', 0.020216221),
    des_vars.add_output('OD5_Rline', 2.0),
    # OFF DESIGN 6
    des_vars.add_output('OD6_MN', 1.6),
    des_vars.add_output('OD6_alt', 35000.0, units='ft'),
    des_vars.add_output('OD6_T4', 2370.0, units='degR'),
    des_vars.add_output('OD6_ab_FAR', 0.038532787),
    des_vars.add_output('OD6_Rline', 2.0),
    # OFF DESIGN 7
    des_vars.add_output('OD7_MN', 1.6),
    des_vars.add_output('OD7_alt', 50000.0, units='ft'),
    des_vars.add_output('OD7_T4', 2370.0, units='degR'),
    des_vars.add_output('OD7_ab_FAR', 0.038532787),
    des_vars.add_output('OD7_Rline', 2.0),
    # OFF DESIGN 8
    des_vars.add_output('OD8_MN', 1.8),
    des_vars.add_output('OD8_alt', 70000.0, units='ft'),
    des_vars.add_output('OD8_T4', 2370.0, units='degR'),
    des_vars.add_output('OD8_ab_FAR', 0.038532787),
    des_vars.add_output('OD8_Rline', 2.0),

    # DESIGN CASE
    prob.model.add_subsystem('DESIGN', Turbojet())

    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')

    prob.model.connect('duct1:dPqP', 'DESIGN.duct1.dPqP')
    prob.model.connect('comp:PRdes', 'DESIGN.comp.PR')
    prob.model.connect('comp:effDes', 'DESIGN.comp.eff')
    prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('turb:effDes', 'DESIGN.turb.eff')
    prob.model.connect('ab:dPqP', 'DESIGN.ab.dPqP')
    prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
    prob.model.connect('shaft:Nmech', 'DESIGN.Nmech')

    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
    prob.model.connect('duct1:MN_out', 'DESIGN.duct1.MN')
    prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
    prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')
    prob.model.connect('ab:MN_out', 'DESIGN.ab.MN')
    prob.model.connect('ab:FAR', 'DESIGN.ab.Fl_I:FAR')

    prob.model.connect('comp:cool1:frac_W', 'DESIGN.comp.cool1:frac_W')
    prob.model.connect('comp:cool1:frac_P', 'DESIGN.comp.cool1:frac_P')
    prob.model.connect('comp:cool1:frac_work', 'DESIGN.comp.cool1:frac_work')

    prob.model.connect('comp:cool2:frac_W', 'DESIGN.comp.cool2:frac_W')
    prob.model.connect('comp:cool2:frac_P', 'DESIGN.comp.cool2:frac_P')
    prob.model.connect('comp:cool2:frac_work', 'DESIGN.comp.cool2:frac_work')

    prob.model.connect('turb:cool1:frac_P', 'DESIGN.turb.cool1:frac_P')
    prob.model.connect('turb:cool2:frac_P', 'DESIGN.turb.cool2:frac_P')

    # # DESIGN CASE (Fixed)
    # prob.root.add('DESIGN', Turbojet_Fixed_Design())

    # OFF DESIGN CASES
    pts = ['OD5',]
    # pts = [] #'OD1','OD2','OD3','OD4','OD5','OD6','OD7','OD8'

    for pt in pts:
        prob.model.add_subsystem(pt, Turbojet(design=False))

        prob.model.connect('duct1:dPqP', pt+'.duct1.dPqP')
        prob.model.connect('burn:dPqP', pt+'.burner.dPqP')
        prob.model.connect('ab:dPqP', pt+'.ab.dPqP')
        prob.model.connect('nozz:Cv', pt+'.nozz.Cv')

        prob.model.connect('comp:cool1:frac_W', pt+'.comp.cool1:frac_W')
        prob.model.connect('comp:cool1:frac_P', pt+'.comp.cool1:frac_P')
        prob.model.connect('comp:cool1:frac_work', pt+'.comp.cool1:frac_work')

        prob.model.connect('comp:cool2:frac_W', pt+'.comp.cool2:frac_W')
        prob.model.connect('comp:cool2:frac_P', pt+'.comp.cool2:frac_P')
        prob.model.connect('comp:cool2:frac_work', pt+'.comp.cool2:frac_work')

        prob.model.connect('turb:cool1:frac_P', pt+'.turb.cool1:frac_P')
        prob.model.connect('turb:cool2:frac_P', pt+'.turb.cool2:frac_P')

        prob.model.connect(pt+'_alt', pt+'.fc.alt')
        prob.model.connect(pt+'_MN', pt+'.fc.MN')

        prob.model.connect('DESIGN.comp.s_PR', pt+'.comp.s_PR')
        prob.model.connect('DESIGN.comp.s_Wc', pt+'.comp.s_Wc')
        prob.model.connect('DESIGN.comp.s_eff', pt+'.comp.s_eff')
        prob.model.connect('DESIGN.comp.s_Nc', pt+'.comp.s_Nc')

        prob.model.connect('DESIGN.turb.s_PR', pt+'.turb.s_PR')
        prob.model.connect('DESIGN.turb.s_Wp', pt+'.turb.s_Wp')
        prob.model.connect('DESIGN.turb.s_eff', pt+'.turb.s_eff')
        prob.model.connect('DESIGN.turb.s_Np', pt+'.turb.s_Np')

    
        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.duct1.Fl_O:stat:area', pt+'.duct1.area')
        prob.model.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')
        prob.model.connect('DESIGN.ab.Fl_O:stat:area', pt+'.ab.area')

        prob.model.connect(pt+'_T4', pt+'.balance.rhs:FAR')
        prob.model.connect(pt+'_Rline', pt+'.balance.rhs:W')
        prob.model.connect(pt+'_ab_FAR', pt+'.ab.Fl_I:FAR')


    prob.setup(check=False)

    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.0175506829934
    prob['DESIGN.balance.W'] = 168.453135137
    prob['DESIGN.balance.turb_PR'] = 4.46138725662
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.665288153

    for pt in pts:
        # prob[pt+'.Rline_balance.indep'] = 1.0 #168.453135137
        # prob[pt+'.temp_balance.indep'] = 0.0175506829934
        # prob[pt+'.shaft_balance.indep'] = 8070.0
        # prob[pt+'.fc.balance.Pt'] = 14.6955113159
        # prob[pt+'.fc.balance.Tt'] = 518.665288153
        # prob[pt+'.turb.PR'] = 4.46138725662

        # OD3 Guesses
        prob[pt+'.balance.W'] = 166.073
        prob[pt+'.balance.FAR'] = 0.01680
        prob[pt+'.balance.Nmech'] = 4000 #8197.38
        prob[pt+'.fc.balance.Pt'] = 15.703
        prob[pt+'.fc.balance.Tt'] = 558.31
        prob[pt+'.turb.PR'] = 4.6690


        print(pt, prob[pt+'.balance.W'], prob[pt+'.balance.Nmech'])

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)

