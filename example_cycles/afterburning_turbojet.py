import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

class ABTurbojet(pyc.Cycle):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        self.pyc_add_element('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_ELEMENTS))
        self.pyc_add_element('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_ELEMENTS))
        self.pyc_add_element('duct1', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_ELEMENTS))
        self.pyc_add_element('comp', pyc.Compressor(map_data=pyc.AXI5, design=design, thermo_data=thermo_spec, elements=pyc.AIR_ELEMENTS,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=['Nmech'])
        self.pyc_add_element('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_ELEMENTS,
                                        air_fuel_elements=pyc.AIR_FUEL_ELEMENTS,
                                        fuel_type='JP-7'))
        self.pyc_add_element('turb', pyc.Turbine(map_data=pyc.LPT2269, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_ELEMENTS,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=['Nmech'])
        self.pyc_add_element('ab', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_FUEL_ELEMENTS,
                                        air_fuel_elements=pyc.AIR_FUEL_ELEMENTS,
                                        fuel_type='JP-7'))
        self.pyc_add_element('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_ELEMENTS, internal_solver=True))
        self.pyc_add_element('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.pyc_add_element('perf', pyc.Performance(num_nozzles=1, num_burners=2))

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


        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'duct1.Fl_I', connect_stat=False)
        self.pyc_connect_flow('duct1.Fl_O', 'comp.Fl_I', connect_stat=False)
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I', connect_stat=False)
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I', connect_stat=False)
        self.pyc_connect_flow('turb.Fl_O', 'ab.Fl_I', connect_stat=False)
        self.pyc_connect_flow('ab.Fl_O', 'nozz.Fl_I', connect_stat=False)

        self.pyc_connect_flow('comp.cool1', 'turb.cool1', connect_stat=False)
        self.pyc_connect_flow('comp.cool2', 'turb.cool2', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
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

    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'], prob[pt+'.inlet.Fl_O:stat:W'],
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'], prob[pt+'.perf.OPR'],
                    prob[pt+'.perf.TSFC'])

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" %summary_data, file=file, flush=True)


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

    bleed_names = ['comp']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)



class MPABTurbojet(pyc.MPCycle):

    def setup(self):

        # DESIGN CASE
        self.pyc_add_pnt('DESIGN', ABTurbojet(design=True))

        self.set_input_defaults('DESIGN.Nmech', 8070.0, units='rpm'),
        self.set_input_defaults('DESIGN.inlet.MN', 0.60),
        self.set_input_defaults('DESIGN.duct1.MN', 0.60),
        self.set_input_defaults('DESIGN.comp.MN', 0.20),
        self.set_input_defaults('DESIGN.burner.MN', 0.20),
        self.set_input_defaults('DESIGN.turb.MN', 0.4),
        self.set_input_defaults('DESIGN.ab.MN',0.4),
        self.set_input_defaults('DESIGN.ab.Fl_I:FAR', 0.000),

        self.pyc_add_cycle_param('duct1.dPqP', 0.02)
        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('ab.dPqP', 0.06)
        self.pyc_add_cycle_param('nozz.Cv', 0.99)
        self.pyc_add_cycle_param('comp.cool1:frac_W', 0.0789)
        self.pyc_add_cycle_param('comp.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('comp.cool1:frac_work', 1.0)
        self.pyc_add_cycle_param('comp.cool2:frac_W', 0.0383)
        self.pyc_add_cycle_param('comp.cool2:frac_P', 1.0)
        self.pyc_add_cycle_param('comp.cool2:frac_work', 1.0)
        self.pyc_add_cycle_param('turb.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('turb.cool2:frac_P', 0.0)

        # define the off_design conditions we want to run
        self.od_pts = ['OD1','OD2','OD1dry','OD2dry','OD3dry','OD4dry','OD5dry','OD6dry','OD7dry','OD8dry'] 
        self.od_MNs = [0.000001, 0.8, 0.000001, 0.8, 1.00001, 1.2, 0.6, 1.6, 1.6, 1.8]
        self.od_alts = [0.0, 0.0, 0.0, 0.0, 15000.0, 25000.0, 35000.0, 35000.0, 50000.0, 70000.0]
        self.od_T4s = [2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0, 2370.0]
        self.od_ab_FARs = [0.031523391, 0.022759941, 0, 0, 0, 0, 0, 0, 0, 0]
        self.od_Rlines = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, ABTurbojet(design=False))

            self.set_input_defaults(pt+'.fc.MN', val=self.od_MNs[i])
            self.set_input_defaults(pt+'.fc.alt', val=self.od_alts[i], units='ft')
            self.set_input_defaults(pt+'.balance.rhs:FAR', val=self.od_T4s[i], units='degR'),
            self.set_input_defaults(pt+'.balance.rhs:W', val=self.od_Rlines[i]),
            self.set_input_defaults(pt+'.ab.Fl_I:FAR', val=self.od_ab_FARs[i]),

        self.pyc_connect_des_od('comp.s_PR', 'comp.s_PR')
        self.pyc_connect_des_od('comp.s_Wc', 'comp.s_Wc')
        self.pyc_connect_des_od('comp.s_eff', 'comp.s_eff')
        self.pyc_connect_des_od('comp.s_Nc', 'comp.s_Nc')

        self.pyc_connect_des_od('turb.s_PR', 'turb.s_PR')
        self.pyc_connect_des_od('turb.s_Wp', 'turb.s_Wp')
        self.pyc_connect_des_od('turb.s_eff', 'turb.s_eff')
        self.pyc_connect_des_od('turb.s_Np', 'turb.s_Np')

        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('duct1.Fl_O:stat:area', 'duct1.area')
        self.pyc_connect_des_od('comp.Fl_O:stat:area', 'comp.area')
        self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.pyc_connect_des_od('turb.Fl_O:stat:area', 'turb.area')
        self.pyc_connect_des_od('ab.Fl_O:stat:area', 'ab.area')

if __name__ == "__main__":

    import time

    prob = om.Problem()

    prob.model = mp_abturbojet = MPABTurbojet()

    prob.setup()

    #Define the design point
    prob.set_val('DESIGN.fc.alt', 0.0, units='ft'),
    prob.set_val('DESIGN.fc.MN', 0.000001),
    prob.set_val('DESIGN.balance.rhs:W', 11800.0, units='lbf'),
    prob.set_val('DESIGN.balance.rhs:FAR', 2370.0, units='degR'),
    prob.set_val('DESIGN.comp.PR', 13.5),
    prob.set_val('DESIGN.comp.eff', 0.83),
    prob.set_val('DESIGN.turb.eff', 0.86),

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.01755078
    prob['DESIGN.balance.W'] = 168.00454616
    prob['DESIGN.balance.turb_PR'] = 4.46131867
    prob['DESIGN.fc.balance.Pt'] = 14.6959
    prob['DESIGN.fc.balance.Tt'] = 518.67

    W_guess = [168.0, 225.917, 168.005, 225.917, 166.074, 141.2, 61.70780608, 145.635, 71.53855266, 33.347]
    FAR_guess = [.01755, .016289, .01755, .01629, .0168, .01689, 0.01872827, .016083, 0.01619524, 0.015170]
    Nmech_guess = [8070., 8288.85, 8070, 8288.85, 8197.39, 8181.03, 8902.24164717, 8326.586, 8306.00268554, 8467.2404]
    Pt_guess = [14.696, 22.403, 14.696, 22.403, 15.7034, 13.230, 4.41149502, 14.707, 7.15363767, 3.7009]
    Tt_guess = [518.67, 585.035, 518.67, 585.04, 558.310, 553.409, 422.29146617, 595.796, 589.9425019, 646.8115]
    PR_guess = [4.4613, 4.8185, 4.4613, 4.8185, 4.669, 4.6425, 4.42779036, 4.8803, 4.84652723, 5.11582]

    for i, pt in enumerate(mp_abturbojet.od_pts):

        # initial guesses
        prob[pt+'.balance.W'] = W_guess[i]
        prob[pt+'.balance.FAR'] = FAR_guess[i]
        prob[pt+'.balance.Nmech'] = Nmech_guess[i]
        prob[pt+'.fc.balance.Pt'] = Pt_guess[i]
        prob[pt+'.fc.balance.Tt'] = Tt_guess[i]
        prob[pt+'.turb.PR'] = PR_guess[i]

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+mp_abturbojet.od_pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)