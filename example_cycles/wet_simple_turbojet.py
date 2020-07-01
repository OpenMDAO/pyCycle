import sys

import openmdao.api as om

import pycycle.api as pyc


class WetTurbojet(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('WAR', default=.001, desc='water to air ratio')

    def setup(self):

        wet_thermo_spec = pyc.species_data.wet_air #special species library is called that allows for using initial compositions that include both H and C
        janaf_thermo_spec = pyc.species_data.janaf #standard species library is called for use in and after burner
        design = self.options['design']
        WAR = self.options['WAR']

        # Add engine elements
        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=wet_thermo_spec, WAR=WAR,
                                    elements=pyc.WET_AIR_MIX))#WET_AIR_MIX contains standard dry air compounds as well as H2O
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=wet_thermo_spec,
                                    elements=pyc.WET_AIR_MIX))
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, design=design,
                                    thermo_data=wet_thermo_spec, elements=pyc.WET_AIR_MIX,),
                                    promotes_inputs=['Nmech'])

        ###Note###
        #The Combustor element automatically assumes that the thermo data to use for both the inflowing air 
        #and the outflowing mixed air and fuel is the data specified by the thermo_data option
        #unless the inflow_thermo_data option is set. If the inflow_thermo_data option is set,
        #the Combustor element will use the thermo data specified by inflow_thermo_data for the inflowing air
        #to the burner, and it will use the thermo data specified by thermo_data for the outflowing mixed
        #air and fuel. This is necessary to do if the airflow upstream of the burner contains both C and H
        #within its compounds, because without the addition of the hydrocarbons from fuel, the solver has
        #a difficult time converging the trace amount of hydrocarbons "present" in the original flow.

        self.add_subsystem('burner', pyc.Combustor(design=design,inflow_thermo_data=wet_thermo_spec,
                                    thermo_data=janaf_thermo_spec, inflow_elements=pyc.WET_AIR_MIX,
                                    air_fuel_elements=pyc.AIR_FUEL_MIX,
                                    fuel_type='JP-7'))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=janaf_thermo_spec, elements=pyc.AIR_FUEL_MIX,),
                                    promotes_inputs=['Nmech'])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv',
                                    thermo_data=janaf_thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'comp.Fl_I')
        pyc.connect_flow(self, 'comp.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'turb.Fl_I')
        pyc.connect_flow(self, 'turb.Fl_O', 'nozz.Fl_I')

        # Connect turbomachinery elements to shaft
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # Connnect nozzle exhaust to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # Connect outputs to pefromance element
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        # Add balances for design and off-design
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

        else:

            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.3)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            balance.add_balance('W', val=168.0, units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        # Setup solver to converge engine
        self.set_order(['balance', 'fc', 'inlet', 'comp', 'burner', 'turb', 'nozz', 'shaft', 'perf'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch = ArmijoGoldsteinLS()
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


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'comp.Fl_O', 'burner.Fl_O',
                'turb.Fl_O', 'nozz.Fl_O']
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

class MPWetTurbojet(om.Group):

    def setup(self):

        des_vars = self.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

        # Design point inputs
        des_vars.add_output('alt', 0.0, units='ft'),
        des_vars.add_output('MN', 0.000001),
        des_vars.add_output('T4max', 2370.0, units='degR'),
        des_vars.add_output('Fn_des', 11800.0, units='lbf'),
        des_vars.add_output('comp:PRdes', 13.5),
        des_vars.add_output('comp:effDes', 0.83),
        des_vars.add_output('burn:dPqP', 0.03),
        des_vars.add_output('turb:effDes', 0.86),
        des_vars.add_output('nozz:Cv', 0.99),
        des_vars.add_output('shaft:Nmech', 8070.0, units='rpm'),
        des_vars.add_output('inlet:MN_out', 0.60),
        des_vars.add_output('comp:MN_out', 0.20),
        des_vars.add_output('burner:MN_out', 0.20),
        des_vars.add_output('turb:MN_out', 0.4),

        # Off-design (point 1) inputs
        des_vars.add_output('OD1_MN', 0.000001),
        des_vars.add_output('OD1_alt', 0.0, units='ft'),
        des_vars.add_output('OD1_Fn', 11000.0, units='lbf')

        # Create design instance of model
        self.add_subsystem('DESIGN', WetTurbojet(WAR=.001))

        # Connect design point inputs to model
        self.connect('alt', 'DESIGN.fc.alt')
        self.connect('MN', 'DESIGN.fc.MN')
        self.connect('Fn_des', 'DESIGN.balance.rhs:W')
        self.connect('T4max', 'DESIGN.balance.rhs:FAR')

        self.connect('comp:PRdes', 'DESIGN.comp.PR')
        self.connect('comp:effDes', 'DESIGN.comp.eff')
        self.connect('burn:dPqP', 'DESIGN.burner.dPqP')
        self.connect('turb:effDes', 'DESIGN.turb.eff')
        self.connect('nozz:Cv', 'DESIGN.nozz.Cv')
        self.connect('shaft:Nmech', 'DESIGN.Nmech')

        self.connect('inlet:MN_out', 'DESIGN.inlet.MN')
        self.connect('comp:MN_out', 'DESIGN.comp.MN')
        self.connect('burner:MN_out', 'DESIGN.burner.MN')
        self.connect('turb:MN_out', 'DESIGN.turb.MN')

        # Connect off-design and required design inputs to model
        pts = ['OD1']

        for pt in pts:
            self.add_subsystem(pt, WetTurbojet(design=False, WAR=.001))

            self.connect('burn:dPqP', pt+'.burner.dPqP')
            self.connect('nozz:Cv', pt+'.nozz.Cv')

            self.connect(pt+'_alt', pt+'.fc.alt')
            self.connect(pt+'_MN', pt+'.fc.MN')

            self.connect('DESIGN.comp.s_PR', pt+'.comp.s_PR')
            self.connect('DESIGN.comp.s_Wc', pt+'.comp.s_Wc')
            self.connect('DESIGN.comp.s_eff', pt+'.comp.s_eff')
            self.connect('DESIGN.comp.s_Nc', pt+'.comp.s_Nc')

            self.connect('DESIGN.turb.s_PR', pt+'.turb.s_PR')
            self.connect('DESIGN.turb.s_Wp', pt+'.turb.s_Wp')
            self.connect('DESIGN.turb.s_eff', pt+'.turb.s_eff')
            self.connect('DESIGN.turb.s_Np', pt+'.turb.s_Np')

            self.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
            self.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
            self.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
            self.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')

            # self.connect(pt+'_T4', pt+'.balance.rhs:FAR')
            self.connect(pt+'_Fn', pt+'.balance.rhs:FAR')
            self.connect('DESIGN.nozz.Throat:stat:area', pt+'.balance.rhs:W')


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    prob.model = MPWetTurbojet()
    
    pts = ['OD1']

    prob.setup(check=False)

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.0175506829934
    prob['DESIGN.balance.W'] = 168.453135137
    prob['DESIGN.balance.turb_PR'] = 4.46138725662
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.665288153

    for pt in pts:
        prob[pt+'.balance.W'] = 166.073
        prob[pt+'.balance.FAR'] = 0.01680
        prob[pt+'.balance.Nmech'] = 8197.38
        prob[pt+'.fc.balance.Pt'] = 15.703
        prob[pt+'.fc.balance.Tt'] = 558.31
        prob[pt+'.turb.PR'] = 4.6690

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)