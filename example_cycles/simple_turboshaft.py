from __future__ import print_function

import sys

import openmdao.api as om

import pycycle.api as pyc

class Turboshaft(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        # Add engine elements
        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True),
                                    promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                    inflow_elements=pyc.AIR_MIX,
                                    air_fuel_elements=pyc.AIR_FUEL_MIX,
                                    fuel_type='JP-7'))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, map_extrap=True),
                                    promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('pt', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, map_extrap=True),
                                    promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv',
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('HP_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('LP_shaft', pyc.Shaft(num_ports=1),promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'comp.Fl_I')
        pyc.connect_flow(self, 'comp.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'turb.Fl_I')
        pyc.connect_flow(self, 'turb.Fl_O', 'pt.Fl_I')
        pyc.connect_flow(self, 'pt.Fl_O', 'nozz.Fl_I')

        # Connect turbomachinery elements to shaft
        self.connect('comp.trq', 'HP_shaft.trq_0')
        self.connect('turb.trq', 'HP_shaft.trq_1')
        self.connect('pt.trq', 'LP_shaft.trq_0')

        # Connnect nozzle exhaust to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # Connect outputs to pefromance element
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')
        self.connect('LP_shaft.pwr_net', 'perf.power')

        # Add balances for design and off-design
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', val=27.0, units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.PR', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=3.0, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('HP_shaft.pwr_net', 'balance.lhs:turb_PR')

            balance.add_balance('pt_PR', val=3.0, lower=1.001, upper=8, eq_units='hp')
            self.connect('balance.pt_PR', 'pt.PR')
            self.connect('LP_shaft.pwr_net', 'balance.lhs:pt_PR')

        else:

            balance.add_balance('FAR', eq_units='hp', lower=1e-4, val=.3)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('LP_shaft.pwr_net', 'balance.lhs:FAR')

            balance.add_balance('HP_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('HP_shaft.pwr_net', 'balance.lhs:HP_Nmech')

            balance.add_balance('W', val=27.0, units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        # Setup solver to converge engine
        self.set_order(['balance', 'fc', 'inlet', 'comp', 'burner', 'turb', 'pt', 'nozz', 'HP_shaft', 'LP_shaft', 'perf'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
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

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     PSFC ")
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" \
                %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'], \
                prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.PSFC']))    


    fs_names = ['fc.Fl_O','inlet.Fl_O','comp.Fl_O',
                'burner.Fl_O','turb.Fl_O','pt.Fl_O',
                'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['turb','pt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['HP_shaft','LP_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    # Design point inputs
    des_vars.add_output('alt', 0.0, units='ft'),
    des_vars.add_output('MN', 0.000001),
    des_vars.add_output('T4max', 2370.0, units='degR'),
    # des_vars.add_output('Fn_des', 11800.0, units='lbf'),
    des_vars.add_output('pwr_des', 4000.0, units='hp')
    des_vars.add_output('nozz_PR', 1.2)
    des_vars.add_output('comp:PRdes', 13.5),
    des_vars.add_output('comp:effDes', 0.83),
    des_vars.add_output('burn:dPqP', 0.03),
    des_vars.add_output('turb:effDes', 0.86),
    des_vars.add_output('pt:effDes', 0.9),
    des_vars.add_output('nozz:Cv', 0.99),
    des_vars.add_output('HP_shaft:Nmech', 8070.0, units='rpm'),
    des_vars.add_output('LP_shaft:Nmech', 5000.0, units='rpm'),
    des_vars.add_output('inlet:MN_out', 0.60),
    des_vars.add_output('comp:MN_out', 0.20),
    des_vars.add_output('burner:MN_out', 0.20),
    des_vars.add_output('turb:MN_out', 0.4),
    des_vars.add_output('pt:MN_out', 0.5),

    # Off-design (point 1) inputs
    des_vars.add_output('OD1_MN', 0.000001),
    des_vars.add_output('OD1_alt', 0.0, units='ft'),
    des_vars.add_output('OD1_pwr', 3500.0, units='hp')
    des_vars.add_output('OD1_LP_Nmech', 5000., units='rpm')

    # Create design instance of model
    prob.model.add_subsystem('DESIGN', Turboshaft())

    # Connect design point inputs to model
    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    # prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
    prob.model.connect('pwr_des', 'DESIGN.balance.rhs:pt_PR')
    prob.model.connect('nozz_PR', 'DESIGN.balance.rhs:W')

    prob.model.connect('comp:PRdes', 'DESIGN.comp.PR')
    prob.model.connect('comp:effDes', 'DESIGN.comp.eff')
    prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('turb:effDes', 'DESIGN.turb.eff')
    prob.model.connect('pt:effDes', 'DESIGN.pt.eff')
    prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
    prob.model.connect('HP_shaft:Nmech', 'DESIGN.HP_Nmech')
    prob.model.connect('LP_shaft:Nmech', 'DESIGN.LP_Nmech')

    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
    prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
    prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')

    # Connect off-design and required design inputs to model
    pts = ['OD1']

    for pt in pts:
        prob.model.add_subsystem(pt, Turboshaft(design=False))

        prob.model.connect('burn:dPqP', pt+'.burner.dPqP')
        prob.model.connect('nozz:Cv', pt+'.nozz.Cv')

        prob.model.connect(pt+'_alt', pt+'.fc.alt')
        prob.model.connect(pt+'_MN', pt+'.fc.MN')
        prob.model.connect(pt+'_LP_Nmech', pt+'.LP_Nmech')
        prob.model.connect(pt+'_pwr', pt+'.balance.rhs:FAR')

        prob.model.connect('DESIGN.comp.s_PR', pt+'.comp.s_PR')
        prob.model.connect('DESIGN.comp.s_Wc', pt+'.comp.s_Wc')
        prob.model.connect('DESIGN.comp.s_eff', pt+'.comp.s_eff')
        prob.model.connect('DESIGN.comp.s_Nc', pt+'.comp.s_Nc')

        prob.model.connect('DESIGN.turb.s_PR', pt+'.turb.s_PR')
        prob.model.connect('DESIGN.turb.s_Wp', pt+'.turb.s_Wp')
        prob.model.connect('DESIGN.turb.s_eff', pt+'.turb.s_eff')
        prob.model.connect('DESIGN.turb.s_Np', pt+'.turb.s_Np')

        prob.model.connect('DESIGN.pt.s_PR', pt+'.pt.s_PR')
        prob.model.connect('DESIGN.pt.s_Wp', pt+'.pt.s_Wp')
        prob.model.connect('DESIGN.pt.s_eff', pt+'.pt.s_eff')
        prob.model.connect('DESIGN.pt.s_Np', pt+'.pt.s_Np')

        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')
        prob.model.connect('DESIGN.pt.Fl_O:stat:area', pt+'.pt.area')

        # prob.model.connect(pt+'_T4', pt+'.balance.rhs:FAR')
        # prob.model.connect(pt+'_pwr', pt+'.balance.rhs:FAR')
        prob.model.connect('DESIGN.nozz.Throat:stat:area', pt+'.balance.rhs:W')


    prob.setup(check=False)

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.0175506829934
    prob['DESIGN.balance.W'] = 27.265
    prob['DESIGN.balance.turb_PR'] = 3.8768
    prob['DESIGN.balance.pt_PR'] = 2.8148
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.665288153

    for pt in pts:
        prob[pt+'.balance.W'] = 27.265
        prob[pt+'.balance.FAR'] = 0.0175506829934
        # prob[pt+'.balance.Nmech'] = 8070.0
        prob[pt+'.fc.balance.Pt'] = 15.703
        prob[pt+'.fc.balance.Tt'] = 558.31
        prob[pt+'.turb.PR'] = 3.8768
        prob[pt+'.pt.PR'] = 2.8148

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)
