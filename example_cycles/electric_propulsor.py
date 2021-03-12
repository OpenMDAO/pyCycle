import openmdao.api as om

import pycycle.api as pyc


class Propulsor(pyc.Cycle):

    def setup(self):

        design = self.options['design']

        USE_TABULAR = True
        if USE_TABULAR: 
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
        else: 
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = 'JP-7'


        self.add_subsystem('fc', pyc.FlightConditions())

        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap, map_extrap=True))
        self.add_subsystem('nozz', pyc.Nozzle())
        
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=0))


        balance = om.BalanceComp()
        if design:
            self.add_subsystem('shaft', om.IndepVarComp('Nmech', 1., units='rpm'))
            self.connect('shaft.Nmech', 'fan.Nmech')

            balance.add_balance('W', units='lbm/s', eq_units='hp', val=50., lower=1., upper=500.)
            self.add_subsystem('balance', balance,
                               promotes_inputs=[('rhs:W', 'pwr_target')])
            self.connect('fan.power', 'balance.lhs:W')



        else:
            # vary mass flow till the nozzle area matches the design values
            balance.add_balance('W', units='lbm/s', eq_units='inch**2', val=50, lower=1., upper=500.)
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('Nmech', val=1., units='rpm', lower=0.1, upper=2.0, eq_units='hp')
            self.connect('balance.Nmech', 'fan.Nmech')
            self.connect('fan.power', 'balance.lhs:Nmech')

            # self.add_subsystem('shaft', om.IndepVarComp('Nmech', 1., units='rpm'))
            # self.connect('shaft.Nmech', 'fan.Nmech')

            self.add_subsystem('balance', balance,
                               promotes_inputs=[('rhs:Nmech', 'pwr_target')])

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'nozz.Fl_I')


        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('balance.W', 'fc.W')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['reraise_child_analysiserror'] = False
        #
        # newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['maxiter'] = 3
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = om.DirectSolver()

        # base_class setup should be called as the last thing in your setup
        super().setup()

def viewer(prob, pt):
    """
    print a report of all the relevant cycle properties
    """

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names)

    pyc.print_compressor(prob, [f'{pt}.fan'])

    pyc.print_nozzle(prob, [f'{pt}.nozz'])

def map_plots(prob, pt):
    comp_names = ['fan']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)



class MPpropulsor(pyc.MPCycle):

    def setup(self):

        design = self.pyc_add_pnt('design', Propulsor(design=True, thermo_method='CEA'))
        self.pyc_add_cycle_param('pwr_target', 100.)

        # define the off-design conditions we want to run
        self.od_pts = ['off_design']
        self.od_MNs = [0.8,]
        self.od_alts = [10000,]
        self.od_Rlines = [2.2,]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, Propulsor(design=False, thermo_method='CEA'))

            self.set_input_defaults(pt+'.fc.MN', val=self.od_MNs[i])
            self.set_input_defaults(pt+'.fc.alt', val=self.od_alts, units='m') 
            self.set_input_defaults(pt+'.fan.map.RlineMap', val=self.od_Rlines[i])        

        self.pyc_use_default_des_od_conns()

        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()
        


if __name__ == "__main__":
    import time

    import numpy as np

    prob = om.Problem()

    prob.model = mp_propulsor = MPpropulsor()


    prob.setup()

    #Define the design point
    prob.set_val('design.fc.alt', 10000, units='m')
    prob.set_val('design.fc.MN', 0.8)
    prob.set_val('design.inlet.MN', 0.6)
    prob.set_val('design.fan.PR', 1.2)
    prob.set_val('pwr_target', -3486.657, units='hp')
    prob.set_val('design.fan.eff', 0.96)

    # Set initial guesses for balances
    prob['design.balance.W'] = 200.
    
    for i, pt in enumerate(mp_propulsor.od_pts):
    
        # initial guesses
        prob['off_design.fan.PR'] = 1.2
        prob['off_design.balance.W'] = 406.790
        prob['off_design.balance.Nmech'] = 1. # normalized value

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)
    prob.model.design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

    prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6


    prob.run_model()
    run_time = time.time() - st

    for pt in ['design']+mp_propulsor.od_pts:
        print('\n', '#'*10, pt, '#'*10)
        viewer(prob, pt)

    map_plots(prob,'design')


    print("Run time", run_time)
