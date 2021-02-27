import openmdao.api as om

import pycycle.api as pyc


class WetPropulsor(pyc.Cycle):

    def setup(self):

        design = self.options['design']

        # NOTE: DEFAULT TABULAR thermo doesn't include WAR, so must use CEA here
        # (or build your own thermo tables)
        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data'] = pyc.species_data.wet_air

        self.add_subsystem('fc', pyc.FlightConditions(composition=pyc.CEA_AIR_COMPOSITION, 
                                                      reactant='Water',
                                                      mix_ratio_name='WAR'))

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
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = om.DirectSolver(assemble_jac=True)

        # base_class setup should be called as the last thing in your setup
        super().setup()


def viewer(prob, pt):


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names)

    pyc.print_compressor(prob, [f'{pt}.fan'])

    pyc.print_nozzle(prob, [f'{pt}.nozz'])

class MPWetPropulsor(pyc.MPCycle):

    def setup(self):

        design = self.pyc_add_pnt('design', WetPropulsor(design=True, thermo_method='CEA'))

        self.set_input_defaults('design.fc.alt', 10000., units="m")
        self.set_input_defaults('design.fc.MN', .72)
        self.set_input_defaults('design.inlet.MN', .6)
        self.set_input_defaults('design.fc.WAR', .001)

        self.pyc_add_cycle_param('pwr_target', -2600., units='kW')

        self.od_pts = ['off_design']
        self.od_alts = [10000,]
        self.od_MNs = [0.72,]
        self.od_WARs = [.001,]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt('off_design', WetPropulsor(design=False, thermo_method='CEA'))

            self.set_input_defaults(pt+'.fc.alt', self.od_alts[i], units='m')
            self.set_input_defaults(pt+'.fc.MN', self.od_MNs[i])
            self.set_input_defaults(pt+'.fc.WAR', self.od_WARs[i])

        self.pyc_use_default_des_od_conns()

        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()

if __name__ == "__main__":
    import time

    import numpy as np
    np.set_printoptions(precision=5)

    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()
    prob.model = mp_wet_propulsor = MPWetPropulsor()

    prob.setup()

    #Define the design point
    prob.set_val('design.fan.PR', 1.2)
    prob.set_val('design.fan.eff', 0.96)

    # Set initial guesses for balances
    prob['design.fc.MN'] = .8
    prob['design.balance.W'] = 200.


    for i, pt in enumerate(mp_wet_propulsor.od_pts):

        # initial guesses    
        prob['off_design.fc.MN'] = .8
        prob['off_design.balance.W'] = 406.790
        prob['off_design.balance.Nmech'] = 1.
        prob['off_design.fan.PR'] = 1.2
        prob['off_design.fan.map.RlineMap'] = 2.2

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)

    prob.model.design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

    prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.off_design.nonlinear_solver.options['maxiter'] = 10

    prob.run_model()

    run_time = time.time() - st

    print("design")

    viewer(prob, 'design')

    print("######"*10)
    print("######"*10)
    print("######"*10)

    viewer(prob, 'off_design')

    print("Run time", run_time)