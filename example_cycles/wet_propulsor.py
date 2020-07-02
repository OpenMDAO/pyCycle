import openmdao.api as om

import pycycle.api as pyc


class WetPropulsor(om.Group):

    def initialize(self):
        self.options.declare('design', types=bool, default=True)

    def setup(self):

        thermo_spec = pyc.species_data.wet_air #special species library is called that allows for using initial compositions that include both H and C
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec,
                                                  elements=pyc.WET_AIR_MIX))#WET_AIR_MIX contains standard dry air compounds as well as H2O

        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.WET_AIR_MIX))
        self.add_subsystem('fan', pyc.Compressor(thermo_data=thermo_spec, elements=pyc.WET_AIR_MIX,
                                                 design=design, map_data=pyc.FanMap, map_extrap=True))
        self.add_subsystem('nozz', pyc.Nozzle(thermo_data=thermo_spec, elements=pyc.WET_AIR_MIX))
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

        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        pyc.connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        pyc.connect_flow(self, 'fan.Fl_O', 'nozz.Fl_I')


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


def viewer(prob, pt):


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names)

    pyc.print_compressor(prob, [f'{pt}.fan'])

    pyc.print_nozzle(prob, [f'{pt}.nozz'])

class MPWetPropulsor(om.Group):

    def setup(self):

        des_vars = self.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])
        des_vars.add_output('des:alt', 10000., units="m")
        des_vars.add_output('des:MN', .72)
        des_vars.add_output('des:inlet_MN', .6)
        des_vars.add_output('des:FPR', 1.2)
        des_vars.add_output('des:eff', 0.96)
        des_vars.add_output('des:WAR', .001)

        des_vars.add_output('pwr_target', -2600., units='kW')


        design = self.add_subsystem('design', WetPropulsor(design=True))

        self.connect('des:alt', 'design.fc.alt')
        self.connect('des:MN', 'design.fc.MN')
        self.connect('des:inlet_MN', 'design.inlet.MN')
        self.connect('des:FPR', 'design.fan.PR')
        self.connect('pwr_target', ['design.pwr_target', 'off_design.pwr_target'])
        self.connect('des:eff', 'design.fan.eff')
        self.connect('des:WAR', 'design.fc.WAR')


        des_vars.add_output('OD:alt', 10000, units='m')
        des_vars.add_output('OD:MN', 0.72)
        des_vars.add_output('OD:WAR', .001)

        od = self.add_subsystem('off_design', WetPropulsor(design=False))

        self.connect('OD:alt', 'off_design.fc.alt')
        self.connect('OD:MN', 'off_design.fc.MN')
        self.connect('OD:WAR', 'off_design.fc.WAR')

        # need to pass some design values to the OD point
        # prob.model.connect('design.inlet:ram_recovery', 'off_design.inlet.ram_recovery')
        self.connect('design.inlet.Fl_O:stat:area', 'off_design.inlet.area')

        self.connect('design.fan.s_PR', 'off_design.fan.s_PR')
        self.connect('design.fan.s_Wc', 'off_design.fan.s_Wc')
        self.connect('design.fan.s_eff', 'off_design.fan.s_eff')
        self.connect('design.fan.s_Nc', 'off_design.fan.s_Nc')
        self.connect('design.fan.Fl_O:stat:area', 'off_design.fan.area')

        self.connect('design.nozz.Throat:stat:area', 'off_design.balance.rhs:W')

if __name__ == "__main__":
    import time

    import numpy as np
    np.set_printoptions(precision=5)

    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()
    prob.model = MPWetPropulsor()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)

    prob.setup(check=False)

    prob.model.design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

    prob.model.off_design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.off_design.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.off_design.nonlinear_solver.options['maxiter'] = 10

    # parameters
    prob['des:MN'] = .8
    prob['OD:MN'] = .8

    # initial guess
    prob['design.balance.W'] = 200.

    prob['off_design.balance.W'] = 406.790
    prob['off_design.balance.Nmech'] = 1. # normalized value
    prob['off_design.fan.PR'] = 1.2
    prob['off_design.fan.map.RlineMap'] = 2.2

    st = time.time()
    prob.run_model()
    run_time = time.time() - st

    print("design")

    viewer(prob, 'design')

    print("######"*10)
    print("######"*10)
    print("######"*10)

    viewer(prob, 'off_design')

    print("Run time", run_time)