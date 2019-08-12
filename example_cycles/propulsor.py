from __future__ import print_function

from openmdao.api import Group, IndepVarComp, BalanceComp
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from pycycle.constants import AIR_MIX
from pycycle.connect_flow import connect_flow
from pycycle.cea import species_data
from pycycle.elements.api import FlightConditions, Inlet, Compressor, Nozzle, Performance


class Propulsor(Group):

    def setup(self):

        thermo_spec = species_data.janaf

        self.add_subsystem('fc', FlightConditions(thermo_data=thermo_spec,
                                                  elements=AIR_MIX))

        self.add_subsystem('inlet', Inlet(design=True, thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('fan', Compressor(thermo_data=thermo_spec, elements=AIR_MIX, design=True))
        self.add_subsystem('nozz', Nozzle(thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('perf', Performance(num_nozzles=1, num_burners=0))

        self.add_subsystem('shaft', IndepVarComp('Nmech', 1., units='rpm'))

        self.add_subsystem('pwr_balance', BalanceComp('W', units='lbm/s', eq_units='hp', val=50., lower=1., upper=500.),
                           promotes_inputs=[('rhs:W', 'pwr_target')])

        # self.set_order(['pwr_balance', 'pwr_target', 'fc', 'inlet', 'shaft', 'fan', 'nozz', 'perf'])

        connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        connect_flow(self, 'fan.Fl_O', 'nozz.Fl_I')

        self.connect('shaft.Nmech', 'fan.Nmech')

        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('pwr_balance.W', 'fc.W')
        self.connect('fan.power', 'pwr_balance.lhs:W')

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        #
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = DirectSolver(assemble_jac=True)


if __name__ == "__main__":
    import time

    import numpy as np
    np.set_printoptions(precision=5)

    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = Problem()

    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])
    des_vars.add_output('alt', 10000., units="m")
    des_vars.add_output('MN', .72)
    des_vars.add_output('inlet_MN', .6)
    des_vars.add_output('FPR', 1.2)
    des_vars.add_output('pwr_target', -2600., units='kW')
    des_vars.add_output('eff', 0.96)

    design = prob.model.add_subsystem('design', Propulsor())

    prob.model.connect('alt', 'design.fc.alt')
    prob.model.connect('MN', 'design.fc.MN')
    prob.model.connect('inlet_MN', 'design.inlet.MN')
    prob.model.connect('FPR', 'design.fan.PR')
    prob.model.connect('pwr_target', 'design.pwr_target')
    prob.model.connect('eff', 'design.fan.eff')

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)

    prob.setup(check=False)

    design.nonlinear_solver.options['atol'] = 1e-6
    design.nonlinear_solver.options['rtol'] = 1e-6

    # from openmdao.api import view_model
    # view_model(prob)
    # exit()

    # parameters
    prob['MN'] = .8

    # initial guess
    prob['design.pwr_balance.W'] = 200.

    st = time.time()
    prob.run_model()
    run_time = time.time() - st

    # prob.check_partials(comps=['design.pwr_balance'])
    # exit()
    # prob.model.list_states()


    print("foo FS Fl_O:tot:P", prob['design.fc.Fl_O:tot:P'])
    print("foo FS Fl_O:stat:P", prob['design.fc.Fl_O:stat:P'])


    print("design")

    print("shaft power (hp)", prob['design.fan.power'])
    print("W (lbm/s)", prob['design.pwr_balance.W'])
    print()
    print("MN", prob['MN'])
    print("FS Fl_O:stat:P", cu(prob['design.fc.Fl_O:stat:P'], 'lbf/inch**2', 'Pa'))
    print("FS Fl_O:stat:T", cu(prob['design.fc.Fl_O:stat:T'], 'degR', 'degK'))
    print("FS Fl_O:tot:P", cu(prob['design.fc.Fl_O:tot:P'], 'lbf/inch**2', 'Pa'))
    print("FS Fl_O:tot:T", cu(prob['design.fc.Fl_O:stat:T'], 'degR', 'degK'))
    print()
    print("Inlet Fl_O:stat:P", cu(prob['design.inlet.Fl_O:stat:P'], 'lbf/inch**2', 'Pa'))
    print("Inlet Fl_O:stat:area", prob['design.inlet.Fl_O:stat:area'])
    print("Fan Fl_O:stat:W", prob['design.inlet.Fl_O:stat:W'])
    # print('NPR', prob['design.fan.Fl_O:tot:P']/prob['design.fc.Fl_O:stat:P'])

    print("Run time", run_time)
