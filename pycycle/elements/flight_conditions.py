import openmdao.api as om

from pycycle.thermo.cea import species_data
from pycycle.constants import AIR_ELEMENTS
from pycycle.elements.ambient import Ambient
from pycycle.elements.flow_start import FlowStart


class FlightConditions(om.Group):
    """Determines total and static flow properties given an altitude and Mach number using the input atmosphere model"""

    def initialize(self):
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_ELEMENTS,
                              desc='set of elements present in the flow')
        self.options.declare('use_WAR', default=False, values=[True, False], 
                              desc='If True, includes WAR calculation')

    def setup(self):
        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        use_WAR = self.options['use_WAR']

        self.add_subsystem('ambient', Ambient(), promotes=('alt', 'dTs'))  # inputs

        conv = self.add_subsystem('conv', om.Group(), promotes=['*'])
        if use_WAR == True:
            proms = ['Fl_O:*', 'MN', 'W', 'WAR']
        else:
            proms = ['Fl_O:*', 'MN', 'W']
        conv.add_subsystem('fs', FlowStart(thermo_data=thermo_data, 
                                           elements=elements, 
                                           use_WAR=use_WAR), 
                           promotes=proms)
        balance = conv.add_subsystem('balance', om.BalanceComp())
        balance.add_balance('Tt', val=500.0, lower=1e-4, units='degR', desc='Total temperature', eq_units='degR')
        balance.add_balance('Pt', val=14.696, lower=1e-4, units='psi', desc='Total pressure', eq_units='psi')
        # sub.set_order(['fs','balance'])

        newton = conv.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-10
        newton.options['rtol'] = 1e-10
        newton.options['maxiter'] = 10
        newton.options['iprint'] = -1
        newton.options['solve_subsystems'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'

        newton.linesearch.options['iprint'] = -1
        # newton.linesearch.options['solve_subsystems'] = True

        conv.linear_solver = om.DirectSolver(assemble_jac=True)

        self.connect('ambient.Ps', 'balance.rhs:Pt')
        self.connect('ambient.Ts', 'balance.rhs:Tt')

        self.connect('balance.Pt', 'fs.P')
        self.connect('balance.Tt', 'fs.T')

        self.connect('Fl_O:stat:P', 'balance.lhs:Pt')
        self.connect('Fl_O:stat:T', 'balance.lhs:Tt')

        # self.set_order(['ambient', 'subgroup'])


if __name__ == "__main__":

    p1 = om.Problem()
    p1.model = om.Group()

    des_vars = p1.model.add_subsystem('des_vars', om.IndepVarComp())
    des_vars.add_output('W', 0.0, units='lbm/s')
    des_vars.add_output('alt', 1., units='ft')
    des_vars.add_output('MN', 0.5)
    des_vars.add_output('dTs', 0.0, units='degR')


    fc = p1.model.add_subsystem("fc", FlightConditions())

    p1.model.connect('des_vars.W', 'fc.W')
    p1.model.connect('des_vars.alt', 'fc.alt')
    p1.model.connect('des_vars.MN', 'fc.MN')
    p1.model.connect('des_vars.dTs', 'fc.dTs')

    p1.setup()

    # p1.root.list_connections()

    p1['des_vars.alt'] = 17868.79060515557
    p1['des_vars.MN'] = 2.101070288213628
    p1['des_vars.dTs'] = 0.0
    p1['des_vars.W'] = 1.0

    p1.run_model()

    print('Ts_atm: ', p1['fc.ambient.Ts'])
    print('Ts_set: ', p1['fc.Fl_O:stat:T'])
    print('Ps_atm: ', p1['fc.ambient.Ps'])
    print('Ps_set: ', p1['fc.Fl_O:stat:P'])
    print('rhos_atm: ', p1['fc.ambient.rhos']*32.175)
    print('rhos_set: ', p1['fc.Fl_O:stat:rho'])
    print('W', p1['fc.Fl_O:stat:W'])

    print('Pt: ', p1['fc.Fl_O:tot:P'])
