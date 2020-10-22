import openmdao.api as om

from pycycle.constants import AIR_ELEMENTS
from pycycle.thermo.cea import species_data
from pycycle.elements.flow_start import FlowStart


class CFDStart(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', default=species_data.janaf,
                             desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the flow')

    def setup(self):
        thermo_data = self.options['thermo_data']
        elements = self.options['elements']

        self.add_subsystem('fs', FlowStart(thermo_data=thermo_data, elements=elements), promotes_outputs=['Fl_O:*'],
                           promotes_inputs=['W'])

        balance = om.BalanceComp()
        balance.add_balance('P', val=10., units='psi', eq_units='psi', lhs_name='Ps_computed', rhs_name='Ps',
                            lower=1e-1)
                            #guess_func=lambda inputs, resids: 5.)
        balance.add_balance('T', val=800., units='degR', eq_units='ft/s', lhs_name='V_computed', rhs_name='V',
                            lower=1e-1)
                            #guess_func=lambda inputs, resids: 400.)
        balance.add_balance('MN', val=.3, eq_units='inch**2', lhs_name='area_computed', rhs_name='area', lower=1e-6)
                            #guess_func=lambda inputs, resids: .6)

        self.add_subsystem('balance', balance, promotes_inputs=['Ps', 'V', 'area'])
        self.connect('Fl_O:stat:P', 'balance.Ps_computed')
        self.connect('Fl_O:stat:V', 'balance.V_computed')
        self.connect('Fl_O:stat:area', 'balance.area_computed')

        self.connect('balance.P', 'fs.P')
        self.connect('balance.T', 'fs.T')
        self.connect('balance.MN', 'fs.MN')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['solve_subsystems'] = True
        newton.options['maxiter'] = 10
        # newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['print_bound_enforce'] = True

        self.linear_solver = om.DirectSolver(assemble_jac=True)



if __name__ == "__main__":

    p = om.Problem()

    params = p.model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    params.add_output('Ps', units='Pa', val=22845.15677648)
    params.add_output('V', units='m/s', val=158.83851913)
    params.add_output('area', units='m**2', val=0.87451328)
    params.add_output('W', units='kg/s', val=50.2454107)

    p.model.add_subsystem('cfd_start', CFDStart(), promotes_inputs=['Ps', 'V', 'area', 'W'])

    p.setup(check=False)

    p.set_solver_print(level=-1)
    p.set_solver_print(level=2, depth=1)


    p.run_model()

    p.model.list_outputs(residuals=True)





