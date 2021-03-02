import openmdao.api as om

from pycycle.constants import THERMO_DEFAULT_COMPOSITIONS
from pycycle.thermo.cea import species_data
from pycycle.elements.flow_start import FlowStart
from pycycle.element_base import Element

class CFDStart(Element):

    def initialize(self):
        self.options.declare('composition', default=None,
                              desc='composition of the flow. If None, default for thermo package is used')
        super().initialize()
        

    def pyc_setup_output_ports(self): 
        thermo_method = self.options['thermo_method']
        composition = self.options['composition']
        if composition is None: 
            composition = THERMO_DEFAULT_COMPOSITIONS[thermo_method]
        self.init_output_flow('Fl_O', composition)


    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        
        composition = self.Fl_O_data['Fl_O']


        fs = self.add_subsystem('fs', FlowStart(thermo_method=thermo_method,thermo_data=thermo_data, 
                                composition=composition), promotes_outputs=['Fl_O:*'],promotes_inputs=['W'])
        fs.pyc_setup_output_ports()


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





