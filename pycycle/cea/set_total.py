import openmdao.api as om

from pycycle.constants import AIR_MIX
from pycycle.cea.chem_eq import ChemEq
from pycycle.cea.props_rhs import PropsRHS
from pycycle.cea.props_calcs import PropsCalcs
from pycycle.cea.static_ps_resid import PsResid
from pycycle.cea.static_ps_calc import PsCalc
from pycycle.cea.unit_comps import EngUnitProps
from pycycle.cea.species_data import Thermo


class Properties(om.Group):

    def initialize(self):
        self.options.declare('thermo', desc='thermodynamic data object', recordable=False)

    def setup(self):
        thermo = self.options['thermo']

        num_element = thermo.num_element

        self.add_subsystem('TP2ls', PropsRHS(thermo), promotes_inputs=('T', 'n', 'n_moles', 'b0'))

        ne1 = num_element+1
        self.add_subsystem('ls2t', om.LinearSystemComp(size=ne1))
        self.add_subsystem('ls2p', om.LinearSystemComp(size=ne1))

        self.add_subsystem('tp2props', PropsCalcs(thermo=thermo),
                           promotes_inputs=['n', 'n_moles', 'T', 'P'],
                           promotes_outputs=['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']
                           )
        self.connect('TP2ls.lhs_TP', ['ls2t.A', 'ls2p.A'])
        self.connect('TP2ls.rhs_T', 'ls2t.b')
        self.connect('TP2ls.rhs_P', 'ls2p.b')
        self.connect('ls2t.x', 'tp2props.result_T')
        self.connect('ls2p.x', 'tp2props.result_P')


class SetTotal(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False)
        self.options.declare('fl_name',
                              default="flow",
                              desc='flowstation name of the output flow variables')
        self.options.declare('mode',
                              desc='the input variable that defines the total properties',
                              default='T',
                              values=('T', 'S', 'h'))
        self.options.declare('init_reacts',
                              default=AIR_MIX,
                              desc='initial amounts of each species in the flow')
        self.options.declare('for_statics',
                              default=False,
                              values=(False, 'Ps', 'MN', 'area'),
                              desc='flag that alters configuration if being used for a static calculation')


    def setup(self):
        #, thermo_data, mode='T', fl_name='flow', init_reacts=AIR_MIX):

        thermo_data = self.options['thermo_data']
        init_reacts = self.options['init_reacts']
        fl_name = self.options['fl_name']
        mode = self.options['mode']
        for_statics = self.options['for_statics']

        thermo = Thermo(thermo_data, init_reacts)

        # chem_eq calculations
        in_vars = ('b0', 'P')
        out_vars = ('n', 'n_moles')
        if mode == 'T':
            in_vars += ('T', )
        elif mode == 'h':
            in_vars += ('h',)
            out_vars += ('T', )
        elif mode == 'S':
            in_vars += ('S', )
            out_vars += ('T', )

        self.ceq = self.add_subsystem('chem_eq', ChemEq(thermo=thermo, mode=mode),
                           promotes_inputs=in_vars,
                           promotes_outputs=out_vars,
                           )

        out_vars = ('gamma', 'Cp', 'Cv', 'rho', 'R')
        if mode == 'h':
            out_vars += ('S',)
        elif mode == 'S':
            out_vars += ('h',)
        else:
            out_vars += ('S', 'h')

        self.add_subsystem('props', Properties(thermo=thermo),
                           promotes_inputs=('T', 'P', 'n', 'n_moles', 'b0'),
                           promotes_outputs=out_vars)

        if for_statics:  # created after props to keep the execution order
            if for_statics == 'MN':
                self.add_subsystem('ps_resid', PsResid(mode=for_statics),
                                   promotes_inputs=['ht', 'n_moles', 'gamma', 'W',
                                                    'rho', 'MN', 'guess:*', ('Ts', 'T'), ('hs', 'h')],
                                   promotes_outputs=['V', 'Vsonic', 'area', 'Ps'])

                self.connect('Ps', 'P')  # create the cyclic data connection for the static solve
            elif for_statics == 'area':
                self.add_subsystem('ps_resid', PsResid(mode=for_statics),
                                   promotes_inputs=['ht', 'n_moles', 'gamma', 'W',
                                                    'rho', 'area', 'guess:*', ('Ts', 'T'), ('hs', 'h')],
                                   promotes_outputs=['V', 'Vsonic', 'MN', 'Ps'])

                self.connect('Ps', 'P') # create the cyclic data connection for the static solve
            else:
                self.add_subsystem('ps_calc', PsCalc(thermo=thermo),
                                   promotes_inputs=['P', 'gamma', 'n_moles', 'ht', 'W', 'rho',
                                                    ('Ts', 'T'), ('hs', 'h')],
                                   promotes_outputs=['MN', 'V', 'Vsonic', 'area']
                                   )

            self.set_input_defaults('W', units='kg/s')
        else:
            self.add_subsystem('flow', EngUnitProps(thermo=thermo, fl_name=fl_name),
                               promotes_inputs=('T', 'P', 'h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles', 'R'),
                               promotes_outputs=('{}:*'.format(fl_name),))


            # self.set_order(['chem_eq', 'props', 'flow'])


    def configure(self):

        for_statics = self.options['for_statics']
        if for_statics and for_statics != 'Ps':

            self.ceq.nonlinear_solver.options['atol'] = 1e-10
            self.ceq.nonlinear_solver.options['rtol'] = 1e-6

            # statics need an newton solver to converge the outer loop with Ps
            newton = self.nonlinear_solver = om.NewtonSolver()
            newton.options['atol'] = 1e-10
            newton.options['rtol'] = 1e-10
            newton.options['maxiter'] = 50
            newton.options['iprint'] = 2
            newton.options['solve_subsystems'] = True
            newton.options['max_sub_solves'] = 50
            newton.options['reraise_child_analysiserror'] = False

            self.options['assembled_jac_type'] = 'dense'
            newton.linear_solver = om.DirectSolver(assemble_jac=True)

            ln_bt = newton.linesearch = om.BoundsEnforceLS()
            ln_bt.options['bound_enforcement'] = 'scalar'
            ln_bt.options['iprint'] = -1


if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data

    thermo = species_data.Thermo(species_data.co2_co_o2)
    # thermo = species_data.Thermo(species_data.janaf)

    prob = om.Problem()
    prob.model = om.Group()

    prob.model.add_subsystem('totals',
                             SetTotal(thermo_data=species_data.janaf,
                                      fl_name="flow",
                                      init_reacts=species_data.co2_co_o2.init_prod_amounts,
                                      mode="h"),
                             promotes_inputs=['h', 'P', 'b0'],
                             promotes_outputs=['flow:*'])

    prob.model.totals.set_input_defaults('b0', thermo.b0)
    prob.model.totals.set_input_defaults('P', 1.034210, units="bar")
    prob.model.totals.set_input_defaults('h', 1., units="Btu/lbm")
    prob.model.suppress_solver_output = True
    prob.setup()

    # from openmdao.api import view_model
    # view_model(prob)
    # exit(0)

    prob.run_model()
    print('gamma', prob['flow:gamma'])
    print('P', prob['flow:P'])
    print('h', prob['flow:h'])
    print('rho', prob['flow:rho'])
    print('S', prob['flow:S'])

    prob['P'] = 4.0


    prob.run_model()
    print("#"*50)
    print('gamma', prob['flow:gamma'])
    print('T', prob['flow:T'])
    print('P', prob['flow:P'])
    print('h', prob['flow:h'])
    print('n', prob['flow:n'])



    prob = om.Problem()
    prob.model = om.Group()

    prob.model.add_subsystem('totals',
                             SetTotal(thermo_data=species_data.janaf,
                                      fl_name="flow", for_statics='area',
                                      init_reacts=species_data.co2_co_o2.init_prod_amounts,
                                      mode="h"),
                             promotes_inputs=['h', 'P', 'b0'])

    prob.model.totals.set_input_defaults('b0', thermo.b0)
    prob.model.totals.set_input_defaults('P', 1.034210, units="bar")
    prob.model.totals.set_input_defaults('h', 100., units="Btu/lbm")
    prob.model.suppress_solver_output = True
    prob.setup()

    # from openmdao.api import view_model
    # view_model(prob)
    # exit(0)

    prob.run_model()
    prob.model.list_inputs()