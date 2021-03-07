import numpy as np
import openmdao.api as om

from pycycle.thermo.thermo import Thermo, ThermoAdd

from pycycle.thermo.cea.species_data import janaf
from pycycle.flow_in import FlowIn
from pycycle.element_base import Element


class MixImpulse(om.ExplicitComponent):
    """
    Compute the combined impulse of two streams
    """        

    def setup(self):


        self.add_input('Fl_I1:stat:W', val=0.0, units='kg/s', desc='mass flow for flow 1')
        self.add_input('Fl_I1:stat:P', val=0.0, units='Pa', desc='static pressure for flow 1')
        self.add_input('Fl_I1:stat:V', val=0.0, units='m/s', desc='velocity for flow 1')
        self.add_input('Fl_I1:stat:area', val=0.0, units='m**2', desc='area for flow 1')


        self.add_input('Fl_I2:stat:W', val=0.0, units='kg/s', desc='mass flow for flow 2')
        self.add_input('Fl_I2:stat:P', val=0.0, units='Pa', desc='static pressure for flow 2')
        self.add_input('Fl_I2:stat:V', val=0.0, units='m/s', desc='velocity for flow 2')
        self.add_input('Fl_I2:stat:area', val=0.0, units='m**2', desc='area for flow 2')

        self.add_output('impulse_mix', val=0., units='N', desc='impulse of the outgoing flow')


        
        self.declare_partials('impulse_mix', ['Fl_I1:stat:P', 'Fl_I1:stat:area', 'Fl_I1:stat:W', 'Fl_I1:stat:V',
                                              'Fl_I2:stat:P', 'Fl_I2:stat:area', 'Fl_I2:stat:W', 'Fl_I2:stat:V'])

        self.set_check_partial_options('*', method='cs')


    def compute(self, inputs, outputs):

        outputs['impulse_mix'] = (inputs['Fl_I1:stat:P']*inputs['Fl_I1:stat:area'] + inputs['Fl_I1:stat:W']*inputs['Fl_I1:stat:V']) +\
                                 (inputs['Fl_I2:stat:P']*inputs['Fl_I2:stat:area'] + inputs['Fl_I2:stat:W']*inputs['Fl_I2:stat:V'])


    def compute_partials(self, inputs, J):

        J['impulse_mix', 'Fl_I1:stat:P'] = inputs['Fl_I1:stat:area']
        J['impulse_mix', 'Fl_I1:stat:area'] = inputs['Fl_I1:stat:P']
        J['impulse_mix', 'Fl_I1:stat:W'] = inputs['Fl_I1:stat:V']
        J['impulse_mix', 'Fl_I1:stat:V'] = inputs['Fl_I1:stat:W']

        J['impulse_mix', 'Fl_I2:stat:P'] = inputs['Fl_I2:stat:area']
        J['impulse_mix', 'Fl_I2:stat:area'] = inputs['Fl_I2:stat:P']
        J['impulse_mix', 'Fl_I2:stat:W'] = inputs['Fl_I2:stat:V']
        J['impulse_mix', 'Fl_I2:stat:V'] = inputs['Fl_I2:stat:W']


class AreaSum(om.ExplicitComponent):

    def setup(self):

        self.add_input('Fl_I1:stat:area', val=1., units='m**2')
        self.add_input('Fl_I2:stat:area', val=1., units='m**2')
        self.add_output('area_sum', val=1., units='m**2')

        self.declare_partials('area_sum', ['Fl_I1:stat:area', 'Fl_I2:stat:area'], val=1.)
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):

        outputs['area_sum'] = inputs['Fl_I1:stat:area'] + inputs['Fl_I2:stat:area']


class Impulse(om.ExplicitComponent):

    def setup(self):
        self.add_input('P', units='Pa')
        self.add_input('area', units='m**2')
        self.add_input('V', units='m/s')
        self.add_input('W', units='kg/s')

        self.add_output('impulse', units='N')

        self.declare_partials('impulse', '*')

    def compute(self, inputs, outputs):
        outputs['impulse'] = inputs['P']*inputs['area'] + inputs['W']*inputs['V']
        self.set_check_partial_options('*', method='cs')

    def compute_partials(self, inputs, J):

        J['impulse', 'P'] = inputs['area']
        J['impulse', 'area'] = inputs['P']
        J['impulse', 'V'] = inputs['W']
        J['impulse', 'W'] = inputs['V']


class Mixer(Element):
    """
    Combines two incomming flows into a single outgoing flow
    using a conservation of momentum approach

    --------------
    Flow Stations
    --------------
    Fl_I1
    FL_I2
    Fl_O

    -------------
    Design
    -------------
        inputs
        --------

        implicit states
        ---------------
        balance.P_tot

        outputs
        --------
        ER
    -------------
    Off-Design
    -------------
        inputs
        --------
        Fl_I1_stat_calc.area | Fl_I2_stat_calc.area
        area

        implicit states
        ---------------
        balance.P_tot

    """

    def initialize(self):

        self.options.declare('designed_stream', default=1, values=(1,2),
                              desc='control for which stream has its area varied to match static pressure (1 means, you vary Fl_I1)')
        self.options.declare('internal_solver', default=True,
                              desc='If True, a newton solver is used inside the mixer to converge the impulse balance')

        

        super().initialize()

    def pyc_setup_output_ports(self): 

        flow1_composition = self.Fl_I_data['Fl_I1']
        flow2_composition = self.Fl_I_data['Fl_I2']

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        self.flow_add = ThermoAdd(method=thermo_method, mix_mode='flow', mix_names='mix', 
                                  thermo_kwargs={'spec':thermo_data,
                                                 'inflow_composition':flow1_composition, 
                                                 'mix_composition':flow2_composition})

        self.copy_flow(self.flow_add, 'Fl_O')

    def setup(self):
        
        design = self.options['design']        
        thermo_data = self.options['thermo_data']
        thermo_method = self.options['thermo_method']

        flow1_composition = self.Fl_I_data['Fl_I1']
        in_flow = FlowIn(fl_name='Fl_I1')
        self.add_subsystem('in_flow1', in_flow, promotes=['Fl_I1:*'])

        flow2_composition = self.Fl_I_data['Fl_I1']
        in_flow = FlowIn(fl_name='Fl_I2')
        self.add_subsystem('in_flow2', in_flow, promotes=['Fl_I2:*'])

        if self.options['designed_stream'] == 1:
            self.default_des_od_conns = [
                ('Fl_O:stat:area', 'area'), 
                ('Fl_I1_calc:stat:area', 'Fl_I1_stat_calc.area')
            ]
        else:
            self.default_des_od_conns = [
                ('Fl_O:stat:area', 'area'), 
                ('Fl_I2_calc:stat:area', 'Fl_I2_stat_calc.area')
            ]

        if design:
            # internal flow station to compute the area that is needed to match the static pressures
            if self.options['designed_stream'] == 1:
                Fl1_stat = Thermo(mode='static_Ps', fl_name="Fl_I1_calc:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':flow1_composition, 
                                                 'spec':thermo_data})
                self.add_subsystem('Fl_I1_stat_calc', Fl1_stat,
                                   promotes_inputs=[('composition', 'Fl_I1:tot:composition'), ('S', 'Fl_I1:tot:S'),
                                                    ('ht', 'Fl_I1:tot:h'), ('W', 'Fl_I1:stat:W'), ('Ps', 'Fl_I2:stat:P')],
                                   promotes_outputs=['Fl_I1_calc:stat*'])

                self.add_subsystem('area_calc', AreaSum(), promotes_inputs=['Fl_I2:stat:area'],
                                   promotes_outputs=[('area_sum', 'area')])
                self.connect('Fl_I1_calc:stat:area', 'area_calc.Fl_I1:stat:area')

            else:
                Fl2_stat = Thermo(mode='static_Ps', fl_name="Fl_I2_calc:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':flow2_composition, 
                                                 'spec':thermo_data})
                self.add_subsystem('Fl_I2_stat_calc', Fl2_stat,
                                   promotes_inputs=[('composition', 'Fl_I2:tot:composition'), ('S', 'Fl_I2:tot:S'),
                                                    ('ht', 'Fl_I2:tot:h'), ('W', 'Fl_I2:stat:W'), ('Ps', 'Fl_I1:stat:P')],
                                   promotes_outputs=['Fl_I2_calc:stat:*'])

                self.add_subsystem('area_calc', AreaSum(), promotes_inputs=['Fl_I1:stat:area'],
                                   promotes_outputs=[('area_sum', 'area')])
                self.connect('Fl_I2_calc:stat:area', 'area_calc.Fl_I2:stat:area')

        else:
            if self.options['designed_stream'] == 1:
                Fl1_stat = Thermo(mode='static_A', fl_name="Fl_I1_calc:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':flow1_composition, 
                                                 'spec':thermo_data})
                self.add_subsystem('Fl_I1_stat_calc', Fl1_stat,
                                    promotes_inputs=[('composition', 'Fl_I1:tot:composition'), ('S', 'Fl_I1:tot:S'),
                                                     ('ht', 'Fl_I1:tot:h'), ('W', 'Fl_I1:stat:W'),
                                                     ('guess:Pt', 'Fl_I1:tot:P'), ('guess:gamt', 'Fl_I1:tot:gamma')],
                                    promotes_outputs=['Fl_I1_calc:stat*'])

            else:
                Fl2_stat = Thermo(mode='static_A', fl_name="Fl_I2_calc:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':flow2_composition, 
                                                 'spec':thermo_data})
                self.add_subsystem('Fl_I2_stat_calc', Fl2_stat,
                                    promotes_inputs=[('composition', 'Fl_I2:tot:composition'), ('S', 'Fl_I2:tot:S'),
                                                     ('ht', 'Fl_I2:tot:h'), ('W', 'Fl_I2:stat:W'),
                                                     ('guess:Pt', 'Fl_I2:tot:P'), ('guess:gamt', 'Fl_I2:tot:gamma')],
                                    promotes_outputs=['Fl_I2_calc:stat*'])

        self.add_subsystem('extraction_ratio', om.ExecComp('ER=Pt1/Pt2', Pt1={'units':'Pa'}, Pt2={'units':'Pa'}),

                            promotes_inputs=[('Pt1', 'Fl_I1:tot:P'), ('Pt2', 'Fl_I2:tot:P')],
                            promotes_outputs=['ER'])

        self.add_subsystem('flow_add', self.flow_add,
                          promotes_inputs=[('Fl_I:stat:W', 'Fl_I1:stat:W'), ('Fl_I:tot:composition', 'Fl_I1:tot:composition'), ('Fl_I:tot:h', 'Fl_I1:tot:h'), 
                                           ('mix:W', 'Fl_I2:stat:W'), ('mix:composition', 'Fl_I2:tot:composition'), ('mix:h', 'Fl_I2:tot:h')])


        if self.options['designed_stream'] == 1:
            self.add_subsystem('impulse_mix', MixImpulse(),
                               promotes_inputs=[('Fl_I1:stat:W','Fl_I1_calc:stat:W'), ('Fl_I1:stat:P','Fl_I1_calc:stat:P'),
                                                ('Fl_I1:stat:V','Fl_I1_calc:stat:V'), ('Fl_I1:stat:area','Fl_I1_calc:stat:area'),
                                                'Fl_I2:stat:W', 'Fl_I2:stat:P', 'Fl_I2:stat:V', 'Fl_I2:stat:area'])
        else:
            self.add_subsystem('impulse_mix', MixImpulse(),
                               promotes_inputs=['Fl_I1:stat:W', 'Fl_I1:stat:P', 'Fl_I1:stat:V', 'Fl_I1:stat:area',
                                                ('Fl_I2:stat:W','Fl_I2_calc:stat:W'), ('Fl_I2:stat:P','Fl_I2_calc:stat:P'),
                                                ('Fl_I2:stat:V','Fl_I2_calc:stat:V'), ('Fl_I2:stat:area','Fl_I2_calc:stat:area')])


        # group to converge for the impulse balance
        conv = self.add_subsystem('impulse_converge', om.Group(), promotes=['*'])

        if self.options['internal_solver']:
            newton = conv.nonlinear_solver = om.NewtonSolver()
            newton.options['maxiter'] = 30
            newton.options['atol'] = 1e-5
            newton.options['rtol'] = 1e-99
            newton.options['solve_subsystems'] = True
            newton.options['max_sub_solves'] = 20
            newton.options['reraise_child_analysiserror'] = False
            newton.linesearch = om.BoundsEnforceLS()
            newton.linesearch.options['iprint'] = -1
            conv.linear_solver = om.DirectSolver()

        out_tot = Thermo(mode='total_hP', fl_name='Fl_O:tot', 
                         method=thermo_method, 
                         thermo_kwargs={'composition':flow1_composition, 
                                        'spec':thermo_data})
        conv.add_subsystem('out_tot', out_tot, promotes_outputs=['Fl_O:tot:*'])
        self.connect('flow_add.composition_out', 'out_tot.composition')
        self.connect('flow_add.mass_avg_h', 'out_tot.h')
        # note: gets Pt from the balance comp

        out_stat = Thermo(mode='static_A', fl_name='Fl_O:stat', 
                          method=thermo_method, 
                          thermo_kwargs={'composition':flow1_composition, 
                                         'spec':thermo_data})
        conv.add_subsystem('out_stat', out_stat, promotes_outputs=['Fl_O:stat:*'], promotes_inputs=['area', ])
        self.connect('flow_add.composition_out', 'out_stat.composition')
        self.connect('flow_add.Wout','out_stat.W')
        conv.connect('Fl_O:tot:S', 'out_stat.S')
        self.connect('flow_add.mass_avg_h', 'out_stat.ht')
        conv.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
        conv.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

        conv.add_subsystem('imp_out', Impulse())
        conv.connect('Fl_O:stat:P', 'imp_out.P')
        conv.connect('Fl_O:stat:area', 'imp_out.area')
        conv.connect('Fl_O:stat:V', 'imp_out.V')
        conv.connect('Fl_O:stat:W', 'imp_out.W')

        balance = conv.add_subsystem('balance', om.BalanceComp())
        balance.add_balance('P_tot', val=100, units='psi', eq_units='N', lower=1e-3, upper=10000)
        conv.connect('balance.P_tot', 'out_tot.P')
        conv.connect('imp_out.impulse', 'balance.lhs:P_tot')
        self.connect('impulse_mix.impulse_mix', 'balance.rhs:P_tot') #note that this connection comes from outside the convergence group

        super().setup()

