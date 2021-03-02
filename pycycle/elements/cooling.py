import openmdao.api as om

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo, ThermoAdd

from pycycle.constants import ALLOWED_THERMOS, THERMO_DEFAULT_COMPOSITIONS
from pycycle.flow_in import FlowIn
from pycycle.element_base import Element


class CombineCooling(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_ins', types=int, desc='number of input flow')

    def setup(self):
        n_ins = self.options['n_ins']
        for i in range(1,n_ins+1):
            self.add_input('W_{}'.format(i), units='lbm/s')

        self.add_output('W_cool', units='lbm/s')
        self.declare_partials('W_cool', '*', val=1) #constant values

    def compute(self, inputs, outputs):
        W_cool = 0
        for i in range(1, self.options['n_ins']+1):
            W_cool += inputs['W_{}'.format(i)]
        outputs['W_cool'] = W_cool


class CoolingCalcs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_stages', types=int, desc="number of stages in the turbine")
        self.options.declare('i_row', types=int, desc="row number")
        self.options.declare('T_safety', types=float, default=150., desc='safety factor applied') # units=degR
        self.options.declare('T_metal', types=float, default=2460., desc='safety factor applied') # units=degR


    def setup(self):
        self.add_input('turb_pwr', val=1, units='Btu/s', desc='power produced by the whole turbine')
        self.add_input('Pt_in', val=1, units='psi', desc='turbine inlet pressure') # note: NOT the pressure at the row. Across the whole turbine!
        self.add_input('Pt_out', val=1, units='psi', desc='turbine exit pressure')
        self.add_input('x_factor', val=1, desc='technology factor. 1 is current technology, lower is more advanced technology')
        self.add_input('W_primary', val=1, units='lbm/s', desc="flow into the row")
        self.add_input('Tt_primary', val=1, units='degR', desc='total temperature of primary flow coming into the row')
        self.add_input('Tt_cool', val=1, units='degR', desc='total temperature of cooling flow coming into the row')
        self.add_input('ht_primary', val=1, units='Btu/lbm', desc='total enthalpy of primary flow coming into the row')
        self.add_input('ht_cool', val=1, units='Btu/lbm', desc='total enthalpy of cooling flow coming into the row')

        self.add_output('W_cool', val=1, units='lbm/s', desc="flow requires to cool the turbine")
        # not the same as the Pt_out thats an input, which is for the whole turbine
        self.add_output('Pt_stage', val=1, lower=1e-5, units='psi', desc="exit total pressure of the row")

        self.add_output('ht_out', val=1, units='Btu/lbm', desc="exit total enthalpy")

        # integer math to so you get 1,1,2,2 for the rows of 2 stage machine
        self.i_stage = (((self.options['i_row'])//2) + 1)//self.options['n_stages']

        self.declare_partials('W_cool', ['x_factor', 'W_primary', 'Tt_primary', 'Tt_cool'])
        self.declare_partials('ht_out', ['x_factor', 'W_primary', 'Tt_primary', 'Tt_cool', 'turb_pwr', 'ht_cool', 'ht_primary'])
        self.declare_partials('Pt_stage', ['Pt_in', 'Pt_out'])


    def compute(self, inputs, outputs):

        n_stages = self.options['n_stages']
        i_row = self.options['i_row']

        if i_row % 2 == 0: # even rows are stators
            T_gas = inputs['Tt_primary'] + self.options['T_safety']
            dh = 0
        else: # rotor
            T_gas = .92*inputs['Tt_primary'] + self.options['T_safety']
            dh = inputs['turb_pwr']/n_stages # only rotors do work


        if i_row == 0:
            profile_factor = .3
        else:
            profile_factor = .13

        W_primary = inputs['W_primary']
        if T_gas < self.options['T_metal']:
            outputs['W_cool'] = W_cool = 0
            phi_prime = 0
        else:
            phi = (T_gas - self.options['T_metal'])/(T_gas-inputs['Tt_cool'])
            phi_prime = (phi + profile_factor)/(profile_factor + 1.)

            # print(self.pathname, W_primary, inputs['Tt_primary'], inputs['Tt_cool'], phi_prime)

            try:
                outputs['W_cool'] = W_cool = .022*inputs['x_factor']*(4./3.)*W_primary*(phi_prime/(1-phi_prime))**1.25
            except FloatingPointError:
                raise om.AnalysisError('bad flow values in {}; W: {}'.format(self.pathname, W_primary))

        outputs['ht_out'] = (W_primary*inputs['ht_primary'] + W_cool*inputs['ht_cool'])/(W_primary+W_cool) - dh/W_primary

        Pt_out = inputs['Pt_out']
        Pt_in = inputs['Pt_in']
        outputs['Pt_stage'] = Pt_out + (Pt_in-Pt_out)*self.i_stage


    def compute_partials(self, inputs, J):

        n_stages = self.options['n_stages']
        i_row = self.options['i_row']

        if i_row % 2 == 0: # even rows are stators
            T_gas = inputs['Tt_primary'] + self.options['T_safety']
            dh = 0
            ddh_dturb_pwr = 0
            dTgas_dTprimary = 1.
        else: # rotor
            T_gas = .92*inputs['Tt_primary'] + self.options['T_safety']
            dh = inputs['turb_pwr']/n_stages # only rotors do work
            ddh_dturb_pwr = 1./n_stages
            dTgas_dTprimary = .92

        if i_row == 0:
            profile_factor = .3
        else:
            profile_factor = .13

        W_primary = inputs['W_primary']

        if T_gas < self.options['T_metal']:
            dWc_dx_factor   = 0
            dWc_dWp  = 0
            dWc_dTt_primary = 0
            dWc_dTt_cool    = 0
            W_cool = 0
        else:
            T_cool = inputs['Tt_cool']
            T_metal = self.options['T_metal']
            phi = (T_gas - T_metal)/(T_gas-T_cool)
            phi_prime = (phi + profile_factor)/(profile_factor + 1.)
            x_factor = inputs['x_factor']
            const = .022*4./3.
            phi_prime_term = (phi_prime/(1.-phi_prime))**1.25
            dphi_prime_dphi = 1./(profile_factor+1.)
            dphi_dTgas = 1./(T_gas-T_cool) - (T_gas-T_metal)/(T_gas-T_cool)**2
            dphi_dTcool = (T_gas-T_metal)/(T_gas-T_cool)**2
            dphi_prime_term_dphi_prime = (1.25*(phi_prime/(1.-phi_prime))**0.25)*(1./(1.-phi_prime)+phi_prime/(1.-phi_prime)**2)

            dWc_dx_factor = const*phi_prime_term*W_primary
            dWc_dWp = const*phi_prime_term*x_factor
            dWc_dTt_primary = const*W_primary*x_factor*dphi_prime_term_dphi_prime*dphi_prime_dphi*dphi_dTgas*dTgas_dTprimary
            dWc_dTt_cool    = const*W_primary*x_factor*dphi_prime_term_dphi_prime*dphi_prime_dphi*dphi_dTcool
            W_cool = const*x_factor*W_primary*(phi_prime/(1.-phi_prime))**1.25

        ht_primary = inputs['ht_primary']
        ht_cool = inputs['ht_cool']

        WpWc = W_primary + W_cool
        dht_out_dW_cool = -W_primary*ht_primary/WpWc**2 + ht_cool/WpWc - W_cool*ht_cool/WpWc**2

        J['W_cool', 'W_primary'] = dWc_dWp
        J['W_cool', 'x_factor'] = dWc_dx_factor
        J['W_cool', 'Tt_primary'] = dWc_dTt_primary
        J['W_cool', 'Tt_cool'] = dWc_dTt_cool

        J['ht_out', 'x_factor'] = dht_out_dW_cool * dWc_dx_factor
        J['ht_out', 'W_primary'] = (ht_primary-ht_cool)*(W_cool - W_primary*dWc_dWp)/WpWc**2 + dh/W_primary**2
        J['ht_out', 'Tt_primary'] = dht_out_dW_cool * dWc_dTt_primary
        J['ht_out', 'Tt_cool'] = dht_out_dW_cool * dWc_dTt_cool
        J['ht_out', 'ht_primary'] = W_primary/WpWc
        J['ht_out', 'ht_cool'] = W_cool/WpWc
        J['ht_out', 'turb_pwr'] = -ddh_dturb_pwr/W_primary

        J['Pt_stage', 'Pt_in'] = self.i_stage
        J['Pt_stage', 'Pt_out'] = 1 - self.i_stage


class Row(om.Group):

    def initialize(self):
        self.options.declare('n_stages', types=int, desc="number of stages in the turbine")
        self.options.declare('i_row', types=int, desc="row number")
        self.options.declare('T_metal', types=float, default=2460., desc='safety factor applied') # units=degR
        self.options.declare('T_safety', types=float, default=150., desc='safety factor applied') # units=degR

        self.options.declare('thermo_method', default='CEA', values=ALLOWED_THERMOS,
                              desc='Method for computing thermodynamic properties')
        self.options.declare('thermo_data', default=None,
                               desc='thermodynamic data set', recordable=False)
        self.options.declare('main_flow_composition')
        self.options.declare('bld_flow_composition')
        self.options.declare('mix_flow_composition')

    def setup(self):

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']

        if thermo_data is None: 
            thermo_data = THERMO_DEFAULT_COMPOSITIONS[thermo_method]

        main_flow_composition = self.options['main_flow_composition']
        bld_flow_composition = self.options['bld_flow_composition']
        mix_flow_composition = self.options['mix_flow_composition']

        self.add_subsystem('cooling_calcs', CoolingCalcs(n_stages=self.options['n_stages'],
                                                         i_row=self.options['i_row'],
                                                         T_safety=self.options['T_safety'],
                                                         T_metal=self.options['T_metal']),
                          promotes_inputs=['Pt_in', 'Pt_out', 'W_primary', 'Tt_primary', 'Tt_cool', 'ht_primary', 'ht_cool', 'x_factor', 'turb_pwr'],
                          promotes_outputs=['W_cool'])

        consts = self.add_subsystem('consts', om.IndepVarComp()) # values that should not be changed ever
        consts.add_output('bld_frac_P', val=1)

        self.add_subsystem('mix_n', ThermoAdd(method=thermo_method, mix_mode='flow', mix_names='cool',
                                              thermo_kwargs={'spec':thermo_data, 
                                                             'inflow_composition':main_flow_composition, 
                                                             'mix_composition':bld_flow_composition,}),
                           promotes_inputs=[('Fl_I:stat:W','W_primary'), 
                                            ('Fl_I:tot:composition', 'composition_primary'), 'cool:composition'], 
                           promotes_outputs=[('Wout','W_out'),]
                           )

        mixed_flow = Thermo(mode='total_hP', fl_name='Fl_O:tot', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':mix_flow_composition, 
                                           'spec':thermo_data})
        self.add_subsystem('mixed_flow', mixed_flow,
                           promotes_outputs=['Fl_O:tot:*'])

        # promoted
        # self.connect('', 'mix_n.Pt_in')
        # self.connect('', 'mix_n.Pt_out')
        # self.connect('', 'mix_n.W_primary')
        # self.connect('', 'mix_n.n_in')
        # self.connect('', 'mix_n.cool:n')

        self.connect('W_cool', 'mix_n.cool:W')

        # self.connect('consts.bld_frac_P', 'mix_n.cool:frac_P')

        self.connect('mix_n.composition_out', 'mixed_flow.composition')
        self.connect('cooling_calcs.ht_out', 'mixed_flow.h')
        self.connect('cooling_calcs.Pt_stage', 'mixed_flow.P')


class TurbineCooling(Element):

    def initialize(self):
        self.options.declare('n_stages', types=int, desc="number of stages in the turbine")
        self.options.declare('T_metal', types=float, default=2460., desc='safety factor applied') # units=degR
        self.options.declare('T_safety', types=float, default=150., desc='safety factor applied') # units=degR

        self.options.declare('owns_x_factor', types=bool, default=True, desc='if True, x_factor will be connected to an IndepVarComp inside this element')

        super().initialize()

    def pyc_setup_output_ports(self): 

        # inflow_composition = self.Fl_I_data['Fl_turb_I']
        # mix_composition = self.Fl_I_data['Fl_cool']

        n_stages = self.options['n_stages']
        n_rows = 2 * n_stages
        for i in range(n_rows):
            row_port_name = r'row_{i}.Fl_O'
            self.copy_flow('Fl_turb_I', row_port_name)

    def setup(self):

        thermo_data = self.options['thermo_data']
        n_stages = self.options['n_stages']
        n_rows = 2 * n_stages

        if self.options['owns_x_factor']:
            indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
            indeps.add_output('x_factor', val=1.0)

        in_flow = FlowIn(fl_name='Fl_turb_I')
        self.add_subsystem('turb_in_flow', in_flow, promotes_inputs=['Fl_turb_I:tot:*', 'Fl_turb_I:stat:*'])

        in_flow = FlowIn(fl_name='Fl_turb_O')
        self.add_subsystem('turb_out_flow', in_flow, promotes_inputs=['Fl_turb_O:tot:*', 'Fl_turb_O:stat:*'])

        in_flow = FlowIn(fl_name='Fl_cool')
        self.add_subsystem('cool_in_flow', in_flow, promotes_inputs=['Fl_cool:tot:*', 'Fl_cool:stat:*'])


        # these are the inputs to the component
        p_inputs_all = ['x_factor', ('Pt_in', 'Fl_turb_I:tot:P'), ('Pt_out', 'Fl_turb_O:tot:P'),
                        ('Tt_cool','Fl_cool:tot:T'), ('ht_cool','Fl_cool:tot:h'), ('cool:composition','Fl_cool:tot:composition'), 'turb_pwr']

        p_row_inputs = [('W_primary',  'Fl_turb_I:stat:W'),
                        ('Tt_primary', 'Fl_turb_I:tot:T'),
                        ('ht_primary', 'Fl_turb_I:tot:h'),
                        ('composition_primary',  'Fl_turb_I:tot:composition')]
        self.add_subsystem('row_0', Row(n_stages=n_stages, i_row=0,
                                        T_safety=self.options['T_safety'], T_metal=self.options['T_metal'],
                                        thermo_data=thermo_data, 
                                        thermo_method=self.options['thermo_method'],
                                        main_flow_composition=self.Fl_I_data['Fl_turb_I'], 
                                        bld_flow_composition=self.Fl_I_data['Fl_cool'], 
                                        mix_flow_composition=self.Fl_I_data['Fl_turb_O']),
                           promotes_inputs=p_inputs_all+p_row_inputs)

        for i in range(1,n_rows):

            prev_row = 'row_{}'.format(i-1)
            curr_row = 'row_{}'.format(i)
            self.add_subsystem('row_{}'.format(i),
                               Row(n_stages=n_stages, i_row=i,
                                   T_safety=self.options['T_safety'], T_metal=self.options['T_metal'],
                                   thermo_data=thermo_data, 
                                   thermo_method=self.options['thermo_method'],
                                   main_flow_composition=self.Fl_I_data['Fl_turb_I'], 
                                   bld_flow_composition=self.Fl_I_data['Fl_cool'], 
                                   mix_flow_composition=self.Fl_I_data['Fl_turb_O']),
                               promotes_inputs=p_inputs_all)

            self.connect('{}.W_out'.format(prev_row), '{}.W_primary'.format(curr_row))
            self.connect('{}.Fl_O:tot:T'.format(prev_row), '{}.Tt_primary'.format(curr_row))
            self.connect('{}.Fl_O:tot:h'.format(prev_row), '{}.ht_primary'.format(curr_row))
            self.connect('{}.Fl_O:tot:composition'.format(prev_row), '{}.composition_primary'.format(curr_row))

        super().setup()

if __name__ == "__main__":

    prob = om.Problem()
    prob.model = TurbineCooling(n_stages=1)

    prob.model.set_input_defaults('Fl_turb_I:tot:T', val=518., units='degR')
    prob.model.set_input_defaults('Fl_turb_I:tot:P', val=1., units='lbf/inch**2')
    prob.model.set_input_defaults('Fl_turb_I:stat:W', val= 1.0, units='lbm/s')
    prob.model.set_input_defaults('Fl_turb_O:tot:P', val=1., units='lbf/inch**2')
    prob.model.set_input_defaults('Fl_cool:tot:T', val=518., units='degR')

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
