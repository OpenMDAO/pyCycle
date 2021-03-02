from collections.abc import Iterable
from copy import copy

import numpy as np

import openmdao.api as om

from pycycle.constants import BTU_s2HP, HP_per_RPM_to_FT_LBF
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.thermo.cea import species_data
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element

from pycycle.elements.turbine_map import TurbineMap
from pycycle.maps.lpt2269 import LPT2269


class CorrectedInputsCalc(om.ExplicitComponent):
    """Compute design corrected flow (Wp) and design corrected speed (Np)"""

    def setup(self):
        self.add_input('Tt', val=500., units='degR',
                       desc='incoming temperature')
        self.add_input('Pt', val=14., units='psi', desc='incoming pressure')
        self.add_input('W_in', val=30.0, units='lbm/s', desc='mass flow')
        self.add_input('Nmech', val=1000.0, units='rpm', desc='shaft speed')

        self.add_output('Wp', val=30.0, units='lbm/s',
                        desc='corrected mass flow')
        self.add_output('Np', val=100., units='rpm',
                        desc='corrected shaft speed')

        self.declare_partials('Wp', ['Tt', 'Pt', 'W_in'])
        self.declare_partials('Np', ['Nmech', 'Tt'])

    def compute(self, inputs, outputs):

        try:
            outputs['Wp'] = inputs['W_in'] * inputs['Tt']**0.5 / inputs['Pt']
            outputs['Np'] = inputs['Nmech'] * inputs['Tt']**-0.5
        except FloatingPointError:
            raise AnalysisError('Bad values flow states in {}: T={}  P={}'.format(self.pathname, inputs['Tt'], inputs['Pt']))

    def compute_partials(self, inputs, J):
        W_in = inputs['W_in']
        Tt = inputs['Tt']
        Pt = inputs['Pt']

        J['Wp', 'Tt'] = 0.5 * W_in / Pt * Tt**-0.5
        J['Wp', 'Pt'] = -W_in * Tt**0.5 * Pt**-2.
        J['Wp', 'W_in'] = Tt**0.5 / Pt
        J['Np', 'Nmech'] = Tt**-0.5
        J['Np', 'Tt'] = -0.5 * inputs['Nmech'] * Tt**-1.5


class eff_poly_calc(om.ExplicitComponent):
    """ Calculate polytropic efficiency for turbine"""

    def setup(self):
        # list_inputs
        # Note, Cp - Cv method for caluclating Rt was used because resulting values matched NPSS well.
        # calculating Rt with molecular weight and the universal R is also valid.
        # self.add_input(     'Cp', 1.0,units='Btu/(lbm*degR)',desc='specific heat at constant pressure')
        # self.add_input(     'Cv', 1.0,units='Btu/(lbm*degR)',desc='specific heat at constant volume')
        self.add_input(     'PR', 1.0,units=None            ,desc='turbine pressure ratio (Pin/Pout)')
        self.add_input(   'S_in', 1.0,units='Btu/(lbm*degR)',desc='element input entropy')
        self.add_input(  'S_out', 1.0,units='Btu/(lbm*degR)',desc='element output entropy')
        self.add_input(     'Rt', val=0.0686, units='Btu/(lbm*degR)', desc='specific gas constant')
        # list_outputs
        self.add_output('eff_poly',    val=1.0,             units=None, desc='polytropic efficiency', lower=1e-6)
        # self.add_output(     'Rt', val=0.0686, units='Btu/(lbm*degR)', desc='specific gas constant', lower=1e-6)
        # define partials
        self.declare_partials('eff_poly','*')
        # self.declare_partials('Rt','Cp',val=1.0)
        # self.declare_partials('Rt','Cv',val=-1.0)

    def compute(self, inputs, outputs):
        PR     = inputs['PR']
        S_in   = inputs['S_in']
        S_out  = inputs['S_out']
        # Cp     = inputs['Cp']
        # Cv     = inputs['Cv']
        Rt = inputs['Rt']

        # outputs['Rt'] = Rt = Cp - Cv
        invPR = 1/PR

        outputs['eff_poly'] = 1 + (S_out-S_in)/(Rt*np.log(invPR))

    def compute_partials(self, inputs, J):
        # J['Rt', 'Cp'] = 1.0
        # J['Rt', 'Cv'] = -1.0
        PR  = inputs['PR']
        S_in   = inputs['S_in']
        S_out  = inputs['S_out']
        # Cp     = inputs['Cp']
        # Cv     = inputs['Cv']
        Rt = inputs['Rt']

        invPR = 1/PR
        log_PR = np.log(invPR)

        # J['eff_poly', 'Cp'] =  (S_in - S_out)/(np.log(invPR)*(Cp - Cv)**2)
        # J['eff_poly', 'Cv'] = -(S_in - S_out)/(np.log(invPR)*(Cp - Cv)**2)

        J['eff_poly', 'PR'] = -(S_in - S_out)/(PR*Rt*log_PR**2)

        J['eff_poly', 'S_in']  = -1/(Rt*log_PR)
        J['eff_poly', 'S_out'] =  1/(Rt*log_PR)

        J['eff_poly', 'Rt'] = -(S_out-S_in)/(Rt**2 * log_PR)


class PressureDrop(om.ExplicitComponent):
    """Calculates pressure drop across the turbine"""

    def setup(self):
        # inputs
        self.add_input('PR', val=3.0, desc='Design PR')
        self.add_input('Pt_in', val=5.0, units='psi',
                       desc='Inlet total pressure')

        # outputs
        self.add_output('Pt_out', shape=1, units='psi',
                        desc='Exit total pressure', lower=1e-3)

        self.declare_partials('Pt_out', ['Pt_in', 'PR'])

    def compute(self, inputs, outputs):
        outputs['Pt_out'] = inputs['Pt_in'] / inputs['PR']

    def compute_partials(self, inputs, J):

        J['Pt_out', 'PR'] = -inputs['Pt_in'] * (inputs['PR']**-2)
        J['Pt_out', 'Pt_in'] = 1 / inputs['PR']


class EnthalpyDrop(om.ExplicitComponent):
    """EnthalpyDrop is a component that calculates the actual enthalpy drop"""

    def setup(self):
        # inputs
        self.add_input('ht_in', val=10.0, units='Btu/lbm',
                       desc='incoming enthalpy')
        self.add_input('ht_out_ideal', val=10.0, units='Btu/lbm',
                       desc='incoming ideal enthalpy')
        self.add_input('eff', val=0.8, desc='isentropic efficiency')

        # outputs
        self.add_output('ht_out', shape=1, units='Btu/lbm',
                        desc='actual enthalpy')

        self.declare_partials('ht_out', ['ht_in', 'ht_out_ideal', 'eff'])

    def compute(self, inputs, outputs):
        outputs['ht_out'] = inputs['ht_in'] - \
            (inputs['ht_in'] - inputs['ht_out_ideal']) * inputs['eff']

    def compute_partials(self, inputs, J):
        J['ht_out', 'ht_in'] = 1 - inputs['eff']
        J['ht_out', 'ht_out_ideal'] = inputs['eff']
        J['ht_out', 'eff'] = -inputs['ht_in'] + inputs['ht_out_ideal']


class BleedPressure(om.ExplicitComponent):
    """Computes bleed flow parameters"""

    def initialize(self):
        self.options.declare('bleed_names', types=Iterable, 
                              desc='list of names for the bleed ports',
                              default=[])

    def setup(self):

        bleeds = self.options['bleed_names']


        # primary inputs and outputs
        self.add_input('Pt_in', val=0.0, units='psi',
                       desc='turbine entrance pressure')
        self.add_input('Pt_out', val=0.0, units='psi',
                       desc='turbine exit pressure')
        # self.add_input('W_in', val=0.0, units='lbm/s',
        #                desc='turbine entrance mass flow rate')
        
        # bleed inputs and outputs
        for BN in bleeds:
            self.add_input(BN + ':frac_P', val=0.0,
                           desc='fraction of pressure drop where bleed flow introduced')
            self.add_input(BN + ':W', val=0.0, units='lbm/s',
                           desc='bleed mass flow rate')
         
            self.add_output(BN + ':Pt', shape=1, units='psi',
                            desc='pressure of incomming bleed flow', lower=1e-3)

            self.declare_partials(BN+':Pt', ['Pt_in', 'Pt_out', BN+':frac_P'])

    def compute(self, inputs, outputs):

        # calculate W_out and composition based on primary flow
       
        Pt_out = inputs['Pt_out']
        delta_Pt = inputs['Pt_in'] - Pt_out

        bleeds = self.options['bleed_names']
        for BN in bleeds:
            outputs[BN + ':Pt'] = Pt_out + inputs[BN+':frac_P'] * delta_Pt

    def compute_partials(self, inputs, J):

        bleeds = self.options['bleed_names']

        Pt_in = inputs['Pt_in']
        Pt_out = inputs['Pt_out']
       
        # Jacobian elements and modifications due to bleed flows
        for BN in bleeds:
            # J['W_out', BN + ':W'] = 1.0
            BN_pt = BN + ':Pt'
            BN_frac_P = BN + ':frac_P'

            frac_P = inputs[BN_frac_P]

            J[BN_pt, 'Pt_in'] = frac_P
            J[BN_pt, 'Pt_out'] = 1.0 - frac_P
            J[BN_pt, BN_frac_P] = Pt_in - Pt_out

class EnthalpyAndPower(om.ExplicitComponent):
    """Calculates exit enthalpy and shaft power for the turbine"""

    def initialize(self):
        self.options.declare('bleed_names', types=Iterable, desc='list of names for the bleed ports',
                              default=[])

    def setup(self):
        bleeds = self.options['bleed_names']
        # primary inputs and outputs
        self.add_input('W_in', val=30.0, units='lbm/s',
                       desc='entrance mass flow')
        self.add_input('W_out', val=30.0, units='lbm/s', desc='exit mass flow')
        self.add_input('ht_in', val=10.0, units='Btu/lbm',
                       desc='entrance enthalpy')
        self.add_input('ht_out_ideal', val=10.0,
                       units='Btu/lbm', desc='ideal exit enthalpy')
        self.add_input('eff', val=1.0, desc='turbine efficiency')
        self.add_input('Nmech', val=1000.0, units='rpm', desc='shaft speed')

        self.add_output('ht_out_b4bld', shape=1, units='Btu/lbm',
                        desc='downstream enthalpy')
        self.add_output('ht_out', shape=1, units='Btu/lbm',
                        desc='downstream enthalpy')
        self.add_output('power', shape=1, units='hp', desc='turbine power', res_ref=1e3)
        self.add_output('trq', shape=1, units='ft*lbf', desc='turbine torque', res_ref=1e3)

        self._bleed_tups = []

        # bleed specific inputs
        for BN in bleeds:
            BN_W = BN + ':W'
            BN_ht = BN + ':ht'
            BN_ht_ideal = BN + ':ht_ideal'

            self.add_input(BN_W, val=0.0, units='lbm/s',
                           desc='bleed mass flow rate')
            self.add_input(BN_ht, val=0.0, units='Btu/lbm',
                           desc='bleed total enthalpy')
            self.add_input(BN_ht_ideal, val=0.0, units='Btu/lbm',
                           desc='ideally expanded bleed total enthalpy')

            bleed_tup = (BN_W, BN_ht, BN_ht_ideal)
            self._bleed_tups.append(bleed_tup)

            self.declare_partials(['ht_out', 'power', 'trq'], bleed_tup)

        self.declare_partials('ht_out_b4bld', ['ht_in', 'ht_out_ideal', 'eff'])
        self.declare_partials('ht_out', ['W_in', 'W_out', 'ht_in', 'ht_out_ideal', 'eff'])
        self.declare_partials('power', ['W_in', 'ht_in', 'ht_out_ideal', 'eff'])
        self.declare_partials('trq', ['W_in', 'ht_in', 'ht_out_ideal', 'eff', 'Nmech'])

    def compute(self, inputs, outputs):
        W_out = inputs['W_out']
        W_in = inputs['W_in']
        eff = inputs['eff']
        ht_out_ideal = inputs['ht_out_ideal']
        ht_in = inputs['ht_in']

        # calculate ht_out and power based on only primary flow
        ht_out_b4bld = (ht_in * (1.0 - eff) + ht_out_ideal * eff)
        ht_out = W_in / W_out * ht_out_b4bld
        power = W_in * eff * (ht_in - ht_out_ideal) * BTU_s2HP

        # modify ht_out and power due to bleed flows
        for BN_W, BN_ht, BN_ht_ideal in self._bleed_tups:
            W = inputs[BN_W]
            ht = inputs[BN_ht]
            ht_ideal = inputs[BN_ht_ideal]

            ht_out += W / W_out * (ht * (1.0 - eff) + ht_ideal * eff)
            power += W * eff * (ht - ht_ideal) * BTU_s2HP

        # calculate torque based on revised power and shaft speed
        outputs['power'] = power
        outputs['ht_out_b4bld'] = ht_out_b4bld
        outputs['ht_out'] = ht_out
        outputs['trq'] = power / inputs['Nmech'] * HP_per_RPM_to_FT_LBF

    def compute_partials(self, inputs, J):
        ht_in = inputs['ht_in']
        W_out = inputs['W_out']
        W_in = inputs['W_in']
        eff = inputs['eff']
        ht_out_ideal = inputs['ht_out_ideal']
        Nmech = inputs['Nmech']

        # Jacobian elements for the primary flow
        J['ht_out_b4bld', 'ht_in'] = (1.0 - eff)
        J['ht_out_b4bld', 'ht_out_ideal'] = eff
        J['ht_out_b4bld', 'eff'] = (ht_out_ideal - ht_in)

        J['ht_out', 'W_in'] = (ht_in * (1.0 - eff) + ht_out_ideal * eff) / W_out
        dht_out_dW_out = -W_in / W_out**2 * (ht_in * (1.0 - eff) + ht_out_ideal * eff)
        J['ht_out', 'ht_in'] = W_in / W_out * (1.0 - eff)
        J['ht_out', 'ht_out_ideal'] = W_in / W_out * eff
        dht_out_deff = W_in / W_out * (ht_out_ideal - ht_in)

        J['power', 'W_in'] = (ht_in - ht_out_ideal) * eff * BTU_s2HP
        J['power', 'ht_in'] = W_in * eff * BTU_s2HP
        J['power', 'ht_out_ideal'] = -W_in * eff * BTU_s2HP
        dpower_deff = W_in * (ht_in - ht_out_ideal) * BTU_s2HP

        J['trq', 'W_in'] = (ht_in - ht_out_ideal) * eff / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
        J['trq', 'ht_in'] = W_in * eff / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
        J['trq', 'ht_out_ideal'] = -W_in * eff / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
        dtrq_deff = W_in * (ht_in - ht_out_ideal) / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
        dtrq_dNmech = -W_in * eff * (ht_in - ht_out_ideal) / Nmech**2 * BTU_s2HP * HP_per_RPM_to_FT_LBF

        # Jacobian elements and modifications due to bleed flows
        for BN_W, BN_ht, BN_ht_ideal in self._bleed_tups:
            W = inputs[BN_W]
            ht = inputs[BN_ht]
            ht_ideal = inputs[BN_ht_ideal]

            J['ht_out', BN_W] = (ht * (1.0 - eff) + ht_ideal * eff) / W_out
            dht_out_dW_out += -W / W_out**2 * (ht * (1.0 - eff) + ht_ideal * eff)
            J['ht_out', BN_ht] = W / W_out * (1.0 - eff)
            J['ht_out', BN_ht_ideal] = W / W_out * eff
            dht_out_deff += W / W_out * (ht_ideal - ht)

            J['power', BN_W] = eff * (ht - ht_ideal) * BTU_s2HP
            J['power', BN_ht] = W * eff * BTU_s2HP
            J['power', BN_ht_ideal] = -W * eff * BTU_s2HP
            dpower_deff += W * (ht - ht_ideal) * BTU_s2HP

            J['trq', BN_W] = eff * (ht - ht_ideal) / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
            J['trq', BN_ht] = W * eff / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
            J['trq', BN_ht_ideal] = -W * eff / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
            dtrq_deff += W * (ht - ht_ideal) / Nmech * BTU_s2HP * HP_per_RPM_to_FT_LBF
            dtrq_dNmech += -W * eff * (ht - ht_ideal) / Nmech**2 * BTU_s2HP * HP_per_RPM_to_FT_LBF

        J['ht_out', 'W_out'] = dht_out_dW_out
        J['ht_out', 'eff'] = dht_out_deff
        J['power', 'eff'] = dpower_deff
        J['trq', 'eff'] = dtrq_deff
        J['trq', 'Nmech'] = dtrq_dNmech


class Turbine(Element):
    """
    An Assembly that models a turbine

    --------------
    Flow Stations
    --------------
    Fl_I
    Fl_O

    -------------
    Design
    -------------
        inputs
        --------
        map.PRdes
        map.effDes
        alphaMap
        MN

        outputs
        --------
        s_PR
        s_Wc
        s_eff
        s_Nc

    -------------
    Off-Design
    -------------
        inputs
        --------
        s_PR
        s_Wc
        s_eff
        s_Nc
        area
        
        outputs
        --------
        Wp
        PR
        eff
        eff_poly
        Np
        power
        trq
    """

    def initialize(self):
        self.options.declare('map_data', default=LPT2269)
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('bleed_names', types=(list,tuple), desc='list of names for the bleed ports',
                              default=[])
        self.options.declare('map_interp_method', default='slinear',
                              desc='Method to use for map interpolation. \
                              Options are `slinear`, `cubic`, `quintic`.')
        self.options.declare('map_extrap', default=False, desc='Switch to allow extrapoloation off map')

        self.default_des_od_conns = [
            # (design src, off-design target)
            ('s_Wp', 's_Wp'),
            ('s_PR', 's_PR'),
            ('s_eff', 's_eff'), 
            ('s_Np', 's_Np'), 
            ('Fl_O:stat:area', 'area')
        ]

        super().initialize()

    def pyc_setup_output_ports(self): 

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        bleeds = self.options['bleed_names']

        inflow_composition = self.Fl_I_data['Fl_I']
        # thermo add expects a list of bleed_element, one for each bleed
        bleed_element_list = []
        for bleed_name in bleeds: 
            bleed_element_list.append(self.Fl_I_data[bleed_name])
        
        self.bld_add = ThermoAdd(method=thermo_method, mix_names=bleeds, mix_mode='flow',
                                 thermo_kwargs={'spec':thermo_data, 
                                                'inflow_composition':inflow_composition, 
                                                'mix_composition':bleed_element_list})
        
        self.copy_flow(self.bld_add, 'Fl_O')


    def setup(self):

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        map_data = self.options['map_data']
        designFlag = self.options['design']
        bleeds = self.options['bleed_names']
        statics = self.options['statics']
        interp_method = self.options['map_interp_method']
        map_extrap = self.options['map_extrap']

        composition = self.Fl_I_data['Fl_I']

        # Create inlet flow station
        in_flow = FlowIn(fl_name='Fl_I')
        self.add_subsystem('in_flow', in_flow, promotes_inputs=['Fl_I:*'])

        self.add_subsystem('corrinputs', CorrectedInputsCalc(),
                           promotes_inputs=[
                               'Nmech', ('W_in', 'Fl_I:stat:W'), ('Pt', 'Fl_I:tot:P'), ('Tt', 'Fl_I:tot:T')],
                           promotes_outputs=['Np', 'Wp'])

        turb_map = TurbineMap(map_data=map_data, design=designFlag,
                              interp_method=interp_method, extrap=map_extrap)
        if designFlag:
            self.add_subsystem('map', turb_map, promotes_inputs=['Np', 'Wp', 'PR', 'eff'],
                               promotes_outputs=['s_PR', 's_Wp', 's_eff', 's_Np'])
        else:
            self.add_subsystem('map', turb_map,
                               promotes_inputs=[
                                   'Np', 'Wp',  's_PR', 's_Wp', 's_eff', 's_Np'],
                               promotes_outputs=['PR', 'eff'])

        # Calculate pressure drop across turbine
        self.add_subsystem('press_drop', PressureDrop(), promotes_inputs=[
                           'PR', ('Pt_in', 'Fl_I:tot:P')])

        # Calculate ideal flow station properties
        ideal_flow = Thermo(mode='total_SP', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':composition, 
                                           'spec':thermo_data})
        self.add_subsystem('ideal_flow', ideal_flow,
                           promotes_inputs=[('S', 'Fl_I:tot:S'), ('composition', 'Fl_I:tot:composition')])
        self.connect("press_drop.Pt_out", "ideal_flow.P")

        # # Calculate enthalpy drop across turbine
        # self.add_subsystem("enth_drop", EnthalpyDrop(), promotes=['eff'])
        # self.connect("Fl_I:tot:h", "enth_drop.ht_in")
        # self.connect("ideal_flow.h", "enth_drop.ht_out_ideal")

        for BN in bleeds:
            bld_flow = FlowIn(fl_name=BN)
            self.add_subsystem(BN, bld_flow, promotes_inputs=[
                               f'{BN}:*'])

        # # Calculate bleed parameters
        blds = BleedPressure(bleed_names=bleeds)
        self.add_subsystem('blds', blds, 
                           promotes_inputs=[('Pt_in', 'Fl_I:tot:P'),] + [f'{BN}:frac_P' for BN in bleeds]
                           )
        self.connect('press_drop.Pt_out', 'blds.Pt_out')

        self.add_subsystem('bld_add', self.bld_add, 
                           promotes_inputs=['Fl_I:stat:W', 'Fl_I:tot:composition'] + 
                                           [(f'{BN}:W', f'{BN}:stat:W') for BN in bleeds] + 
                                           [(f'{BN}:composition', f'{BN}:tot:composition') for BN in bleeds], 
                           promotes_outputs=[('Wout', 'W_out')]
                           )

        bleed_names2 = []
        for BN in bleeds:
            # Determine bleed inflow properties
            bleed_names2.append(BN + '_inflow')
            inflow = Thermo(mode='total_hP', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':self.Fl_I_data[BN], 
                                           'spec':thermo_data})
            self.add_subsystem(BN + '_inflow', inflow,
                               promotes_inputs=[('composition', BN + ":tot:composition"), ('h', BN + ':tot:h')])
            self.connect( f'blds.{BN}:Pt', f'{BN}_inflow.P')

            # Ideally expand bleeds to exit pressure
            bleed_names2.append(f'{BN}_ideal')
            ideal = Thermo(mode='total_SP', 
                           method=thermo_method, 
                           thermo_kwargs={'composition':self.Fl_I_data[BN], 
                                          'spec':thermo_data})
            self.add_subsystem(f'{BN}_ideal', ideal,
                               promotes_inputs=[('composition', BN + ":tot:composition")])
            self.connect(f"{BN}_inflow.flow:S", f"{BN}_ideal.S")
            self.connect("press_drop.Pt_out", f"{BN}_ideal.P")

        # Calculate shaft power and exit enthalpy with cooling flows production
        self.add_subsystem('pwr_turb', EnthalpyAndPower(bleed_names=bleeds),
                           promotes_inputs=['Nmech', 'eff', 'W_out', ('W_in', 'Fl_I:stat:W'), ('ht_in', 'Fl_I:tot:h')] +
                                           [(BN + ':W', BN + ':stat:W') for BN in bleeds] +
                                           [(BN + ':ht', BN + ':tot:h') for BN in bleeds] +
                                           [(BN + ':ht_ideal', BN + '_ideal.h')
                                            for BN in bleeds],
                           promotes_outputs=['power', 'trq', 'ht_out_b4bld'])
        self.connect('ideal_flow.h', 'pwr_turb.ht_out_ideal')

        # Calculate real flow station properties before bleed air is added
        real_flow_b4bld = Thermo(mode='total_hP', fl_name="Fl_O_b4bld:tot",
                                 method=thermo_method, 
                                 thermo_kwargs={'composition':composition, 
                                                'spec':thermo_data})
        self.add_subsystem('real_flow_b4bld', real_flow_b4bld,
                           promotes_inputs=[('composition', 'Fl_I:tot:composition')])
        self.connect('ht_out_b4bld', 'real_flow_b4bld.h')
        self.connect('press_drop.Pt_out', 'real_flow_b4bld.P')

        # Calculate Polytropic efficiency
        self.add_subsystem('eff_poly_calc',eff_poly_calc(),promotes_inputs=['PR',('S_in','Fl_I:tot:S'),
                            ('Rt','Fl_I:tot:R')],
                            promotes_outputs=['eff_poly'])
        self.connect('real_flow_b4bld.Fl_O_b4bld:tot:S','eff_poly_calc.S_out')

        # Calculate real flow station properties
        real_flow = Thermo(mode='total_hP', fl_name="Fl_O:tot",
                                 method=thermo_method, 
                                 thermo_kwargs={'composition':composition, 
                                                'spec':thermo_data})
        self.add_subsystem('real_flow', real_flow,
                           promotes_outputs=['Fl_O:tot:*'])
        self.connect("pwr_turb.ht_out", "real_flow.h")
        self.connect("press_drop.Pt_out", "real_flow.P")
        self.connect("bld_add.composition_out", "real_flow.composition")

       # Calculate static properties
        if statics:
            if designFlag:
                #   SetStaticMN
                out_stat = Thermo(mode='static_MN', fl_name="Fl_O:stat",
                                 method=thermo_method, 
                                 thermo_kwargs={'composition':composition, 
                                                'spec':thermo_data})
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=['MN'],
                                   promotes_outputs=['Fl_O:stat:*'])
                self.connect('bld_add.composition_out', 'out_stat.composition')
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            else:
                #   SetStaticArea
                out_stat = Thermo(mode='static_A', fl_name="Fl_O:stat",
                                 method=thermo_method, 
                                 thermo_kwargs={'composition':composition, 
                                                'spec':thermo_data})
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=['area'],
                                   promotes_outputs=['Fl_O:stat:*'])
                self.connect('bld_add.composition_out', 'out_stat.composition')
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            self.set_order(['in_flow', 'corrinputs', 'map', 'press_drop', 'ideal_flow'] + bleeds + ['bld_add', 'blds'] + bleed_names2 +
                           ['pwr_turb','real_flow_b4bld', 'eff_poly_calc' ,'real_flow', 'out_stat'])

        else:
            self.add_subsystem('W_passthru', PassThrough(
                'W_out', 'Fl_O:stat:W', 1.0, units="lbm/s"), promotes=['*'])
            self.set_order(['in_flow', 'corrinputs', 'map', 'press_drop', 'ideal_flow'] + bleeds + ['bld_add', 'blds'] + bleed_names2 +
                           ['pwr_turb','real_flow_b4bld', 'eff_poly_calc', 'real_flow', 'W_passthru'])

        self.set_input_defaults('eff', val=0.99, units=None)
        # if not designFlag: 
        #     self.set_input_defaults('area', val=1, units='in**2')
        thermo_method = self.options['thermo_method']

        super().setup()
