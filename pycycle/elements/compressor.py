import numpy as np
from collections.abc import Iterable
import itertools

import openmdao.api as om

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.constants import BTU_s2HP, HP_per_RPM_to_FT_LBF, T_STDeng, P_STDeng
from pycycle.elements.compressor_map import CompressorMap
from pycycle.maps.ncp01 import NCP01
from pycycle.element_base import Element



class CorrectedInputsCalc(om.ExplicitComponent):
    """Compute design corrected flow (Wc) and design corrected speed (Nc)"""

    def setup(self):
        # inputs
        self.add_input('Tt', val=500., units='degR',
                       desc='incoming temperature')
        self.add_input('Pt', val=14., units='psi', desc='incoming pressure')
        self.add_input('W_in', val=30.0, units='lbm/s', desc='mass flow')
        self.add_input('Nmech', val=1000.0, units='rpm', desc='shaft speed')
        # outputs
        self.add_output('Wc', val=30.0, units='lbm/s',
                        desc='corrected mass flow')
        self.add_output('Nc', val=100., lower=1e-5,
                        units='rpm', desc='corrected shaft speed')

        self.declare_partials('Wc', ['Tt', 'Pt', 'W_in'])
        self.declare_partials('Nc', ['Nmech', 'Tt'])

    def compute(self, inputs, outputs):

        self.delta = inputs['Pt'] / P_STDeng
        self.W = inputs['W_in']
        self.theta = inputs['Tt'] / T_STDeng

        outputs['Wc'] = self.W * self.theta**0.5 / self.delta
        outputs['Nc'] = inputs['Nmech'] * self.theta**-0.5

    def compute_partials(self, inputs, J):

        theta = self.theta
        delta = self.delta

        J['Wc', 'Tt'] = 0.5 * self.W / delta * theta**-0.5 / T_STDeng
        J['Wc', 'Pt'] = -self.W * theta**0.5 * delta**-2. / P_STDeng
        J['Wc', 'W_in'] = theta**0.5 / delta
        J['Nc', 'Nmech'] = theta**-0.5
        J['Nc', 'Tt'] = -0.5 * inputs['Nmech'] * theta**-1.5 / T_STDeng

class eff_poly_calc(om.ExplicitComponent):
    """ Calculate polytropic efficiency for compressor"""

    def setup(self):
        self.add_input('PR', 1.0,units=None, desc='element pressure ratio Pt_out/Pt_in')
        self.add_input('S_in', 1.0,units='Btu/(lbm*degR)', desc='element input entropy')
        self.add_input('S_out', 1.0,units='Btu/(lbm*degR)', desc='element output entropy')
        self.add_input('Rt', val=0.0686, units='Btu/(lbm*degR)', desc='specific gas constant')
        # list_outputs
        self.add_output('eff_poly', val=1.0, units=None, desc='polytropic efficiency', lower=1e-6)
        # define partials
        self.declare_partials('eff_poly','*')
    def compute(self, inputs, outputs):
        PR = inputs['PR']
        S_in = inputs['S_in']
        S_out = inputs['S_out']
        Rt = inputs['Rt']

        outputs['eff_poly'] = Rt * np.log(PR) / ( Rt*np.log(PR) + S_out - S_in )

    def compute_partials(self, inputs, J):
        PR     = inputs['PR']
        S_in   = inputs['S_in']
        S_out  = inputs['S_out']
        Rt = inputs['Rt']

        J['eff_poly', 'PR'] = (Rt*(S_out - S_in))/(PR*(np.log(PR)*Rt+S_out-S_in)**2)

        J['eff_poly', 'S_in']  =  (np.log(PR)*Rt)/(np.log(PR)*Rt+S_out-S_in)**2
        J['eff_poly', 'S_out'] = -(np.log(PR)*Rt)/(np.log(PR)*Rt+S_out-S_in)**2

        J['eff_poly', 'Rt'] = (np.log(PR)*(S_out-S_in))/(np.log(PR)*Rt+S_out-S_in)**2


class Power(om.ExplicitComponent):
    """Power calculates shaft power for the compressor or turbine"""

    def setup(self):
        # inputs
        self.add_input('W', val=30.0, units='lbm/s', desc='mass flow')
        self.add_input('ht_out', val=20.0, units='Btu/lbm',
                       desc='downstream enthalpy')
        self.add_input('ht_in', val=10.0, units='Btu/lbm',
                       desc='incoming enthalpy')
        self.add_input('Nmech', val=1000.0, units='rpm', desc='shaft speed')
        # self.add_input('Tt_in', val=500., units='degR', desc='incoming temperature')
        # outputs
        self.add_output('power', shape=1, units='hp', desc='turbine power')
        self.add_output('trq', shape=1, units='ft*lbf', desc='turbine torque')

        self.declare_partials('power', ['W', 'ht_out', 'ht_in'])
        self.declare_partials('trq', '*')

    def compute(self, inputs, outputs):

        outputs['power'] = inputs['W'] * (inputs['ht_in'] - inputs['ht_out']) * BTU_s2HP
        outputs['trq'] = HP_per_RPM_to_FT_LBF * outputs['power'] / inputs['Nmech']


    def compute_partials(self, inputs, J):
        ht_in = inputs['ht_in']
        ht_out = inputs['ht_out']
        W = inputs['W']
        Nmech = inputs['Nmech']

        J['power', 'W'] = (ht_in - ht_out) * BTU_s2HP
        J['power', 'ht_out'] = -W * BTU_s2HP
        J['power', 'ht_in'] = W * BTU_s2HP
        # J['power','Nmech'] = 0.

        J['trq', 'W'] = (ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        J['trq', 'ht_out'] = -W * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        J['trq', 'ht_in'] = W * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        J['trq', 'Nmech'] = -W * (ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF * Nmech**-2.


class BleedsAndPower(om.ExplicitComponent):
    """BleedsAndPower calculates the bleed flows and shaft power for the compressor"""

    def initialize(self):
        self.options.declare('bleed_names', types=Iterable,
                              desc='list of names for the bleed ports')

    def setup(self):
        self.add_input('W_in', val=30.0, units='lbm/s',
                       desc='entrance mass flow')
        self.add_input('ht_out', val=20.0, units='Btu/lbm',
                       desc='exit total enthalpy')
        self.add_input('ht_in', val=10.0, units='Btu/lbm',
                       desc='entrance total enthalpy')
        self.add_input('Pt_out', val=20.0, units='psi',
                       desc='exit total pressure')
        self.add_input('Pt_in', val=10.0, units='psi',
                       desc='entrance total pressure')
        self.add_input('Nmech', val=1000.0, units='rpm', desc='shaft speed')

        self.add_output('W_out', shape=1, units='lbm/s', desc='exit mass flow')
        self.add_output('power', shape=1, units='hp', desc='shaft power')
        self.add_output('trq', shape=1, units='ft*lbf', desc='shaft torque')

        self.declare_partials('W_out', 'W_in')
        self.declare_partials('power', ['W_in', 'ht_in', 'ht_out'])
        self.declare_partials('trq', ['W_in', 'ht_in', 'ht_out', 'Nmech'])

        # bleed inputs and outputs
        for BN in self.options['bleed_names']:
            self.add_input(BN + ':frac_W', val=0.0,
                           desc='bleed mass flow fraction (W_bld/W_in)')
            self.add_input(BN + ':frac_P', val=0.0,
                           desc='bleed pressure fraction ((P_bld-P_in)/(P_out-P_in))')
            self.add_input(BN + ':frac_work', val=0.0,
                           desc='bleed work fraction ((h_bld-h_in)/(h_out-h_in))')

            self.add_output(BN + ':stat:W', shape=1, lower=0.0,
                            units='lbm/s', desc='bleed mass flow')
            self.add_output(BN + ':Pt', shape=1, lower=1e-6,
                            units='psi', desc='bleed total pressure')
            self.add_output(BN + ':ht', shape=1, units='Btu/lbm',
                            desc='bleed total enthalpy')
            # self.add_output(BN+':power', shape=1, desc='bleed power reduction')

            self.declare_partials('W_out', BN+':frac_W')
            self.declare_partials('power', [BN+':frac_W', BN+':frac_work'])
            self.declare_partials(BN+':stat:W', ['W_in', BN+':frac_W'])
            self.declare_partials(BN+':Pt', ['Pt_in', BN+':frac_P', 'Pt_out'])
            self.declare_partials(BN+':ht', ['ht_in', BN+':frac_work', 'ht_out'])
            self.declare_partials('trq', [BN+':frac_W', BN+':frac_work'])

    def compute(self, inputs, outputs):

        Pt_in = inputs['Pt_in']
        Pt_out = inputs['Pt_out']
        ht_in = inputs['ht_in']
        ht_out = inputs['ht_out']
        W_in = inputs['W_in']

        # calculate flow and power without bleed flows
        outputs['W_out'] = W_in
        outputs['power'] = W_in * (ht_in - ht_out) * BTU_s2HP

        # calculate bleed specific outputs and modify exit flow and power
        for BN in self.options['bleed_names']:
            BN_stat_W = BN + ':stat:W'
            BN_ht = BN + ':ht'

            stat_W = W_in * inputs[BN + ':frac_W']
            outputs[BN + ':Pt'] = Pt_in + inputs[BN + ':frac_P'] * (Pt_out - Pt_in)
            ht = ht_in + inputs[BN + ':frac_work'] * (ht_out - ht_in)

            outputs['W_out'] -= stat_W
            outputs['power'] -= stat_W * (ht - ht_out) * BTU_s2HP
            outputs[BN_stat_W] = stat_W
            outputs[BN_ht] = ht

        # calculate torque based on revised power and shaft speed
        outputs['trq'] = HP_per_RPM_to_FT_LBF * outputs['power'] / inputs['Nmech']

    def compute_partials(self, inputs, J):

        ht_in = inputs['ht_in']
        ht_out = inputs['ht_out']
        W_in = inputs['W_in']
        Nmech = inputs['Nmech']
        delta_Pt = inputs['Pt_out'] - inputs['Pt_in']

        # Jacobian elements without bleed flows
        dW_out_dW_in = 1.0

        dpower_dW_in = (ht_in - ht_out) * BTU_s2HP
        dpower_dht_in = W_in * BTU_s2HP
        dpower_dht_out = -W_in * BTU_s2HP

        dtrq_dW_in = (ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        dtrq_dht_in = W_in * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        dtrq_dht_out = -W_in * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
        dtrq_dNmech = -W_in * (ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF * Nmech**-2.

        # Jacobian elements and modifications due to bleed flows
        for BN in self.options['bleed_names']:
            BN_frac_W = BN + ':frac_W'
            BN_frac_work = BN + ':frac_work'
            BN_Pt = BN + ':Pt'
            BN_stat_W = BN + ':stat:W'
            BN_frac_P = BN + ':frac_P'
            BN_ht = BN + ':ht'

            frac_W = inputs[BN_frac_W]
            frac_work = inputs[BN_frac_work]
            frac_P = inputs[BN_frac_P]

            dW_out_dW_in -= frac_W
            J['W_out', BN_frac_W] = -W_in

            dpower_dW_in -= frac_W * (1.0 - frac_work) * (ht_in - ht_out) * BTU_s2HP
            dpower_dht_in -= W_in * frac_W * (1.0 - frac_work) * BTU_s2HP
            dpower_dht_out -= -W_in * frac_W * (1.0 - frac_work) * BTU_s2HP
            J['power', BN_frac_W] = -W_in * (1.0 - frac_work) * (ht_in - ht_out) * BTU_s2HP
            J['power', BN_frac_work] = W_in * frac_W * (ht_in - ht_out) * BTU_s2HP

            J[BN_stat_W, 'W_in'] = frac_W
            J[BN_stat_W, BN_frac_W] = W_in

            J[BN_Pt, 'Pt_in'] = 1.0 - frac_P
            J[BN_Pt, BN_frac_P] = delta_Pt
            J[BN_Pt, 'Pt_out'] = frac_P

            J[BN_ht, 'ht_in'] = 1.0 - frac_work
            J[BN_ht, BN_frac_work] = ht_out - ht_in
            J[BN_ht, 'ht_out'] = frac_work

            dtrq_dW_in -= frac_W * (1.0 - frac_work) * (
                ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
            dtrq_dht_in -= W_in * frac_W * (1.0 - frac_work) * BTU_s2HP * \
                HP_per_RPM_to_FT_LBF / Nmech
            dtrq_dht_out -= -W_in * frac_W * (1.0 - frac_work) * BTU_s2HP * \
                HP_per_RPM_to_FT_LBF / Nmech
            J['trq', BN_frac_W] = -W_in * (1.0 - frac_work) * (
                ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
            J['trq', BN_frac_work] = W_in * frac_W * (ht_in - ht_out) * \
                BTU_s2HP * HP_per_RPM_to_FT_LBF / Nmech
            dtrq_dNmech -= -W_in * frac_W * (1.0 - frac_work) * (
                ht_in - ht_out) * BTU_s2HP * HP_per_RPM_to_FT_LBF * Nmech**-2

        J['W_out', 'W_in'] = dW_out_dW_in
        J['power', 'W_in'] = dpower_dW_in
        J['power', 'ht_in'] = dpower_dht_in
        J['power', 'ht_out'] = dpower_dht_out
        J['trq', 'W_in'] = dtrq_dW_in
        J['trq', 'ht_in'] = dtrq_dht_in
        J['trq', 'ht_out'] = dtrq_dht_out
        J['trq', 'Nmech'] = dtrq_dNmech

class EnthalpyRise(om.ExplicitComponent):
    """Calculates enthalpy rise across a compressor"""

    def setup(self):

        self.add_input('ideal_ht', val=2.0, units='Btu/lbm',
                       desc='ideal exit total enthalpy')
        self.add_input('inlet_ht', val=1.0, units='Btu/lbm',
                       desc='entrance total enthalpy')
        self.add_input('eff', val=0.5, desc='design efficiency')

        self.add_output('ht_out', shape=1, units='Btu/lbm',
                        desc='exit total enthalpy')

        self.declare_partials('ht_out', '*')

    def compute(self, inputs, outputs):
        inlet_ht = inputs['inlet_ht']
        outputs['ht_out'] = (inputs['ideal_ht'] - inlet_ht) / inputs['eff'] + inlet_ht

    def compute_partials(self, inputs, J):
        eff = inputs['eff']

        J['ht_out', 'ideal_ht'] = 1. / eff
        J['ht_out', 'inlet_ht'] = 1 - 1. / eff
        J['ht_out', 'eff'] = - (inputs['ideal_ht'] -
                                inputs['inlet_ht']) * eff**(-2)


class PressureRise(om.ExplicitComponent):
    """A Component that calculates ..."""

    def setup(self):
        self.add_input('PR', 3.0, desc="design pressure ratio")
        self.add_input('Pt_in', 5.0, units='lbf/inch**2',
                       desc="incomming total pressure")

        self.add_output('Pt_out', shape=1, lower=1e-5,
                        units='lbf/inch**2', desc="exit total pressure")

        self.declare_partials('Pt_out', '*')

    def compute(self, inputs, outputs):
        outputs['Pt_out'] = inputs['PR'] * inputs['Pt_in']

    def compute_partials(self, inputs, J):
        J['Pt_out', 'Pt_in'] = inputs['PR']
        J['Pt_out', 'PR'] = inputs['Pt_in']


class Compressor(Element):
    """
    Calculates pressure and temperature rise of a flow through a non-ideal compressors,
    using turbomachinery performance maps.

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
        map.RlineMap
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
        Wc
        PR
        eff_poly
        Nc
        power
        map.RlineMap
        map.readmap.NcMap
    """

    def initialize(self):
        self.options.declare('map_data', default=NCP01,
                              desc='data container for raw compressor map data')
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
            ('s_Wc', 's_Wc'),
            ('s_PR', 's_PR'),
            ('s_eff', 's_eff'),
            ('s_Nc', 's_Nc'),
            ('Fl_O:stat:area', 'area')
        ]

        super().initialize()

    def pyc_setup_output_ports(self):

        self.copy_flow('Fl_I', 'Fl_O')

        bleeds = self.options['bleed_names']
        for BN in bleeds:
            self.copy_flow('Fl_I', BN)

    def setup(self):

        map_data = self.options['map_data']
        interp_method = self.options['map_interp_method']
        map_extrap = self.options['map_extrap']
        # self.linear_solver = ScipyGMRES()
        # self.linear_solver.options['atol'] = 2e-8
        # self.linear_solver.options['maxiter'] = 100
        # self.linear_solver.options['restart'] = 100

        # self.nonlinear_solver = Newton()
        # self.nonlinear_solver.options['utol'] = 1e-9

        thermo_method = self.options['thermo_method']
        design = self.options['design']
        bleeds = self.options['bleed_names']
        thermo_data = self.options['thermo_data']
        statics = self.options['statics']

        composition = self.Fl_I_data['Fl_I']

        # Create inlet flow station
        flow_in = FlowIn(fl_name='Fl_I')
        self.add_subsystem('flow_in', flow_in, promotes_inputs=['Fl_I:*'])

        self.add_subsystem('corrinputs', CorrectedInputsCalc(),
                           promotes_inputs=(
                               'Nmech', ('W_in', 'Fl_I:stat:W'),
                               ('Pt', 'Fl_I:tot:P'), ('Tt', 'Fl_I:tot:T')),
                           promotes_outputs=('Nc', 'Wc'))

        map_calcs = CompressorMap(map_data=self.options['map_data'], design=design,
                            interp_method=interp_method, extrap=map_extrap)
        self.add_subsystem('map', map_calcs,
                            promotes=['s_Nc','s_eff','s_Wc','s_PR','Nc','Wc',
                                    'PR','eff','SMN','SMW'])

        # Calculate pressure rise across compressor
        self.add_subsystem('press_rise', PressureRise(), promotes_inputs=[
                           'PR', ('Pt_in', 'Fl_I:tot:P')])

        # Calculate ideal flow station properties
        ideal_flow = Thermo(mode='total_SP',
                            method=thermo_method,
                            thermo_kwargs={'composition':composition,
                                           'spec':thermo_data})
        self.add_subsystem('ideal_flow', ideal_flow,
                           promotes_inputs=[('S', 'Fl_I:tot:S'),
                                            ('composition', 'Fl_I:tot:composition')])
        self.connect("press_rise.Pt_out", "ideal_flow.P")

        # Calculate enthalpy rise across compressor
        self.add_subsystem("enth_rise", EnthalpyRise(),
                           promotes_inputs=['eff', ('inlet_ht', 'Fl_I:tot:h')])
        self.connect("ideal_flow.h", "enth_rise.ideal_ht")

        # Calculate real flow station properties
        real_flow = Thermo(mode='total_hP', fl_name='Fl_O:tot',
                                  method=thermo_method,
                                  thermo_kwargs={'composition':composition,
                                                 'spec':thermo_data})
        self.add_subsystem('real_flow', real_flow,
                           promotes_inputs=[
                               ('composition', 'Fl_I:tot:composition')],
                           promotes_outputs=['Fl_O:tot:*'])
        self.connect("enth_rise.ht_out", "real_flow.h")
        self.connect("press_rise.Pt_out", "real_flow.P")
        #clculate Polytropic Efficiency
        self.add_subsystem('eff_poly_calc', eff_poly_calc(),
                            promotes_inputs=[('PR','PR'),
                                             ('S_in','Fl_I:tot:S'),
                                             ('S_out','Fl_O:tot:S'),
                                             # ('Cp','Fl_I:tot:Cp'),
                                             # ('Cv','Fl_I:tot:Cv'),
                                             ('Rt', 'Fl_I:tot:R')],
                            promotes_outputs=['eff_poly'] )

        # Calculate shaft power consumption
        blds_pwr = BleedsAndPower(bleed_names=bleeds)
        bld_inputs = ['frac_W', 'frac_P', 'frac_work']
        bld_in_vars = ['{0}:{1}'.format(
            bn, in_name) for bn, in_name in itertools.product(bleeds, bld_inputs)]
        bld_out_globs = ['{}:*'.format(bn) for bn in bleeds]

        self.add_subsystem('blds_pwr', blds_pwr,
                           promotes_inputs=['Nmech', ('W_in', 'Fl_I:stat:W'),
                                            ('ht_in', 'Fl_I:tot:h'),
                                            ('Pt_in', 'Fl_I:tot:P'),
                                            ('Pt_out', 'Fl_O:tot:P'), ] + bld_in_vars,
                           promotes_outputs=['power', 'trq', 'W_out'] + bld_out_globs)
        self.connect('enth_rise.ht_out', 'blds_pwr.ht_out')

        bleed_names = []
        for BN in bleeds:

            bleed_names.append(f'{BN}_flow')
            bleed_flow = Thermo(mode='total_hP', fl_name=BN + ":tot",
                                  method=thermo_method,
                                  thermo_kwargs={'composition':composition,
                                                 'spec':thermo_data})
            self.add_subsystem(BN + '_flow', bleed_flow,
                               promotes_inputs=[
                                   ('composition', 'Fl_I:tot:composition')],
                               promotes_outputs=[f'{BN}:tot:*'])
            self.connect(BN + ':ht', BN + "_flow.h")
            self.connect(BN + ':Pt', BN + "_flow.P")


        if statics:
            if design:
                #   Calculate static properties
                out_stat = Thermo(mode='static_MN', fl_name='Fl_O:stat',
                                  method=thermo_method,
                                  thermo_kwargs={'composition':composition,
                                                 'spec':thermo_data})
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=[
                                       'MN', ('composition', 'Fl_I:tot:composition')],
                                   promotes_outputs=['Fl_O:stat:*'])
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            else:  # Calculate static properties
                out_stat = Thermo(mode='static_A', fl_name='Fl_O:stat',
                                  method=thermo_method,
                                  thermo_kwargs={'composition':composition,
                                                 'spec':thermo_data})
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=[
                                       'area', ('composition', 'Fl_I:tot:composition')],
                                   promotes_outputs=['Fl_O:stat:*'])

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            self.set_order(['flow_in', 'corrinputs', 'map',
                            'press_rise','ideal_flow', 'enth_rise',
                            'real_flow','eff_poly_calc' ,'blds_pwr',]
                            + bleed_names + ['out_stat'])

        else:
            self.add_subsystem('W_passthru', PassThrough('W_out',
                                                         'Fl_O:stat:W',
                                                         1.0,
                                                         units="lbm/s"),
                               promotes=['*'])
            self.set_order(['flow_in', 'corrinputs', 'map',
                            'press_rise','ideal_flow', 'enth_rise',
                            'real_flow','eff_poly_calc' , 'blds_pwr']
                            + bleed_names + ['W_passthru'])


        # define the group level defaults
        self.set_input_defaults('Fl_I:FAR', val=0., units=None)
        self.set_input_defaults('PR', val=2., units=None)
        self.set_input_defaults('eff', val=0.99, units=None)

        # if not design:
        #     self.set_input_defaults('area', val=1, units='inch**2')

        super().setup()

