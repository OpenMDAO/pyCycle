from collections.abc import Iterable
from copy import copy

import numpy as np

import openmdao.api as om

from pycycle.constants import BTU_s2HP, HP_per_RPM_to_FT_LBF, AIR_MIX, AIR_FUEL_MIX
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle.cea import species_data
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
# from pycycle.components.compressor import Power

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


class Bleeds(om.ExplicitComponent):
    """Computes bleed flow parameters"""

    def initialize(self):
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('main_flow_elements',
                              desc='set of elements present in the flow')
        self.options.declare('bld_flow_elements', default=AIR_MIX,
                              desc='set of elements present in the flow')
        self.options.declare('bleed_names', types=Iterable, desc='list of names for the bleed ports',
                              default=[])

    def setup(self):

        bleeds = self.options['bleed_names']

        self.main_flow_elements = self.options['main_flow_elements']
        thermo_data = self.options['thermo_data']
        self.mixed_elements = self.main_flow_elements.copy()
        self.mixed_elements.update(self.options['bld_flow_elements'])

        main_flow_thermo = species_data.Thermo(
            thermo_data, init_reacts=self.mixed_elements)
        self.main_flow_prods = main_flow_thermo.products
        self.main_flow_wt_mole = main_flow_thermo.wt_mole
        self.n_main_flow_prods = len(self.main_flow_prods)

        bld_flow_thermo = species_data.Thermo(
            thermo_data, init_reacts=self.options['bld_flow_elements'])
        self.bld_flow_prods = bld_flow_thermo.products
        self.bld_flow_wt_mole = bld_flow_thermo.wt_mole
        self.n_bld_flow_prods = len(self.bld_flow_prods)

        # primary inputs and outputs
        self.add_input('Pt_in', val=0.0, units='psi',
                       desc='turbine entrance pressure')
        self.add_input('Pt_out', val=0.0, units='psi',
                       desc='turbine exit pressure')
        self.add_input('W_in', val=0.0, units='lbm/s',
                       desc='turbine entrance mass flow rate')
        self.add_input('n_in', val=np.zeros(self.n_main_flow_prods),
                       desc='turbine entrance flow composition')

        self.add_output('W_out', shape=1, units='lbm/s',
                        desc='turbine exit mass flow rate')
        self.add_output('n_out', shape=self.n_main_flow_prods,
                        desc='turbine exit flow composition')

        # bleed inputs and outputs
        for BN in bleeds:
            self.add_input(BN + ':frac_P', val=0.0,
                           desc='fraction of pressure drop where bleed flow introduced')
            self.add_input(BN + ':W', val=0.0, units='lbm/s',
                           desc='bleed mass flow rate')
            self.add_input(
                BN + ':n', val=np.zeros(self.n_bld_flow_prods), desc='bleed flow composition')

            self.add_output(BN + ':Pt', shape=1, units='psi',
                            desc='pressure of incomming bleed flow', lower=1e-3)

            self.declare_partials(BN+':Pt', ['Pt_in', 'Pt_out', BN+':frac_P'])
            self.declare_partials('W_out', BN+':W', val=1.0)
            self.declare_partials('n_out', [BN+':W', BN+':n'])

        # create mapping for main and bleed flow
        self.main_flow_idx_map = {prod: i for i, prod in enumerate(self.main_flow_prods)}
        self.bld_flow_idx_map = {prod: i for i, prod in enumerate(self.bld_flow_prods)}

        n_main = len(self.main_flow_prods)
        n_bld = len(self.bld_flow_prods)
        self.mix_mat = np.zeros((n_main, n_bld), dtype=int)

        for j, prod in enumerate(self.bld_flow_prods):
            i = self.main_flow_idx_map[prod]
            self.mix_mat[i,j] = 1

        self.declare_partials('W_out', 'W_in', val=1.0)
        self.declare_partials('n_out', ['W_in', 'n_in'])

    def compute(self, inputs, outputs):

        # calculate W_out and composition based on primary flow
        W_in = inputs['W_in']
        W_out = copy(W_in)
        Pt_out = inputs['Pt_out']
        delta_Pt = inputs['Pt_in'] - Pt_out

        n_mass = inputs['n_in'] * self.main_flow_wt_mole

        flow_mass = n_mass / np.sum(n_mass) * W_in

        # calculate W_out, bleed total pressure and composition based on
        # primary flow
        W_bld = 0.0
        n_bld_tot = np.zeros(self.n_bld_flow_prods, dtype=inputs._data.dtype)

        bleeds = self.options['bleed_names']
        for BN in bleeds:
            outputs[BN + ':Pt'] = Pt_out + inputs[BN+':frac_P'] * delta_Pt
            W_bld += inputs[BN + ':W']
            n_bld_tot += inputs[BN + ':n']

        W_out += W_bld

        if len(bleeds) > 0:
            n_bld_mass = n_bld_tot * self.bld_flow_wt_mole
            bld_mass = n_bld_mass / np.sum(n_bld_mass) * W_bld

            flow_mass += self.mix_mat.dot(bld_mass)

        # determine the exit composition
        flow_mass_norm = flow_mass / W_out
        outputs['n_out'] = flow_mass_norm / self.main_flow_wt_mole
        outputs['W_out'] = W_out

    def compute_partials(self, inputs, J):

        bleeds = self.options['bleed_names']

        n_in = inputs['n_in']
        Pt_in = inputs['Pt_in']
        Pt_out = inputs['Pt_out']
        W_in = inputs['W_in']
        n_len = self.n_main_flow_prods

        mfwm = self.main_flow_wt_mole
        bfwm = self.bld_flow_wt_mole

        n_mass = n_in * mfwm
        n_mass_sum = np.sum(n_mass)
        flow_mass = n_mass / n_mass_sum * W_in
        exit_total_mass = 0.0
        exit_total_mass += W_in
        W_bld = 0.0
        n_bld_tot = np.zeros(self.n_bld_flow_prods, dtype=inputs._data.dtype)

        for BN in bleeds:
            W_bld += inputs[BN + ':W']
            n_bld_tot += inputs[BN + ':n']

        if len(bleeds) > 0:
            n_bld_mass = n_bld_tot * bfwm
            bld_mass_sum = np.sum(n_bld_mass)
            bld_mass = n_bld_mass / bld_mass_sum * W_bld
            bld_mass_flw = self.mix_mat.dot(bld_mass)

            flow_mass += bld_mass_flw

        exit_total_mass += W_bld

        # Jacobian elements without bleed flows
        J['n_out', 'W_in'] = (n_mass/n_mass_sum - flow_mass/exit_total_mass)/(exit_total_mass*mfwm)

        A = (np.diag(mfwm) - np.outer(n_mass, mfwm)/n_mass_sum)
        J['n_out', 'n_in'] = (A.T * W_in/(exit_total_mass*n_mass_sum*mfwm)).T

        # Jacobian elements and modifications due to bleed flows
        for BN in bleeds:
            # J['W_out', BN + ':W'] = 1.0
            BN_W = BN + ':W'
            BN_n = BN + ':n'
            BN_pt = BN + ':Pt'
            BN_frac_P = BN + ':frac_P'

            W = inputs[BN_W]
            n = inputs[BN_n]
            frac_P = inputs[BN_frac_P]

            J[BN_pt, 'Pt_in'] = frac_P
            J[BN_pt, 'Pt_out'] = 1.0 - frac_P
            J[BN_pt, BN_frac_P] = Pt_in - Pt_out

            n_bld_mass = n * bfwm
            bld_mass_i = n_bld_mass / np.sum(n_bld_mass)

            bld_mass_new = self.mix_mat.dot(bld_mass_i)
            J['n_out', BN_W] = (-flow_mass/exit_total_mass + bld_mass_new)/(exit_total_mass*mfwm)

            A = self.mix_mat*bfwm*W_bld - np.outer(bld_mass_flw, bfwm)
            J['n_out', BN_n] = (A.T / (exit_total_mass*bld_mass_sum*mfwm)).T


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


class Turbine(om.Group):
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
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_MIX,
                              desc='set of elements present in the flow')
        self.options.declare('bleed_elements', default=AIR_MIX,
                              desc='set of elements present in the flow')
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('bleed_names', types=(list,tuple), desc='list of names for the bleed ports',
                              default=[])
        self.options.declare('map_interp_method', default='slinear',
                              desc='Method to use for map interpolation. \
                              Options are `slinear`, `cubic`, `quintic`.')
        self.options.declare('map_extrap', default=False, desc='Switch to allow extrapoloation off map')

        self.default_des_od_conns = [
            # (design src, off-design target)
            ('s_WpDes', 's_WpDes'),
            ('s_PRdes', 's_PRdes'),
            ('s_effDes', 's_effDes'), 
            ('s_NpDes', 's_NpDes'), 
            ('Fl_O:stat:area', 'area')
        ]



    def setup(self):

        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        bleed_elements = self.options['bleed_elements']
        map_data = self.options['map_data']
        designFlag = self.options['design']
        bleeds = self.options['bleed_names']
        statics = self.options['statics']
        interp_method = self.options['map_interp_method']
        map_extrap = self.options['map_extrap']

        gas_thermo = species_data.Thermo(thermo_data, init_reacts=elements)
        self.gas_prods = gas_thermo.products
        self.num_prod = len(self.gas_prods)

        bld_thermo = species_data.Thermo(
            thermo_data, init_reacts=bleed_elements)
        self.bld_prods = bld_thermo.products
        self.num_bld_prod = len(self.bld_prods)

        # Create inlet flow station
        in_flow = FlowIn(fl_name='Fl_I', num_prods=self.num_prod)
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
        self.add_subsystem('ideal_flow', SetTotal(thermo_data=thermo_data, mode='S', init_reacts=elements),
                           promotes_inputs=[('S', 'Fl_I:tot:S'), ('init_prod_amounts', 'Fl_I:tot:n')])
        self.connect("press_drop.Pt_out", "ideal_flow.P")

        # # Calculate enthalpy drop across turbine
        # self.add_subsystem("enth_drop", EnthalpyDrop(), promotes=['eff'])
        # self.connect("Fl_I:tot:h", "enth_drop.ht_in")
        # self.connect("ideal_flow.h", "enth_drop.ht_out_ideal")

        for BN in bleeds:
            bld_flow = FlowIn(fl_name=BN, num_prods=self.num_bld_prod)
            self.add_subsystem(BN, bld_flow, promotes_inputs=[
                               '{}:*'.format(BN)])

        # Calculate bleed parameters
        blds = Bleeds(bleed_names=bleeds, main_flow_elements=elements)
        self.add_subsystem('blds', blds,
                           promotes_inputs=[('W_in', 'Fl_I:stat:W'),
                                            ('Pt_in', 'Fl_I:tot:P'), ('n_in', 'Fl_I:tot:n')] +
                           ['{}:frac_P'.format(BN) for BN in bleeds] +
                           [('{}:W'.format(BN), '{}:stat:W'.format(BN)) for BN in bleeds] +
                           [('{}:n'.format(BN), '{}:tot:n'.format(BN))
                            for BN in bleeds],
                           promotes_outputs=['W_out'])
        self.connect('press_drop.Pt_out', 'blds.Pt_out')

        bleed_names2 = []
        for BN in bleeds:
            # self.connect(BN+':stat:W','blds.{}:W'.format(BN))
            # self.connect(BN+':tot:n','blds.{}:n'.format(BN))
            # self.connect(BN+':stat:W','blds.%s:'%BN)

            # Determine bleed inflow properties
            bleed_names2.append(BN + '_inflow')
            self.add_subsystem(BN + '_inflow', SetTotal(thermo_data=thermo_data, mode='h', init_reacts=bleed_elements),
                               promotes_inputs=[('init_prod_amounts', BN + ":tot:n"), ('h', BN + ':tot:h')])
            self.connect('blds.' + BN + ':Pt', BN + "_inflow.P")

            # Ideally expand bleeds to exit pressure
            bleed_names2.append(BN + '_ideal')
            self.add_subsystem(BN + '_ideal', SetTotal(thermo_data=thermo_data, mode='S', init_reacts=bleed_elements),
                               promotes_inputs=[('init_prod_amounts', BN + ":tot:n")])
            self.connect(BN + "_inflow.flow:S", BN + "_ideal.S")
            self.connect("press_drop.Pt_out", BN + "_ideal.P")

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
        real_flow_b4bld = SetTotal(thermo_data=thermo_data, mode='h',
                     init_reacts=elements, fl_name="Fl_O_b4bld:tot")
        self.add_subsystem('real_flow_b4bld', real_flow_b4bld,
                           promotes_inputs=[('init_prod_amounts', 'Fl_I:tot:n')])
        self.connect('ht_out_b4bld', 'real_flow_b4bld.h')
        self.connect('press_drop.Pt_out', 'real_flow_b4bld.P')

        # Calculate Polytropic efficiency
        self.add_subsystem('eff_poly_calc',eff_poly_calc(),promotes_inputs=['PR',('S_in','Fl_I:tot:S'),
                            ('Rt','Fl_I:tot:R')],
                            promotes_outputs=['eff_poly'])
        self.connect('real_flow_b4bld.Fl_O_b4bld:tot:S','eff_poly_calc.S_out')

        # Calculate real flow station properties
        real_flow = SetTotal(thermo_data=thermo_data, mode='h',
                             init_reacts=elements, fl_name="Fl_O:tot")
        self.add_subsystem('real_flow', real_flow,
                           promotes_outputs=['Fl_O:tot:*'])
        self.connect("pwr_turb.ht_out", "real_flow.h")
        self.connect("press_drop.Pt_out", "real_flow.P")
        self.connect("blds.n_out", "real_flow.init_prod_amounts")

        self.add_subsystem('FAR_passthru', PassThrough(
            'Fl_I:FAR', 'Fl_O:FAR', 1.0), promotes=['*'])

       # Calculate static properties
        if statics:
            if designFlag:
                #   SetStaticMN
                out_stat = SetStatic(
                    mode='MN', thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O:stat")
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=['MN'],
                                   promotes_outputs=['Fl_O:stat:*'])
                self.connect('blds.n_out', 'out_stat.init_prod_amounts')
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            else:
                #   SetStaticArea
                out_stat = SetStatic(
                    mode='area', thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O:stat")
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=['area'],
                                   promotes_outputs=['Fl_O:stat:*'])
                self.connect('blds.n_out', 'out_stat.init_prod_amounts')
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('W_out', 'out_stat.W')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            self.set_order(['in_flow', 'corrinputs', 'map', 'press_drop', 'ideal_flow'] + bleeds + ['blds'] + bleed_names2 +
                           ['pwr_turb','real_flow_b4bld', 'eff_poly_calc' ,'real_flow', 'FAR_passthru', 'out_stat'])

        else:
            self.add_subsystem('W_passthru', PassThrough(
                'W_out', 'Fl_O:stat:W', 1.0, units="lbm/s"), promotes=['*'])
            self.set_order(['in_flow', 'corrinputs', 'map', 'press_drop', 'ideal_flow'] + bleeds + ['blds'] + bleed_names2 +
                           ['pwr_turb','real_flow_b4bld', 'eff_poly_calc', 'real_flow', 'FAR_passthru', 'W_passthru'])

        self.set_input_defaults('Fl_I:FAR', val=0., units=None)
        self.set_input_defaults('eff', val=0.99, units=None)
        # if not designFlag: 
        #     self.set_input_defaults('area', val=1, units='in**2')


if __name__ == "__main__":
    from pycycle.api import FlowStart
    from pycycle.cea import species_data
    from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
    from pycycle.connect_flow import connect_flow
    from openmdao.api import IndepVarComp
    from pycycle.maps.lpt2269 import LPT2269

    gas = AIR_FUEL_MIX

    prob = om.Problem()
    prob.model = om.Group()

    dv = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    dv.add_output('P', 15.8172, units='lbf/inch**2')
    dv.add_output('T', 2644.02, units='degR')
    dv.add_output('W', 100.0, units='lbm/s')
    dv.add_output('eff',0.9, units=None)
    dv.add_output('P_bld', 16.000, units='lbf/inch**2')
    dv.add_output('T_bld', 2000.0, units='degR')
    dv.add_output('W_bld', 1.0, units='lbm/s')
    dv.add_output('PR', 4.0, units=None)
    dv.add_output('frac_P', 0.5),

    prob.model.add_subsystem('flow_start', FlowStart(
        thermo_data=species_data.janaf, elements=AIR_MIX))
    prob.model.add_subsystem('bld_start', FlowStart(
        thermo_data=species_data.janaf, elements=AIR_MIX))
    prob.model.add_subsystem('turbine', Turbine(
        map_data=LPT2269, design=True, elements=AIR_MIX,
        bleed_names=['bld1']))

    connect_flow(prob.model, 'flow_start.Fl_O', 'turbine.Fl_I')
    connect_flow(prob.model, 'bld_start.Fl_O', 'turbine.bld1', connect_stat=False)

    prob.model.connect("P", "flow_start.P")
    prob.model.connect("T", "flow_start.T")
    prob.model.connect("W", "flow_start.W")
    prob.model.connect("P_bld", "bld_start.P")
    prob.model.connect("T_bld", "bld_start.T")
    prob.model.connect("W_bld", "bld_start.W")
    prob.model.connect("PR", "turbine.PR")
    prob.model.connect('eff', 'turbine.eff')
    prob.model.connect("frac_P", "turbine.bld1:frac_P")

    prob.setup()
    # prob.model.flow_start.list_connections()
    prob.run_model()

    print(prob['turbine.Fl_O:tot:T'])
    print(prob['turbine.Fl_O:tot:P'])
    print(prob['turbine.Fl_O:stat:W'])
    print(prob['bld_start.Fl_O:stat:W'])

    prob.check_partials(compact_print=True, abs_err_tol=1e-3, rel_err_tol=1e-3)
