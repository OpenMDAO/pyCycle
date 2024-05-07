import numpy as np
import openmdao.api as om

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC
from numpy import log as ln

class ThermoCalcs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('spec', recordable=False)
        # self.options.declare('composition')
        


    def setup(self):
        self.add_input('P', 101325.0, units='Pa')
        self.add_input('T', 273.0, units='degK')
        self.add_input('FAR', 0.015, units=None)

        self.add_output('h', 1.0, units='J/kg')
        self.add_output('S', 1.0, units='J/kg/degK')
        self.add_output('gamma', 1.4, units=None)
        self.add_output('Cp', 1.0, units='J/kg/degK')
        self.add_output('Cv', 1.0, units='J/kg/degK')
        self.add_output('rho', 1.0, units='kg/m**3')
        self.add_output('R', 287.0, units='J/kg/degK')

        self.declare_partials(of=['h','Cp','Cv'], wrt=['T','FAR'], method='fd')
        self.declare_partials(of=['R'], wrt=['FAR'], method='fd')
       
        self.declare_partials(of=['S','rho'], wrt=['T','P','FAR'], method='fd')

    def compute(self, inputs, outputs):
        spec = self.options['spec']
        
        a_T = spec.products['Air']['coeffs'][0]
        b_T = spec.products['JET_A']['coeffs'][0]

        P = inputs['P']
        FAR = inputs['FAR']
        TZ = inputs['T']/1000

        CP1 = a_T[0] + a_T[1] * TZ + a_T[2] * TZ**2 + a_T[3] * TZ**3 + a_T[4] * TZ**4 + a_T[5] * TZ**5 + a_T[6] * TZ**6 + a_T[7] * TZ**7 + a_T[8] * TZ**8
        CP2 = b_T[0] + b_T[1] * TZ + b_T[2] * TZ**2 + b_T[3] * TZ**3  + b_T[4] * TZ**4 + b_T[5] * TZ**5 + b_T[6] * TZ**6 + b_T[7] * TZ**7

        H1 =  a_T[0] *TZ + a_T[1]/2 * TZ**2 + a_T[2]/3 * TZ**3 + a_T[3]/4 * TZ**4 + a_T[4]/5 * TZ**5 + a_T[5]/6 * TZ**6 + a_T[6]/7 * TZ**7 + a_T[7]/8 * TZ**8 + a_T[8]/9 * TZ**9 + a_T[9]
        H2 =  b_T[0] *TZ + b_T[1]/2 * TZ**2 + b_T[2]/3 * TZ**3 + b_T[3]/4 * TZ**4 + b_T[4]/5 * TZ**5 + b_T[5]/6 * TZ**6 + b_T[6]/7 * TZ**7 + b_T[8]

        CpdT_T1 = a_T[0] * ln(TZ) + a_T[1] * TZ + a_T[2]/2 * TZ**2 + a_T[3]/3 * TZ**3 + a_T[4]/4 * TZ**4 + a_T[5]/5 * TZ**5 + a_T[6]/6 * TZ**6 + a_T[7]/7 * TZ**7 + a_T[8]/8 * TZ**8 + a_T[10]
        CpdT_T2 = b_T[0] * ln(TZ) + b_T[1] * TZ + b_T[2]/2 * TZ**2 + b_T[3]/3 * TZ**3 + b_T[4]/4 * TZ**4 + b_T[5]/5 * TZ**5 + b_T[6]/6 * TZ**6 + b_T[7]/7 * TZ**7 + b_T[9]
        CpdT_T = CpdT_T1  + (FAR/(FAR+1))*CpdT_T2

        outputs['h'] = (H1  + (FAR/(FAR+1))*H2)*1000000
        outputs['Cp'] = (CP1  + (FAR/(FAR+1))*CP2 )*1000
        outputs['R'] = 287.05 - 0.00990 * FAR + 1E-07 * FAR**2
        outputs['Cv'] =  outputs['Cp'] - outputs['R']
        outputs['S'] =  CpdT_T*1000 - outputs['R'] * ln(P/101.325)
        outputs['gamma'] = outputs['Cp']/outputs['Cv']
        outputs['rho'] = P/(inputs['T']*outputs['R'])

        

class SetTotalTP(om.Group):

    def initialize(self):
        self.options.declare('spec', recordable=False)
        self.options.declare('composition')

    def setup(self):
        spec = self.options['spec']
        composition = self.options['composition']

        print(composition)

        if composition is None:
            composition = TAB_AIR_FUEL_COMPOSITION

        sorted_compo = sorted(composition.keys())

        print(composition[sorted_compo[0]])
        self.add_subsystem('tab', ThermoCalcs(spec = spec), 
                                          promotes_inputs=['P', 'T', (sorted_compo[0], composition[sorted_compo[0]])],
                                          promotes_outputs=['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])
        
        
    
        # for i, param in enumerate(sorted_compo):
        #     ThermoCalcs.add_input(param,composition[param])
        #     self.promotes('tab', inputs=[(param, 'composition')], src_indices=[i,])
        # self.set_input_defaults('composition', src_shape=len(composition))
        
        

        # required part of the SetTotalTP API for flow setup
        # use a sorted list of keys, so dictionary hash ordering doesn't bite us
        # loop over keys and create a vector of mass fractions
        self.composition = [composition[k] for k in sorted_compo]

        
