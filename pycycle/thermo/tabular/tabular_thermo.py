import numpy as np
import openmdao.api as om

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC


class SetTotalTP(om.Group):

    def initialize(self):
        self.options.declare('interp_method', default='slinear')
        self.options.declare('spec', recordable=False)
        self.options.declare('composition')

    def setup(self):
        interp_method = self.options['interp_method']
        spec = self.options['spec']
        composition = self.options['composition']

        if composition is None: 
            composition = TAB_AIR_FUEL_COMPOSITION

        sorted_compo = sorted(composition.keys())

        interp = om.MetaModelStructuredComp(method=interp_method, extrapolate=True)
        self.add_subsystem('tab', interp, promotes_inputs=['P', 'T'], 
                                          promotes_outputs=['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])

        for i, param in enumerate(sorted_compo): 
            interp.add_input(param, composition[param], training_data=spec[param])
            self.promotes('tab', inputs=[(param, 'composition')], src_indices=[i,])
        self.set_input_defaults('composition', src_shape=len(composition))

        interp.add_input('P', 101325.0, units='Pa', training_data=spec['P'])
        interp.add_input('T', 273.0, units='degK', training_data=spec['T'])

        interp.add_output('h', 1.0, units='J/kg', training_data=spec['h'])
        interp.add_output('S', 1.0, units='J/kg/degK', training_data=spec['S'])
        interp.add_output('gamma', 1.4, units=None, training_data=spec['gamma'])
        interp.add_output('Cp', 1.0, units='J/kg/degK', training_data=spec['Cp'])
        interp.add_output('Cv', 1.0, units='J/kg/degK', training_data=spec['Cv'])
        interp.add_output('rho', 1.0, units='kg/m**3', training_data=spec['rho'])
        interp.add_output('R', 287.0, units='J/kg/degK', training_data=spec['R'])

        # required part of the SetTotalTP API for flow setup
        # use a sorted list of keys, so dictionary hash ordering doesn't bite us 
        # loop over keys and create a vector of mass fractions
        self.composition = [composition[k] for k in sorted_compo]
