import numpy as np
import openmdao.api as om

from pycycle.cea.species_data import Thermo, wet_air
from pycycle.constants import WET_AIR_MIX

class SetWAR(om.ExplicitComponent):

    """
    Set initial product amounts based on specified WAR

    --------------
    inputs
    --------------
        WAR (water to air ratio by mass)

    --------------
    outputs
    --------------
        init_prod_amounts (ratios of initial compounds present in the flow by volume)


    """

    def initialize(self):
        self.options.declare('thermo_data', default=wet_air, 
                            desc='thermodynamic data set')
        self.options.declare('WAR', default=0.0001,
                            desc='water to air ratio by mass (specific humidity)')
        self.options.declare('elements', default=WET_AIR_MIX,
                              desc='set of elements present in the flow')

    def setup(self):

        thermo_data = self.options['thermo_data']
        WAR = self.options['WAR']
        elements = self.options['elements']

        self.original_init_reacts = thermo_data.init_prod_amounts
        thermo = Thermo(thermo_data, self.original_init_reacts) #call Thermo function with incorrect ratios to get the number of products in the output
        shape = len(thermo.products)

        #make sure provided elements and data contain the same compounds
        num_elements = len(set(elements.keys()))
        num_init_prods = len(set(self.original_init_reacts.keys()))
        num_intersection = len(set(elements.keys()).intersection(set(self.original_init_reacts.keys())))
        deviations = 2*num_intersection - num_elements - num_init_prods
        
        if 'H2O' not in elements:
            raise ValueError('The provided elements to FlightConditions do not contain H2O. In order to specify a nonzero WAR the elements must contain H2O.')

        if 'H2O' not in self.original_init_reacts:
            raise ValueError(f'H2O must be present in `{thermo_data}`.init_prod_amounts to have a nonzero WAR. The provided thermo_data has no H2O present.')

        if deviations != 0:
            raise ValueError('The compounds present in the provided elements must be the same as the compounds present in the init_prod_amounts of the provided thermo data, and are not')

        self.add_input('WAR', val=WAR, desc='water to air ratio by mass') #note: if WAR is set to 1 the equation becomes singular
        
        self.add_output('init_prod_amounts', shape=(shape,), val=thermo.init_prod_amounts,
                       desc="stoichiometric ratios by mass of the initial compounds present in the flow, scaled to desired WAR")

        self.declare_partials('init_prod_amounts', 'WAR')

    def compute(self, inputs, outputs):

        thermo_data = self.options['thermo_data']
        prod_data = thermo_data.products

        WAR = inputs['WAR']

        if WAR == 1:
            raise ValueError('Cannot specify WAR to have a value of 1. This is a physical impossibility and creates a singularity.')

        self.dry_wt = 0 #total weight of dry air
        self.init_react_amounts = [] #amounts of initial compounds scaled to desired WAR, not including zero value initial trace species

        for i, p in enumerate(self.original_init_reacts): #calculate total weight of dry air and include non-water values in init_react_amounts
            if p is not 'H2O':
                self.dry_wt += self.original_init_reacts[p] * prod_data[p]['wt']
                self.init_react_amounts.append(self.original_init_reacts[p])

            else:
                self.init_react_amounts.append(0)
                location = i

        self.water_wt = prod_data['H2O']['wt'] #molar weight of water

        n_water = WAR*self.dry_wt/((1 - WAR)*self.water_wt) #volumentric based ratio of water scaled to desired WAR

        self.init_react_amounts[location] = n_water #add in the amount of water scaled to the correct WAR

        init_reacts = self.original_init_reacts.copy() #dictionary containing the initial reactants with water scaled to desired WAR (used for passing to Thermo())
        init_reacts['H2O'] = n_water #update with correct water amount

        thermo = Thermo(thermo_data, init_reacts) #call Thermo function with correct ratios to get output values including zero value trace species
        self.products = thermo.products #get list of all products

        outputs['init_prod_amounts'] = thermo.init_prod_amounts

    def compute_partials(self, inputs, J):

        water_wt = self.water_wt
        dry_wt = self.dry_wt

        for i, p in enumerate(self.original_init_reacts):
            location = self.products.index(p)
            if p is 'H2O':
                J['init_prod_amounts', 'WAR'][location] = 1/water_wt

            else:
                J['init_prod_amounts', 'WAR'][location] = -self.init_react_amounts[i]/dry_wt


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = om.Group()

    WAR = prob.model.add_subsystem('WAR', SetWAR(thermo_data=wet_air), promotes=['*'])

    prob.setup(force_alloc_complex=True)

    prob.run_model()

    prob.check_partials(method='cs', compact_print=True)
    print(prob['init_prod_amounts'])