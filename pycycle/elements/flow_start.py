from openmdao.api import Group, ExplicitComponent

from pycycle.cea import species_data
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle.constants import AIR_MIX, WET_AIR_MIX

class SetWAR(ExplicitComponent):

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
        self.options.declare('thermo_data', default=species_data.wet_air, 
                            desc='thermodynamic data set')
        self.options.declare('elements', default=WET_AIR_MIX,
                              desc='set of elements present in the flow')

    def setup(self):

        thermo_data = self.options['thermo_data']
        elements = self.options['elements']

        self.original_init_reacts = thermo_data.init_prod_amounts
        thermo = species_data.Thermo(thermo_data, elements) #call Thermo function with incorrect ratios to get the number of products in the output
        shape = len(thermo.products)

        self.add_input('WAR', val=0.0, desc='water to air ratio by mass') #note: if WAR is set to 1 the equation becomes singular
        
        self.add_output('init_prod_amounts', shape=(shape,), val=thermo.init_prod_amounts,
                       desc="stoichiometric ratios by mass of the initial compounds present in the flow, scaled to desired WAR")

        self.declare_partials('init_prod_amounts', 'WAR')

    def compute(self, inputs, outputs):

        WAR = inputs['WAR']
        thermo_data = self.options['thermo_data']

        prod_data = thermo_data.products

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

        init_reacts = self.original_init_reacts.copy() #dictionary containing the initial reactants with water scaled to desired WAR (used for passing to species_data.Thermo())
        init_reacts['H2O'] = n_water #update with correct water amount

        thermo = species_data.Thermo(thermo_data, init_reacts) #call Thermo function with correct ratios to get output values including zero value trace species
        self.products = thermo.products #get list of all products

        outputs['init_prod_amounts'] = thermo.init_prod_amounts

    def compute_partials(self, inputs, J):

        WAR = inputs['WAR']

        water_wt = self.water_wt
        dry_wt = self.dry_wt

        for i, p in enumerate(self.original_init_reacts):
            location = self.products.index(p)
            if p is 'H2O':
                J['init_prod_amounts', 'WAR'][location] = 1/water_wt

            else:
                J['init_prod_amounts', 'WAR'][location] = -self.init_react_amounts[i]/dry_wt

class FlowStart(Group):

    def initialize(self):

        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_MIX,
                              desc='set of elements present in the flow')

        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('use_WAR', default=False, values=[True, False], 
                              desc='If True, includes WAR calculation')

    def setup(self):
        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        use_WAR = self.options['use_WAR']

        if use_WAR == True:
            if 'H2O' not in elements:
                raise ValueError('The provided elements to FlightConditions do not contain H2O. In order to specify a nonzero WAR the elements must contain H2O.')

            elif 'H2O' not in thermo_data.init_prod_amounts:
                raise ValueError(f'H2O must be present in `{thermo_data}`.init_prod_amounts to have a nonzero WAR. The provided thermo_data has no H2O present.')

        elif use_WAR == False:
            if 'H2O' in elements.keys():

                raise ValueError('In order to provide elements containing H2O, a nonzero water to air ratio (WAR) must be specified. Please set the option use_WAR to True.')

        thermo = species_data.Thermo(thermo_data, init_reacts=elements)
        self.air_prods = thermo.products
        self.num_prod = len(self.air_prods)

        # inputs
        if use_WAR == True:
            set_WAR = SetWAR(thermo_data=thermo_data, elements=elements)
            self.add_subsystem('WAR', set_WAR, promotes_inputs=('WAR',), promotes_outputs=('init_prod_amounts',))
        
        set_TP = SetTotal(mode="T", fl_name="Fl_O:tot",
                          thermo_data=thermo_data,
                          init_reacts=elements)

        params = ('T','P', 'init_prod_amounts')

        self.add_subsystem('totals', set_TP, promotes_inputs=params,
                           promotes_outputs=('Fl_O:tot:*',))


        # if self.options['statics']:
        set_stat_MN = SetStatic(mode="MN", thermo_data=thermo_data,
                                init_reacts=elements, fl_name="Fl_O:stat")
        set_stat_MN.set_input_defaults('W', val=1.0, units='kg/s')

        self.add_subsystem('exit_static', set_stat_MN, promotes_inputs=('MN', 'W', 'init_prod_amounts'),
                           promotes_outputs=('Fl_O:stat:*', ))

        self.connect('totals.h','exit_static.ht')
        self.connect('totals.S','exit_static.S')
        self.connect('Fl_O:tot:P','exit_static.guess:Pt')
        self.connect('totals.gamma', 'exit_static.guess:gamt')


if __name__ == "__main__": 
    from collections import OrderedDict

    from openmdao.api import Problem, IndepVarComp

    print('\n-----\nFlowStart\n-----\n')

    p = Problem()
    p.model = FlowStart(elements=WET_AIR_MIX, use_WAR=True, thermo_data=species_data.wet_air)
    p.model.add_subsystem('WAR_start', IndepVarComp('WAR', .1), promotes=['*'])
    p.model.add_subsystem('temp', IndepVarComp('T', 4000., units="degR"), promotes=["*"])
    p.model.add_subsystem('pressure', IndepVarComp('P', 1.0342, units="bar"), promotes=["*"])
    p.model.add_subsystem('W', IndepVarComp('W', 100.0), promotes=['*'])

    p.setup()

    def find_order(group):
        subs = OrderedDict()

        for s in group.subsystems():
            if isinstance(s, Group):
                subs[s.name] = find_order(s)
            else:
                subs[s.name] = {}
        return subs

    # order = find_order(p.root)
    # import json
    # print(json.dumps(order, indent=4))
    # exit()

    # p['exit_static.mach_calc.Ps_guess'] = .97
    import time
    st = time.time()
    p.run_model()
    print("time", time.time() - st)

    print("Temp", p['T'], p['Fl_O:tot:T'])
    print("Pressure", p['P'], p['Fl_O:tot:P'])
    print("h", p['totals.h'], p['Fl_O:tot:h'])
    print("S", p['totals.S'])
    print("actual Ps", p['exit_static.Ps'], p['Fl_O:stat:P'])
    print("Mach", p['Fl_O:stat:MN'])
    print("n tot", p['Fl_O:tot:n'])
    print("n stat", p['Fl_O:stat:n'])


    print('\n-----\nWAR\n-----\n')

    prob = Problem()
    prob.model = Group()

    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('WAR', .0001),

    prob.model.add_subsystem('WAR', SetWAR(thermo_data=species_data.wet_air, elements=WET_AIR_MIX), promotes=['*'])

    prob.setup(force_alloc_complex=True)

    prob.run_model()

    # prob.check_partials(method='cs', compact_print=True)
    print('init_prod_amounts', prob['init_prod_amounts'])
