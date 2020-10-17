import numpy as np

from collections import OrderedDict

from pycycle.cea.thermo_data import janaf

big_range = np.array([200,1000,6000,20000])
small_range = np.array([200,1000,6000])


products = dict(janaf.products)
# remove these, because they cause numerical
products.pop('CH4')
products.pop('C2H4')

element_wts = {
  'C':12.0170, 'O': 15.99940, 'Ar': 39.948, 'H': 1.00794, 'N':14.00674
}

reactants = { # used to compute the correct amounts of each product for an initial condition
  'air': OrderedDict([('N2', 78.084), ('O2', 20.9476), ('Ar', .9365), ('CO2', .0319)]), #percentage by volume
  # 'air': {'N': 1.5616, 'O':.41959, 'Ar': .00936, 'C': .00032},
  #'JP-7': {'C': 1.00, 'H': 2.0044},
  'JP-7': OrderedDict([('C2H4', 1.00), ('H', 0.0044)]), 
  'Jet-A(g)': OrderedDict([('C2H4', 6), ('H', -1.)]), 
  'H2': OrderedDict([('H2', 1.00)]), 
  'Methane': OrderedDict([('CH4', 1.00)])
}



# init_prod_amounts = reactants['air'].copy() # initial value used to set the atomic fractions in the mixture
# default_elements = {'Ar':3.23319258e-04, 'C':1.10132241e-05, 'N':5.39157736e-02, 'O':1.44860147e-02}

# tot_amount = 0
# for r in products.keys(): # assume pure air by default
#     r_amount = reactants['air'].get(r,0) * products[r]['wt'] # initial amounts need to be given in mass ratios
#     tot_amount += r_amount
#     init_prod_amounts[r] = r_amount


# for r in products.keys():
#     if r == "O2" or r == "N2":
#       init_prod_amounts[r] = 1
#     else:
#       init_prod_amounts[r] = 0