import numpy as np

from collections import OrderedDict

from pycycle.thermo.cea.thermo_data import janaf

big_range = janaf.big_range
small_range = janaf.small_range


products = dict(janaf.products)
# remove these, because they cause numerical
products.pop('CH4')
products.pop('C2H4')

element_wts = janaf.element_wts

reactants = janaf.reactants