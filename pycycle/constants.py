import warnings
import os
import os.path
import pickle

class DeprecatedDict(dict): 

    def __init__(self, old_name, new_name, *args, **kwargs): 
        self.old_name = old_name 
        self.new_name = new_name 
        self._has_warned = False
        super().__init__(*args, **kwargs)

    def __getitem__(self, key): 
        if not self._has_warned: 
            self._has_warned = True
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f"Deprecation warning: `{self.old_name}` will be replaced by `{self.new_name}` in pyCycle 4.0", DeprecationWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
        return super().__getitem__(key)

# these elemental ratios matter! 
CEA_AIR_FUEL_COMPOSITION = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05, 'H':1e-8}
CEA_AIR_COMPOSITION = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05}
CEA_WET_AIR_COMPOSITION = {'Ar':3.21320739e-04, 'C':1.09451485e-05, 'H':6.86216207e-04, 'N':5.35825063e-02, 'O':1.47395810e-02}
CEA_CO2_CO_O2_COMPOSITION = {'C':0.02272237, 'O':0.04544473}

TAB_AIR_FUEL_COMPOSITION = {'FAR': 0.0}
# A little fancy code to find the default thermo data in the python package, wherever its installed
pkg_path = os.path.dirname(os.path.realpath(__file__))
tab_spec_path = os.path.join(pkg_path, 'thermo', 'tabular', 'air_jetA.pkl')
with open(tab_spec_path, 'rb') as spec_data:
    AIR_JETA_TAB_SPEC = pickle.load(spec_data)


THERMO_DEFAULT_COMPOSITIONS = {
    'CEA': CEA_AIR_COMPOSITION, 
    'TABULAR': TAB_AIR_FUEL_COMPOSITION
}


# these elemental ratios matter! 
AIR_FUEL_ELEMENTS = DeprecatedDict('AIR_FUEL_ELEMENTS', 'CEA_AIR_FUEL_COMPOSITION', CEA_AIR_FUEL_COMPOSITION)
AIR_ELEMENTS = DeprecatedDict('AIR_ELEMENTS', 'CEA_AIR_COMPOSITION', CEA_AIR_COMPOSITION)
WET_AIR_ELEMENTS = DeprecatedDict('WET_AIR_ELEMENTS', 'CEA_WET_AIR_COMPOSITION', CEA_WET_AIR_COMPOSITION)
CO2_CO_O2_ELEMENTS = DeprecatedDict('CO2_CO_O2_ELEMENTS', 'CEA_CO2_CO_O2_COMPOSITION', CEA_CO2_CO_O2_COMPOSITION)


AIR_FUEL_MIX = DeprecatedDict('AIR_FUEL_MIX', 'CEA_AIR_FUEL_COMPOSITION', CEA_AIR_FUEL_COMPOSITION)
AIR_MIX = DeprecatedDict('AIR_MIX', 'CEA_AIR_COMPOSITION', CEA_AIR_COMPOSITION)
WET_AIR_MIX = DeprecatedDict('WET_AIR_MIX', 'CEA_WET_AIR_COMPOSITION', CEA_WET_AIR_COMPOSITION)
CO2_CO_O2_MIX = DeprecatedDict('CO2_CO_O2_MIX', 'CEA_CO2_CO_O2_COMPOSITION', CEA_CO2_CO_O2_COMPOSITION)


BTU_s2HP = 1.4148532
HP_per_RPM_to_FT_LBF = 5252.11

R_UNIVERSAL_SI = 8314.4598 # (m**3 * Pa)/(mol*degK)
R_UNIVERSAL_ENG = 1.9872035 # (Btu lbm)/(mol*degR)

g_c = 32.174

MIN_VALID_CONCENTRATION = 1e-10

T_STDeng = 518.67 #degR
P_STDeng = 14.695951 #psi

P_REF = 1.01325 # 1 atm
# P_REF = 1.0162 # Not sure why, but this seems to match the SP set to the TP better


ALLOWED_THERMOS = ('CEA', 'TABULAR')