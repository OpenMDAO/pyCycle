import warnings


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
AIR_FUEL_ELEMENTS = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05, 'H':1e-8}
# AIR_ELEMENTS = {'Ar':3.23319258e-04, 'C':1.10132241e-05, 'N':5.39157736e-02, 'O':1.44860147e-02}
AIR_ELEMENTS = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05}
WET_AIR_ELEMENTS = {'Ar':3.21320739e-04, 'C':1.09451485e-05, 'H':6.86216207e-04, 'N':5.35825063e-02, 'O':1.47395810e-02}
CO2_CO_O2_ELEMENTS = {'C':0.02272237, 'O':0.04544473}


AIR_FUEL_MIX = DeprecatedDict('AIR_FUEL_MIX', 'AIR_FUEL_ELEMENTS', AIR_FUEL_ELEMENTS)
AIR_MIX = DeprecatedDict('AIR_MIX', 'AIR_ELEMENTS', AIR_ELEMENTS)
WET_AIR_MIX = DeprecatedDict('WET_AIR_MIX', 'WET_AIR_ELEMENTS', WET_AIR_ELEMENTS)
CO2_CO_O2_MIX = DeprecatedDict('CO2_CO_O2_MIX', 'CO2_CO_O2_ELEMENTS', CO2_CO_O2_ELEMENTS)


OXYGEN = {'O': 1}
OXYGEN_METHANE_MIX = {'O': 1, 'CH4': 1} # can't use elemental 'C' because its not a valid species
OXYGEN_HYDROGEN_MIX = {'O': 1, 'H': 1}

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