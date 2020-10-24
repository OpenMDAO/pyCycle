import warnings

def __getattr__(name):
    if name == 'AIR_FUEL_MIX':
        warnings.warn("Deprecation warning: `AIR_FUEL_MIX` will be replaced by `AIR_FUEL_ELEMENTS` in pyCycle 4.0", DeprecationWarning)
        return AIR_FUEL_ELEMENTS
    if name == 'AIR_MIX': 
        warnings.warn("Deprecation warning: `AIR_MIX` will be replaced by `AIR_ELEMENTS` in pyCycle 4.0", DeprecationWarning)
        return AIR_ELEMENTS
    if name == 'WET_AIR_MIX': 
        warnings.warn("Deprecation warning: `WET_AIR_MIX` will be replaced by `WET_AIR_ELEMENTS` in pyCycle 4.0", DeprecationWarning)
        return WET_AIR_ELEMENTS
    if name == 'CO2_CO_O2_MIX':
        warnings.warn("Deprecation warning: `CO2_CO_O2_MIX` will be replaced by `CO2_CO_O2_ELEMENTS` in pyCycle 4.0", DeprecationWarning)
        return CO2_CO_O2_ELEMENTS

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# these elemental ratios matter! 
AIR_FUEL_ELEMENTS = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05, 'H':1e-8}
# AIR_ELEMENTS = {'Ar':3.23319258e-04, 'C':1.10132241e-05, 'N':5.39157736e-02, 'O':1.44860147e-02}
AIR_ELEMENTS = {'N': 5.39157698e-02, 'O':1.44860137e-02, 'Ar': 3.23319235e-04, 'C': 1.10132233e-05}
WET_AIR_ELEMENTS = {'Ar':3.21320739e-04, 'C':1.09451485e-05, 'H':6.86216207e-04, 'N':5.35825063e-02, 'O':1.47395810e-02}
CO2_CO_O2_ELEMENTS = {'C':0.02272237, 'O':0.04544473}

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