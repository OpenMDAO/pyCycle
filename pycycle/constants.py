AIR_FUEL_MIX = {'O':1, 'H':1, 'CO2':1, 'N':1, 'Ar':1}
AIR_MIX = {'N2': 78.084, 'O2': 20.9476, 'Ar': .9365, 'CO2': .0319} 
WET_AIR_MIX = {'N2':78.084, 'O2':20.9476, 'Ar':.9365, 'CO2':.0319, 'H2O':1}
CO2_CO_O2_MIX = {'CO':0, 'CO2':1, 'O2':0}

AIR_ELEMENTS = {'Ar':3.23319258e-04, 'C':1.10132241e-05, 'N':5.39157736e-02, 'O':1.44860147e-02}
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

MIN_VALID_CONCENTRATION = 1e-12

T_STDeng = 518.67 #degR
P_STDeng = 14.695951 #psi

P_REF = 1.01325 # 1 atm
# P_REF = 1.0162 # Not sure why, but this seems to match the SP set to the TP better