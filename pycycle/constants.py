AIR_FUEL_MIX = {'O':1, 'H':1, 'CO2':1, 'N':1, 'Ar':1}
AIR_MIX = {'N2': 78.084, 'O2': 20.9476, 'Ar': .9365, 'CO2': .0319} # {'O':1, 'C':1, 'N':1, 'Ar':1}
WET_AIR_MIX = {'N2': 78.2, 'O2': 20.78, 'H2O':1.0, 'CO2':.01, 'Ar':0.01}

# Note: the actual amounts don't matter here. All that matters is that you have some amount of the correct atoms
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

