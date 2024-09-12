### Import Libraries ###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

### Specify directory and file paths of .txt files ###
Unalloyed_W = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\Pure_W_A\Unalloyed_W_All_Frequencies.csv"
W99_Mo1 = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\99_1_A\99W-1Mo_All_Frequencies.csv"
W98_Mo2 = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\98_2_A\98W-2Mo_All_Frequencies.csv"
W94_Mo6 = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\94_6_A\94W-6Mo_All_Frequencies.csv"
W75_Mo25 = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\75_25_A\75W-25Mo_All_Frequencies.csv"
W50_Mo50 = r"C:\Users\j48032ja\OneDrive - The University of Manchester\Fusion CDT PhD\Project\Physical\CRAN-YR9-012\Results\50_50_A\50W-50Mo_All_Frequencies.csv"

### Read files .csv files ###
Unalloyed_W_df = pd.read_csv(Unalloyed_W, encoding='unicode_escape', skiprows = 13, header = 0) # First 13 rows are skipped as they contain metadata, and the 14th row is used as the header
W99_Mo1_df = pd.read_csv(W99_Mo1, encoding='unicode_escape', skiprows = 13, header = 0)
W98_Mo2_df = pd.read_csv(W98_Mo2, encoding='unicode_escape', skiprows = 13, header = 0)
W94_Mo6_df = pd.read_csv(W94_Mo6, encoding='unicode_escape', skiprows = 13, header = 0)
W75_Mo25_df = pd.read_csv(W75_Mo25, encoding='unicode_escape', skiprows = 13, header = 0)
W50_Mo50_df = pd.read_csv(W50_Mo50, encoding='unicode_escape', skiprows = 13, header = 0)

### Temperature Lists ###
Temp_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 24].to_numpy()
Temp_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 24].to_numpy()
Temp_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 24].to_numpy()
Temp_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 24].to_numpy()
Temp_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 24].to_numpy()
Temp_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 24].to_numpy()
Temp_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 24].to_numpy()
Temp_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 24].to_numpy()
Temp_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 24].to_numpy()
Temp_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 24].to_numpy()
Temp_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 24].to_numpy()
Temp_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 24].to_numpy()

### Reference temperature list from 0 to 800 ###
Temp_Reference = np.linspace(0, 800, 80) # 0 to 800 in 80 steps

### Flexural Frequency Lists ###
Flex_Hz_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 27].to_numpy()
Flex_Hz_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 27].to_numpy()
Flex_Hz_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 8].to_numpy()
Flex_Hz_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 8].to_numpy()
Flex_Hz_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 8].to_numpy()
Flex_Hz_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 8].to_numpy()
Flex_Hz_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 8].to_numpy()
Flex_Hz_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 8].to_numpy()
Flex_Hz_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 8].to_numpy()
Flex_Hz_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 8].to_numpy()
Flex_Hz_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 8].to_numpy()
Flex_Hz_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 8].to_numpy()

### E-Modulus Lists ###
E_mod_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 3].to_numpy()
E_mod_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 3].to_numpy()
E_mod_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 3].to_numpy()
E_mod_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 3].to_numpy()
E_mod_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 3].to_numpy()
E_mod_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 3].to_numpy()
E_mod_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 3].to_numpy()
E_mod_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 3].to_numpy()
E_mod_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 3].to_numpy()
E_mod_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 3].to_numpy()
E_mod_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 3].to_numpy()
E_mod_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 3].to_numpy()

### E-Mod Error Lists ###
E_mod_Err_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 4].to_numpy()
E_mod_Err_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 4].to_numpy()
E_mod_Err_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 4].to_numpy()
E_mod_Err_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 4].to_numpy()
E_mod_Err_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 4].to_numpy()
E_mod_Err_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 4].to_numpy()
E_mod_Err_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 4].to_numpy()
E_mod_Err_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 4].to_numpy()
E_mod_Err_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 4].to_numpy()
E_mod_Err_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 4].to_numpy()
E_mod_Err_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 4].to_numpy()
E_mod_Err_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 4].to_numpy()

### Input Poisson Lists ###
IP_Poisson_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 7].to_numpy()
IP_Poisson_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 7].to_numpy()
IP_Poisson_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 7].to_numpy()
IP_Poisson_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 7].to_numpy()
IP_Poisson_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 7].to_numpy()
IP_Poisson_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 7].to_numpy()
IP_Poisson_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 7].to_numpy()
IP_Poisson_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 7].to_numpy()
IP_Poisson_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 7].to_numpy()
IP_Poisson_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 7].to_numpy()
IP_Poisson_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 7].to_numpy()
IP_Poisson_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 7].to_numpy()

### Damping Lists ###
Damping_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 10].to_numpy()
Damping_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 10].to_numpy()
Damping_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 10].to_numpy()
Damping_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 10].to_numpy()
Damping_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 10].to_numpy()
Damping_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 10].to_numpy()
Damping_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 10].to_numpy()
Damping_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 10].to_numpy()
Damping_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 10].to_numpy()
Damping_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 10].to_numpy()
Damping_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 10].to_numpy()
Damping_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 10].to_numpy()

### Length Lists ###
Length_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 16].to_numpy()
Length_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 16].to_numpy()
Length_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 16].to_numpy()
Length_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 16].to_numpy()
Length_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 16].to_numpy()
Length_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 16].to_numpy()
Length_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 16].to_numpy()
Length_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 16].to_numpy()
Length_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 16].to_numpy()
Length_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 16].to_numpy()
Length_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 16].to_numpy()
Length_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 16].to_numpy()

### Width Lists ###
Width_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 18].to_numpy()
Width_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 18].to_numpy()
Width_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 18].to_numpy()
Width_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 18].to_numpy()
Width_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 18].to_numpy()
Width_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 18].to_numpy()
Width_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 18].to_numpy()
Width_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 18].to_numpy()
Width_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 18].to_numpy()
Width_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 18].to_numpy()
Width_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 18].to_numpy()
Width_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 18].to_numpy()

### Thickness Lists ###
Thks_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 20].to_numpy()
Thks_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 20].to_numpy()
Thks_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 20].to_numpy()
Thks_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 20].to_numpy()
Thks_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 20].to_numpy()
Thks_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 20].to_numpy()
Thks_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 20].to_numpy()
Thks_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 20].to_numpy()
Thks_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 20].to_numpy()
Thks_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 20].to_numpy()
Thks_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 20].to_numpy()
Thks_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 20].to_numpy()

### Mass Lists ###
Mass_Unalloyed_W_Heating = Unalloyed_W_df.iloc[0:80, 22].to_numpy()
Mass_Unalloyed_W_Cooling = Unalloyed_W_df.iloc[81:169, 22].to_numpy()
Mass_99W_1Mo_Heating = W99_Mo1_df.iloc[0:129, 22].to_numpy()
Mass_99W_1Mo_Cooling = W99_Mo1_df.iloc[130:274, 22].to_numpy()
Mass_98W_2Mo_Heating = W98_Mo2_df.iloc[0:216, 22].to_numpy()
Mass_98W_2Mo_Cooling = W98_Mo2_df.iloc[217:361, 22].to_numpy()
Mass_94W_6Mo_Heating = W94_Mo6_df.iloc[0:212, 22].to_numpy()
Mass_94W_6Mo_Cooling = W94_Mo6_df.iloc[213:477, 22].to_numpy()
Mass_75W_25Mo_Heating = W75_Mo25_df.iloc[0:92, 22].to_numpy()
Mass_75W_25Mo_Cooling = W75_Mo25_df.iloc[93:241, 22].to_numpy()
Mass_50W_50Mo_Heating = W50_Mo50_df.iloc[0:121, 22].to_numpy()
Mass_50W_50Mo_Cooling = W50_Mo50_df.iloc[122:270, 22].to_numpy()

### Calculations ###

# Material Constants
Avogadro = 6.022e23 # atoms / mol
Atomic_Weight_W = 183.84 # g/mol
Atomic_Weight_Mo = 95.95 # g/mol
Molar_Volume_W = 9.55  # cm³/mol
Molar_Volume_Mo = 9.334  # cm³/mol
Atomic_Dens_W = 6.338e22 # atoms / cm^3
Atomic_Dens_Mo = 6.022e22 # atoms / cm^3

# Constants and Parameters
T_ref = 20  # Reference temperature (room temperature in Celsius)

# Alloy dimensions at T_ref
dimensions_ref = {
    'W': {'L': 98.003, 'W': 8.074, 'T': 4.976},
    'W99': {'L': 97.997, 'W': 8.036, 'T': 4.961},
    'W98': {'L': 97.983, 'W': 7.998, 'T': 4.972},
    'W94': {'L': 97.980, 'W': 8.056, 'T': 4.976},
    'W75': {'L': 97.983, 'W': 8.096, 'T': 4.969},
    'W50': {'L': 97.977, 'W': 8.064, 'T': 4.980}
}

# Measure Alloy masses
W_m = 75.469
W99_m = 74.271
W98_m = 73.111
W94_m = 71.290
W75_m = 61.645
W50_m  = 52.333

# Thermal expansion coefficients as functions of temperature
def alpha_W(T):
    return 4.35e-6 + 0.3e-9 * T + 0.51e-12 * T**2

def alpha_Mo(T):
    return 0.075e-8 * T + 5.1e-6

# Atomic fractions
fractions = {
    'W': (1.0, 0.0),  # (Fraction of W, Fraction of Mo)
    'W99': (0.981, 0.019),
    'W98': (0.962, 0.038),
    'W94': (0.891, 0.109),
    'W75': (0.61, 0.39),
    'W50': (0.343, 0.657)
}

# Atomic Percentage of W
Atomic_Percentages_W = {
    'W': 100,
    'W99': 98.1,
    'W98': 96.2,
    'W94': 89.1,
    'W75': 61,
    'W50': 34.3
}

# Atomic Percentage of Mo
Atomic_Percentages_Mo = {
    'W': 0,
    'W99': 1.9,
    'W98': 3.8,
    'W94': 10.9,
    'W75': 39,
    'W50': 65.7
}

# Weight Percentage of Mo
Weight_Percentages_Mo = {
    'W': 0,
    'W99': 1.0,
    'W98': 2.0,
    'W94': 6.0,
    'W75': 25.0,
    'W50': 50.0
}

# Function to calculate the effective thermal expansion coefficient
def alpha_eff(T, fraction_W, fraction_Mo):
    return fraction_W * alpha_W(T) + fraction_Mo * alpha_Mo(T)

# Function to calculate new dimension based on thermal expansion
def calculate_new_dimension(D0, T, fraction_W, fraction_Mo):
    alpha = alpha_eff(T, fraction_W, fraction_Mo)
    D_new = D0 * (1 + alpha * (T - T_ref))
    return D_new

# Define a dictionary to hold the temperature data
temperature_lists = {
    'W': {'Heating': Temp_Unalloyed_W_Heating, 'Cooling': Temp_Unalloyed_W_Cooling},
    'W99': {'Heating': Temp_99W_1Mo_Heating, 'Cooling': Temp_99W_1Mo_Cooling},
    'W98': {'Heating': Temp_98W_2Mo_Heating, 'Cooling': Temp_98W_2Mo_Cooling},
    'W94': {'Heating': Temp_94W_6Mo_Heating, 'Cooling': Temp_94W_6Mo_Cooling},
    'W75': {'Heating': Temp_75W_25Mo_Heating, 'Cooling': Temp_75W_25Mo_Cooling},
    'W50': {'Heating': Temp_50W_50Mo_Heating, 'Cooling': Temp_50W_50Mo_Cooling}
}

# Calculate updated dimensions for each alloy and temperature list
updated_dimensions = {}

for alloy, temps in temperature_lists.items():
    fraction_W, fraction_Mo = fractions[alloy]
    updated_dimensions[alloy] = {'Heating': {}, 'Cooling': {}}
    
    for condition, temp_list in temps.items():
        updated_dimensions[alloy][condition]['L'] = [calculate_new_dimension(dimensions_ref[alloy]['L'], T, fraction_W, fraction_Mo) for T in temp_list]
        updated_dimensions[alloy][condition]['W'] = [calculate_new_dimension(dimensions_ref[alloy]['W'], T, fraction_W, fraction_Mo) for T in temp_list]
        updated_dimensions[alloy][condition]['T'] = [calculate_new_dimension(dimensions_ref[alloy]['T'], T, fraction_W, fraction_Mo) for T in temp_list]

# Temperature dependent Poisson's ratio
Unalloyed_W_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_Unalloyed_W_Heating]
Unalloyed_W_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_Unalloyed_W_Cooling]
W99_1Mo_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_99W_1Mo_Heating]
W99_1Mo_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_99W_1Mo_Cooling]
W98_2Mo_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_98W_2Mo_Heating]
W98_2Mo_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_98W_2Mo_Cooling]
W94_6Mo_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_94W_6Mo_Heating]
W94_6Mo_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_94W_6Mo_Cooling]
W75_25Mo_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_75W_25Mo_Heating]
W75_25Mo_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_75W_25Mo_Cooling]
W50_50Mo_Poissons_Temp_Heating = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_50W_50Mo_Heating]
W50_50Mo_Poissons_Temp_Cooling = [0.28247+6.1902*10**(-6)*T+3.162*10**(-9)*T**2 for T in Temp_50W_50Mo_Cooling]

# Calculate correction factor (T) for each material
T_W_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(Unalloyed_W_Poissons_Temp_Heating) + 0.8109 * np.array(Unalloyed_W_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W']['Heating']['T']) / np.array(updated_dimensions['W']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W']['Heating']['T']) / np.array(updated_dimensions['W']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(Unalloyed_W_Poissons_Temp_Heating) + 2.173 * np.array(Unalloyed_W_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W']['Heating']['T']) / np.array(updated_dimensions['W']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(Unalloyed_W_Poissons_Temp_Heating) + 1.539 * np.array(Unalloyed_W_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W']['Heating']['T']) / np.array(updated_dimensions['W']['Heating']['L'])) ** 2))
T_W_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(Unalloyed_W_Poissons_Temp_Cooling) + 0.8109 * np.array(Unalloyed_W_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W']['Cooling']['T']) / np.array(updated_dimensions['W']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W']['Cooling']['T']) / np.array(updated_dimensions['W']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(Unalloyed_W_Poissons_Temp_Cooling) + 2.173 * np.array(Unalloyed_W_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W']['Cooling']['T']) / np.array(updated_dimensions['W']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(Unalloyed_W_Poissons_Temp_Cooling) + 1.539 * np.array(Unalloyed_W_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W']['Cooling']['T']) / np.array(updated_dimensions['W']['Cooling']['L'])) ** 2))
T_W99_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(W99_1Mo_Poissons_Temp_Heating) + 0.8109 * np.array(W99_1Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W99']['Heating']['T']) / np.array(updated_dimensions['W99']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W99']['Heating']['T']) / np.array(updated_dimensions['W99']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W99_1Mo_Poissons_Temp_Heating) + 2.173 * np.array(W99_1Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W99']['Heating']['T']) / np.array(updated_dimensions['W99']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W99_1Mo_Poissons_Temp_Heating) + 1.539 * np.array(W99_1Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W99']['Heating']['T']) / np.array(updated_dimensions['W99']['Heating']['L'])) ** 2))
T_W99_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(W99_1Mo_Poissons_Temp_Cooling) + 0.8109 * np.array(W99_1Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W99']['Cooling']['T']) / np.array(updated_dimensions['W99']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W99']['Cooling']['T']) / np.array(updated_dimensions['W99']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W99_1Mo_Poissons_Temp_Cooling) + 2.173 * np.array(W99_1Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W99']['Cooling']['T']) / np.array(updated_dimensions['W99']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W99_1Mo_Poissons_Temp_Cooling) + 1.539 * np.array(W99_1Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W99']['Cooling']['T']) / np.array(updated_dimensions['W99']['Cooling']['L'])) ** 2))
T_W98_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(W98_2Mo_Poissons_Temp_Heating) + 0.8109 * np.array(W98_2Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W98']['Heating']['T']) / np.array(updated_dimensions['W98']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W98']['Heating']['T']) / np.array(updated_dimensions['W98']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W98_2Mo_Poissons_Temp_Heating) + 2.173 * np.array(W98_2Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W98']['Heating']['T']) / np.array(updated_dimensions['W98']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W98_2Mo_Poissons_Temp_Heating) + 1.539 * np.array(W98_2Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W98']['Heating']['T']) / np.array(updated_dimensions['W98']['Heating']['L'])) ** 2))
T_W98_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(W98_2Mo_Poissons_Temp_Cooling) + 0.8109 * np.array(W98_2Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W98']['Cooling']['T']) / np.array(updated_dimensions['W98']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W98']['Cooling']['T']) / np.array(updated_dimensions['W98']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W98_2Mo_Poissons_Temp_Cooling) + 2.173 * np.array(W98_2Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W98']['Cooling']['T']) / np.array(updated_dimensions['W98']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W98_2Mo_Poissons_Temp_Cooling) + 1.539 * np.array(W98_2Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W98']['Cooling']['T']) / np.array(updated_dimensions['W98']['Cooling']['L'])) ** 2))
T_W94_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(W94_6Mo_Poissons_Temp_Heating) + 0.8109 * np.array(W94_6Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W94']['Heating']['T']) / np.array(updated_dimensions['W94']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W94']['Heating']['T']) / np.array(updated_dimensions['W94']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W94_6Mo_Poissons_Temp_Heating) + 2.173 * np.array(W94_6Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W94']['Heating']['T']) / np.array(updated_dimensions['W94']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W94_6Mo_Poissons_Temp_Heating) + 1.539 * np.array(W94_6Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W94']['Heating']['T']) / np.array(updated_dimensions['W94']['Heating']['L'])) ** 2))
T_W94_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(W94_6Mo_Poissons_Temp_Cooling) + 0.8109 * np.array(W94_6Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W94']['Cooling']['T']) / np.array(updated_dimensions['W94']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W94']['Cooling']['T']) / np.array(updated_dimensions['W94']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W94_6Mo_Poissons_Temp_Cooling) + 2.173 * np.array(W94_6Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W94']['Cooling']['T']) / np.array(updated_dimensions['W94']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W94_6Mo_Poissons_Temp_Cooling) + 1.539 * np.array(W94_6Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W94']['Cooling']['T']) / np.array(updated_dimensions['W94']['Cooling']['L'])) ** 2))
T_W75_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(W75_25Mo_Poissons_Temp_Heating) + 0.8109 * np.array(W75_25Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W75']['Heating']['T']) / np.array(updated_dimensions['W75']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W75']['Heating']['T']) / np.array(updated_dimensions['W75']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W75_25Mo_Poissons_Temp_Heating) + 2.173 * np.array(W75_25Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W75']['Heating']['T']) / np.array(updated_dimensions['W75']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W75_25Mo_Poissons_Temp_Heating) + 1.539 * np.array(W75_25Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W75']['Heating']['T']) / np.array(updated_dimensions['W75']['Heating']['L'])) ** 2))
T_W75_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(W75_25Mo_Poissons_Temp_Cooling) + 0.8109 * np.array(W75_25Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W75']['Cooling']['T']) / np.array(updated_dimensions['W75']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W75']['Cooling']['T']) / np.array(updated_dimensions['W75']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W75_25Mo_Poissons_Temp_Cooling) + 2.173 * np.array(W75_25Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W75']['Cooling']['T']) / np.array(updated_dimensions['W75']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W75_25Mo_Poissons_Temp_Cooling) + 1.539 * np.array(W75_25Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W75']['Cooling']['T']) / np.array(updated_dimensions['W75']['Cooling']['L'])) ** 2))
T_W50_Heating = 1 + 6.585 * (1 + 0.0752 * np.array(W50_50Mo_Poissons_Temp_Heating) + 0.8109 * np.array(W50_50Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W50']['Heating']['T']) / np.array(updated_dimensions['W50']['Heating']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W50']['Heating']['T']) / np.array(updated_dimensions['W50']['Heating']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W50_50Mo_Poissons_Temp_Heating) + 2.173 * np.array(W50_50Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W50']['Heating']['T']) / np.array(updated_dimensions['W50']['Heating']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W50_50Mo_Poissons_Temp_Heating) + 1.539 * np.array(W50_50Mo_Poissons_Temp_Heating) ** 2) * (np.array(updated_dimensions['W50']['Heating']['T']) / np.array(updated_dimensions['W50']['Heating']['L'])) ** 2))
T_W50_Cooling = 1 + 6.585 * (1 + 0.0752 * np.array(W50_50Mo_Poissons_Temp_Cooling) + 0.8109 * np.array(W50_50Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W50']['Cooling']['T']) / np.array(updated_dimensions['W50']['Cooling']['L'])) ** 2 - 0.868 * (np.array(updated_dimensions['W50']['Cooling']['T']) / np.array(updated_dimensions['W50']['Cooling']['L'])) ** 4 - ((8.340 * (1 + 0.2023 * np.array(W50_50Mo_Poissons_Temp_Cooling) + 2.173 * np.array(W50_50Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W50']['Cooling']['T']) / np.array(updated_dimensions['W50']['Cooling']['L'])) ** 4) / (1 + 6.338 * (1 + 0.1408 * np.array(W50_50Mo_Poissons_Temp_Cooling) + 1.539 * np.array(W50_50Mo_Poissons_Temp_Cooling) ** 2) * (np.array(updated_dimensions['W50']['Cooling']['T']) / np.array(updated_dimensions['W50']['Cooling']['L'])) ** 2))

# Calculate the E-Modulus for each material
Calc_E_mod_Unalloyed_W_Heating = 0.9465 * ((W_m * np.array(Flex_Hz_Unalloyed_W_Heating) ** 2) / np.array(updated_dimensions['W']['Heating']['W'])) * ((np.array(updated_dimensions['W']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W']['Heating']['T']) ** 3)) * T_W_Heating * 10**(-9)
Calc_E_mod_Unalloyed_W_Cooling = 0.9465 * ((W_m * np.array(Flex_Hz_Unalloyed_W_Cooling) ** 2) / np.array(updated_dimensions['W']['Cooling']['W'])) * ((np.array(updated_dimensions['W']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W']['Cooling']['T']) ** 3)) * T_W_Cooling * 10**(-9)
Calc_E_mod_W99_1Mo_Heating = 0.9465 * ((W99_m * np.array(Flex_Hz_99W_1Mo_Heating) ** 2) / np.array(updated_dimensions['W99']['Heating']['W'])) * ((np.array(updated_dimensions['W99']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W99']['Heating']['T']) ** 3)) * T_W99_Heating * 10**(-9)
Calc_E_mod_W99_1Mo_Cooling = 0.9465 * ((W99_m * np.array(Flex_Hz_99W_1Mo_Cooling) ** 2) / np.array(updated_dimensions['W99']['Cooling']['W'])) * ((np.array(updated_dimensions['W99']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W99']['Cooling']['T']) ** 3)) * T_W99_Cooling * 10**(-9)
Calc_E_mod_W98_2Mo_Heating = 0.9465 * ((W98_m * np.array(Flex_Hz_98W_2Mo_Heating) ** 2) / np.array(updated_dimensions['W98']['Heating']['W'])) * ((np.array(updated_dimensions['W98']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W98']['Heating']['T']) ** 3)) * T_W98_Heating * 10**(-9)
Calc_E_mod_W98_2Mo_Cooling = 0.9465 * ((W98_m * np.array(Flex_Hz_98W_2Mo_Cooling) ** 2) / np.array(updated_dimensions['W98']['Cooling']['W'])) * ((np.array(updated_dimensions['W98']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W98']['Cooling']['T']) ** 3)) * T_W98_Cooling * 10**(-9)
Calc_E_mod_W94_6Mo_Heating = 0.9465 * ((W94_m * np.array(Flex_Hz_94W_6Mo_Heating) ** 2) / np.array(updated_dimensions['W94']['Heating']['W'])) * ((np.array(updated_dimensions['W94']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W94']['Heating']['T']) ** 3)) * T_W94_Heating * 10**(-9)
Calc_E_mod_W94_6Mo_Cooling = 0.9465 * ((W94_m * np.array(Flex_Hz_94W_6Mo_Cooling) ** 2) / np.array(updated_dimensions['W94']['Cooling']['W'])) * ((np.array(updated_dimensions['W94']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W94']['Cooling']['T']) ** 3)) * T_W94_Cooling * 10**(-9)
Calc_E_mod_W75_25Mo_Heating = 0.9465 * ((W75_m * np.array(Flex_Hz_75W_25Mo_Heating) ** 2) / np.array(updated_dimensions['W75']['Heating']['W'])) * ((np.array(updated_dimensions['W75']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W75']['Heating']['T']) ** 3)) * T_W75_Heating * 10**(-9)
Calc_E_mod_W75_25Mo_Cooling = 0.9465 * ((W75_m * np.array(Flex_Hz_75W_25Mo_Cooling) ** 2) / np.array(updated_dimensions['W75']['Cooling']['W'])) * ((np.array(updated_dimensions['W75']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W75']['Cooling']['T']) ** 3)) * T_W75_Cooling * 10**(-9)
Calc_E_mod_W50_50Mo_Heating = 0.9465 * ((W50_m * np.array(Flex_Hz_50W_50Mo_Heating) ** 2) / np.array(updated_dimensions['W50']['Heating']['W'])) * ((np.array(updated_dimensions['W50']['Heating']['L']) ** 3) / (np.array(updated_dimensions['W50']['Heating']['T']) ** 3)) * T_W50_Heating * 10**(-9)
Calc_E_mod_W50_50Mo_Cooling = 0.9465 * ((W50_m * np.array(Flex_Hz_50W_50Mo_Cooling) ** 2) / np.array(updated_dimensions['W50']['Cooling']['W'])) * ((np.array(updated_dimensions['W50']['Cooling']['L']) ** 3) / (np.array(updated_dimensions['W50']['Cooling']['T']) ** 3)) * T_W50_Cooling * 10**(-9)

# E-Modulus as per Robert Lowrie and A. M. Gonas https://doi.org/10.1063/1.1714447
E_mod_LowGon = (4.0761*10**12 - 3.5521*10**8*np.array(Temp_Reference) - 5.871*10**4*np.array(Temp_Reference)**2) * 10**(-10) # in GPa, * 10**(-10) as the original formula is in dynes/cm^2

# E-Modulus as per G.P. Škoro et al. https://doi.org/10.1016/j.jnucmat.2010.12.222
E_mod_Skoro = 391.448 - 1.316*10**(-2)*np.array(Temp_Reference) -1.4838*10**(-5)*np.array(Temp_Reference)**2


# ### Plotting ###

# def filter_valid_data(x, y):
#     """Filter out NaN, non-numeric, and zero values from x and y arrays."""
#     # Convert input to numeric, forcing errors to NaN
#     x = pd.to_numeric(x, errors='coerce')
#     y = pd.to_numeric(y, errors='coerce')
    
#     # Create a mask for valid data points
#     valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    
#     # Apply mask and return filtered data
#     return x[valid_mask], y[valid_mask]

# ## Plotting ###

# # print the legnth of each array to check if they are the same
# # print(len(Temp_Unalloyed_W_Heating))
# # print(len(E_mod_Unalloyed_W_Heating))
# # print(len(Calc_E_mod_Unalloyed_W_Heating))
# # print(Calc_E_mod_W50_50Mo_Heating)

# # Create a figure with two subplots

# textheight = 556 # in pt 
# textwidth = 469
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(textwidth/72.27, textheight/72.27/2))

# # Set the title for both plots
# fig1.suptitle('E-Modulus vs. Temperature for W and W-Mo Alloys', fontsize=10)

# # Move suptitle closer to the plots
# plt.subplots_adjust(top=0.9)

# # Filter and plot for heating data
# x, y = filter_valid_data(Temp_Unalloyed_W_Heating, E_mod_Unalloyed_W_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_Unalloyed_W_Heating, Calc_E_mod_Unalloyed_W_Heating)

# # Plot the original data
# ax1.scatter(x, y, label='Unalloyed W', marker='o', color='blue', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='Unalloyed W (Calculated)', marker='o', color='cyan')
# # ax1.plot(Temp_Reference, E_mod_LowGon, label='Unalloyed W (Lowrie and Gonas)', color='red')
# # ax1.plot(Temp_Reference, E_mod_Skoro, label='Unalloyed W (Škoro et al.)', color='green')

# # Fit and plot the linear trendline
# m_W_heat, b_W_heat = np.polyfit(x, y, 1)  # Fit a linear trendline
# ax1.plot(x, m_W_heat*x + b_W_heat, color='blue')

# # Fit and plot the third-order polynomial trendline
# # poly_coeffs = np.polyfit(x, y, 2)  # Fit a third-degree polynomial
# # poly_trendline = np.polyval(poly_coeffs, x)
# # ax1.plot(x, poly_trendline, color='green', linestyle='--', label='2nd Order Polynomial')

# # Repeat for the calculated data
# # n, c = np.polyfit(x_calc, y_calc, 1)
# # ax1.plot(x_calc, n*x_calc + c, color='cyan', label='Linear Trendline (Calculated)')

# # Fit and plot the third-order polynomial trendline for the calculated data
# # poly_coeffs_calc = np.polyfit(x_calc, y_calc, 2)
# # poly_trendline_calc = np.polyval(poly_coeffs_calc, x_calc)
# # ax1.plot(x_calc, poly_trendline_calc, color='magenta', linestyle='--', label='2nd Order Polynomial (Calculated)')

# x, y = filter_valid_data(Temp_99W_1Mo_Heating, E_mod_99W_1Mo_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_99W_1Mo_Heating, Calc_E_mod_W99_1Mo_Heating)
# ax1.scatter(x, y, label='99W-1Mo', marker='x', color='red', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='99W-1Mo (Calculated)', marker='x', color='pink')
# m_99W_heat, b_99W_heat = np.polyfit(x, y, 1)
# ax1.plot(x, m_99W_heat*x + b_99W_heat, color='red')

# x, y = filter_valid_data(Temp_98W_2Mo_Heating, E_mod_98W_2Mo_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_98W_2Mo_Heating, Calc_E_mod_W98_2Mo_Heating)
# ax1.scatter(x, y, label='98W-2Mo', marker='s', color='green', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='98W-2Mo (Calculated)', marker='s', color='lime')
# m_98W_heat, b_98W_heat = np.polyfit(x, y, 1)
# ax1.plot(x, m_98W_heat*x + b_98W_heat, color='green')

# x, y = filter_valid_data(Temp_94W_6Mo_Heating, E_mod_94W_6Mo_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_94W_6Mo_Heating, Calc_E_mod_W94_6Mo_Heating)
# ax1.scatter(x, y, label='94W-6Mo', marker='^', color='orange', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='94W-6Mo (Calculated)', marker='^', color='gold')
# m_94W_heat, b_94W_heat = np.polyfit(x, y, 1)
# ax1.plot(x, m_94W_heat*x + b_94W_heat, color='orange')

# x, y = filter_valid_data(Temp_75W_25Mo_Heating, E_mod_75W_25Mo_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_75W_25Mo_Heating, Calc_E_mod_W75_25Mo_Heating)
# ax1.scatter(x, y, label='75W-25Mo', marker='d', color='purple', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='75W-25Mo (Calculated)', marker='d', color='violet')
# m_75W_heat, b_75W_heat = np.polyfit(x, y, 1)
# ax1.plot(x, m_75W_heat*x + b_75W_heat, color='purple')

# x, y = filter_valid_data(Temp_50W_50Mo_Heating, E_mod_50W_50Mo_Heating)
# # x_calc, y_calc = filter_valid_data(Temp_50W_50Mo_Heating, Calc_E_mod_W50_50Mo_Heating)
# ax1.scatter(x, y, label='50W-50Mo', marker='v', color='brown', alpha = 0.3)
# # ax1.scatter(x_calc, y_calc, label='50W-50Mo (Calculated)', marker='v', color='tan')
# m_50W_heat, b_50W_heat = np.polyfit(x, y, 1)
# ax1.plot(x, m_50W_heat*x + b_50W_heat, color='brown')

# # Set labels and title for heating plot
# ax1.set_xlabel('Temperature (°C)', fontsize=7)
# ax1.set_ylabel('E-Modulus (GPa)', fontsize=7)
# ax1.set_title('During Heating Stage', fontsize=9)
# ax1.legend(fontsize=7)
# ax1.grid(True)
# ax1.set_ylim(310, 420)

# # set the font size of the ticks
# ax1.tick_params(axis='both', which='major', labelsize=7)

# # Filter and plot for cooling data
# x, y = filter_valid_data(Temp_Unalloyed_W_Cooling, E_mod_Unalloyed_W_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_Unalloyed_W_Cooling, Calc_E_mod_Unalloyed_W_Cooling)

# # Plot the original data
# ax2.scatter(x, y, label='Unalloyed W', marker='o', color='blue', alpha = 0.3)
# # ax2.scatter(x_calc, y_calc, label='Unalloyed W (Calculated)', marker='o', color='cyan')
# # ax2.plot(Temp_Reference, E_mod_LowGon, label='Unalloyed W (Lowrie and Gonas)', color='red')
# # ax2.plot(Temp_Reference, E_mod_Skoro, label='Unalloyed W (Škoro et al.)', color='green')

# # # Fit and plot the linear trendline
# m_W_cool, b_W_cool = np.polyfit(x, y, 1)  # Fit a linear trendline
# ax2.plot(x, m_W_cool*x + b_W_cool, color='blue')

# # # Fit and plot the third-order polynomial trendline
# # poly_coeffs = np.polyfit(x, y, 2)  # Fit a second-degree polynomial
# # poly_trendline = np.polyval(poly_coeffs, x)
# # ax2.plot(x, poly_trendline, color='green', linestyle='--', label='2nd Order Polynomial')

# # # Repeat for the calculated data
# # n, c = np.polyfit(x_calc, y_calc, 1)
# # ax2.plot(x_calc, n*x_calc + c, color='cyan', label='Linear Trendline (Calculated)')

# # # Fit and plot the third-order polynomial trendline for the calculated data
# # poly_coeffs_calc = np.polyfit(x_calc, y_calc, 2)
# # poly_trendline_calc = np.polyval(poly_coeffs_calc, x_calc)
# # ax2.plot(x_calc, poly_trendline_calc, color='magenta', linestyle='--', label='2nd Order Polynomial (Calculated)')

# x, y = filter_valid_data(Temp_99W_1Mo_Cooling, E_mod_99W_1Mo_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_99W_1Mo_Cooling, Calc_E_mod_W99_1Mo_Cooling)
# ax2.scatter(x, y, label='99W-1Mo', marker='x', color='red', alpha=0.3)
# # ax2.scatter(x_calc, y_calc, label='99W-1Mo (Calculated)', marker='x', color='pink', alpha=0.5)
# m_99W_cool, b_99W_cool = np.polyfit(x, y, 1)
# ax2.plot(x, m_99W_cool*x + b_99W_cool, color='red')

# x, y = filter_valid_data(Temp_98W_2Mo_Cooling, E_mod_98W_2Mo_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_98W_2Mo_Cooling, Calc_E_mod_W98_2Mo_Cooling)
# ax2.scatter(x, y, label='98W-2Mo', marker='s', color='green', alpha=0.3)
# # ax2.scatter(x_calc, y_calc, label='98W-2Mo (Calculated)', marker='s', color='lime', alpha=0.5)
# m_98W_cool, b_98W_cool = np.polyfit(x, y, 1)
# ax2.plot(x, m_98W_cool*x + b_98W_cool, color='green')

# x, y = filter_valid_data(Temp_94W_6Mo_Cooling, E_mod_94W_6Mo_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_94W_6Mo_Cooling, Calc_E_mod_W94_6Mo_Cooling)
# ax2.scatter(x, y, label='94W-6Mo', marker='^', color='orange', alpha=0.3)
# # ax2.scatter(x_calc, y_calc, label='94W-6Mo (Calculated)', marker='^', color='gold', alpha=0.5)
# m_94W_cool, b_94W_cool = np.polyfit(x, y, 1)
# ax2.plot(x, m_94W_cool*x + b_94W_cool, color='orange')

# x, y = filter_valid_data(Temp_75W_25Mo_Cooling, E_mod_75W_25Mo_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_75W_25Mo_Cooling, Calc_E_mod_W75_25Mo_Cooling)
# ax2.scatter(x, y, label='75W-25Mo', marker='d', color='purple', alpha=0.3)
# # ax2.scatter(x_calc, y_calc, label='75W-25Mo (Calculated)', marker='d', color='magenta', alpha=0.5)
# m_75W_cool, b_75W_cool = np.polyfit(x, y, 1)
# ax2.plot(x, m_75W_cool*x + b_75W_cool, color='purple')

# x, y = filter_valid_data(Temp_50W_50Mo_Cooling, E_mod_50W_50Mo_Cooling)
# # x_calc, y_calc = filter_valid_data(Temp_50W_50Mo_Cooling, Calc_E_mod_W50_50Mo_Cooling)
# ax2.scatter(x, y, label='50W-50Mo', marker='v', color='brown', alpha=0.3)
# # ax2.scatter(x_calc, y_calc, label='50W-50Mo (Calculated)', marker='v', color='tan', alpha=0.5)
# m_50W_cool, b_50W_cool = np.polyfit(x, y, 1)
# ax2.plot(x, m_50W_cool*x + b_50W_cool, color='brown')

# # Set labels and title for cooling plot
# ax2.set_xlabel('Temperature (°C)', fontsize=7)
# # ax2.set_ylabel('E-Modulus (GPa)')
# ax2.set_title('During Cooling Stage', fontsize=9)
# # ax2.legend()
# ax2.grid(True)
# ax2.set_ylim(310, 420)
# ax2.tick_params(axis='both', which='major', labelsize=7)

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()

def filter_valid_data(x, y):
    """Filter out NaN, non-numeric, and zero values from x and y arrays."""
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    return x[valid_mask], y[valid_mask]

# def calculate_r_squared(x, y, m, b):
def calculate_r_squared(x, y, poly_coeffs):
    """Calculate the R^2 value for a linear fit."""
    # y_pred = m * x + b
    y_pred = np.polyval(poly_coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Define data and their labels
data_dict = {
    "Unalloyed W": {
        "color": "blue", "marker": "o", "heating": (Temp_Unalloyed_W_Heating, E_mod_Unalloyed_W_Heating),
        "cooling": (Temp_Unalloyed_W_Cooling, E_mod_Unalloyed_W_Cooling)
    },
    "99W-1Mo": {
        "color": "red", "marker": "x", "heating": (Temp_99W_1Mo_Heating, E_mod_99W_1Mo_Heating),
        "cooling": (Temp_99W_1Mo_Cooling, E_mod_99W_1Mo_Cooling)
    },
    "98W-2Mo": {
        "color": "green", "marker": "s", "heating": (Temp_98W_2Mo_Heating, E_mod_98W_2Mo_Heating),
        "cooling": (Temp_98W_2Mo_Cooling, E_mod_98W_2Mo_Cooling)
    },
    "94W-6Mo": {
        "color": "orange", "marker": "^", "heating": (Temp_94W_6Mo_Heating, E_mod_94W_6Mo_Heating),
        "cooling": (Temp_94W_6Mo_Cooling, E_mod_94W_6Mo_Cooling)
    },
    "75W-25Mo": {
        "color": "purple", "marker": "d", "heating": (Temp_75W_25Mo_Heating, E_mod_75W_25Mo_Heating),
        "cooling": (Temp_75W_25Mo_Cooling, E_mod_75W_25Mo_Cooling)
    },
    "50W-50Mo": {
        "color": "brown", "marker": "v", "heating": (Temp_50W_50Mo_Heating, E_mod_50W_50Mo_Heating),
        "cooling": (Temp_50W_50Mo_Cooling, E_mod_50W_50Mo_Cooling)
    }
}

# Plot setup
textheight = 556  # in pt 
textwidth = 469
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(textwidth/72.27, textheight/72.27/2))
fig1.suptitle('Measured E-Modulus vs. Temperature for W and W-Mo Alloys', fontsize=10)
plt.subplots_adjust(top=0.9)

# Iterate over datasets for heating and cooling stages
r_squared_values = {"heating": {}, "cooling": {}}
for label, properties in data_dict.items():
    
    # Heating Stage
    x, y = filter_valid_data(*properties["heating"])
    ax1.scatter(x, y, label=label, marker=properties["marker"], color=properties["color"], alpha=0.5)
    # m, b = np.polyfit(x, y, 1)
    # ax1.plot(x, m * x + b, color=properties["color"])
    # r_squared_values["heating"][label] = calculate_r_squared(x, y, m, b)
    poly_coeffs = np.polyfit(x, y, 2)
    ax1.plot(x, np.polyval(poly_coeffs, x), color=properties["color"])
    r_squared_values["heating"][label] = calculate_r_squared(x, y, poly_coeffs)

    # Cooling Stage
    x, y = filter_valid_data(*properties["cooling"])
    ax2.scatter(x, y, label=label, marker=properties["marker"], color=properties["color"], alpha=0.5)
    # m, b = np.polyfit(x, y, 1)
    # ax2.plot(x, m * x + b, color=properties["color"])
    # r_squared_values["cooling"][label] = calculate_r_squared(x, y, m, b)
    poly_coeffs = np.polyfit(x, y, 2)
    ax2.plot(x, np.polyval(poly_coeffs, x), color=properties["color"])
    r_squared_values["cooling"][label] = calculate_r_squared(x, y, poly_coeffs)

# Heating plot adjustments
ax1.set_xlabel('Temperature (°C)', fontsize=7)
ax1.set_ylabel('E-Modulus (GPa)', fontsize=7)
ax1.set_title('During Heating Stage', fontsize=9)
ax1.legend(fontsize=7)
ax1.grid(True)
ax1.set_ylim(310, 420)
ax1.tick_params(axis='both', which='major', labelsize=7)

# Cooling plot adjustments
ax2.set_xlabel('Temperature (°C)', fontsize=7)
ax2.set_title('During Cooling Stage', fontsize=9)
ax2.grid(True)
ax2.set_ylim(310, 420)
ax2.tick_params(axis='both', which='major', labelsize=7)

# Adjust spacing between subplots and show plot
plt.tight_layout()
plt.show()

# Print the R^2 values for reference
print("R^2 Values:")
for stage, results in r_squared_values.items():
    print(f"{stage.capitalize()} Stage:")
    for label, r_squared in results.items():
        print(f"  {label}: R^2 = {r_squared:.4f}")

# Initialize lists to collect residuals and temperatures for all alloys
all_residuals_linear = []
all_residuals_quadratic = []
all_temperatures = []

# Iterate over datasets for heating and cooling stages
for label, properties in data_dict.items():
    # Heating Stage
    x, y = filter_valid_data(*properties["heating"])
    
    # Linear fit
    linear_coeffs = np.polyfit(x, y, 1)
    residuals_linear = y - np.polyval(linear_coeffs, x)
    all_residuals_linear.extend(residuals_linear)  # Store linear residuals
    
    # Quadratic fit
    quadratic_coeffs = np.polyfit(x, y, 2)
    residuals_quadratic = y - np.polyval(quadratic_coeffs, x)
    all_residuals_quadratic.extend(residuals_quadratic)  # Store quadratic residuals
    
    all_temperatures.extend(x)  # Store corresponding temperatures
    
    # Cooling Stage
    x, y = filter_valid_data(*properties["cooling"])
    
    # Linear fit
    linear_coeffs = np.polyfit(x, y, 1)
    residuals_linear = y - np.polyval(linear_coeffs, x)
    all_residuals_linear.extend(residuals_linear)  # Store linear residuals
    
    # Quadratic fit
    quadratic_coeffs = np.polyfit(x, y, 2)
    residuals_quadratic = y - np.polyval(quadratic_coeffs, x)
    all_residuals_quadratic.extend(residuals_quadratic)  # Store quadratic residuals
    
    all_temperatures.extend(x)  # Store corresponding temperatures

# Create subplots for residuals plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Linear residuals plot
ax1.scatter(all_temperatures, all_residuals_linear)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Residuals (GPa)')
ax1.set_title('Residuals Plot for Linear Trendline')

# Quadratic residuals plot
ax2.scatter(all_temperatures, all_residuals_quadratic)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Residuals (GPa)')
ax2.set_title('Residuals Plot for Quadratic Trendline')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# ### Calculate max difference ###

# # Define the temperature range over which to compare the trendlines
# x_values_W = np.linspace(min(min(Temp_Unalloyed_W_Heating), min(Temp_Unalloyed_W_Cooling)),
#                        max(max(Temp_Unalloyed_W_Heating), max(Temp_Unalloyed_W_Cooling)), 100)
# # Calculate the trendline values for both heating and cooling
# y_W_heat = m_W_heat * x_values_W + b_W_heat
# y_W_cool = m_W_cool * x_values_W + b_W_cool
# # Calculate the percentage difference
# percentage_difference = np.abs(y_W_heat - y_W_cool) / ((y_W_heat + y_W_cool) / 2) * 100
# # Find the maximum percentage difference
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling W is: {max_percentage_difference:.2f}%")

# x_values_99W = np.linspace(min(min(Temp_99W_1Mo_Heating), min(Temp_99W_1Mo_Cooling)),
#                         max(max(Temp_99W_1Mo_Heating), max(Temp_99W_1Mo_Cooling)), 100)
# y_99W_heat = m_99W_heat * x_values_99W + b_99W_heat
# y_99W_cool = m_99W_cool * x_values_99W + b_99W_cool
# percentage_difference = np.abs(y_99W_heat - y_99W_cool) / ((y_99W_heat + y_99W_cool) / 2) * 100
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling 99W-1Mo is: {max_percentage_difference:.2f}%")

# x_values_98W = np.linspace(min(min(Temp_98W_2Mo_Heating), min(Temp_98W_2Mo_Cooling)),
#                         max(max(Temp_98W_2Mo_Heating), max(Temp_98W_2Mo_Cooling)), 100)
# y_98W_heat = m_98W_heat * x_values_98W + b_98W_heat
# y_98W_cool = m_98W_cool * x_values_98W + b_98W_cool
# percentage_difference = np.abs(y_98W_heat - y_98W_cool) / ((y_98W_heat + y_98W_cool) / 2) * 100
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling 98W-2Mo is: {max_percentage_difference:.2f}%")

# x_values_94W = np.linspace(min(min(Temp_94W_6Mo_Heating), min(Temp_94W_6Mo_Cooling)),
#                         max(max(Temp_94W_6Mo_Heating), max(Temp_94W_6Mo_Cooling)), 100)
# y_94W_heat = m_94W_heat * x_values_94W + b_94W_heat
# y_94W_cool = m_94W_cool * x_values_94W + b_94W_cool
# percentage_difference = np.abs(y_94W_heat - y_94W_cool) / ((y_94W_heat + y_94W_cool) / 2) * 100
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling 94W-6Mo is: {max_percentage_difference:.2f}%")

# x_values_75W = np.linspace(min(min(Temp_75W_25Mo_Heating), min(Temp_75W_25Mo_Cooling)),
#                         max(max(Temp_75W_25Mo_Heating), max(Temp_75W_25Mo_Cooling)), 100)
# y_75W_heat = m_75W_heat * x_values_75W + b_75W_heat
# y_75W_cool = m_75W_cool * x_values_75W + b_75W_cool
# percentage_difference = np.abs(y_75W_heat - y_75W_cool) / ((y_75W_heat + y_75W_cool) / 2) * 100
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling 75W-25Mo is: {max_percentage_difference:.2f}%")

# x_values_50W = np.linspace(min(min(Temp_50W_50Mo_Heating), min(Temp_50W_50Mo_Cooling)),
#                         max(max(Temp_50W_50Mo_Heating), max(Temp_50W_50Mo_Cooling)), 100)
# y_50W_heat = m_50W_heat * x_values_50W + b_50W_heat
# y_50W_cool = m_50W_cool * x_values_50W + b_50W_cool
# percentage_difference = np.abs(y_50W_heat - y_50W_cool) / ((y_50W_heat + y_50W_cool) / 2) * 100
# max_percentage_difference = np.max(percentage_difference)
# #print(f"Maximum Percentage Difference Between Heating and Cooling 50W-50Mo is: {max_percentage_difference:.2f}%")

