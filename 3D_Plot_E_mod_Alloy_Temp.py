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

# Define a dictionary to hold the temperature data
temperature_lists = {
    'W': {'Heating': Temp_Unalloyed_W_Heating, 'Cooling': Temp_Unalloyed_W_Cooling},
    'W99': {'Heating': Temp_99W_1Mo_Heating, 'Cooling': Temp_99W_1Mo_Cooling},
    'W98': {'Heating': Temp_98W_2Mo_Heating, 'Cooling': Temp_98W_2Mo_Cooling},
    'W94': {'Heating': Temp_94W_6Mo_Heating, 'Cooling': Temp_94W_6Mo_Cooling},
    'W75': {'Heating': Temp_75W_25Mo_Heating, 'Cooling': Temp_75W_25Mo_Cooling},
    'W50': {'Heating': Temp_50W_50Mo_Heating, 'Cooling': Temp_50W_50Mo_Cooling}
}

# Define a dictionary to hold the E-Modulus data
alloy_e_mods = {
    'W': {'Heating': E_mod_Unalloyed_W_Heating, 'Cooling': E_mod_Unalloyed_W_Cooling},
    'W99': {'Heating': E_mod_99W_1Mo_Heating, 'Cooling': E_mod_99W_1Mo_Cooling},
    'W98': {'Heating': E_mod_98W_2Mo_Heating, 'Cooling': E_mod_98W_2Mo_Cooling},
    'W94': {'Heating': E_mod_94W_6Mo_Heating, 'Cooling': E_mod_94W_6Mo_Cooling},
    'W75': {'Heating': E_mod_75W_25Mo_Heating, 'Cooling': E_mod_75W_25Mo_Cooling},
    'W50': {'Heating': E_mod_50W_50Mo_Heating, 'Cooling': E_mod_50W_50Mo_Cooling}
}

def filter_valid_data(x, y):
    """Filter out NaN, non-numeric, and zero values from x and y arrays."""
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    return x[valid_mask], y[valid_mask]

# Arrays to store all the temperatures, Mo contents, and calculated E-Modulus values
all_temps = []
all_mo_content = []
all_e_mods = []

# Store the fit lines for highlighting
fit_lines = {}

# Perform linear fits for each alloy and collect data
# for alloy, mo_percent in Atomic_Percentages_Mo.items():
for alloy, mo_percent in Weight_Percentages_Mo.items():
    # Combine Heating and Cooling data
    temp_combined = np.concatenate([temperature_lists[alloy]['Heating'], temperature_lists[alloy]['Cooling']])
    e_mod_combined = np.concatenate([alloy_e_mods[alloy]['Heating'], alloy_e_mods[alloy]['Cooling']])
    
    # Filter valid data
    x, y = filter_valid_data(temp_combined, e_mod_combined)
    
    if len(x) > 0:  # Check if there's valid data after filtering
        # Linear fit for combined data
        linear_fit = np.polyfit(x, y, 1)  # Linear fit (degree 1)
        fitted_y = np.polyval(linear_fit, x)  # Calculate fitted y values
        
        all_temps.extend(x)
        all_mo_content.extend([mo_percent] * len(x))
        all_e_mods.extend(fitted_y)
        
        # Store the fit line data
        fit_lines[alloy] = (x, fitted_y, mo_percent)

# Convert lists to numpy arrays
all_temps = np.array(all_temps)
all_mo_content = np.array(all_mo_content)
all_e_mods = np.array(all_e_mods)

# Create a grid for interpolation
temp_grid, mo_content_grid = np.meshgrid(
    np.linspace(min(all_temps), max(all_temps), 100),
    np.linspace(min(all_mo_content), max(all_mo_content), 100)
)

# Interpolate the E-Modulus values over the grid
e_mod_grid = griddata(
    (all_temps, all_mo_content),
    all_e_mods,
    (temp_grid, mo_content_grid),
    method='linear'
)

# Plot the 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(temp_grid, mo_content_grid, e_mod_grid, cmap='plasma', edgecolor='none', alpha=0.8)

# Highlight the linear fit lines
colors = ['red', 'green', 'blue', 'orange', 'purple', 'magenta']
for i, (alloy, (x, y, mo_percent)) in enumerate(fit_lines.items()):
    ax.plot(x, [mo_percent]*len(x), y, color='black', linewidth=2, label=f'{alloy} Linear Fit')
    ax.text(x[-1], mo_percent, y[-1], f'{alloy}', color='black', fontsize=10, weight='bold')

# Labels and title
ax.set_xlabel('Temperature (°C)')
# ax.set_ylabel('Atomic Percentage of Molybdenum (%)')
ax.set_ylabel('Weight Percentage of Molybdenum (%)')
ax.set_zlabel('Measured E-Modulus (GPa)')
ax.set_title('3D Surface Plot with Linear Fits: E-Modulus vs Temperature vs Molybdenum Content')

# Show plot
plt.show()