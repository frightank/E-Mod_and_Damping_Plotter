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

# Weight Percentage of Mo
Weight_Percentages_Mo = {
    'W': 0,
    'W99': 1.0,
    'W98': 2.0,
    'W94': 6.0,
    'W75': 25.0,
    'W50': 50.0
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

# Define data and their labels
data_dict = {
    "Unalloyed W": {
        "color": "blue", "marker": "o", "heating": (Temp_Unalloyed_W_Heating, Damping_Unalloyed_W_Heating),
        "cooling": (Temp_Unalloyed_W_Cooling, Damping_Unalloyed_W_Cooling)
    },
    "99W-1Mo": {
        "color": "red", "marker": "x", "heating": (Temp_99W_1Mo_Heating, Damping_99W_1Mo_Heating),
        "cooling": (Temp_99W_1Mo_Cooling, Damping_99W_1Mo_Cooling)
    },
    "98W-2Mo": {
        "color": "green", "marker": "s", "heating": (Temp_98W_2Mo_Heating, Damping_98W_2Mo_Heating),
        "cooling": (Temp_98W_2Mo_Cooling,  Damping_98W_2Mo_Cooling)
    },
    "94W-6Mo": {
        "color": "orange", "marker": "^", "heating": (Temp_94W_6Mo_Heating, Damping_94W_6Mo_Heating),
        "cooling": (Temp_94W_6Mo_Cooling, Damping_94W_6Mo_Cooling)
    },
    "75W-25Mo": {
        "color": "purple", "marker": "d", "heating": (Temp_75W_25Mo_Heating, Damping_75W_25Mo_Heating),
        "cooling": (Temp_75W_25Mo_Cooling, Damping_75W_25Mo_Cooling)
    },
    "50W-50Mo": {
        "color": "brown", "marker": "v", "heating": (Temp_50W_50Mo_Heating, Damping_50W_50Mo_Heating),
        "cooling": (Temp_50W_50Mo_Cooling, Damping_50W_50Mo_Cooling)
    }
}

def filter_valid_data(x, y):
    """Filter out NaN, non-numeric, and zero values from x and y arrays."""
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    return x[valid_mask], y[valid_mask]

def calculate_r_squared(x, y, poly_coeffs):
    """Calculate the R^2 value for a specified order fit."""
    # y_pred = m * x + b
    y_pred = np.polyval(poly_coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Plot setup
textheight = 556  # in pt 
textwidth = 469
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(textwidth/72.27, textheight/72.27/2))
fig1.suptitle('Measured Damping vs. Temperature for W and W-Mo Alloys', fontsize=10)
plt.subplots_adjust(top=0.9)

# Iterate over datasets for heating and cooling stages
r_squared_values = {"heating": {}, "cooling": {}}
for label, properties in data_dict.items():
    
    # Heating Stage
    x, y = filter_valid_data(*properties["heating"])
    ax1.scatter(x, y, label=label, marker=properties["marker"], color=properties["color"], alpha=0.1)
    # m, b = np.polyfit(x, y, 1)
    # ax1.plot(x, m * x + b, color=properties["color"])
    # r_squared_values["heating"][label] = calculate_r_squared(x, y, m, b)
    poly_coeffs = np.polyfit(x, y, 2)
    ax1.plot(x, np.polyval(poly_coeffs, x), color=properties["color"])
    r_squared_values["heating"][label] = calculate_r_squared(x, y, poly_coeffs)

    # Cooling Stage
    x, y = filter_valid_data(*properties["cooling"])
    ax2.scatter(x, y, label=label, marker=properties["marker"], color=properties["color"], alpha=0.1)
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
#ax1.set_ylim(0, 0.006)
ax1.tick_params(axis='both', which='major', labelsize=7)

# Cooling plot adjustments
ax2.set_xlabel('Temperature (°C)', fontsize=7)
ax2.set_title('During Cooling Stage', fontsize=9)
ax2.grid(True)
#ax2.set_ylim(0, 0.006)
ax2.tick_params(axis='both', which='major', labelsize=7)

# Adjust spacing between subplots and show plot
plt.tight_layout()
plt.show()