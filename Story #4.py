# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:29:09 2025

@author: AConrard
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Load the data
url = "https://raw.githubusercontent.com/Aconrard/DATA608/refs/heads/main/state_M2023_dl.csv"
df = pd.read_csv(url)


# Let's examine the data structure
print("DataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())


# Filtered for Fifty States, excluding Guam, Virgin islands, and Purto Rico
excluded_states=['Guam', 'Puerto Rico', 'Virgin Islands']


# Filtered for Computer and Mathematics Occupation
df_filtered_15 = df[df['OCC_CODE'].str.startswith('15-', na=False)]


# Filter for Business Occupation Code and Analysts in title
df_filtered_13 = df[(df['OCC_CODE'].str.startswith('13-', na=False)) & 
                    (df['OCC_TITLE'].str.contains('analyst', case=False, na=False))]


# Combine the dfs for analysis
df_combined = pd.concat([df_filtered_15, df_filtered_13], ignore_index=True)
df_combined = df_combined[~df_combined['AREA_TITLE'].isin(excluded_states)]


# Set function to clean values
def clean_salary(x):
    if pd.isna(x) or x == '*' or x == '#':  
        return np.nan
    return x.replace(',', '')


# Appply the function to the df
df_combined['A_MEAN'] = df_combined['A_MEAN'].apply(clean_salary)
df_combined['A_MEDIAN'] = df_combined['A_MEDIAN'].apply(clean_salary)
df_combined['A_PCT25'] = df_combined['A_PCT25'].apply(clean_salary)
df_combined['A_PCT75'] = df_combined['A_PCT75'].apply(clean_salary)


# Print summary of combined dataset
print("Combined Dataset Summary:")
print("Total number of records:", len(df_combined))
print("\nUnique occupation codes and titles:")
print(df_combined[['OCC_CODE', 'OCC_TITLE']].drop_duplicates().sort_values('OCC_CODE'))


# Show sample of cleaned salary data
print("\nSample of cleaned salary data:")
print(df_combined[['OCC_TITLE', 'A_MEAN', 'A_MEDIAN', 'A_PCT25', 'A_PCT75']].head())


# Map the relevent OCC_CODE to assignment Job Titles
role_mappings = {
    'Data Scientist': ['15-2051', '15-1221'],
    'Data Engineer': ['15-1252', '15-1242'],
    'Data Analyst': ['15-1211', '15-2031', '15-2041', '13-2031', '13-2041', '13-2051'],
    'Business Analyst': ['13-1111', '13-1161', '15-1211'],
    'Data Architect': ['15-1243']
}


# Flatten the list of codes
selected_roles = [code for codes in role_mappings.values() for code in codes]


# Filter the dataframe
df_analysis = df_combined[df_combined['OCC_CODE'].isin(selected_roles)].copy()


# Print categorization
print("Data Practitioner Categories:")
for role, codes in role_mappings.items():
    print(f"\n{role}:")
    role_data = df_analysis[df_analysis['OCC_CODE'].isin(codes)][['OCC_CODE', 'OCC_TITLE']].drop_duplicates()
    print(role_data.to_string(index=False))


# Show count of records for each role
print("\nNumber of records for each occupation:")
print(df_analysis.groupby(['OCC_CODE', 'OCC_TITLE']).size().reset_index(name='count'))


# Create a role category column
def assign_role_category(row):
    occ_code = row['OCC_CODE']
    for category, codes in role_mappings.items():
        if occ_code in codes:
            return category
    return 'Other'


# Initial Visualization
df_analysis = df_combined[df_combined['OCC_CODE'].isin(selected_roles)].copy()
df_analysis['A_MEAN'] = pd.to_numeric(df_analysis['A_MEAN'].str.replace(',', ''))


# Add role category
df_analysis['Role_Category'] = df_analysis.apply(assign_role_category, axis=1)
# Import seaborn and use their colorblind palette


colors = sns.color_palette('colorblind')
role_colors = {
    'Data Scientist': colors[0],
    'Data Engineer': colors[1],
    'Data Analyst': colors[2],
    'Business Analyst': colors[3],
    'Data Architect': colors[4]
}


# Define regions with states
regions = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}


# Create the figure
#fig, axs = plt.subplots(2, 2, figsize=(15, 15))
#axs = axs.ravel()


for idx, (region, states) in enumerate(regions.items()):
    region_data = df_analysis[df_analysis['PRIM_STATE'].isin(states)]
    
    # Get role categories for this region
    role_categories = sorted(region_data['Role_Category'].unique())

def convert_salary(x):
    if isinstance(x, str):
        if x in ['*', '#', '']:
            return np.nan
        return float(x.replace(',', ''))
    return x
# Convert columns
for col in ['A_PCT25', 'A_PCT75']:
    df_analysis[col] = df_analysis[col].apply(convert_salary)


# Add role category
df_analysis['Role_Category'] = df_analysis.apply(assign_role_category, axis=1)


# Create IQR ranges by state and role
def format_range(row):
    return f"${row['A_PCT25']:,.0f} - ${row['A_PCT75']:,.0f}"


# Calculate IQR ranges for each state and role
state_role_ranges = df_analysis.groupby(['PRIM_STATE', 'Role_Category']).agg({
    'A_PCT25': 'mean',
    'A_PCT75': 'mean'
}).round(0)


# Format the ranges
state_role_ranges['Salary_Range'] = state_role_ranges.apply(format_range, axis=1)


# Pivot the table to get roles as columns
salary_table = state_role_ranges['Salary_Range'].unstack()


# Reset index to make state a column
salary_table = salary_table.reset_index()


# Rename the state column
salary_table = salary_table.rename(columns={'PRIM_STATE': 'State'})


for role in role_colors.keys():
    role_data = df_analysis[df_analysis['Role_Category'] == role]
    
    # Calculate averages of the 25th and 75th percentiles
    low_avg = role_data['A_PCT25'].mean()
    high_avg = role_data['A_PCT75'].mean()
    
    print(f"\n{role}:")
    print(f"Low Average: ${low_avg:,.0f}")
    print(f"High Average: ${high_avg:,.0f}")
    print(f"Overall Average Range: ${low_avg:,.0f} - ${high_avg:,.0f}")

# Store these values in a dictionary for later use
salary_ranges = {
    role: {
        'low': df_analysis[df_analysis['Role_Category'] == role]['A_PCT25'].mean(),
        'high': df_analysis[df_analysis['Role_Category'] == role]['A_PCT75'].mean()
    }
    for role in role_colors.keys()
}

# Create the comparison DataFrame with sorted state indexes
states = sorted(df_analysis['PRIM_STATE'].unique())
comparison_matrix = pd.DataFrame(index=states)

# First, ensure our salary values are numeric
df_analysis['A_PCT25'] = pd.to_numeric(df_analysis['A_PCT25'], errors='coerce')
df_analysis['A_PCT75'] = pd.to_numeric(df_analysis['A_PCT75'], errors='coerce')

# Create two matrices - one for values (display) and one for differences (coloring)
value_matrix = pd.DataFrame(index=states)
diff_matrix = pd.DataFrame(index=states)

# For each role, get the state-specific data
for role in role_colors.keys():
    # Get national averages from the original data
    role_data = df_analysis[df_analysis['Role_Category'] == role]
    nat_low = role_data['A_PCT25'].mean()
    nat_high = role_data['A_PCT75'].mean()
    
    # Get state-specific data
    for state in states:
        # Get state data for this role
        state_data = df_analysis[
            (df_analysis['PRIM_STATE'] == state) & 
            (df_analysis['Role_Category'] == role)
        ]
        
        if len(state_data) > 0:
            # Calculate state averages
            state_low = state_data['A_PCT25'].mean()
            state_high = state_data['A_PCT75'].mean()
            
            # Store actual values
            value_matrix.loc[state, f'{role}_Low'] = state_low
            value_matrix.loc[state, f'{role}_High'] = state_high
            
            # Store differences for coloring
            if not pd.isna(state_low) and not pd.isna(nat_low) and nat_low != 0:
                diff_matrix.loc[state, f'{role}_Low'] = (state_low - nat_low) / nat_low
            
            if not pd.isna(state_high) and not pd.isna(nat_high) and nat_high != 0:
                diff_matrix.loc[state, f'{role}_High'] = (state_high - nat_high) / nat_high


# Create the heatmap with adjusted figure size
plt.figure(figsize=(15, 12))

# Create custom colormap (light blue to white to light red)
colors = ['#ADD8E6', 'white', '#FFB6C6']
cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

# Format colorbar ticks to show percentages with consistent formatting
def percentage_formatter(x, pos):
    return f'{int(x*100):+}%'


# Analyze states for all-positive or all-negative values
state_colors = {}
for state in diff_matrix.index:
    state_values = diff_matrix.loc[state]
    if all(v < 0 for v in state_values if not pd.isna(v)):
        state_colors[state] = '#0066CC'  # bright blue
    elif all(v > 0 for v in state_values if not pd.isna(v)):
        state_colors[state] = '#CC0000'  # bright red
    else:
        state_colors[state] = '#A9A9A9'



# Create heatmap with percentage formatter
ax = sns.heatmap(diff_matrix, 
            cmap=cmap,
            center=0,
            vmin=-0.3, 
            vmax=0.3,
            cbar_kws={'label': 'Percentage Difference from National Average',
                     'format': FuncFormatter(percentage_formatter),
                     'ticks': [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]},
            mask=diff_matrix.isna(),
            annot=value_matrix,  
            fmt=',.0f',   
            annot_kws={'size': 7},
            xticklabels=True,
            yticklabels=True)

# Set the tick labels with smaller font size
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', size=6)

# Set y-tick labels with colors
y_ticks = ax.get_yticklabels()
for tick in y_ticks:
    state = tick.get_text()
    tick.set_color(state_colors[state])
ax.set_yticklabels(y_ticks, rotation=0, size=8)

# Add main title and subtitle with adjusted positions
plt.suptitle('State Salary Ranges Compared to National Average', 
             size=12, y=0.98)
ax.text(0.5, 1.04, '(Light Red = Higher, Light Blue = Lower)', 
        ha='center', va='bottom', transform=ax.transAxes, size=9)

ax.set_xlabel('Salary Range (Low/High) by Role', size=10)
ax.set_ylabel('State', size=10)

# Adjust layout - increase top margin to reduce gap
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)

plt.show()
# Print states with all values above or below national average
print("\nStates with all salaries below national average:")
print([state for state, color in state_colors.items() if color == '#0066CC'])
print("\nStates with all salaries above national average:")
print([state for state, color in state_colors.items() if color == '#CC0000'])