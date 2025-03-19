# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:29:09 2025

@author: AConrard
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import seaborn as sns

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
    if pd.isna(x) or x == '*' or x == '#':  # Handle missing values and special characters
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
    'Data Scientist': colors[0],    # blue
    'Data Engineer': colors[1],     # orange
    'Data Analyst': colors[2],      # green
    'Business Analyst': colors[3],  # red
    'Data Architect': colors[4]     # purple
}

# Define regions with states
regions = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# Create the figure
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for idx, (region, states) in enumerate(regions.items()):
    region_data = df_analysis[df_analysis['PRIM_STATE'].isin(states)]
    
    # Get role categories for this region
    role_categories = sorted(region_data['Role_Category'].unique())
    
    # Create boxplot for this region
    bp = axs[idx].boxplot([group['A_MEAN'].dropna() for name, group in 
                          region_data.groupby('Role_Category')],
                         tick_labels=role_categories,
                         patch_artist=True)
    
    # Color each box according to role
    for patch, role in zip(bp['boxes'], role_categories):
        patch.set_facecolor(role_colors[role])
        patch.set_alpha(0.7)
    
    # Customize the plot
    axs[idx].set_title(f'{region} Region\n(States: {", ".join(sorted(states))})', 
                       fontsize=10, pad=20)
    axs[idx].set_ylabel('Average Annual Salary ($)')
    axs[idx].tick_params(axis='x', rotation=45)
    axs[idx].grid(True, alpha=0.3)
    
    # Format y-axis with comma separator
    axs[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Median Lines to Black
    plt.setp(bp['medians'], color='black', linewidth=1.5)


# Add a legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) 
                  for color in role_colors.values()]
fig.legend(legend_elements, role_colors.keys(), 
          loc='upper center',  # Change location to upper center
          bbox_to_anchor=(0.5, 1.05),  # Adjust these values to move legend up
          ncol=5)  # Display all items in one row

plt.suptitle('Salary Distribution by Role and Region', fontsize=16, y=1.08)  # Adjust y to make room for legend
plt.tight_layout()
plt.show()


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

# Print table using tabulate
print("\nSalary Ranges by State and Role:")
print(tabulate(salary_table, headers='keys', tablefmt='grid', showindex=False))

# Also save to CSV for reference
salary_table.to_csv('salary_ranges_by_state.csv', index=False)
