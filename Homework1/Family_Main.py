"""
Assignment: Python Programming Homework 1 Family
Author: Jason Zhao
Student ID: T12611201
Date created: April 1, 2024
Description: This script demonstrates the usage of a Python file header.
"""

import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("family_data.csv")

#%% Question 1: Family with highest/lowest annual income

# Group the data by family and aggregate income and spend
families = df.groupby('Family').agg({'Income': 'sum', 'Spend': 'sum'})

# Reset index to make 'Family' a regular column
families.reset_index(inplace=True)

# Sorting families by income
ranked_incomes = families.sort_values(by='Income')

# Lowest and highest incomes
lowest_income = ranked_incomes.iloc[0]
highest_income = ranked_incomes.iloc[-1]

print('The family with the lowest income is: ')
print(lowest_income, "\n")

print('The family with the highest income is: ')
print(highest_income)


#%% Question 2: Family with inadequate income to cover spending

broke_families = families[families['Spend'] > families['Income']]

# Check if there are any families where spend exceeds income
if broke_families.empty:
    print("From the given dataset, it appears there are no families" ,
          "where spending exceeds income.")
else:
    print("There are families where spending exceeds income.")
    

print()
    

#%% Question 3: Presence of single-parent and childless families

#%%% Single-Parent

# Group the data by family and count the number of unique adult members
family_adult_counts = df[df['Member'].str.startswith('Adult')].groupby('Family').size()

# Filter families with only one adult
single_parent_families = family_adult_counts[family_adult_counts == 1]

if single_parent_families.empty:
    print("There are no single-parent families in the dataset.")
else:
    print("Single-parent families:")
    print(single_parent_families)
    
print()    
    
#%%% Childless

# Group the data by family and check if any member starts with 'Child'
childless_families = df.groupby('Family').filter(lambda x: not any(x['Member'].str.startswith('Child')))

if childless_families.empty:
    print("There are no childless families in the dataset.")
else:
    print("Childless families:")
    print(childless_families['Family'].unique())
    
print()


#%% Question 4: Errors in the dataset


#%%% Only child familes
child_only_families = df.groupby('Family').filter(lambda x: not any(x['Member'].str.startswith('Adult')))

if child_only_families.empty:
    print("There are no families with only children in the dataset.")
else:
    print("Families with only children:")
    print(child_only_families)
    
print()


#%%% Children earn income
# Filter the dataset to include only child members
child_df = df[df['Member'].str.startswith('Child')]

# Check if any child member has non-zero income
child_income_families = child_df[child_df['Income'] > 0]['Family'].unique()

if len(child_income_families) == 0:
    print("There are no families where children earn income in the dataset.")
else:
    print("Families where children earn income:")
    print(child_income_families)

