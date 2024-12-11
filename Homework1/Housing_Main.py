"""
Assignment: Python Programming Homework 1 Housing
Author: Jason Zhao
Student ID: T12611201
Date created: April 1, 2024
Description: This script demonstrates the usage of a Python file header.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("housing_data.csv")


#%% Question 1 and 2: Data Pre-processing

#%%% Exploratory Data Analysis

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Data types of columns
print(df.info())

print() # new line


#%%% Data Cleaning
# Drop rows with missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print()

#%% Question 3: Renting a house

# Filtering based on # rooms, location, rent price, and distance to school
filtered_rent_df = df[(df['No. of Rooms'] >= 3) &
                 ((df['Location'] == 'Suburb') |
                 (df['Location'] == 'City Center') ) & 
                 (df['Rent Price per Month'] <= 16000) & 
                 (df['Miles (dist. between school and house)'] <= 50)]

# Ranking the filtered options (for example, based on lowest rent price)
ranked_rent_options = filtered_rent_df.sort_values(by='Rent Price per Month')

# Final selection based on personal preferences
selected_rent_house = ranked_rent_options.iloc[0]  # Selecting the top-ranked option

print("Selected House for Renting:")
print(selected_rent_house)


print()

#%% Question 4: Renting vs Buying

#%%% Buying

# Filtering based on location, distance to school, and sell price
filtered_buy_df = df[((df['Location'] == 'Suburb') |
                 (df['Location'] == 'City Center') ) &
                 (df['Miles (dist. between school and house)'] <= 50) &
                 (df['Sell Price'] <= 50000000)]

# Ranking the filtered options (for example, based on lowest rent price)
ranked_buy_options = filtered_buy_df.sort_values(by='Sell Price')

# Final selection based on personal preferences
selected_buy_house = ranked_buy_options.iloc[0]  # Selecting the top-ranked option

print("Selected House for Purchase:")
print(selected_buy_house)

print()

#%%% When to buy vs rent a house

# Rent price of the selected renting house
selected_rent_price = selected_rent_house.at["Rent Price per Month"]

# Sell price of the selected buying house
selected_sell_price = selected_buy_house.at["Sell Price"]

# Time until it becomes more worth it to buy the house instead of renting
time_till_buy = 0.25*(selected_sell_price) / selected_rent_price

print("You should buy the house if the you decide to rent for more than ",
      time_till_buy.round(3), "months, or ",
      (time_till_buy/12).round(3), "years")


#%% Question 5: Outlier rent/sell prices

#%% Statistical Outliers

# Calculate summary statistics
mean_rent_price = df['Rent Price per Month'].mean()
median_rent_price = df['Rent Price per Month'].median()
std_rent_price = df['Rent Price per Month'].std()

mean_sell_price = df['Sell Price'].mean()
median_sell_price = df['Sell Price'].median()
std_sell_price = df['Sell Price'].std()

# Define thresholds for identifying outliers (e.g., 3 standard deviations from the mean)
rent_price_ceiling = mean_rent_price + (3 * std_rent_price)
rent_price_floor = mean_rent_price - (3 * std_rent_price)

sell_price_ceiling = mean_sell_price + (3 * std_sell_price)
sell_price_floor = mean_sell_price - (3 * std_sell_price)


# Identify properties with unusually high or low rent or selling prices
high_rent_properties = df[df['Rent Price per Month'] > rent_price_ceiling]
low_rent_properties = df[df['Rent Price per Month'] < rent_price_floor]

high_sell_properties = df[df['Sell Price'] > sell_price_ceiling]
low_sell_properties = df[df['Sell Price'] < sell_price_floor]

# Print the properties with unusually high or low prices
print("Properties with unusually high rent prices:")
print(high_rent_properties)

print("Properties with unusually low rent prices:")
print(low_rent_properties)

print("Properties with unusually high selling prices:")
print(high_sell_properties)

print("Properties with unusually low selling prices:")
print(low_sell_properties)


print() # New Line


#%% Plotting rent price

# Plot rent prices as individual points
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['Rent Price per Month'], color='skyblue', alpha=0.6)
plt.title('Rent Prices')
plt.xlabel('Index')
plt.ylabel('Rent Price per Month')
plt.show()


#%% Plotting sell price

# Plot sell prices as individual points
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['Sell Price'], color='lightcoral', alpha=0.6)
plt.title('Purchasing prices')
plt.xlabel('Index')
plt.ylabel('Sell Price per Month')
plt.show()







