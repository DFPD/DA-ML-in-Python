"""
Assignment: Python Programming Midterm 1 Question 3 Part 3
Author: Jason Zhao
Student ID: T12611201
Date created: April 9, 2024
"""


#%% Initilization

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"Motor ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "Lifespan (hours)": [980, 1050, 990, 1100, 1000, 980, 1020, 990, 950, 1020]})



#%% a
print(df.describe(), "\n")



#%% b

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(df["Lifespan (hours)"], bins=5, edgecolor='black', alpha=0.8)
plt.title('Histogram of Motor Lifespans')
plt.xlabel('Lifespan (hours)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



#%% c

# Filter motors with lifespan >= 1000 hours
exceeded_lifespan = df[df["Lifespan (hours)"] >= 1000]

# Count the # of motors
num_exceeded_lifespan = exceeded_lifespan.shape[0]

print("Number of products in the batch that have reached or exceeded the expected lifespan:", num_exceeded_lifespan)
print()


#%% d

# Calculated in part a)



#%% e

from scipy.stats import shapiro

# Perform Shapiro-Wilk test
statistic, p_value = shapiro(df["Lifespan (hours)"])

print("Shapiro-Wilk test statistic:", statistic)
print("p-value:", p_value)

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The lifespan of the batch of motors follows a normal distribution (fail to reject H0)")
else:
    print("The lifespan of the batch of motors does not follow a normal distribution (reject H0)")

