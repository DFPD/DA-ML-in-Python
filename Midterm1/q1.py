"""
Assignment: Python Programming Midterm 1 Question 3 Part 1
Author: Jason Zhao
Student ID: T12611201
Date created: April 9, 2024
"""

import pandas as pd

# Read the data from the CSV file
df = pd.read_csv("question.csv")

# Group the data by "Group" and sum the occurrence of each category within each group
df_grouped = df.groupby('Group').sum()

# Count the occurrences of category "D" being 0
result = (df_grouped['D'] == 0).sum()

# Print the result
print(result)

