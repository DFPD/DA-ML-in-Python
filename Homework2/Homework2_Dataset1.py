"""
Assignment: Python Programming Homework 2 Dataset 1
Author: Jason Zhao
Student ID: T12611201
Date created: May 27th, 2024
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("maintenance_prediction.csv")

# Display the first few rows to verify
print(df.head())
print()


#%% Question 1 Unique Device IDs

unique_devices = df['device'].nunique()

print("There are", unique_devices, " unique device IDs in the dataset")


#%% Question 2 Data Analysis

# 1. Basic Descriptive Statistics
print("\nBasic Descriptive Statistics:\n")
print(df.describe())

# 2. Distribution Analysis
metrics = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
df[metrics].hist(bins=50, figsize=(15, 10))
plt.suptitle('Distribution of Metrics')
plt.show()

# 3. Correlation Analysis
correlation_matrix = df[metrics].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Metrics')
plt.show()

# 4. Failure Analysis
# Visualizing the average value of metrics for failed (failure=1) and non-failed (failure=0) devices
failed_devices = df[df['failure'] == 1]
non_failed_devices = df[df['failure'] == 0]

avg_metrics_failed = failed_devices[metrics].mean()
avg_metrics_non_failed = non_failed_devices[metrics].mean()

metrics_df = pd.DataFrame({
    'Failed Devices': avg_metrics_failed,
    'Non-Failed Devices': avg_metrics_non_failed
})

metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Average Value of Metrics for Failed and Non-Failed Devices')
plt.ylabel('Average Value')
plt.show()


#%% Question 3

# Define features and target variable
X = df[['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']]
y = df['failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Logistic Regression Results:\n")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classifier Results:\n")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



#%% Question 4


# Logistic Regression
# Extract coefficients
coefficients = logreg.coef_[0]

# Pair feature names with coefficients
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

print("Logistic Regression Coefficients:\n")
print(feature_importance)


# Random Forest
# Extract feature importances
importances = rf.feature_importances_

# Pair feature names with importances
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importances:\n")
print(feature_importance)



#%% Question 5 Predictions 


# New data points
new_data = pd.DataFrame({
    'metric1': [127175526, 4527376],
    'metric2': [4109.434, 0],
    'metric3': [3.90566, 0],
    'metric4': [54.63208, 0],
    'metric5': [15.46226, 3],
    'metric6': [258303.5, 24],
    'metric7': [30.62264, 0],
    'metric8': [30.62264, 0],
    'metric9': [23.08491, 0]
})

# Standardize the new data points
new_data_scaled = scaler.transform(new_data)


# Logistic Regression Predictions
logreg_predictions = logreg.predict(new_data_scaled)
print("Logistic Regression Predictions for new data points:")
print(logreg_predictions)

# Random Forest Predictions
rf_predictions = rf.predict(new_data_scaled)
print("Random Forest Predictions for new data points:")
print(rf_predictions)