"""
Assignment: Python Programming Homework 2 Dataset 3
Author: Jason Zhao
Student ID: T12611201
Date created: May 27th, 2024
"""

#%% Question 1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import Counter
import string
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


# Load the dataset
df = pd.read_csv("feedback_sentiment.csv")

# Splitting the rows by commas into different columns
df = df['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'].str.split(', ', expand=True)
df.columns = ['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']


# Display the first few rows to verify
print(df.head())

#%% Question 2

#%%% Clean Data, drop NaN values


# Drop rows with missing values
df.dropna(inplace=True)

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Remove duplicate rows, done as makes sense in context
df.drop_duplicates(inplace=True)

print()


#%%% Convert 'Date/Time' column to datetime format

df['Date/Time'] = pd.to_datetime(df['Date/Time'])


#%%% Create new features for month, day, and hour

df['Month'] = df['Date/Time'].dt.month
df['Day'] = df['Date/Time'].dt.day
df['Hour'] = df['Date/Time'].dt.hour


#%%% Create new other features

# Function to calculate total words
def count_words(text):
    return len(text.split())

# Function to calculate total characters
def count_chars(text):
    return len(text)

# Apply functions to create new features
df['Total Words'] = df['Text'].apply(count_words)
df['Total Chars'] = df['Text'].apply(count_chars)
df['Total Words After Transformation'] = np.log(df['Total Words'])



#%% Question 3 Plots

#%%% 1) by Sentiment 'Positive' and 'Negative'


# Count of Positive and Negative Sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Bar Plotting
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel')
plt.title('Count of Positive vs Negative Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


#%%% 2) by 'Source' and 'Sentiment'

# Grouped bar plot by Source and Sentiment
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Source', hue='Sentiment', palette='pastel')
plt.title('Distribution of Sentiments by Source')
plt.xlabel('Source')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

#%%% 3) by 'Location' and 'Sentiment'

# Grouped bar plot by Location and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Location', hue='Sentiment', palette='pastel')
plt.title('Distribution of Sentiments by Location')
plt.xlabel('Location')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

#%%% 4) by 'Confidence Score' and 'Sentiment'

# Sort the dataframe by 'Confidence Score'
df_sorted = df.sort_values(by='Confidence Score')

# Histogram with KDE by Sentiment, sorted by Confidence Score
plt.figure(figsize=(10, 6))
sns.histplot(df_sorted, x='Confidence Score', hue='Sentiment', kde=True, multiple='stack', palette='pastel')
plt.title('Distribution of Confidence Score by Sentiment')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()
#%%% 5) by 'Month' and 'Sentiment'

# Grouped bar plot by Month and Sentiment
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Month', hue='Sentiment', palette='pastel')
plt.title('Distribution of Sentiments by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

#%%% 6) by 'Day' and 'Sentiment'

# Grouped bar plot by Day and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Day', hue='Sentiment', palette='pastel')
plt.title('Distribution of Sentiments by Day')
plt.xlabel('Day of the Month')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

#%%% 7) by 'hour' and 'Sentiment'

# Grouped bar plot by Hour and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Hour', hue='Sentiment', palette='pastel')
plt.title('Distribution of Sentiments by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

#%%% 8) by 'Total Words' and 'Sentiment'

# Box plot by Sentiment
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Sentiment', y='Total Words', palette='pastel')
plt.title('Box Plot of Total Words by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Total Words')
plt.show()


#%%% 9) by 'Total Chars' and 'Sentiment'

# Box plot by Sentiment
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Sentiment', y='Total Chars', palette='pastel')
plt.title('Box Plot of Total Characters by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Total Characters')
plt.show()

#%%% 10) Wordcloud by Sentiment = Negative

# Filter texts with Negative sentiment
negative_texts = df[df['Sentiment'] == 'Negative']['Text']

# Concatenate all texts into a single string
negative_text_combined = ' '.join(negative_texts)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text_combined)

# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')
plt.show()


#%%% 11) Wordcloud by Sentiment = Positive

# Filter texts with Positive sentiment
positive_texts = df[df['Sentiment'] == 'Positive']['Text']

# Concatenate all texts into a single string
positive_text_combined = ' '.join(positive_texts)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(positive_text_combined)

# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')
plt.show()

#%%% 12) by Top 25 Negative Words

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Apply preprocessing and filter by Negative sentiment
negative_texts = df[df['Sentiment'] == 'Negative']['Text'].apply(preprocess_text)

# Combine all texts into a single string
negative_text_combined = ' '.join(negative_texts)

# Split text into words
words = negative_text_combined.split()

# Remove stopwords (optional, depends on your data and analysis)
words = [word for word in words if word not in STOPWORDS]

# Count frequencies of each word
word_freq = Counter(words)

# Select top 25 words
top_words = word_freq.most_common(25)

# Extract words and frequencies
top_words, freq = zip(*top_words)

# Plotting the top 25 negative words
plt.figure(figsize=(10, 8))
plt.barh(top_words, freq, color='salmon')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 25 Negative Words')
plt.gca().invert_yaxis()  # Invert y-axis to have the most frequent word at the top
plt.show()


#%%% 13) by Top 25 Positive Words

# Apply preprocessing and filter by Positive sentiment
positive_texts = df[df['Sentiment'] == 'Positive']['Text'].apply(preprocess_text)

# Combine all texts into a single string
positive_text_combined = ' '.join(positive_texts)

# Split text into words
words = positive_text_combined.split()

# Remove stopwords (optional, depends on your data and analysis)
words = [word for word in words if word not in STOPWORDS]

# Count frequencies of each word
word_freq = Counter(words)

# Select top 25 words
top_words = word_freq.most_common(25)

# Extract words and frequencies
top_words, freq = zip(*top_words)

# Plotting the top 25 positive words
plt.figure(figsize=(10, 8))
plt.barh(top_words, freq, color='lightgreen')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 25 Positive Words')
plt.gca().invert_yaxis()  # Invert y-axis to have the most frequent word at the top
plt.show()





#%% Question 4 & 5: 8 Classification Models, Classification Reports and Confusion Matrices


#%%% Tuning first

# Assuming df is your dataframe with 'Text' and 'Sentiment' columns
X = df['Text']
y = df['Sentiment']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the target variable
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)



#%%% Model Evaluation to determine optimal parameters

#%%%% Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# Define the parameter grid
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}

# Define the GridSearchCV
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='accuracy')

# Fit the model
grid_lr.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for Logistic Regression:", grid_lr.best_params_)
print("Best Score for Logistic Regression:", grid_lr.best_score_)


# Predict using the best logistic regression model
y_pred_lr = grid_lr.best_estimator_.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# Generate the confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


#%%%% K-NN
from sklearn.neighbors import KNeighborsClassifier

# Define the model
knn = KNeighborsClassifier()


# Convert sparse matrix to dense matrix
# DONE SPECIFICALLY FOR KNN
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# Define the parameter grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Define the GridSearchCV
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')

# Fit the model
grid_knn.fit(X_train_dense, y_train)

# Best parameters and score
print("Best Parameters for KNN:", grid_knn.best_params_)
print("Best Score for KNN:", grid_knn.best_score_)



# Predict using the best KNN model
y_pred_knn = grid_knn.best_estimator_.predict(X_test_tfidf)

# Print the accuracy and classification report
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# Generate the confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for K-NN')
plt.show()


#%%%% SVC
from sklearn.svm import SVC

# Define the model
svc = SVC()

# Define the parameter grid
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Define the GridSearchCV
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy')

# Fit the model
grid_svc.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for SVC:", grid_svc.best_params_)
print("Best Score for SVC:", grid_svc.best_score_)



# Predict using the best SVC model
y_pred_svc = grid_svc.best_estimator_.predict(X_test_tfidf)

# Print the accuracy and classification report
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))



# Generate the confusion matrix
cm_svc = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for SVC')
plt.show()


#%%%% Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

# Define the model
gnb = GaussianNB()

# Define the parameter grid
param_grid_gnb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
}

# Define the GridSearchCV
grid_gnb = GridSearchCV(gnb, param_grid_gnb, cv=5, scoring='accuracy')

# Fit the model
grid_gnb.fit(X_train_tfidf.toarray(), y_train)  # GaussianNB requires dense arrays

# Best parameters and score
print("Best Parameters for Gaussian Naive Bayes:", grid_gnb.best_params_)
print("Best Score for Gaussian Naive Bayes:", grid_gnb.best_score_)


# Predict using the best GNB model
y_pred_gnb = grid_gnb.best_estimator_.predict(X_test_tfidf.toarray())

# Print the accuracy and classification report
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))


# Generate the confusion matrix
cm_gnb = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Gaussian Naive Bayes')
plt.show()


#%%%% Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

# Define the model
mnb = MultinomialNB()

# Define the parameter grid
param_grid_mnb = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}

# Define the GridSearchCV
grid_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='accuracy')

# Fit the model
grid_mnb.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for Multinomial Naive Bayes:", grid_mnb.best_params_)
print("Best Score for Multinomial Naive Bayes:", grid_mnb.best_score_)


# Predict using the best MNB model
y_pred_mnb = grid_mnb.best_estimator_.predict(X_test_tfidf)

# Print the accuracy and classification report
print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb))




# Generate the confusion matrix
cm_mnb = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Multinomial Naive Bayes')
plt.show()

#%%%% Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Define the model
dt = DecisionTreeClassifier()

# Define the parameter grid
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Define the GridSearchCV
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy')

# Fit the model
grid_dt.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for Decision Tree:", grid_dt.best_params_)
print("Best Score for Decision Tree:", grid_dt.best_score_)


# Predict using the best decision tree model
y_pred_dt = grid_dt.best_estimator_.predict(X_test_tfidf)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# Generate the confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

#%%%% Random Forest
from sklearn.ensemble import RandomForestClassifier

# Define the model
rf = RandomForestClassifier()

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Define the GridSearchCV
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')

# Fit the model
grid_rf.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for Random Forest:", grid_rf.best_params_)
print("Best Score for Random Forest:", grid_rf.best_score_)



# Predict using the best XGBoost model
y_pred_rf = grid_rf.best_estimator_.predict(X_test_tfidf)

# Print the accuracy and classification report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# Generate the confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Random Forest')
plt.show()

#%%%% XGBoost

import xgboost as xgb

# Define the model
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the parameter grid
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.7, 1.0]
}

# Define the GridSearchCV
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=5, scoring='accuracy')

# Fit the model
grid_xgb.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters for XGBoost:", grid_xgb.best_params_)
print("Best Score for XGBoost:", grid_xgb.best_score_)



# Predict using the best XGBoost model
y_pred_xgb = grid_xgb.best_estimator_.predict(X_test_tfidf)

# Print the accuracy and classification report
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))




# Generate the confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for XGBoost')
plt.show()

