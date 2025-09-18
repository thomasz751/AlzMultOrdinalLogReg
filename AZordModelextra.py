import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Load dataset
df = pd.read_csv(r"C:\Users\Thoma\Downloads\endoflife_analysis\analysis_data\combined_heart_attack_data_from_survey01_and_02\heart_attack_studies.csv")

# One hot encoding for categorical variables
df = pd.get_dummies(df, columns=['education', 'race_eth'], drop_first=True)

# Create the target variable as ordered categorical
y = pd.Categorical(df['answer'], ordered=True)

# Select feature columns (adjust if needed)
X = df[[col for col in df.columns if col.startswith('education_') or col.startswith('race_eth_')]]

# Drop rows with missing values in X or y
X = X.dropna()
y = y[X.index]  # Ensure alignment

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build and fit the OrderedModel
model = OrderedModel(y_train, X_train, distr='logit')
model_results = model.fit(method='bfgs', disp=False)

# Predict on test set
predicted_probs = model_results.predict(X_test)
predicted_classes = predicted_probs.idxmax(axis=1)  # index of max prob = class index

# Map to actual category levels
predicted_labels = y.categories[predicted_classes]  # y.categories maintains label order

# Evaluate
accuracy = accuracy_score(y_test, predicted_labels)
kappa = cohen_kappa_score(y_test, predicted_labels, weights='quadratic')

# Output results
print(model_results.summary())
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Weighted Kappa Score: {kappa:.4f}")




value_counts = df['answer'].value_counts()
print(value_counts)
categories = ['Stage 1', 'Stage 2', 'Stage 4', 'Stage 3']
plt.bar(categories, value_counts)
plt.xlabel('Stages of Alzheimers Chosen')
plt.ylabel('Amount of People')
plt.title('Stage Selections')
plt.show()
n = df['answer'].count()
print(n)

