import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
#import data
df = pd.read_csv(r"C:\Users\Thoma\Downloads\endoflife_analysis\analysis_data\combined_heart_attack_data_from_survey01_and_02\heart_attack_studies.csv")

# One hot encoding
df = pd.get_dummies(df, columns=['education', 'race_eth'], drop_first=True)

# Y check
y = pd.Categorical(df['answer'], ordered=True)

# Select all relevant X columns (update as needed)
X = df[[col for col in df.columns if col.startswith('education_') or col.startswith('race_eth_')]]

# Check for NaNs
print(X.isnull().sum())
print(y.isnull().sum())

# Create model using logit and BFGS
model = OrderedModel(y, X, distr='logit')
model_results = model.fit(method='bfgs')
print(model_results.summary())

