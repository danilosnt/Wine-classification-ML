import pandas as pd
from sklearn.datasets import load_wine

# 1. Load the "Wine" Dataset
wine_data = load_wine()

# 2. Create a Pandas DataFrame
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

# 3. Add the Target column (identifying the wine type: 0, 1, or 2)
df['target'] = wine_data.target

# 4. Display the first few rows for verification
print("--- First 5 Rows ---")
print(df.head())

# 5. Generate technical info for the paper
print("\n--- Dataset Technical Info ---")
df.info()

# 6. Check for missing values (preprocessing requirement)
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 7. Descriptive Statistics (Mean, Std Dev - useful for 'Results')
print("\n--- Descriptive Statistics ---")
print(df.describe())