import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Load the "Wine" Dataset directly from the library
wine_data = load_wine()

# Create the DataFrame (Table)
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target # Add the answer key (target)

print("--- Data Loaded Successfully! ---")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}\n")

# Split between Features (X) and Target (y)
X = df.drop('target', axis=1)  # Chemical data
y = df['target']               # Wine type (0, 1, 2)

# Feature Selection (Top 10)
print("Selecting the top 10 features...")
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Show which columns were selected
cols = selector.get_support(indices=True)
selected_features = X.iloc[:,cols].columns.tolist()
print(f"Selected Features: {selected_features}\n")

# Split into Training (70%) and Testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Standardization (StandardScaler)
# Important: Fit on training data, Transform on both training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model A: SVM (Support Vector Machine)
print("Training SVM...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# Model B: Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# SVM Results
print("\n" + "="*40)
print("SVM RESULTS")
print("="*40)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_svm))

# SVM Confusion Matrix Plot
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Decision Tree Results
print("\n" + "="*40)
print("DECISION TREE RESULTS")
print("="*40)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_dt))

# Decision Tree Confusion Matrix Plot
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()