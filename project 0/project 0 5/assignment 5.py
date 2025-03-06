import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\vc studio projects\\python files\\assignment 5\\World_GDP_Population_CO2_Emissions_Dataset.csv")

df.columns = df.columns.str.strip()

print("Dataset Head:\n", df.head())
print("\nDataset Description:\n", df.describe())

sns.pairplot(df, vars=['GDP Real (USD)', 'World Population', 'Fossil CO2 Emissions (tons)'])
plt.show()

X = df[['GDP Real (USD)', 'World Population']]
y = df['Fossil CO2 Emissions (tons)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.xlabel("Actual CO2 Emissions (tons)")
plt.ylabel("Predicted CO2 Emissions (tons)")
plt.title("Linear Regression: Actual vs Predicted CO2 Emissions")
plt.show()

median_co2 = df['Fossil CO2 Emissions (tons)'].median()
y_binary = np.where(df['Fossil CO2 Emissions (tons)'] > median_co2, 1, 0)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_binary, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_log, y_train_log)

y_pred_logistic = logistic_model.predict(X_test_log)

accuracy = accuracy_score(y_test_log, y_pred_logistic)
conf_matrix = confusion_matrix(y_test_log, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()