import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("emissions_data.csv")

# Encode categorical variables
le_fuel = LabelEncoder()
le_industry = LabelEncoder()
le_severity = LabelEncoder()

df['fuel_type_encoded'] = le_fuel.fit_transform(df['fuel_type'])
df['industry_type_encoded'] = le_industry.fit_transform(df['industry_type'])
df['severity_encoded'] = le_severity.fit_transform(df['severity'])

# Define features and target
X = df[['fuel_type_encoded', 'fuel_qty', 'energy_kwh', 'hours', 'industry_type_encoded']]
y = df['severity_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_severity.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, "carbon_model.pkl")
joblib.dump(le_fuel, "le_fuel.pkl")
joblib.dump(le_industry, "le_industry.pkl")
joblib.dump(le_severity, "le_severity.pkl")