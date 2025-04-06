import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load model and encoders
model = joblib.load("carbon_model.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_industry = joblib.load("le_industry.pkl")
le_severity = joblib.load("le_severity.pkl")

# Load labeled test data
df = pd.read_csv("test_data_with_labels.csv")

# Encode input features
df['fuel_type_encoded'] = le_fuel.transform(df['fuel_type'])
df['industry_type_encoded'] = le_industry.transform(df['industry_type'])

# Encode target labels (for evaluation)
df['severity_encoded'] = le_severity.transform(df['severity_level'])

# Prepare feature set
X_test = df[['fuel_type_encoded', 'fuel_qty', 'energy_kwh', 'hours', 'industry_type_encoded']]
y_true = df['severity_encoded']

# Predict
y_pred = model.predict(X_test)
df['predicted_severity'] = le_severity.inverse_transform(y_pred)

# Evaluate
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {accuracy:.2f}")
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le_severity.classes_))

# Save predictions with actual vs predicted
df.to_csv("test_predictions_with_evaluation.csv", index=False)
print("📁 Results saved to test_predictions_with_evaluation.csv")