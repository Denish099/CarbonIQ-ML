from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("carbon_model.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_industry = joblib.load("le_industry.pkl")
le_severity = joblib.load("le_severity.pkl")

# Emission factors in kg CO2 per unit of fuel
emission_factors = {
    "Diesel": 2.68,      # kg CO₂ per liter
    "Coal": 2.86,        # kg CO₂ per kg
    "Natural Gas": 2.04  # kg CO₂ per m³
}

# Suggestion engine
suggestions = {
    "Low": [
        "Maintain current operations",
        "Monitor emissions periodically"
    ],
    "Medium": [
        "Improve energy efficiency",
        "Inspect equipment for leaks",
        "Schedule preventive maintenance"
    ],
    "High": [
        "Switch to renewable energy sources",
        "Redesign manufacturing process",
        "Install carbon capture systems"
    ]
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        fuel_type = data['fuel_type']
        fuel_qty = float(data['fuel_qty'])      # Convert to float
        industry_type = data['industry_type']
        energy_kwh = float(data['energy_kwh'])  # Convert to float
        hours = float(data['hours'])            # Convert to float

      

        # Encode categorical values
        fuel_encoded = le_fuel.transform([fuel_type])[0]
        industry_encoded = le_industry.transform([industry_type])[0]

        # Predict severity
        features = [[
            fuel_encoded,
            fuel_qty,
            energy_kwh,
            hours,
            industry_encoded
        ]]
        pred_encoded = model.predict(features)[0]
        severity = le_severity.inverse_transform([pred_encoded])[0]

        # Calculate emissions
        emission_factor = emission_factors.get(fuel_type, 0)
        emissions_kg = round(fuel_qty * emission_factor, 2)

        return jsonify({
            "fuel_type": fuel_type,
            "industry_type": industry_type,
            "estimated_emissions_kg": emissions_kg,
            "severity": severity,
            "suggestions": suggestions[severity]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)