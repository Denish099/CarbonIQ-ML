import pandas as pd
import random

fuel_types = ["Coal", "Natural Gas", "Diesel"]
industry_types = ["Chemical", "Textile", "Manufacturing", "Automobile", "Paper"]

def assign_severity(fuel_qty, energy_kwh):
    if fuel_qty > 1300 and energy_kwh > 6000:
        return "High"
    elif fuel_qty > 800 and energy_kwh > 3000:
        return "Medium"
    else:
        return "Low"

test_data = []
for _ in range(20):  # Generate more rows for better testing
    fuel_qty = random.randint(300, 1800)
    energy_kwh = random.randint(1000, 8000)
    row = {
        "fuel_type": random.choice(fuel_types),
        "fuel_qty": fuel_qty,
        "energy_kwh": energy_kwh,
        "hours": random.randint(4, 20),
        "industry_type": random.choice(industry_types)
    }
    row["severity_level"] = assign_severity(fuel_qty, energy_kwh)
    test_data.append(row)

df_test = pd.DataFrame(test_data)
df_test.to_csv("test_data_with_labels.csv", index=False)
print("✅ Labeled test file saved as test_data_with_labels.csv")