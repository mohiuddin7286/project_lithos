import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import joblib

print("--- PROJECT LITHOS: Advanced Telemetry Training Initiated ---\n")

# 1. Load the Dynamic Sensor Dataset
data_path = os.path.join('data', 'tabular', 'excavation_risk_dataset.csv')

try:
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded {len(df)} live sensor records.")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {data_path}.")
    exit()

# 2. Clean and Prepare Features
print("\n⚙️ Engineering Dynamic Features...")
df = df.dropna()

# In this dataset, the target is 'Risk_Level'
target_column = 'Risk_Level'

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert Categorical text (like 'Soil_Type' and 'Support_System') into numbers
X = pd.get_dummies(X, drop_first=True)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

# 4. Train the LightGBM Model
print("\n🧠 Training the Multi-Class Sensor AI...")
model = LGBMClassifier(
    n_estimators=250, 
    learning_rate=0.05, 
    max_depth=8, 
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Risk Level Classification Report:")
print(classification_report(y_test, predictions))

# 6. Save the new, dynamic model!
model_filename = 'lithos_telemetry_model.pkl'
joblib.dump(model, model_filename)
print(f"\n💾 SUCCESS: Model saved as '{model_filename}'")

# Print the exact features so we can update React later
print("\n📋 EXACT SENSOR COLUMNS NEEDED FOR REACT:")
for feature in model.feature_name_[:6]: # Just printing the top 6 for reference
    print(f" - {feature}")