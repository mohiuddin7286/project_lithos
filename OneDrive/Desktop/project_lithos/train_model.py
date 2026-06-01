import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import joblib

print("--- PROJECT LITHOS: Model Training Sequence Initiated ---\n")

# ==========================================
# STEP 1: LOAD THE DATA
# ==========================================
data_path = os.path.join('data', 'tabular', 'slope_stability_dataset.csv')

try:
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded {len(df)} records from {data_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {data_path}.")
    exit()

# ==========================================
# STEP 2: PREPARE FEATURES & TARGET (FIXED!)
# ==========================================
print("\n⚙️ Engineering features...")
df = df.dropna()

target_column = 'Factor of Safety (FS)'

X = df.drop(target_column, axis=1)

# THE FIX: Convert the continuous decimal into a binary Category!
# 1 = Unstable/Failed (FS < 1.1)
# 0 = Stable/Safe (FS >= 1.1)
y = (df[target_column] < 1.1).astype(int) 

# Convert text/categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

# ==========================================
# STEP 3: TRAIN THE AI
# ==========================================
print("\n🧠 Training the LightGBM Gradient Booster...")
model = LGBMClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=7, 
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# STEP 4: EVALUATE & SAVE
# ==========================================
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))

# Export the trained model
model_filename = 'lithos_core_model.pkl'
joblib.dump(model, model_filename)
print(f"\n💾 SUCCESS: Model saved as '{model_filename}' in the main project folder.")