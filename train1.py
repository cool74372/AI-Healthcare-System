import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('h.csv')

# Features & Target
# X = data.drop('HeartDisease', axis=1)
# y = data['HeartDisease']

X = pd.get_dummies(data.drop('HeartDisease', axis=1))
y = data['HeartDisease']

# SAVE COLUMNS
pickle.dump(X.columns, open('models/vitals_columns.pkl', 'wb'))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('models/vitals_model.pkl', 'wb'))

print("✅ Model trained successfully!")