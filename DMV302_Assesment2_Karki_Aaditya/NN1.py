import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the training data
data = pd.read_csv('AtRiskStudentsTraining.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize and train the neural network
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
mlp.fit(X_train, y_train)

# Evaluate the model
print("Training Accuracy:", mlp.score(X_train, y_train))
print("Validation Accuracy:", mlp.score(X_val, y_val))

# Assuming continuation from the previous task
test_data = pd.read_csv('AtRiskStudentsTest.csv')
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
X_test = scaler.transform(X_test)

y_pred = mlp.predict(X_test)
accuracy = mlp.score(X_test, y_test)
print("Test Accuracy:", accuracy)
