import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def weighted_nearest_neighbor(X_train, y_train, x_test):
    distances = np.array([euclidean_distance(x, x_test) for x in X_train])
    weights = distances / np.sum(distances)
    weighted_votes = np.zeros(2)

    for i in range(len(weights)):
        weighted_votes[y_train[i]] += weights[i]

    return np.argmax(weighted_votes)

# Example usage
X_train = np.random.rand(10, 5)  # 10 samples in R^5
y_train = np.random.randint(0, 2, 10)  # Class labels 0 or 1
x_test = np.random.rand(5)  # New data point in R^5

predicted_class = weighted_nearest_neighbor(X_train, y_train, x_test)
print(predicted_class)
