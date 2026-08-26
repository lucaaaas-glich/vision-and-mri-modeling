from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np


# Load LFW dataset
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)


# Dataset shape
n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
y = lfw_people.target

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

n_features = X.shape[1]


print("Total dataset size:")
print("n_samples:", n_samples)
print("image height:", h)
print("image width:", w)
print("n_features:", n_features)
print("n_classes:", n_classes)

print()
print("Target names:")
print(target_names)

print()
print("X shape:", X.shape)
print("Images shape:", lfw_people.images.shape)
print("y shape:", y.shape)


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# Number of PCA components
n_components = 150


# Compute mean face using training data only
mean = np.mean(
    X_train,
    axis=0
)

print("Mean shape:", mean.shape)


# Center the data
X_train = X_train - mean
X_test = X_test - mean


# SVD
U, S, V = np.linalg.svd(
    X_train,
    full_matrices=False
)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("V shape:", V.shape)


# Keep first 150 principal components
components = V[:n_components]

print("Components shape:", components.shape)


# Convert components back into image form
eigenfaces = components.reshape(
    (n_components, h, w)
)

print("Eigenfaces shape:", eigenfaces.shape)


# Project training and testing data into PCA space
X_transformed = np.dot(
    X_train,
    components.T
)

X_test_transformed = np.dot(
    X_test,
    components.T
)

print("X transformed:", X_transformed.shape)
print("X test transformed:", X_test_transformed.shape)