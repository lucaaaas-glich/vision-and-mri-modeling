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