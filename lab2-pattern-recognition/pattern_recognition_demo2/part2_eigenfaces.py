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


import matplotlib.pyplot as plt


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))

    plt.subplots_adjust(
        bottom=0,
        left=0.01,
        right=0.99,
        top=0.90,
        hspace=0.35
    )

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(
            images[i].reshape((h, w)),
            cmap=plt.cm.gray
        )

        plt.title(
            titles[i],
            size=12
        )

        plt.xticks(())
        plt.yticks(())


eigenface_titles = [
    f"eigenface {i}"
    for i in range(eigenfaces.shape[0])
]

plot_gallery(
    eigenfaces,
    eigenface_titles,
    h,
    w
)

plt.show()


explained_variance = (S ** 2) / (len(X_train) - 1)

total_var = explained_variance.sum()

explained_variance_ratio = explained_variance / total_var

ratio_cumsum = np.cumsum(
    explained_variance_ratio
)

eigenvalue_count = np.arange(
    n_components
)

plt.figure(figsize=(8, 5))

plt.plot(
    eigenvalue_count,
    ratio_cumsum[:n_components]
)

plt.title("Compactness")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.grid(True)
plt.show()

print(
    "Variance retained by 150 components:",
    ratio_cumsum[n_components - 1]
)


from sklearn.ensemble import RandomForestClassifier


estimator = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    max_features=150,
    random_state=42
)

# Train the classifier
estimator.fit(
    X_transformed,
    y_train
)

# Predict test faces
predictions = estimator.predict(
    X_test_transformed
)

correct = predictions == y_test

total_test = len(
    X_test_transformed
)

accuracy = np.sum(correct) / total_test

print("Total Testing:", total_test)
print("Total Correct:", np.sum(correct))
print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        predictions,
        target_names=target_names
    )
)