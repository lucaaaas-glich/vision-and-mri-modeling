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

# .data 会把每张二维人脸摊平成一维向量；这是 PCA/RF 需要的输入形式。
# 到 P3 的 CNN 时要改用 lfw_people.images，因为 CNN 需要保留二维空间结构。
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


# 只能用训练集计算 mean；若使用测试集信息，会发生 data leakage（数据泄漏）。
# PCA 的中心化使每个样本表示成“相对平均脸的变化”。
mean = np.mean(
    X_train,
    axis=0
)

print("Mean shape:", mean.shape)


# Center the data
X_train = X_train - mean
X_test = X_test - mean


# SVD: X = U S V^T。V 的每一行是像素空间中的主方向，也就是 eigenface。
# 奇异值 S 越大，代表对应主方向解释的数据变化越多。
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


# 与 components.T 相乘，把 1850 维像素投影为 150 维 face-space 特征。
# 测试集必须使用训练集学到的同一组 components，不能重新做 PCA。
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


# 累积解释方差（compactness）说明前 k 个 components 保留了多少总变化。
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


# Random Forest 不直接看原图，而是用 PCA 得到的 150 个特征分类。
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
