import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the dataset (SVMs are sensitive to feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM models with different kernels
models = [
    ("SVC with linear kernel", SVC(kernel="linear")),
    ("LinearSVC (linear kernel)", LinearSVC(max_iter=10000)),
    ("SVC with RBF kernel", SVC(kernel="rbf", gamma=0.7)),
    ("SVC with polynomial (degree 3) kernel", SVC(kernel="poly", degree=3))
]

# Train each model and plot the decision boundary
plt.figure(figsize=(12, 10))

for i, (title, model) in enumerate(models):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print classification report and confusion matrix
    print(f"\n{title}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # Plotting decision boundaries
    plt.subplot(2, 2, i + 1)
    plt.title(title)
    
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Plot the decision boundary by assigning a color to each point in the mesh
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot also the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k", s=30)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

# Manually add a legend for the target classes
for j, class_name in enumerate(iris.target_names):
    plt.scatter([], [], color=plt.cm.coolwarm(j / 2), label=class_name)  # Empty scatter for legend entry

plt.legend(title="Classes")
plt.tight_layout()
plt.show()
