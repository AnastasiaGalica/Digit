from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from Functions import *
import numpy as np


# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Define cross-validation strategy (e.g., 5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

dir_path = 'input_treatment/Splitted/val'
save_to = 'Matrixe'
mat_lab_calculate(dir_path, save_to, 14, 14, True)
val_matrix = np.load('Matrixe/val_matrix/val_matrix_14x14.npy')
val_labels = np.load('Matrixe/val_matrix/val_labels_14x14.npy')


# Perform cross-validation and compute accuracy scores
cv_scores = cross_val_score(knn, val_matrix, val_labels, cv=kf, scoring='accuracy')

# Print average accuracy and other metrics if needed
print("Cross-validated Accuracy:", np.mean(cv_scores))
