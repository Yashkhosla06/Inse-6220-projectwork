# Pistachio Classification: Unveiling Insights with Principal Component Analysis and Machine Learning Optimization

## Overview
This project focuses on optimizing post-harvest processes for pistachio nuts, a key element in the agricultural economy. The research utilizes image processing, artificial intelligence, and the Pistachio Image Dataset to create a robust model that accurately differentiates between two pistachio species.

## Abstract
The study introduces an advanced classification model that combines the K-NN method with Principal Component Analysis (PCA) for dimensionality reduction and weighting. Further improvements are explored by incorporating PCA with three machine learning algorithms: Gaussian, k-Nearest Neighbors (KNN), and Decision Trees.

## Principal Component Analysis (PCA)
### PCA Algorithm
PCA is applied to reduce dimensionality and enhance the discriminatory power of features. The steps involved are:
1. **Standardization**: Standardizing the initial variables.
2. **Covariance Matrix Computation**: Establishing the connections between variables.
3. **Eigen Decomposition**: Calculating the eigenvalues and eigenvectors of the covariance matrix.
4. **Principal Components**: Transforming the dataset using the eigenvector matrix.

## Machine Learning-Based Classification Algorithms
### Gaussian Naive Bayes (GNB)
GNB is effective for datasets with features that follow a normal distribution. It operates on the assumption of conditional independence among features given the class label.

### k-Nearest Neighbors (KNN)
KNN classifies samples based on the majority class of their nearest neighbors. It is versatile and used in various applications such as pattern recognition and recommendation systems.

### Decision Trees
Decision Trees recursively split the dataset based on the most influential attribute at each node. They are known for their interpretability but can risk overfitting.

## Data Set Description
The dataset from Kaggle contains 2148 entries with features such as "Eccentricity," "Solidity," "Extent," "Aspect Ratio," "Roundness," and "Compactness." It includes labels identifying the Siirt and Kirmizi pistachio classes.

## PCA Results
PCA is applied to the dataset to reduce the feature set while retaining the majority of the original dataset’s information. The first two principal components contribute significantly to the variance in the data.

## Classification Results
### Gaussian Naive Bayes (GNB)
GNB calculates the probability of a data point belonging to a particular class based on the probability distributions of its features.

### k-Nearest Neighbors (KNN)
KNN identifies the k-nearest neighbors in the training dataset and makes predictions based on the majority vote.

### Decision Trees
Decision Trees split the dataset into subsets based on the most influential attribute, creating a tree of decision rules.

### Decision Boundaries and Confusion Matrices
Decision boundaries and confusion matrices for the three algorithms demonstrate their effectiveness in classifying pistachio species. Performance metrics such as F1-score, confusion matrix, and ROC curves provide insights into the models’ effectiveness.

## Conclusion
This study leverages advanced techniques in image processing, machine learning, and statistical analysis to classify pistachio species. The combination of PCA with machine learning algorithms effectively distinguishes between species, offering valuable insights for the agricultural industry.

## Authors
- Yash Khosla, Concordia Institute for Information Systems Engineering (CIISE), Concordia University, Montreal, Canada


## References
1. Ozkan, I. A., M. Koklu, and Rıdvan Saraçoğlu. ”Classification of pistachio species using improved k-NN classifier.” Health 23 (2021): e2021044.
2. Singh, Dilbag, et al. "Classification and analysis of pistachio species with pre-trained deep learning models." Electronics 11, no. 7 (2022): 981.
3. [Kaggle: Pistachio Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset/data)
4. Maćkiewicz, Andrzej, and Waldemar Ratajczak. "Principal components analysis (PCA)." Computers & Geosciences 19, no. 3 (1993): 303-342.
5. [PyCaret Tutorials](https://github.com/pycaret/pycaret/tree/master/tutorials)
6. [Python Programming](https://pythonprogramming.net/machine-learning-tutorial-python-introduction/)

---------------------------------------------------------------------------

For more detailed information, please refer to the full report, the dataset in csv file and the program code. 
