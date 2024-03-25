# Machine learning from scratch

By: [M. D. R. Klarqvist](https://twitter.com/marcusklarqvist)

This repository contains a collection of Jupyter notebooks focusing on the understanding and intuition behind the fundamentals of machine learning. The notebooks are written in Python and are intended to be used as a learning resource for those who are new to the field of machine learning. The notebooks are designed to be self-contained and to provide a step-by-step guide to understanding the core concepts of machine learning.

 The notebooks are designed to be interactive and to provide a hands-on learning experience. The notebooks are written in a clear and concise manner and are intended to be accessible to those with a basic understanding of Python and mathematics.

## Table of Contents

1. Linear regression 
    * What is linear regression?
    * Defining a linear regression model
    * Solving for the best possible parameters using calculus
    * Optimizing our model fit: Manually calculating the partial derivatives of a loss function with respect to our parameters
    * Optimizing our model fit: Using gradient descent to train the model using these manually computed partial derivative equations
    * Using stochastic gradient descent and mini-batch stochastic gradient descent to train the model
    * Introduction to higher-order optimization methods exemplified by Newton's method

2. Regularization and the bias-variance trade-off
    * The bias-variance trade-off and the problem of overfitting and underfitting
    * Introduction to regularization as a way to prevent overfitting
    * L1 and L2 regularization in linear regression
    * Feature selection using L1 regularization
    * Elastic net (L1 + L2) regularization
    * Early stopping as a form of regularization

3. Introduction to non-linear models exemplified by logistic regression
    * What is logistic regression?
    * Defining a logistic regression model
    * Understanding log odds and odds ratios and how they relate to logistic regression
    * Understanding logistic regression as a linear model with a non-linear activation function
    * Solving for the best possible parameters using calculus
    * Optimizing our model fit: Manually calculating the partial derivatives of a loss function with respect to our parameters
    * Optimizing our model fit: Using gradient descent to train the model using these manually computed partial derivative equations
    * Using stochastic gradient descent and mini-batch stochastic gradient descent to train the model

4. Optional: Automatic differentiation (part 1)
    * Introducing dual numbers, their relationship to the derivative, and why they are important
    * A brief overview of the two main automatic differentation approaches
    * Implementing an automatic differentation program in Python from scratch
    * Verifying that the results are correct
    * Highlighting the pros and cons of the two main autodifferentiation approaches

5. Optional: Automatic differentiation (part 2)
    * First we will revisit forward-mode automatic differentiation and look at function compositions and the chain rule
    * Breaking down the sequence of elementary operations into lists
    * Visualizing these lists graphically
    * Implementing these changes in Python from scratch
    * Introducing reverse-mode autodiff and an example
    * Implementing reverse-mode autodiff in Python from scratch

6. Stepping into the world of neural networks
    * What is a neural network?
    * Defining a neural network model
    * Understanding the feedforward process
    * Understanding the backpropagation process
    * Implementing a neural network from scratch in Python
    * Training the neural network using gradient descent
    * Using stochastic gradient descent and mini-batch stochastic gradient descent to train the model
    * Introduction to higher-order optimization methods exemplified by Newton's method

7. Dimensionality reduction and unsupervised learning
    * What is dimensionality reduction?
    * The curse of dimensionality
    * Principal component analysis (PCA)
    * Singular value decomposition (SVD)
    * Non-negative matrix factorization (NMF)
    * t-distributed stochastic neighbor embedding (t-SNE)

8. Introduction to clustering
    * Nearest neighbors and K-nearest neighbors
    * K-means clustering
    * Hierarchical clustering
    * Density-based clustering
    * Evaluating clustering performance
    * Autoencoders (AEs)
    * Variational autoencoders (VAEs)

9. Introduction to decision trees and ensemble methods
    * What is a decision tree?
    * How do decision trees work?
    * Cost functions for decision trees
    * How to build a decision tree from scratch
    * Problems with decision trees and advances made to address them
    * What are ensemble methods?
    * Bagging and random forests

More topics TBD.
