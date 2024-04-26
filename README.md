# Machine learning from scratch

By: [M. D. R. Klarqvist](https://twitter.com/marcusklarqvist) ([mdrk.io](https://mdrk.io))

This repository contains a collection of Jupyter notebooks focusing on the understanding and intuition behind the fundamentals of machine learning. The notebooks are written in Python and are intended to be used as a learning resource for those who are new to the field of machine learning. The notebooks are designed to be self-contained and to provide a step-by-step guide to understanding the core concepts of machine learning. The text that goes with the notebooks are available at [https://mdrk.io](https://mdrk.io).

 The notebooks are designed to be interactive and to provide a hands-on learning experience. The notebooks are intended to be accessible to those with a basic understanding of Python and mathematics.

## Table of Contents

* [Introduction to machine learning using linear regression](https://mdrk.io/introduction-to-machine-learning-using-linear-regression/)
    * What is linear regression?
    * Defining a linear regression model
    * Solving for the best possible parameters using calculus
    * Optimizing our model fit: Manually calculating the partial derivatives of a loss function with respect to our parameters
    * Optimizing our model fit: Using gradient descent to train the model using these manually computed partial derivative equations
    * Using stochastic gradient descent and mini-batch stochastic gradient descent to train the model
    * Introduction to higher-order optimization methods exemplified by Newton's method

* [Regularization and the bias-variance trade-off (part 1)](https://mdrk.io/regularization-in-machine-learning-part1/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mklarqvist/machine-learning-from-scratch/blob/main/regularization/regularization.ipynb)
    * Understanding overfitting and underfitting, and the bias-variance trade-off
    * Introducing the train-test split
    * How to add a L2 penalty term to our linear regression model to constrain the model to generalize better to unseen data
    * Optimizing the ridge regression model by computing the gradient manually
    * Optional section: deriving the Normal Equation
    * Understanding the regularization constraint and the method of Lagrange multipliers
    * Changing the L2 penalty to an L1 penalty to perfom Lasso regression
    * Applying what we have learned so far to a real-world application of predicting house prices

* Introduction to non-linear models exemplified by logistic regression
    * What is logistic regression?
    * Defining a logistic regression model
    * Understanding log odds and odds ratios and how they relate to logistic regression
    * Understanding logistic regression as a linear model with a non-linear activation function
    * Solving for the best possible parameters using calculus
    * Optimizing our model fit: Manually calculating the partial derivatives of a loss function with respect to our parameters
    * Optimizing our model fit: Using gradient descent to train the model using these manually computed partial derivative equations
    * Using stochastic gradient descent and mini-batch stochastic gradient descent to train the model

* [Automatic differentiation (part 1)](https://mdrk.io/introduction-to-automatic-differentiation/)
    * Introducing dual numbers, their relationship to the derivative, and why they are important
    * A brief overview of the two main automatic differentation approaches
    * Implementing an automatic differentation program in Python from scratch
    * Verifying that the results are correct
    * Highlighting the pros and cons of the two main autodifferentiation approaches

* [Automatic differentiation (part 2)](https://mdrk.io/introduction-to-automatic-differentiation-part2/)
    * First we will revisit forward-mode automatic differentiation and look at function compositions and the chain rule
    * Breaking down the sequence of elementary operations into lists
    * Visualizing these lists graphically
    * Implementing these changes in Python from scratch
    * Introducing reverse-mode autodiff and an example
    * Implementing reverse-mode autodiff in Python from scratch

* [Optimizers in Deep Learning](https://mdrk.io/optimizers-in-deep-learning/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mklarqvist/machine-learning-from-scratch/blob/main/optimization/optimizer_plots.ipynb)
    * Covers several optimizers with math, intuition, and implementations from scratch in Python
    * SGD with Momentum
    * SGD with Nesterov Accelerated Gradient
    * Adam
    * RMSprop
    * Adagrad
    * Adadelta
    * Adamax
    * Nadam
    * AdamW

* [Interesting Functions for Testing Optimization Methods](https://mdrk.io/interesting-functions-to-optimize/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mklarqvist/machine-learning-from-scratch/blob/main/optimization/loss_surfaces.ipynb)
    * A series of interesting 2D functions to test optimize

* Stepping into the world of neural networks
    * What is a neural network?
    * Defining a neural network model
    * Understanding the feedforward process
    * Understanding the backpropagation process
    * Implementing linear regression neural network from scratch
    * Using automatic differentiation to train the neural network using backpropagation
    * Extending the neural network to include non-linear activation functions
    * Training a logistic regression neural network from scratch


More topics TBD.
