# Table of contents
<!--ts-->
   * [Introduction to Machine learning](#introduction-to-machine-learning)
      * [Analytics](#analytics)
      * [Categories of ML algorithms](#categories-of-ml-algorithms)
      * [A typical ML algorithm uses the following steps](#a-typical-ml-algorithm-uses-the-following-steps)
      * [Framework for develping ML models](#framework-for-develping-ml-models)
   *  [Python](https://github.com/arunsinp/Python-programming)
      * [Google colab help](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/google-colab-help.ipynb)
      * [Python OS and filesystem](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/python-os-and-filesystem.ipynb)
      * [First steps with python](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/0.first-steps-with-python.ipynb)
      * [Python variables and data types](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/1.python-variables-and-data-types.ipynb)
      * [Python conditionals and loops](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/2.python-conditionals-and-loops.ipynb)
      * [Python functions and scope](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/3.%20python-functions-and-scope.ipynb)
      * [Numpy](https://github.com/arunsinp/Python-programming/blob/main/Python-fundamental/NUMPY.ipynb)
   * [Python and data science](#python-and-data-science)
      * [Core Python Libraries for Data Analysis](#core-python-libraries-for-data-analysis)
      * [Descriptive statistics](#descriptive-statisctics)
      * [Probability distributions and hypothesis testing](#probability-diestributions-and-hypotehsis-testing)
      * [Linear Regression](#linear-regression)
      * [Advanced machine learning](#advanced-machine-learning)
   *  [References](#reference)
<!--te-->

# Introduction to Machine learning

## Analytics
Analytics is a collection of techniques and tools used for creating value from data. Techniques include concepts such as artificial intelligence (AI), machine learning (ML), and deep learning (DL) algorithms.

- **Artificial Intelligence (AI):** Algorithms and systems that exhibit human-like intelligence.
- **Machine Learning (ML):** Subset of AI that can learn to perform a task with extracted data and/or
Models
- **Deep Learning (DL):** Subset of machine learning that imitate the functioning of human brain to solve problems.

<img src="AI-vs-ML-vs-Deep-Learning.png"     alt="Markdown Monster icon"     style="float: left; margin-right: 10px;" />


Machine learning is a set of algorithms that have the capability to learn to perform tasks such as prediction and classification effectively using data.


## Categories of ML algorithms
1. **Supervised Learning Algorithms:** These algorithms require the knowledge of both the outcome variable (dependent variable) and the features (independent variable or input variables). The algorithm learns (i.e., estimates the values of the model parameters or feature weights) by defining a loss function which is usually a function of the difference between the predicted value and actual value of the outcome variable
2.	**Unsupervised Learning Algorithms:** These algorithms are set of algorithms, which do not have the knowledge of the outcome variable in the dataset. The algorithms must find the possible values of the outcome variable.
3.	**Reinforcement Learning Algorithms:** In many datasets, there could be uncertainty around
both input as well as the output variables. For example, consider the case of spell check in various text editors. If a person types “buutiful” in Microsoft Word, the spell check in Microsoft Word will immediately identify this as a spelling mistake and give options such as “beautiful”, “bountiful”, and dutiful”. Here the prediction is not one single value, but a set of values.
4.	**Evolutionary Learning Algorithms:** Evolutional algorithms are algorithms that imitate natural evolution to solve a problem. Techniques such as genetic algorithm and ant colony optimization fall under the category of evolutionary learning algorithms.

<img src="category1.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>


## A typical ML algorithm uses the following steps

1. Identify the problem or opportunity for value creation.
2. Identify sources of data (primary as well secondary data sources) and create a data lake
(integrated data set from different sources).
3. Pre-process the data for issues such as missing and incorrect data. Generate derived variables
(feature engineering) and transform the data if necessary. Prepare the data for ML model building.
4. Divide the datasets into subsets of training and validation datasets.
5. Build ML models and identify the best model(s) using model performance in validation data.
6. Implement Solution/Decision/Develop Product.

## Framework for develping ML models

The framework for ML algorithm development can be divided into five integrated stages:
- problem and opportunity identification, 
- collection of relevant data, 
- data pre-processing, 
- ML model building, and 
- model deployment.

<img src="frameworkML.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

# Python and data science

![image](https://user-images.githubusercontent.com/15100077/209703557-f22b143b-8b42-4c5d-b8dd-f180522f33d8.png)


* Data science projects need extraction of data from various sources, data cleaning, data imputation beside model building, validation, and making predictions. 
* Data analysis is mostly an iterative process, where lots of exploration needs to be done in an ad-hoc manner. 
* Python being an interpreted language provides an interactive interface for accomplishing this. Python is an interpreted, high-level, general-purpose programming language. 
* Python’s strong community, continuously evolves its data science libraries and keeps it cutting edge.
* It has libraries for **linear algebra computations**, **statistical analysis**, **machine learning**, **visualization**, **optimization**, **stochastic models**, etc.

<img src="python-libraries.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

(**Reference for the figure:** Machine learning using Python, Manaranjan Pradhan & U Dinesh Kumar)

## Core Python Libraries for Data Analysis

| Areas of Application | Library | Description | 
|----------------------|---------|-------------|
| Statistical Computations | [SciPy](www.scipy.org) | SciPy contains modules for optimization and computation. It provides libraries for several statistical distributions and statistical tests. |
| Statistical Modelling | [StatsModels](www.statsmodels.org/stable/index.html) | StatsModels is a Python module that provides classes and functions for various statistical analyses. | 
| Mathematical Computations | [NumPy](www.numpy.org)| NumPy is the fundamental package for scientific computing involving large arrays and matrices. It provides useful mathematical computation capabilities. |
| Data Structure Operations (Dataframes) | [Pandas](https://pandas.pydata.org/) | Pandas provides high-performance, easy-to-use data structures called DataFrame for exploration and analysis. DataFrames are the key data structures that feed into most of the statistical and machine learning models. |
| Visualization | [Matplotlib](https://matplotlib.org/)| Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python|
| More elegant Visualization | [Seaborn](https://seaborn.pydata.org/) | Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics. |
| Machine Learning Algorithm | [Scikit-learn (aka sklearn)](https://scikit-learn.org/stable/) | Scikit-learn provides a range of supervised and unsupervised learning algorithms. |
| IDE (Integrated Development Environment) | [Jupyter Notebook](jupyter.org) | The Jupyter Notebook is an opensource web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.|

# Statistics

## Descriptive statisctics

## Probability diestributions and hypotehsis testing

## Linear Regression

## Advanced Machine learning






# Reference

1. Machine learning using python, Manarajan Pradhan, U Dinesh Kumar
2. Please follow lecture series provided at [Jovian.ml](https://jovian.ai/), i.e. [Data Analysis with Python: Zero to Pandas](https://jovian.ai/learn/data-analysis-with-python-zero-to-pandas)