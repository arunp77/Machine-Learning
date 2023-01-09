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
* Statistics is the science of collecting, analysing, presenting, and interpreting data.
* Currently the need to turn the large amounts of data available in many applied fields into useful information has stimulated both theoretical and practical developments in statistics.
* Data are the facts and figures that are collected, analysed, and summarized for presentation and interpretation.
* Data may be classified as 
   a) Qualitative
   b) Quantitative
* Quantitative data measure either how much or how many of something, and qualitative data provide labels, or names, for categories of like items.
* Sample survey methods are used to collect data from observational studies, and experimental design methods are used to collect data from experimental studies.
* The area of descriptive statistics is concerned primarily with methods of presenting and interpreting data using graphs, tables, and numerical summaries. Whenever statisticians use data from a sample to make statements about a data, they are performing statistical inference.
* Estimation and hypothesis testing are procedures used to make statistical inferences.
* Methods of probability were developed initially for the analysis of gambling games.
* Probability plays a key role in statistical inference; it is used to provide measures of the quality and precision of the inferences.
* The statistical inference (अनुमान) are used primarily for single-variable studies, while others, such as regression and correlation analysis, are used to make inferences about relationships among two or more variables.

## Descriptive statisctics
- Descriptive statistics are tabular, graphical, and numerical summaries of data, i.e.,
    * tabular,
    * graphics method or
    * numerical (for example central tendency and variability).
- Descriptive statistics are brief informational coefficients that summarize a given data set, which can be either a representation of the entire data or a sample of a data.
- The purpose of descriptive statistics is to facilitate the presentation and interpretation of data.
- Descriptive statistics consists of three basic categories of measures:
    1. measures of central tendency: focus on the average or middle values of data sets
    2. measures of variability (or spread): aid in analysing how dispersed the distribution is for a set of data
    3. frequency distribution.
- Measures of central tendency describe the centre of the data set (mean, median, mode).
- Measures of variability describe the dispersion of the data set (variance, standard deviation).
- Measures of frequency distribution describe the occurrence of data within the data set (count)

1. **Tabular methods:**
    - The most commonly used tabular summary of data for a single variable is a frequency distribution.
    - A frequency distribution shows the number of data values in each of several nonoverlapping classes.
    - Another tabular summary, called a relative frequency distribution, shows the fraction, or percentage, of data values in each class.
    - The most common tabular summary of data for two variables is a cross tabulation, a two-variable analogue of a frequency distribution.
    - Constructing a frequency distribution for a quantitative variable requires more care in defining the classes and the division points between adjacent classes.
    - A frequency distribution would show the number of data values in each of these classes, and a relative frequency distribution would show the fraction of data values in each.
    - A cross tabulation is a two-way table with the rows of the table representing the classes of one variable and the columns of the table representing the classes of another variable.

2. **Graphical Methods:** A number of graphical methods are available for describing data.
    - Dot plots.
    - Histograms.
    - Box-whisker plots.
    - Scatter plots.
    - Bar charts.
    - Pie charts
3. **Numerical statistics:** This is broken down into-
    - Measures of central tendency: include the
        * mean,
        * median,
        * mode,
        * percentiles
    - **Measures of variability (spread):** include
        * standard deviation,
        * variance,
        * minimum and maximum variables,
        * kurtosis, and
        * skewness.
    - **Outliers:** Sometimes data for a variable will include one or more values that appear unusually large or small and out of place when compared with the other data values. These values are known as outliers and often have been erroneously included in the data set.
        * The mean and standard deviation are used to identify outliers.
        * A z-score can be computed for each data value.
        * With x representing the data value, x̄ the sample mean, and s the sample standard deviation, the z-score is given by

            $z=\frac{x-\bar{x}}{s}$
        * The z-score represents the relative position of the data value by indicating the number of standard deviations it is from the mean.
        *  A rule of thumb is that any value with a z-score less than −3 or greater than +3 should be considered an outlier (i.e. 𝑧<−3,𝑜𝑟 𝑧>+3).


## Probability diestributions and hypotehsis testing
### Probability
- Probability is a subject that deals with uncertainty. 
- In everyday terminology, probability can be thought of as a numerical measure of the likelihood that a particular event will occur.
- Probability values are assigned on a scale from 0 to 1, with values near 0 indicating that an event is unlikely to occur and those near 1 indicating that an event is likely to take place.
- Suppose that an event E can happen in h ways out of a total of n possible equally likely ways. Then the probability of occurrence of the event (called its success) is denoted by

    $p=Pr\{E\}=\frac{h}{n}~~~~~~~~~~~~~~~$ (success probability)
- The probability of non-occurrence of the event (called its failure) is denoted by

    $𝑞=1−𝑝 \rightarrow 𝑝+𝑞=1 $

### Conditional probability; Independent and dependent events:

## Linear Regression

## Advanced Machine learning





<!---------------------------- Reference ------------------------------>
# Reference

1. Machine learning using python, Manarajan Pradhan, U Dinesh Kumar
2. Please follow lecture series provided at [Jovian.ml](https://jovian.ai/), i.e. [Data Analysis with Python: Zero to Pandas](https://jovian.ai/learn/data-analysis-with-python-zero-to-pandas)
3. [Python roadmap](https://roadmap.sh/python/)
4. [Python tutorials at w3school](https://www.w3schools.com/python/default.asp)
5. [30 Days of Python](https://github.com/asabeneh/30-days-of-python)
6. [Python official documents](https://docs.python.org/3/tutorial/)
7. [Data science roadmaps](https://github.com/codebasics/py/blob/master/TechTopics/DataScienceRoadMap2020/data_science_roadmap_2020.md)
8. Statistics, Murray R. Spiegel, Larry J. Stephens