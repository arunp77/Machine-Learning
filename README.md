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

- **Mean or expectation value:** 
  The expected value, or mean, of a random variable—denoted by E(x) or μ—is a weighted average of the values the random variable may assume. The formulas for computing the expected values of discrete and continuous random variables are given by:

    E(x)= $\sum_i$ $x_i$ $p_i(x)$ $~~~~~~~~~~~~~~~~~~~~~~~~$  for discrete variables

    E(x)= $\int$ dx x p(x) $~~~~~~~~~~~~~~~~~~~~~~~~~~~$  for continuous variables

    If $x_1$, $x_2$, $x_3$, ..... $x_i$ ...., $x_k$ have frequency $f_1$, $f_2$, $f_3$,…… $f_k$ then 

    𝐸(𝑥) = $\sum_i$ $\frac{f_i x_i}{N}$.

- **Variance:**
    * In statistics, variance refers to the spread of a data set. It’s a measurement used to identify how far each number in the data set is from the mean.
    * The larger the variance, the more spread in the data set.
    * A large variance means that the numbers in a set are far from the mean and each other. A small variance means that the numbers are closer together in value.
    * Variance is calculated by taking the differences between each number in a data set and the mean, squaring those differences to give them positive value, and dividing the sum of the resulting squares by the number of values in the set.
    * Advantage: One of the primary advantages of variance is that it treats all deviations from the mean of the data set in the same way, regardless of direction. This ensures that the squared deviations cannot sum to zero, which would result in giving the appearance that there was no variability in the data set at all.
    * **Disadvantage:** One of the most commonly discussed disadvantages of variance is that it gives added weight to numbers that are far from the mean, or outliers. Squaring these numbers can at times result in skewed interpretations of the data set as a whole.
    * **Formula:** The variance of a random variable, denoted by Var(x) or σ2, is a weighted average of the squared deviations from the mean. The formulas for computing the variances of discrete and continuous random variables are given by:

        Var(x) = $\sigma^2=\sum_i (x_i-\mu)^2 P_i(x) ~~~~~~~~$  (for discrete variables)

        Var(x) = $\int dx ~ (x-\mu)^2 ~ p(x) ~~~~~~~~$  (for continuous variables)
    
    In this formula, $x$ represents an individual data point, $\mu$ represents the mean of the data points, and $n$ represents the total number of data points.

    <img src="https://miro.medium.com/max/720/1*4ct-L3QpNuiAsR10kGKGoQ.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
    
    ([Reference for the figure](https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1))

- **Standard deviation:**
    The standard deviation, denoted σ, is the positive square root of the variance., i.e. $\sigma= \sqrt{Var(x)}$. Since the standard deviation is measured in the same units as the random variable and the variance is measured in squared units, the standard deviation is often the preferred measure.

- **Covariance:** 
    * Covariance provides insight into how two variables are related to one another.
    * More precisely, covariance refers to the measure of how two random variables in a data set will change together.
    * A positive covariance means that the two variables at hand are positively related, and they move in the same direction.
    * A negative covariance means that the variables are inversely related, or that they move in opposite directions.
    * A zero covariance means that the variables are not related to each other.

        <img src="https://media.geeksforgeeks.org/wp-content/uploads/Covar.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

        Cov(X, Y) = $\frac{\sum_i^n (x_i-\bar{x})(y_i-\bar{y})}{N-1}$
     
        In this formula, $X$ represents the independent variable, $Y$ represents the dependent variable, $N$ represents the number of data points in the sample, $\bar{x}$ represents the mean of the $X$, and $\bar{y}$ represents the mean of the dependent variable $Y$. Note that while calculating a sample variance in order to estimate a population variance, the denominator of the variance equation becomes N – 1. This removes bias from the estimation.
- **Correlation:**
    * Covariance and correlation both primarily assess the relationship between variables.
    * The closest analogy to the relationship between them is the relationship between the variance and standard deviation.
    * Covariance measures the total variation of two random variables from their expected values. Using covariance, we can only gauge the direction of the relationship (whether the variables tend to move in tandem or show an inverse relationship). However, it does not indicate the strength of the relationship, nor the dependency between the variables.
    * On the other hand, correlation measures the strength of the relationship between variables. Correlation is the scaled measure of covariance. It is dimensionless. In other words, the correlation coefficient is always a pure value and not measured in any units.

    <img src="https://media.geeksforgeeks.org/wp-content/uploads/Correl.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

     ([for reference click the website](https://www.geeksforgeeks.org/robust-correlation/))

    * **Relation between the covariance and correlation:**

        $\rho(X,Y)=\frac{Cov(X,Y)}{\sigma_X \sigma_Y}$

        Where $ρ(X,Y)$ – is the correlation between the variables X and Y
        
        COV(X,Y) – is the covariance between the variables X and Y

        $\sigma_X$ – is the standard deviation of the X-variable

        $\sigma_Y$– is the standard deviation of the Y-variable

    * **Advantages of the Correlation Coefficient:**
        1. Covariance can take on practically any number while a correlation is limited: -1 to +1.
        2. Because of its numerical limitations, correlation is more useful for determining how strong the relationship is between the two variables.
        3. Correlation does not have units. Covariance always has units.
        4. Correlation isn’t affected by changes in the centre (i.e. mean) or scale of the variables.

# **Probability distributions and hypothesis testing**

### **Probability**

- Probability is a subject that deals with uncertainty. 
- In everyday terminology, probability can be thought of as a numerical measure of the likelihood that a particular event will occur.
- Probability values are assigned on a scale from `0` to `1`, with values near `0` indicating that an event is unlikely to occur and those near `1` indicating that an event is likely to take place.
- Suppose that an event `E` can happen in `h` ways out of a total of `n` possible equally likely ways. Then the probability of occurrence of the event (called its success) is denoted by
    
    $p=Pr\{E\}=\frac{h}{n} ~~~~~~~~~~~~~~~$ (success probability)

- The probability of non-occurrence of the event (called its failure) is denoted by

    $𝑞=1−𝑝 \rightarrow 𝑝+𝑞=1 $

### **Conditional probability; Independent and dependent events**
- If $E_1$ and $E_2$ are two events, the probability that $E_2$ occurs given that $E_1$ has occurred is denoted by $Pr\{E_2|E_1\}$, or $Pr\{E_2 ~\text{given} ~E_1\}$, and is called the conditional probability of $E_2$ given that $E_1$ has occurred.
- If the occurrence or non-occurrence of $E_1$ does not affect the probability of occurrence of $E_2$, then $Pr\{E_2 | E_1\}=Pr\{E_2\}$ and we say that $E_1$ and $E_2$ are independent events, they are dependents.
- If we denote by $(E_1 E_2)$ the event that "both $E_1$ and $E_2$ occur,’’ sometimes called a compound event, then

    $Pr \{𝐸1𝐸2\}=Pr\{𝐸_1\} Pr\{𝐸_2|𝐸_1\}$

- Similarly for three events $(𝐸_1 𝐸_2 𝐸_3)$ 

    $Pr {𝐸1𝐸2𝐸3}=Pr{𝐸1}Pr {𝐸2|𝐸1}Pr {𝐸3|𝐸2𝐸1}$

    If these events are independent, then 
    
    $Pr\{𝐸_1 𝐸_2\}=Pr\{𝐸_1\} Pr\{𝐸_2\}.$
    
    Similarly 
    
    $Pr\{𝐸_1 𝐸_2 𝐸_3\}=Pr\{𝐸_1\} Pr\{𝐸_2\} Pr\{𝐸_3\}$.


### **Mutually exclusive events:**
- Two or more events are called mutually exclusive if the occurrence of any one of them excludes the occurrence of the others. Thus if $E_1$ and $E_2$ are mutually exclusive events, then

    $Pr\{𝐸_1 𝐸_2\}=0.$

- If $E_1 + E_2$ denotes the event that ‘‘either $E_1$ or $E_2$ or both occur’’, then

    $Pr\{𝐸_1 + 𝐸_2\}=Pr\{𝐸_1\} + Pr\{𝐸_2\} − Pr\{𝐸_1 𝐸_2\}$.


## **Probability distribution**
The probability distribution for a random variable describes how the probabilities are distributed over the values of the random variable. Based on the variables, probability distributions are of two type mainly: 
1. Discrete probability distribution, and 
2. Continuous probability distribution.

<img src="https://miro.medium.com/max/720/1*4uD1j7NvakmaLmlpgGwk-A.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

### **Random variables**
- A random variable is a numerical description of the outcome of a statistical experiment.
- A random variable that may assume only a finite number or an infinite sequence of values is said to be discrete; one that may assume any value in some interval on the real number line is said to be continuous
    * Discrete random variables
    * Continuous random variables

    <img src="https://miro.medium.com/max/640/1*7DwXV_h_t7_-TkLAImKBaQ.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>



### **Discrete probability distribution:**
For a discrete random variable, $x$, the probability distribution is defined by a probability mass function, denoted by $p(x)$. This function provides the probability for each value of the random variable.

Following two conditions must be satisfied for $p(x)$
- $p(x)$ must be nonnegative for each value of the random variable, and
- the sum of the probabilities for each value of the random variable must equal one.

### **Continuous probability distribution**

- A continuous random variable may assume any value in an interval on the real number line or in a collection of intervals. Since there is an infinite number of values in any interval, it is not meaningful to talk about the probability that the random variable will take on a specific value; instead, the probability that a continuous random variable will lie within a given interval is considered.

- In the continuous case, the counterpart of the probability mass function is the probability density function, also denoted by $p(x)$. For a continuous random variable, the probability density function provides the height or value of the function at any particular value of $x$; it does not directly give the probability of the random variable taking on a specific value. However, the area under the graph of $p(x)$ corresponding to some interval, obtained by computing the integral of $p(x)$ over that interval, provides the probability that the variable will take on a value within that interval.

- A probability density function must satisfy two requirements:

    * $f(x)$ must be nonnegative for each value of the random variable, and
    * the integral over all values of the random variable must equal one.




























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