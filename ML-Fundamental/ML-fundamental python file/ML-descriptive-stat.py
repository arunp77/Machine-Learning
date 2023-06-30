# %% [markdown]
# # Statistics Tutorial
# 
# **Courtesy to the Notebook:** The Jupyter Notebook and dataset are also available on [Github repository](https://github.com/CarloLepelaars/stats_tutorial).

# %% [markdown]
# ## Table of contents
# 
# - [Statistics](#1)
#   - [Population vs. Sample](#1.1)
#   - [Types of statistics](#1.2)
#   - [Descriptive statistics](#1.3)
#   - [Probabillity](#1.4)
# - [Preparation](#2)
# - [Discrete and Continuous Variables](#3)
#   - PMF (Probability Mass Function)
#   - PDF (Probability Density Function)
#   - CDF (Cumulative Distribution Function)
# - [Distributions](#4)
#   - Uniform Distribution
#   - Normal Distribution
#   - Binomial Distribution
#   - Poisson Distribution
#   - Log-normal Distribution
# - [Summary Statistics and Moments](#5)
# - [Bias, MSE and SE](#6)
# - [Sampling Methods](#7)
# - [Covariance](#8)
# - [Correlation](#9)
# - [Linear Regression](#10)
#   - Anscombe's Quartet
# - [Bootstrapping](#11)
# - [Hypothesis Testing](#12)
#   - p-value
#   - q-q plot
# - [Outliers](#12)
#   - Grubbs Test
#   - Tukey's Method
# - [Overfitting](#20)
#   - Prevention of Overfitting
#   - Cross-Validation
# - [Generalized Linear Models (GLMs)](#13)
#   - Link Functions
#   - Logistic Regression
# - [Frequentist vs. Bayes](#14)
# - [Bonus: Free Statistics Courses](#15)
# - [Sources](#16)

# %% [markdown]
# # Descriptive statistics <a id="1.3"></a>
# 
# <img src="https://www.scribbr.de/wp-content/uploads/2023/01/Descriptive-statistics.webp?_ga=2.28769407.23142110.1678373880-143911539.1678373874" width="800" height="550" />
# 
# Descriptive statistics is a branch of statistics that deals with the collection, organization, analysis, and presentation of data. It involves summarizing and describing the main features of a dataset, such as the central tendency, variability, and distribution of the data.
# 
# Some common measures of descriptive statistics include:
# 
# 1. **Measures of central tendency:** These include the 
#     - mean, 
#     - median, and 
#     - mode,
#      
#     which represent the average or typical value of the data.
# 
# 2. **Measures of variability:** These include the 
#     - range, 
#     - variance, and 
#     - standard deviation, 
# 
#     which measure the spread or variability of the data.
# 
# 3. **Measures of distribution:** These include the 
#     - skewness and 
#     - kurtosis, 
# 
#     which describe the shape of the distribution of the data.
# 
# Descriptive statistics are often used to summarize and describe data in a way that is easily understandable and interpretable. They can be used to make comparisons between groups, identify patterns or trends in the data, and to detect outliers or anomalies that may need further investigation.
# 
# Descriptive statistics are commonly used in fields such as business, economics, psychology, sociology, and healthcare, among others. They are an important tool for making informed decisions and drawing meaningful conclusions from data.

# %% [markdown]
# ## 1. Measures of central tendency
# 
# Measures of central tendency are statistical measures that represent the typical or central value of a dataset. They are used to summarize and describe the main features of the data and to make comparisons between different groups or distributions.
# 
# There are three main measures of central tendency:
# 
# 1. **Mean:** The mean is the arithmetic average of a dataset and is calculated by adding up all the values in the dataset and dividing by the total number of values. If $x_1$, $x_2$, $x_3$, ..... $x_i$ ...., $x_k$ have frequency $f_1$, $f_2$, $f_3$,…… $f_k$ then 
# 
#     $\mu = \sum_i \frac{f_i x_i}{N}$,
# 
#     i.e. $\text{Mean} = \frac{\text{sum of all values}}{\text{total number of values}}$
#     
# 
#     **Example:** if we have a dataset of test scores for a class of students: 70, 80, 90, 85, and 75, we can calculate the mean by adding up all the scores and dividing by the total number of scores: Mean = (70 + 80 + 90 + 85 + 75) / 5 = 80. So the mean test score for the class is 80.
# 
#     The mean is commonly used in statistics to summarize and describe a dataset, and is often used as a benchmark for making comparisons between different groups or distributions. However, the mean can be affected by extreme values or outliers, which can skew the results. In such cases, it may be more appropriate to use other measures of central tendency, such as the median or mode, to represent the typical or central value of the dataset.
# 
# 2. **Median:** The median is the middle value of a dataset when the values are arranged in order of magnitude. It is used to represent the typical or central value when the data are skewed or have outliers.
# 
#     - **How to calculate?:** To calculate the median, follow these steps:
# 
#     1. Arrange the values in the dataset in order from smallest to largest (or vice versa).
#     2. If the dataset has an odd number of values, the median is the middle value. For example, in the dataset {1, 3, 5, 7, 9}, the median is 5 because it is the middle value.
#     3. If the dataset has an even number of values, the median is the average of the two middle values. For example, in the dataset {1, 3, 5, 7, 9, 11}, the two middle values are 5 and 7, so the median is (5+7)/2 = 6.
# 
#     The median is a useful measure of central tendency for datasets that have outliers or extreme values, as it is less sensitive to these values than the mean. Additionally, the median is appropriate for ordinal data, where the values have an inherent order but the distance between values is not meaningful (e.g. ranks, grades).
# 
# 3. **Mode:** The mode is the value that occurs most frequently in a dataset. It is used to represent the most common or typical value when the data are categorical or have a discrete distribution. Unlike mean and median, the mode does not take into account the actual numerical values of the data points, but only their frequencies.
# 
#     - **How to calculate?:** The mode can be calculated for any type of data, including nominal, ordinal, interval, and ratio data. In a dataset with a single mode, there is only one value that occurs more frequently than any other value. However, it is also possible to have datasets with multiple modes, where there are several values that occur with the same highest frequency.
# 
#         - **Nominal data:** In nominal data, each value represents a category or a label, such as colors or types of fruits. To find the mode, simply identify the category that occurs most frequently in the dataset.
#         - **Ordinal data:** In ordinal data, each value represents a category that can be ordered or ranked, such as levels of education or ranks in a competition. To find the mode, identify the category with the *highest frequency*.
#         - **Interval or ratio data:** In interval or ratio data, each value represents a numerical quantity, such as heights or weights. To find the mode, first group the data into intervals or bins, then identify the interval with the *highest frequency*. If there are multiple intervals with the same highest frequency, the mode is considered to be multimodal.
# 
#         **Example:** Here is an example of how to calculate the mode for a dataset of heights:
# 
#         1. Sort the dataset in ascending order: 62, 64, 66, 66, 68, 68, 68, 70, 70, 72.
# 
#         2. Count the frequency of each value: 62 (1), 64 (1), 66 (2), 68 (3), 70 (2), 72 (1).
# 
#         3. Identify the value with the highest frequency: 68.
# 
#         4. The mode of the dataset is 68, indicating that 68 is the most common height in the dataset.
# 
#         Note that in some cases, a dataset may not have a mode if all the values occur with the same frequency. In other cases, the mode may not be a meaningful measure of central tendency if there are extreme values or outliers that skew the distribution.
#         
#     - The mode is often used in conjunction with other measures of central tendency, such as mean and median, to gain a better understanding of the underlying distribution of the data. It is especially useful for describing skewed distributions, where the mean and median may not accurately represent the central tendency of the data.
# 
# The choice of which measure of central tendency to use depends on the nature of the data and the research question. The mean is commonly used when the data are normally distributed and have a symmetrical distribution. The median is used when the data are skewed or have outliers. The mode is used when the data are categorical or have a discrete distribution.
# 
# Measures of central tendency are an important tool for summarizing and describing data and for making comparisons between different groups or distributions. However, they should be used in conjunction with other statistical measures, such as measures of variability and distribution, to provide a more complete picture of the data.

# %% [markdown]
# ## 2. Measures of variability
# 
# Measures of variability are statistical measures that describe the spread or dispersion of a dataset. Some common measures of variability include:
# 
# - **Range:** The range is the difference between the maximum and minimum values in a dataset. It is the simplest measure of variability but can be heavily influenced by outliers. It is calculated using the formula:
# 
#     $\text{Range} = \text{max value} - \text{min value}$
# 
#     **Example:** if a dataset consists of the following values: 2, 5, 7, 8, 12, the range would be calculated as:
# 
#     Range = 12 - 2 = 10
# 
# - **Variance:** The variance measures how much the values in a dataset vary from the mean. It is calculated by taking the average of the squared differences between each value and the mean. It is calculated using the formula:
# 
#     $\text{Variance} = \sum \frac{(x-\mu)^2}{n}$
# 
#     Variance is commonly used in statistical analysis and can be influenced by extreme values.
# 
#     where $\sum$ represents the sum of, $x$ represents each value in the dataset, $\mu$ represents the mean of the dataset, and $n$ represents the number of values in the dataset.
# 
#     **Example:** If a dataset consists of the following values: 10, 15, 20, 25, 30, and the mean is calculated to be 20, the variance would be calculated as:
# 
#     Variance = [(10-20)² + (15-20)² + (20-20)² + (25-20)² + (30-20)²] / 5 = 200 / 5 = 40
# 
# - **Standard deviation:** Standard deviation is a measure of how spread out a set of data is from its mean or average. It tells you how much the data deviates from the average. A low standard deviation indicates that the data is clustered closely around the mean, while a high standard deviation indicates that the data is spread out over a larger range of values. It is a commonly used measure of variability and is often preferred over the variance because it is expressed in the same units as the original data. The formula for standard deviation is:
# 
#     $\sigma = \sqrt{\frac{\sum (x-\mu)^2}{n}}$
# 
#     (Standard deviation of the population)
# 
#     where:
# 
#     - $\sigma$ is the standard deviation
#     - $\sum$ is the sum of all the data points
#     - $x$ is each individual data point
#     - $\mu$ is the mean or average of the data
#     - $n$ is the total number of data points
# 
#     **Method:** To find the standard deviation, you first subtract each data point from the mean, square the result, sum up all the squared differences, divide by the total number of data points, and finally, take the square root of the result.
# 
#     **Example:** let's say you have the following set of data: {2, 4, 6, 8, 10}. 
#     - First, find the mean: $\mu = (2 + 4 + 6 + 8 + 10) / 5 = 6$. 
#     - Next, calculate the difference between each data point and the mean: (2 - 6) = -4, (4 - 6) = -2, (6 - 6) = 0, (8 - 6) = 2, (10 - 6) = 4. 
#     - Then, square each of these differences and add up all the squared differences: (-4)² = 16, (-2)² = 4, (0)² = 0, (2)² = 4, (4)² = 6.
#     - Divide by the total number of data points: 16 + 4 + 0 + 4 + 16 = 40.
#     - Finally, take the square root of the result: 40 / 5 = 8.
#     - So, the standard deviation of this set of data is approximately 2.83.
# 
# 
# 
# - **Interquartile range (IQR):** The IQR is the difference between the third quartile (the value above which 75% of the data falls) and the first quartile (the value below which 25% of the data falls). It is a measure of the spread of the middle 50% of the data and is less influenced by extreme values than the range.
# 
#     The formula for calculating the IQR is as follows:
# 
#     $\text{IQR} =Q_3 -Q_1$
# 
#     Where $Q_3$ is the third quartile and $Q_1$ is the first quartile. The quartiles are calculated by dividing the dataset into four equal parts. The first quartile (i.e. $Q_1$) represents the 25th percentile of the dataset, and the third quartile (i.e. $Q_3$) represents the 75th percentile.
# 
#     <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/1920px-Boxplot_vs_PDF.svg.png" width="450" height="460" />
# 
#     **Example:** Consider the following dataset: 1, 3, 5, 6, 7, 8, 9, 10, 11, 15. 
#     - The first quartile ($Q_1$) is 4 and the third quartile ($Q_3$) is 10. Therefore, the IQR is:
#     $IQR = Q_3 - Q_1 = 10 - 4 = 6$
#     - This means that the middle 50% of the dataset (between the 25th and 75th percentiles) falls within a range of 6.
# 
#     > **Quartiles:** Quartiles are a way to divide a dataset into four equal parts or quarters. Quartiles are used to understand the distribution of a dataset and to calculate other measures of variability such as the interquartile range.
#     > There are three quartiles that divide a dataset into four parts:
#     >   - The first quartile ($Q_1$) is the 25th percentile of the dataset. It divides the dataset into the bottom 25% and the top 75%.
#     >   - The second quartile ($Q_2$) is the median of the dataset. It divides the dataset into two equal parts.
#     >   - The third quartile ($Q_3$) is the 75th percentile of the dataset. It divides the dataset into the bottom 75% and the top 25%.
#     > To calculate the quartiles, the dataset must be sorted in ascending order. The formula for finding the quartiles depends on the number of observations in the dataset. Here are the general formulas:
#     >   - $Q_1$ = ($n+1$)/4th observation
#     >   - $Q_2$ = ($n+1$)/2th observation
#     >   - $Q_3$ = 3($n+1$)/4th observation
#     > where $n$ is the total number of observations in the dataset.
#     > The quartiles are significant because they provide important information about the distribution of the data. They help us to understand the spread and central tendency of the dataset. The interquartile range, which is the difference between the third and first quartiles, is a measure of variability that is useful for identifying outliers and understanding the spread of the data. In addition, quartiles can be used to create box plots, which are a graphical representation of the distribution of the data.
# 
# - **Mean absolute deviation (MAD):** The mean absolute deviation (MAD) is a measure of variability that indicates how much the observations in a dataset deviate, on average, from the mean of the dataset. The MAD is the average of the absolute differences between each value and the mean. It is a robust measure of variability that is less sensitive to outliers than the variance and standard deviation.
# 
#     **Formula:** MAD is calculated by finding the absolute difference between each data point and the mean, then taking the average of those absolute differences. The formula for calculating MAD is as follows:
# 
#     $\text{MAD} = \frac{1}{n}\sum_i^n |x_i - \mu|$
# 
#     Where $n$ is the number of observations in the dataset, $x_i$ is the value of the ith observation, $\mu$ is the mean of the dataset, and $\sum$ represents the sum of the absolute differences.
# 
#     **Example:** For example, consider the following dataset: 2, 3, 5, 6, 7, 8, 9, 10, 11, 15
# 
#     To calculate the MAD, we first find the mean of the dataset:
# 
#     $\mu$ = (2 + 3 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 15) / 10 = 7.6
# 
#     Next, we find the absolute difference between each data point and the mean: |2 - 7.6| = 5.6, |3 - 7.6| = 4.6, |5 - 7.6| = 2.6, |6 - 7.6| = 1.6, |7 - 7.6| = 0.6, |8 - 7.6| = 0.4, |9 - 7.6| = 1.4, |10 - 7.6| = 2.4, |11 - 7.6| = 3.4, |15 - 7.6| = 7.4.
# 
#     Then we take the average of those absolute differences:
# 
#     MAD = (1/10) * (5.6 + 4.6 + 2.6 + 1.6 + 0.6 + 0.4 + 1.4 + 2.4 + 3.4 + 7.4) = 3.34
# 
#     The MAD for this dataset is 3.34, which means that, on average, each observation deviates from the mean by approximately 3.34.
# 
# 
# These measures of variability are useful in providing information about how much the values in a dataset vary from each other. The appropriate measure to use depends on the specific characteristics of the data and the research question being asked.

# %% [markdown]
# ## 3. Measures of distribution
# 
# Skewness and kurtosis are two statistical measures used to describe the shape of a probability distribution.
# 
# - **Skewness:** Skewness measures the degree of asymmetry in a distribution. A distribution with a positive skewness has a longer tail on the positive side of the mean, while a negative skewness means the tail is longer on the negative side of the mean. A perfectly symmetrical distribution has a skewness of zero.
# 
#     <img src="ML-image/Pos-skew.jpeg" width="400" height="230" />
# 
#     <img src="ML-image/neg-skew.jpeg" width="400" height="230" />
#     
#     <img src="ML-image/zero-skew.png" width="420" height="230" />
# 
#     ([Image credit](https://www.analyticsvidhya.com/blog/2021/08/a-guide-to-complete-statistics-for-data-science-beginners/))
# 
#     Here are three common measures of skewness:
#     
#     1. **Pearson's moment coefficient of skewness:**
# 
#         $\text{Skewness} = \frac{3(\text{Mean}-\text{Mode})}{\text{Standard deviation}}$.
# 
#         This is the formula described above that uses the third moment of the distribution to measure skewness.
#     
#     2. **Sample skewness:** This is a formula that uses the sample mean, standard deviation, and third central moment to estimate the skewness of the distribution. The formula for sample skewness is:
# 
#         $\text{Skewness} = \frac{n}{(n - 1) * (n - 2)}\times \left(\frac{\sum(x_i - \mu)^3}{\sigma_s^3}\right)$     
#         
#         (known as Fisher-Pearson standardized moment coefficient)
# 
#         where $n$ is the sample size, $\mu$ is the sample mean, $x_i$ is the $i-th$ observation in the sample, and $\sigma_s$ is the sample standard deviation.
# 
#         > **Sample standard deviation:** The sample standard deviation measures the spread of the data around the mean. It tells you how much the individual data points deviate from the mean, on average. Note that the sample standard deviation is calculated using $n - 1$ in the denominator instead of $n$, which is known as Bessel's correction. This is because using $n$ instead of $n - 1$ tends to underestimate the true variance of the population from which the sample was drawn.
#         > 
#         > Formula: $\sigma_s = \sqrt{\frac{\sum_i^n (x_i-\mu)}{n-1}}$
#         >
#         > Care should be taken when getting the standard deviation because the standard deviation is different from the standard deviation of a sample. If the problem describes a situation dealing with a sample or subset of a group, then the sample standard deviation, s, should be used.
# 
#         **How to Transform Skewed Data?** The graph of skewed data may be transformed into a symmetrical, balanced bell curve shape by changing the data using various methods. The selection of which method to use depends on the characteristic of the data set and its behavior. Here are the most common ways of correcting the skewness of data distribution:
# 
#         - **Logarithmic transformation:** If the data are positively skewed (i.e., skewed to the right), taking the logarithm of the data can help to reduce the skewness. This can be especially useful when dealing with data that are highly variable and cover a wide range of values. A logarithmic transformation compresses the higher values and stretches the lower values, so the distribution can become more symmetrical.
#         - **Square root transformation:** Similar to the logarithmic transformation, taking the square root of the data can also help to reduce positive skewness. This transformation can be useful when the values are strictly positive and the range of values is limited.
#         - **Inverse transformation:** In some cases, taking the reciprocal of the data (i.e., 1/x) can help to reduce negative skewness (i.e., skewed to the left). This transformation can be useful when the values are strictly positive and the distribution is highly skewed.
#         - **Box-Cox transformation:** The Box-Cox transformation is a family of power transformations that can be used to adjust the shape of a distribution to be more normal. The transformation involves raising the data to a power (λ), which is chosen to optimize the normality of the distribution. The optimal value of λ can be found through maximum likelihood estimation.
# 
#         It is important to note that transforming the data may not always be necessary or appropriate. The choice of transformation depends on the distribution of the data, the research question, and the statistical model being used. In addition, some transformations may change the interpretation of the data, so it is important to carefully consider the implications of any transformations before applying them.
# 
#         **Why transformation is done?:** Transformation is required for a skewed data for several reasons:
# 
#         - **To meet statistical assumptions:** Many statistical analyses assume that the data are normally distributed. Skewed data violate this assumption and can lead to biased or incorrect results. Transforming the data can make it more normally distributed and improve the validity of the analysis.
#         - **To improve interpretability:** Skewed data can be difficult to interpret and can make it hard to compare different groups or variables. Transforming the data can make the distribution more symmetrical and easier to understand.
#         - **To reduce the influence of outliers:** Skewed data can be particularly sensitive to outliers, which can have a large impact on the mean and standard deviation. Transforming the data can reduce the influence of outliers and make the analysis more robust.
#         - **To improve model performance:** In some cases, transforming the data can improve the performance of statistical models. For example, linear regression models assume that the residuals are normally distributed. Transforming the response variable or predictor variables can improve the normality of the residuals and improve the model fit.
#         
#         It's important to note that not all skewed data require transformation. The decision to transform the data depends on the specific research question, the nature of the data, and the statistical model being used. In some cases, it may be more appropriate to use non-parametric methods that do not rely on the assumption of normality.
#     
#     3. **Quartile skewness:** This measure of skewness is based on the difference between the median and the mode of the distribution. Specifically, the quartile skewness is defined as:
# 
#         $\text{Skewness} = \frac{Q_1 + Q_3 - 2 * \text{median}}{Q_3 - Q_1}$
# 
#         where $Q_1$ and $Q_3$ are the first and third quartiles of the distribution, and the median is the second quartile.
#     
#     Each of these measures of skewness has its own strengths and weaknesses, and the choice of measure may depend on the context and purpose of the analysis.
# 
# - **Kurtosis:** Kurtosis is a statistical measure that describes the shape of a distribution by measuring the degree of peakedness or flatness of the distribution compared to the normal distribution. A distribution with high kurtosis indicates that the data have many outliers or extreme values, while a distribution with low kurtosis indicates that the data are more spread out and have fewer outliers.
# 
#     **How to calculate kurtosis:** Mathematically speaking, kurtosis is the standardized fourth moment of a distribution. Moments are a set of measurements that tell you about the shape of a distribution.
# 
#     Moments are standardized by dividing them by the standard deviation raised to the appropriate power.
# 
#     - **Kurtosis of a population:** The following formula describes the kurtosis of a population:
#     
#         Kurtosis = $\tilde{\mu}_4 = \frac{\mu_4}{\sigma^4}$. 
# 
#         Where:
# 
#         - $\tilde{\mu}_4$ is the standardized fourth moment
#         - $\mu_4$ is the unstandardized central fourth moment
#         - $\sigma$ is the standard deviation
# 
#     - **Kurtosis of a sample:** The kurtosis of a sample is an estimate of the kurtosis of the population.
# 
#         It might seem natural to calculate a sample’s kurtosis as the fourth moment of the sample divided by its standard deviation to the fourth power. However, this leads to a biased estimate.
# 
#         The formula for the unbiased estimate of excess kurtosis includes a lengthy correction based on the sample size:
# 
#         kurtosis $ = \frac{(n+1)(n-1)}{(n-1)(n-3)}\frac{\sum (x_i -\mu)^4}{(\sum (x_i - \mu)^2)^2}- 3\frac{(n-1)^2}{(n-2)(n-3)}$
# 
#         Where
# 
#         - $n$ is the sample size
#         - $x_i$ are observations of the variable x
#         - $\mu$ is the mean of the variable x.
# 
#     **Types of kurtosis:** Examples of kurtosis include:
# 
#     1. **Mesokurtic distribution:** A mesokurtic distribution has a kurtosis value of zero and is similar in shape to the normal distribution. It has a moderate degree of peakedness and is neither too flat nor too peaked.
# 
#     2. **Leptokurtic distribution:** A leptokurtic distribution has a kurtosis value greater than zero and is more peaked than the normal distribution. It has heavier tails and more outliers than a normal distribution.
# 
#     3. **Platykurtic distribution:** A platykurtic distribution has a kurtosis value less than zero and is flatter than the normal distribution. It has fewer outliers and less extreme values than a normal distribution.
# 
#      <img src="https://cdn.scribbr.com/wp-content/uploads/2022/07/The-difference-between-skewness-and-kurtosis.webp" width="700" height="500" />
# 
#     It's important to note that kurtosis can only be interpreted in the context of the specific distribution being analyzed. A high or low kurtosis value does not necessarily indicate that the data are problematic or that any action needs to be taken. Rather, kurtosis can provide insight into the shape of the distribution and can help to identify potential issues with the data.
#     

# %% [markdown]
# # Probability distributions and hypothesis testing
# 
# ## Probability
# 
# - Probability is a subject that deals with uncertainty. 
# - In everyday terminology, probability can be thought of as a numerical measure of the likelihood that a particular event will occur.
# - Probability values are assigned on a scale from `0` to `1`, with values near `0` indicating that an event is unlikely to occur and those near `1` indicating that an event is likely to take place.
# - Suppose that an event `E` can happen in `h` ways out of a total of `n` possible equally likely ways. Then the probability of occurrence of the event (called its success) is denoted by
#     
#     $p=Pr\{E\}=\frac{h}{n} ~~~~~~~~~~~~~~~$ (success probability)
# 
# - The probability of non-occurrence of the event (called its failure) is denoted by
# 
#     $𝑞=1−𝑝 \rightarrow 𝑝+𝑞=1 $
# 
# ### Conditional probability; Independent and dependent events
# 
# - If $E_1$ and $E_2$ are two events, the probability that $E_2$ occurs given that $E_1$ has occurred is denoted by $Pr\{E_2|E_1\}$, or $Pr\{E_2 ~\text{given} ~E_1\}$, and is called the conditional probability of $E_2$ given that $E_1$ has occurred.
# 
# - If the occurrence or non-occurrence of $E_1$ does not affect the probability of occurrence of $E_2$, then Pr{$E_2$ | $E_1$}=Pr{$E_2$} and we say that $E_1$ and $E_2$ are independent events, they are dependents.
# 
# - If we denote by ( $E_1$ $E_2$) the event that "both $E_1$ and $E_2$ occur,’’ sometimes called a compound event, then
# 
#     Pr{ $𝐸_1$ $𝐸_2$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ | $𝐸_1$ }
# 
# - Similarly for three events $(𝐸_1 𝐸_2 𝐸_3)$ 
# 
#     Pr{ $𝐸_1$ $𝐸_2$ $𝐸_3$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ | $𝐸_1$ } Pr{ $𝐸_3$ | $𝐸_2$ $𝐸_1$ }
# 
#     If these events are independent, then 
# 
#     Pr{ $𝐸_1$ $𝐸_2$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ }.
# 
#     Similarly 
# 
#     Pr{ $𝐸_1$ $𝐸_2$ $𝐸_3$}=Pr{ $𝐸_1$ } Pr{ $𝐸_2$ } Pr{ $𝐸_3$}.
# 
# ### Mutually exclusive events
# 
# - Two or more events are called mutually exclusive if the occurrence of any one of them excludes the occurrence of the others. Thus if $E_1$ and $E_2$ are mutually exclusive events, then
# 
#     Pr{ $𝐸_1$ $𝐸_2$ } = 0.
# 
# - If $E_1 + E_2$ denotes the event that ‘‘either $E_1$ or $E_2$ or both occur’’, then
# 
#     Pr{ $𝐸_1$ + $𝐸_2$ } = Pr{ $𝐸_1$ } + Pr{ $𝐸_2$ } − Pr{ $𝐸_1$ $𝐸_2$ }.

# %% [markdown]
# ## Probability distributions <a id="1.4"></a>
# 
# ### Random variables
# 
# - A random variable is a numerical description of the outcome of a statistical experiment.
# - A random variable that may assume only a finite number or an infinite sequence of values is said to be discrete; one that may assume any value in some interval on the real number line is said to be continuous
# * Discrete random variables
# * Continuous random variables
# 
# <img src="https://miro.medium.com/max/640/1*7DwXV_h_t7_-TkLAImKBaQ.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
# 
# A probability distribution is a function that describes the likelihood of each possible outcome in a random experiment. Variables that follow a probability distribution are called random variables. In other words, it is a mathematical function that assigns probabilities to all possible outcomes of a random variable.

# %% [markdown]
# ## Types of probability distributions
# 
# There are two types of probability distributions: 
# 
# ### 1. Discrete
# 
# A discrete probability distribution assigns probabilities to a finite or countably infinite number of possible outcomes. There are several types of discrete probability distributions, including:
# 
# 1. **Bernoulli distribution:** The Bernoulli distribution is a simple probability distribution that describes the probability of success or failure in a single trial of a binary experiment. The Bernoulli distribution has two possible outcomes: success (with probability $p$) or failure (with probability $1-p$). The formula for the Bernoulli distribution is:
# 
#     $P(X=x) = p^x \times (1-p)^{(1-x)}$
# 
#     where $X$ is the random variable, $x$ is the outcome (either `0` or `1`), and $p$ is the probability of success.
# 
#     <img src="ML-image/Berno-pmf.png" width="450" height="300" />
# 
# 2. **Binomial distribution:** The binomial distribution describes the probability of getting a certain number of successes in a fixed number of independent trials of a binary experiment. The binomial distribution has two parameters: $n$, the number of trials, and $p$, the probability of success in each trial. The formula for the binomial distribution is:
# 
#     $P(X=x) = ^nC_x ~ p^x ~ (1-p)^{(n-x)}$
# 
#     where $X$ is the random variable representing the number of successes, $x$ is the number of successes, $n$ is the number of trials, $p$ is the probability of success, and $^nC_x = \frac{n!}{x! (n-x)!}$ is the binomial coefficient, which represents the number of ways to choose $x$ objects from a set of n objects.
# 
# |     Statistics          |       Formula      |
# |-------------------------|--------------------|
# |            Mean         |        $\mu=n p$   |
# |      Variance           | $\sigma^2 = nn p (1-p)$ |
# | Standard deviation | $\sigma = \sqrt{n p (1-p)}$ |
# | Moment coefficient of skewness | $\alpha_3 = \frac{1-p-p}{\sqrt{n p (1-p)}}$ |
# | Moment coefficient of Kurtosis | $\alpha_4 = 3+ \frac{1-6 p (1-p)}{n p (1-p)}$ |
# 
# <img src="ML-image/Binomial.png" width="550" height="350" />
# 
# [A link to generate the plot](https://homepage.divms.uiowa.edu/~mbognar/applets/bin.html)
# 
# 3. **Poisson distribution:** The Poisson distribution is used to describe the probability of a certain number of events occurring in a fixed time interval when the events occur independently and at a constant rate. The Poisson distribution has one parameter: $\lambda$, which represents the expected number of events in the time interval. The formula for the Poisson distribution is:
# 
#     $P(X=x) = e^{-λ} \frac{λ^x}{x!}$
# 
#     where $X$ is the random variable representing the number of events, $x$ is the number of events, $e$ is the mathematical constant, $\lambda$ is the expected number of events, and $x!$ is the factorial function.
# 
# | Statistics     |    Formula   |
# |----------------|--------------|
# | Mean | $\mu=\lambda $ |
# | Variance | $\sigma^2 = \lambda$ |
# |Standard deviation | $\sigma = \sqrt{\lambda}$ |
# | Moment coefficient of skewness | $\alpha_3 = \frac{1}{\sqrt{\lambda}}$ |
# | Moment coefficient of Kurtosis | $\alpha_4 = 3+ \frac{1}{\lambda}$ |
# 
# <img src="ML-image/Possion.png" width="550" height="350" />
# 
# [A link to generate the plot](https://homepage.divms.uiowa.edu/~mbognar/applets/pois.html)
# 
# The PMF is a function that gives the probability of each possible value of the random variable. The PMF for the Bernoulli distribution has two values ($p$ and $1-p$), the PMF for the binomial distribution has $n+1$ values (corresponding to the number of successes), and the PMF for the Poisson distribution has an infinite number of values (corresponding to the number of events).
# 
# > **probability mass functions (PMFs):** A probability mass function (PMF) is a function that gives the probability of each possible value of a discrete random variable. It is a way of summarizing the probability distribution of a discrete random variable.
# > The PMF is defined for all possible values of the random variable and satisfies the following properties:
# > 
# > - The value of the PMF at any possible value of the random variable is a non-negative number.
# > - The sum of the PMF over all possible values of the random variable is equal to one.
# >
# > The PMF is often represented graphically using a histogram or bar graph. The height of each bar represents the probability of the corresponding value of the random variable.
# > 
# > **Example:** consider a fair six-sided die. The random variable X can take on values of 1, 2, 3, 4, 5, or 6, each with probability 1/6. The PMF for this random variable is:
# > 
# > P(X = 1) = 1/6
# >
# > P(X = 2) = 1/6
# >
# > P(X = 3) = 1/6
# >
# > P(X = 4) = 1/6
# >
# > P(X = 5) = 1/6
# >
# > P(X = 6) = 1/6
# >
# > This PMF is illustrated in the following figure:

# %% [markdown]
# ### 2. Continuous
# 
# Continuous probability distributions are used to model continuous random variables, which can take on any value in a given range. Unlike discrete random variables, which take on only a finite or countably infinite set of possible values, continuous random variables can take on an uncountably infinite set of possible values. 
# 
# There are several common continuous probability distributions, including:
# 
# 1. **Normal distribution:** also known as the Gaussian distribution, this is a bell-shaped distribution that is symmetric around the mean. It is commonly used to model measurements that are expected to be normally distributed, such as heights or weights of individuals in a population. The probability density function (PDF) of the normal distribution is:
# 
#     $$f(x; μ, σ) = \frac{1}{\sigma \sqrt{2\pi}} \text{Exp}\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
# 
#     where $x$ is the random variable, $\mu$ is the mean, $\sigma$ is the standard deviation.
# 
#     <img src="ML-image/normal-df.png" width="850" height="350" />
#     
#     **Empirical rule:** The Empirical Rule, also known as the 68-95-99.7 Rule, is a rule of thumb for the normal distribution. It states that:
# 
#     - Approximately 68% of the data falls within one standard deviation of the mean.
#     - Approximately 95% of the data falls within two standard deviations of the mean.
#     - Approximately 99.7% of the data falls within three standard deviations of the mean.
# 
#     This means that if a distribution is approximately normal, we can use these percentages to estimate the proportion of data that falls within a certain range of values.
# 
#     <img src="ML-image/normal-df2.png" width="600" height="330" />
# 
#     **Example:** if we know that a distribution is approximately normal with a mean of 50 and a standard deviation of 10, we can use the Empirical Rule to estimate the proportion of data that falls within certain ranges:
# 
#     - Approximately 68% of the data falls between 40 and 60 (one standard deviation from the mean).
#     - Approximately 95% of the data falls between 30 and 70 (two standard deviations from the mean).
#     - Approximately 99.7% of the data falls between 20 and 80 (three standard deviations from the mean).
# 
#     It's important to note that the Empirical Rule is only an approximation and may not hold for all normal distributions. It is also not applicable to non-normal distributions.
# 
# | Statistics | Formula |
# |------------|---------|
# | Mean | $\mu$ |
# | Variance | $\sigma^2 $ |
# |Standard deviation | $\sigma $ |
# | Moment coefficient of skewness | $\alpha_3 = 0$ |
# | Moment coefficient of Kurtosis | $\alpha_4 = 3$ |
# | Mean deviation | $\sigma\sqrt{\frac{2}{\pi}} = 0.7979 ~ \sigma $ |
# 
# 
# 2. **Uniform distribution:** this is a distribution in which all values in a given range are equally likely to occur. The PDF of the uniform distribution is:
# 
#     $$f(x)=
#     \begin{cases}
#     \frac{1}{b-a}, & a \leq x \leq b \\
#     0, & \text{otherwise}
#     \end{cases}$$
# 
#     where $x$ is the random variable, $a$ is the lower bound of the range, and $b$ is the upper bound of the range.
# 
#     <img src="ML-image/uni-dist1.png" width="450" height="350" />
# 
# 3. **Exponential distribution:** this is a distribution that is commonly used to model the time between events that occur at a constant rate. The PDF of the exponential distribution is:
# 
#     $$ f(x; \lambda) = 
#     \begin{cases} 
#     \lambda e^{-\lambda x}, & x \geq 0 \\
#     0, & x < 0
#     \end{cases} $$ 
# 
#     where $x$ is the random variable, and $\lambda$ is the rate parameter.
# 
# |Statistics| Formula|
# |----------|--------|
# | Mean | $𝐸[𝑋]=\frac{1}{\lambda}$ |
# | Median | $m[X] =\frac{ln(2)}{\lambda} < E[X]$ |
# | Variance | $𝑉𝑎𝑟[𝑋]=\frac{1}{\lambda^2}$ |
# | Moments | $E[X^n]=\frac{n!}{\lambda^n}$|
# 
# <img src="https://cdn1.byjus.com/wp-content/uploads/2019/08/Exponential-Distribution.png" width="450" height="350" />
# 
# 4. **Gamma distribution:** this is a distribution that is used to model the sum of several exponentially distributed random variables. The PDF of the gamma distribution is:
# 
# $$f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}$$
# 
# where $x$ is the random variable, $k$ is the shape parameter, $\theta$ is the scale parameter, and $\Gamma(k)$ is the gamma function.
# 
# <img src="ML-image/eexpon.png" width="450" height="350" />
# 
# [Image credit](https://en.wikipedia.org/wiki/Gamma_distribution)
# 
# 
# The probability distribution is an essential concept in probability theory and is used to calculate the expected values, variances, and other statistical properties of random variables. Understanding probability distributions is important in fields such as statistics, physics, engineering, finance, and many others where randomness plays a role.

# %% [markdown]
# # Central Limit theorem (CLT)
# 
# The central limit theorem (CLT) is a fundamental concept in statistics and probability theory. It states that under certain conditions, the sampling distribution of the mean of a random sample drawn from any population will approximate a normal distribution, regardless of the shape of the original population distribution.
# 
# Specifically, the CLT states that as the sample size n increases, the sampling distribution of the mean approaches a normal distribution with mean equal to the population mean and standard deviation equal to the population standard deviation divided by the square root of the sample size. This means that even if the population distribution is not normal, the distribution of sample means will tend to be normal if the sample size is sufficiently large.
# 
# The conditions necessary for the CLT to hold are:
# 
# - **Random sampling:** The samples must be drawn at random from the population.
# - **Independence:** Each sample observation must be independent of all the others.
# - **Finite variance:** The population distribution must have a finite variance.
# 
# The CLT has many important practical applications, as it allows us to make inferences about population means and proportions based on samples drawn from the population. It is also used in hypothesis testing, confidence interval estimation, and in the construction of many statistical models.
# 
# ## Application of CLT
# 
# The central limit theorem (CLT) has many important applications in statistics and data analysis. Here are a few examples:
# 
# 1. **Estimating population parameters:** The CLT can be used to estimate population parameters, such as the population mean or proportion, based on a sample drawn from the population. For example, if we want to estimate the average height of all adults in a country, we can take a random sample of heights and use the CLT to construct a confidence interval for the population mean.
# 
# 2. **Hypothesis testing:** The CLT is often used in hypothesis testing to determine whether a sample is likely to have come from a particular population. For example, if we want to test whether the mean salary of a group of employees is different from the mean salary of all employees in the company, we can use the CLT to calculate the probability of observing a sample mean as extreme as the one we observed if the null hypothesis (i.e., the mean salaries are equal) is true.
# 
# 3. **Machine learning:** The CLT is used in many machine learning algorithms that require the assumption of normality, such as linear regression and logistic regression. In these algorithms, the CLT is used to justify the assumption that the errors or residuals of the model are normally distributed.
# 
# **Forumla** The formula for the CLT depends on the specific population distribution and the sample size. In general, if $X$ is a random variable with mean $\mu$ and standard deviation $\sigma$, then the distribution of the sample mean $\mu_X$ of a random sample of size $n$ from $X$ approaches a normal distribution with mean $\mu$ and standard deviation $\sigma/\sqrt{n}$ as $n$ gets larger. This can be expressed mathematically as:
# 
# $$\frac{\mu_X - \mu}{\sigma/\sqrt{n}}\sim N(0,1)$$
# 
# where $N(0,1)$ represents a _**standard normal distribution**_ with mean $0$ and standard deviation $1$.
# 
# In practice, the CLT is often used to calculate confidence intervals for population means or proportions. The formula for a confidence interval for the population mean based on a sample mean $\mu_X$ and a sample standard deviation $s$ is:
# 
# $$\mu_X \pm z^* \left(\frac{s}{\sqrt{n}}\right)$$
# 
# where $z^*$ is the appropriate critical value from the standard normal distribution based on the desired level of confidence.
# 
# **Note:** To calculate the value of $z^*$ for a given level of confidence, we need to use a standard normal distribution table (Z-table or normal probability table) or a statistical software program (R, Python, and GNU Octave to commercial software like SPSS, SAS, and Stata). For example, if we want to find the critical value for a 95% confidence level, we would look up the corresponding value in a standard normal distribution table or use the formula:
# 
# $$z^* = \text{invNorm}(1 - \frac{\alpha}{2})$$
# 
# where invNorm is the inverse cumulative distribution function of the standard normal distribution, and $\alpha$ is the significance level, which is equal to 1 - confidence level.
# 
# > [standard normal distribution table](https://www.mathsisfun.com/data/standard-normal-distribution-table.html)
# 
# For a 95% confidence level, alpha is 0.05, so we would have:
# 
# $$z^* = \text{invNorm}(1 - 0.05/2) = \text{invNorm}(0.975) = 1.96$$
# 
# Therefore, the critical value $z^*$ for a 95% confidence level is 1.96.
# 
# 
# ## Normal distribution vs the standard normal distribution
# 
# - The standard normal distribution, also called the z-distribution, is a special normal distribution where the mean is 0 and the standard deviation is 1.
# - All normal distributions, like the standard normal distribution, are unimodal and symmetrically distributed with a bell-shaped curve.
# - Every normal distribution is a version of the standard normal distribution that’s been stretched or squeezed and moved horizontally right or left.
# - The mean determines where the curve is centered. Increasing the mean moves the curve right, while decreasing it moves the curve left.
# 
# | Curve	| Position or shape (relative to standard normal distribution) |
# |-------|--------------------------------------------------------------|
# | A (M = 0, SD = 1)	| Standard normal distribution |
# | B (M = 0, SD = 0.5)	| Squeezed, because SD < 1 |
# | C (M = 0, SD = 2)	| Stretched, because SD > 1 |
# | D (M = 1, SD = 1)	| Shifted right, because M > 0 |
# | E (M = –1, SD = 1)	| Shifted left, because M < 0 |
# 
# <img src="ML-image/snd-nd.png" width="650" height="450" />
# 
# [☞(For image Reference, click here)](https://www.scribbr.com/statistics/standard-normal-distribution/)
# 
# ## Standardizing a normal distribution
# 
# - When you standardize a normal distribution, the mean becomes 0 and the standard deviation becomes 1. This allows you to easily calculate the probability of certain values occurring in your distribution, or to compare data sets with different means and standard deviations.
# 
# - While data points are referred to as x in a normal distribution, they are called z or z scores in the z distribution. A z score is a standard score that tells you how many standard deviations away from the mean an individual value (x) lies:
# 
#     - A positive z score means that your x value is greater than the mean.
#     - A negative z score means that your x value is less than the mean.
#     - A z score of zero means that your x value is equal to the mean.
# 
#     <img src="ML-image/snd.png" width="650" height="450" />
# 
#     [☞(For image Reference, click here)](https://www.scribbr.com/statistics/normal-distribution/)
# 
# - **Formula:**
# 
#     $z=\frac{x-\mu}{\sigma}$
# 
#     where
#     - $x$ = individual value
#     - $\mu$ = mean
#     - $\sigma$ = standard deviation
# 

# %%
#Example to calculate z in python
from scipy.stats import norm

# Find the critical value for a 95% confidence level
confidence_level = 0.95
alpha = 1 - confidence_level
z_critical = norm.ppf(1 - alpha/2)

print("Critical value:", z_critical)

# %% [markdown]
# # Example <a id="2"></a>
# 
# Here we import necessary python libraries. For example:
# 
# 1. Numpy
# 2. Pandas
# 3. Math
# 4. Matplotlib
# 5. Seaborn
# 6. pyplot
# 7. statistics
# 8. scipy
# 9. sklearn

# %% [markdown]
# ### If not installed
# 
# Type in jupyter cell, and install these libraries (if not installed):
# 
# | Sr.N. | Library | Code |
# |-------|---------|------|
# | 1. | Numpy | `%pip install -q numpy` |
# | 2. | Pandas | `%pip install -q pandas` |
# | 3. | Math | Math is a built-in module in Python, so you don't need to install it separately. |
# | 4. | Matplotlib  | `%pip install -q matplotlib` |
# | 5. | Seaborn | `%pip install -q seaborn` |
# | 6. | pyplot | pyplot is a sublibrary of Matplotlib, so you don't need to install it separately.|
# | 7. | statistics | Statistics is a built-in module in Python 3.4 and above, so you don't need to install it separately. If you're using an older version of Python, you can install it using: `%pip install -q statistics` |
# | 8. | scipy | `%pip install -q scipy` |
# | 9. | sklearn | `%pip install -q scikit-learn` |
# 
# Note: Make sure you have an active internet connection and pip installed in your system before running these commands. You can check if pip is installed by running the following command:
# 
# `%pip --version`
# 
# In terminal reemove % and copy the command and run it to install.

# %% [markdown]
# ### Importing the libraries

# %%
# Dependencies

# Standard Dependencies
import os
import numpy as np
import pandas as pd
from math import sqrt

# Visualization
from pylab import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from statistics import median
from scipy import signal
# from scipy.misc import factorial
import scipy.stats as stats
from scipy.stats import sem, binom, lognorm, poisson, bernoulli, spearmanr
from scipy.fftpack import fft, fftshift

# Scikit-learn for Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Seed for reproducability
seed = 12345
np.random.seed(seed)

# %%
# Read in csv of Toy Dataset
# We will use this dataset throughout the tutorial
toy_df = pd.read_csv('ml-data/toy_dataset.csv')

# %% [markdown]
# ## Discrete and Continuous Variables <a id="2"></a>

# %% [markdown]
# A discrete variable is a variable that can only take on a "countable" number of values. If you can count a set of items, then it’s a discrete variable. An example of a discrete variable is the outcome of a dice. It can only have 1 of 6 different possible outcomes and is therefore discrete. A discrete random variable can have an infinite number of values. For example, the whole set of natural numbers (1,2,3,etc.) is countable and therefore discrete. 
# 
# A continuous variable takes on an "uncountable" number of values. An example of a continuous variable is length. Length can be measured to an arbitrary degree and is therefore continuous.
# 
# In statistics we represent a distribution of discrete variables with PMF's (Probability Mass Functions) and CDF's (Cumulative Distribution Functions). We represent distributions of continuous variables with PDF's (Probability Density Functions) and CDF's. 
# 
# The PMF defines the probability of all possible values x of the random variable. A PDF is the same but for continuous values.
# The CDF represents the probability that the random variable X will have an outcome less or equal to the value x. The name CDF is used for both discrete and continuous distributions.
# 
# The functions that describe PMF's, PDF's and CDF's can be quite daunting at first, but their visual counterpart often looks quite intuitive.

# %% [markdown]
# ### PMF (Probability Mass Function)

# %% [markdown]
# Here we visualize a PMF of a binomial distribution. You can see that the possible values are all integers. For example, no values are between 50 and 51. 
# 
# The PMF of a binomial distribution in function form:
# 
# $P(X=x)= p^x\left(\frac{N}{x}\right)(1-p)^{N-x}$
# 
# See the "[Distributions](#3)" sections for more information on binomial distributions.

# %%
# PMF Visualization
n = 100
p = 0.5

fig, ax = plt.subplots(1, 1, figsize=(17,5))
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='Binomial PMF')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
rv = binom(n, p)
#ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1, label='frozen PMF')
ax.legend(loc='best', frameon=False, fontsize='xx-large')
plt.title('PMF of a binomial distribution (n=100, p=0.5)', fontsize='xx-large')
plt.show()

# %% [markdown]
# ### PDF (Probability Density Functions)

# %% [markdown]
# The PDF is the same as a PMF, but continuous. It can be said that the distribution has an infinite number of possible values. Here we visualize a simple normal distribution with a mean of 0 and standard deviation of 1.
# 
# PDF of a normal distribution in formula form:
# 
# ![](https://www.mhnederlof.nl/images/normalpdf.jpg)

# %%
# Plot normal distribution
mu = 0
variance = 1
sigma = sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.figure(figsize=(16,5))
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal Distribution')
plt.title('Normal Distribution with mean = 0 and std = 1')
plt.legend(fontsize='xx-large')
plt.show()

# %% [markdown]
# ### CDF (Cumulative Distribution Function)

# %% [markdown]
# The CDF maps the probability that a random variable X will take a value of less than or equal to a value x (P(X ≤  x)). CDF's can be discrete or continuous. In this section we visualize the continuous case. You can see in the plot that the CDF accumulates all probabilities and is therefore bounded between 0 ≤ x ≤ 1.
# 
# The CDF of a normal distribution as a formula:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/187f33664b79492eedf4406c66d67f9fe5f524ea)
# 
# *Note: erf means "[error function](https://en.wikipedia.org/wiki/Error_function)".*

# %%
# Data
X  = np.arange(-2, 2, 0.01)
Y  = exp(-X ** 2)

# Normalize data
Y = Y / (0.01 * Y).sum()

# Plot the PDF and CDF
plt.figure(figsize=(15,5))
plt.title('Continuous Normal Distributions', fontsize='xx-large')
plot(X, Y, label='Probability Density Function (PDF)')
plot(X, np.cumsum(Y * 0.01), 'r', label='Cumulative Distribution Function (CDF)')
plt.legend(fontsize='xx-large')
plt.show()

# %% [markdown]
# ## Distributions <a id="3"></a>

# %% [markdown]
# A Probability distribution tells us something about the likelihood of each value of the random variable.
# 
# A random variable X is a function that maps events to real numbers.
# 
# The visualizations in this section are of discrete distributions. Many of these distributions can however also be continuous.

# %% [markdown]
# ### Uniform Distribution

# %% [markdown]
# A Uniform distribution is pretty straightforward. Every value has an equal change of occuring. Therefore, the distribution consists of random values with no patterns in them. In this example we generate random floating numbers between 0 and 1.
# 
# The PDF of a Uniform Distribution:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/648692e002b720347c6c981aeec2a8cca7f4182f)
# 
# CDF:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/eeeeb233753cfe775b24e3fec2f371ee8cdc63a6)

# %%
# Uniform distribution (between 0 and 1)
uniform_dist = np.random.random(1000)
uniform_df = pd.DataFrame({'value' : uniform_dist})
uniform_dist = pd.Series(uniform_dist)

# %%
plt.figure(figsize=(18,5))
sns.scatterplot(data=uniform_df)
plt.legend(fontsize='xx-large')
plt.title('Scatterplot of a Random/Uniform Distribution', fontsize='xx-large')

# %%
plt.figure(figsize=(18,5))
sns.distplot(uniform_df)
plt.title('Random/Uniform distribution', fontsize='xx-large')

# %% [markdown]
# ### Normal Distribution

# %% [markdown]
# A normal distribution (also called Gaussian or Bell Curve) is very common and convenient. This is mainly because of the [Central Limit Theorem (CLT)](https://en.wikipedia.org/wiki/Central_limit_theorem), which states that as the amount independent random samples (like multiple coin flips) goes to infinity the distribution of the sample mean tends towards a normal distribution.
# 
# PDF of a normal distribution:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2ce7e315b02666699e0cd8ea5fb1a3e0c287cd9d)
# 
# CDF:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/187f33664b79492eedf4406c66d67f9fe5f524ea)
# 

# %%
# Generate Normal Distribution
normal_dist = np.random.randn(10000)
normal_df = pd.DataFrame({'value' : normal_dist})
# Create a Pandas Series for easy sample function
normal_dist = pd.Series(normal_dist)

normal_dist2 = np.random.randn(10000)
normal_df2 = pd.DataFrame({'value' : normal_dist2})
# Create a Pandas Series for easy sample function
normal_dist2 = pd.Series(normal_dist)

normal_df_total = pd.DataFrame({'value1' : normal_dist, 
                                'value2' : normal_dist2})

# %%
# Scatterplot
plt.figure(figsize=(18,5))
sns.scatterplot(data=normal_df)
plt.legend(fontsize='xx-large')
plt.title('Scatterplot of a Normal Distribution', fontsize='xx-large')

# %%
# Normal Distribution as a Bell Curve
plt.figure(figsize=(18,5))
sns.displot(normal_df, kde=True)
plt.title('Normal distribution (n=1000)', fontsize='xx-large')
plt.show()

# %%
plt.figure(figsize=(7,5))
sns.histplot(normal_df, kde=True)
plt.title('Normal distribution (n=1000)', fontsize='xx-large')
plt.show()

# %% [markdown]
# ### Binomial Distribution

# %% [markdown]
# A Binomial Distribution has a countable number of outcomes and is therefore discrete.
# 
# Binomial distributions must meet the following three criteria:
# 
# 1. The number of observations or trials is fixed. In other words, you can only figure out the probability of something happening if you do it a certain number of times.
# 2. Each observation or trial is independent. In other words, none of your trials have an effect on the probability of the next trial.
# 3. The probability of success is exactly the same from one trial to another.
# 
# An intuitive explanation of a binomial distribution is flipping a coin 10 times. If we have a fair coin our chance of getting heads (p) is 0.50. Now we throw the coin 10 times and count how many times it comes up heads. In most situations we will get heads 5 times, but there is also a change that we get heads 9 times. The PMF of a binomial distribution will give these probabilities if we say N = 10 and p = 0.5. We say that the x for heads is 1 and 0 for tails.
# 
# PMF:
# 
# ![](http://reliabilityace.com/formulas/binomial-pmf.png)
# 
# CDF:
# 
# ![](http://reliabilityace.com/formulas/binomial-cpf.png)
# 
# 
# A **Bernoulli Distribution** is a special case of a Binomial Distribution.
# 
# All values in a Bernoulli Distribution are either 0 or 1. 
# 
# For example, if we take an unfair coin which falls on heads 60 % of the time, we can describe the Bernoulli distribution as follows:
# 
# p (change of heads) = 0.6
# 
# 1 - p (change of tails) = 0.4
# 
# heads = 1
# 
# tails = 0
# 
# Formally, we can describe a Bernoulli distribution with the following PMF (Probability Mass Function):
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a9207475ab305d280d2958f5c259f996415548e9)
# 

# %%
# Change of heads (outcome 1)
p = 0.6

# Create Bernoulli samples
bern_dist = bernoulli.rvs(p, size=1000)
bern_df = pd.DataFrame({'value' : bern_dist})
bern_values = bern_df['value'].value_counts()

# Plot Distribution
plt.figure(figsize=(10,4))
bern_values.plot(kind='bar', rot=0)

plt.title('Bernoulli Distribution: p = 0.6, n = 1000')

# %% [markdown]
# ### Poisson Distribution

# %% [markdown]
# The Poisson distribution is a discrete distribution and is popular for modelling the number of times an event occurs in an interval of time or space. 
# 
# It takes a value lambda, which is equal to the mean of the distribution.
# 
# PMF: 
# 
# ![](https://study.com/cimages/multimages/16/poisson1a.jpg)
# 
# CDF: 
# ![](http://www.jennessent.com/images/cdf_poisson.gif)

# %%
x = np.arange(0, 20, 0.1)
y = np.exp(-5)*np.power(5, x)/factorial(x)

plt.figure(figsize=(15,8))
plt.title('Poisson distribution with lambda=5', fontsize='xx-large')
plt.plot(x, y, 'bs')
plt.show()

# %% [markdown]
# ### Log-Normal Distribution

# %% [markdown]
# A log-normal distribution is continuous. The main characteristic of a log-normal distribution is that it's logarithm is normally distributed. It is also referred to as Galton's distribution.
# 
# PDF: 
# 
# ![](https://www.mhnederlof.nl/images/lognormaldensity.jpg)
# 
# CDF:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/29095d9cbd6539833d549c59149b9fc5bd06339b)
# 
# Where Phi is the CDF of the standard normal distribution.

# %%
# Specify standard deviation and mean
std = 1
mean = 5

# Create log-normal distribution
dist=lognorm(std,loc=mean)
x=np.linspace(0,15,200)

# Visualize log-normal distribution
plt.figure(figsize=(15,6))
plt.xlim(5, 10)
plt.plot(x,dist.pdf(x), label='log-normal PDF')
plt.plot(x,dist.cdf(x), label='log-normal CDF')
plt.legend(fontsize='xx-large')
plt.title('Visualization of log-normal PDF and CDF', fontsize='xx-large')
plt.show()

# %% [markdown]
# ## Summary Statistics and Moments <a id="4"></a>

# %% [markdown]
# **Mean, Median and Mode** 
# 
# Note: The mean is also called the first moment.
# 
# 
# ![](https://qph.fs.quoracdn.net/main-qimg-29a4925034e075f16e1c743a4b3dda8b)

# %% [markdown]
# ### Moments
# 
# A moment is a quantitative measure that says something about the shape of a distribution. There are central moments and non-central moments. This section is focused on the central moments.
# 
# The 0th central moment is the total probability and is always equal to 1.
# 
# The 1st moment is the mean (expected value).
# 
# The 2nd central moment is the variance.
# 
# **Variance** = The average of the squared distance of the mean. Variance is interesting in a mathematical sense, but the standard deviation is often a much better measure of how spread out the distribution is.
# 
# ![](http://www.visualmining.com/wp-content/uploads/2013/02/analytics_formula_variance.png)
# 
# **Standard Deviation** = The square root of the variance
# 
# ![](http://www.visualmining.com/wp-content/uploads/2013/02/analytics_formula_std_dev.png)
# 
# The 3rd central moment is the skewness.
# 
# **Skewness** = A measure that describes the contrast of one tail versus the other tail. For example, if there are more high values in your distribution than low values then your distribution is 'skewed' towards the high values.
# 
# ![](http://www.visualmining.com/wp-content/uploads/2013/02/analytics_formula_skewness.png)
# 
# The 4th central moment is the kurtosis.
# 
# **Kurtosis** = A measure of how 'fat' the tails in the distribution are.
# 
# ![](http://www.visualmining.com/wp-content/uploads/2013/02/analytics_formula_kurtosis.png)
# 
# The higher the moment, the harder it is to estimate with samples. Larger samples are required in order to obtain good estimates.

# %%
# Summary
print('Summary Statistics for a normal distribution: ')
# Median
medi = median(normal_dist)
print('Median: ', medi)
display(normal_df.describe())

# Standard Deviation
std = sqrt(np.var(normal_dist))

print('The first four calculated moments of a normal distribution: ')
# Mean
mean = normal_dist.mean()
print('Mean: ', mean)

# Variance
var = np.var(normal_dist)
print('Variance: ', var)

# Return unbiased skew normalized by N-1
skew = normal_df['value'].skew()
print('Skewness: ', skew)

# Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis 
# (kurtosis of normal == 0.0) normalized by N-1
kurt = normal_df['value'].kurtosis()
print('Kurtosis: ', kurt)

# %% [markdown]
# ## Bias, MSE and SE <a id="5"></a>

# %% [markdown]
# **Bias** is a measure of how far the sample mean deviates from the population mean. The sample mean is also called **Expected value**.
# 
# Formula for Bias:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/82a9c6501a54260ed0edd2f03923719b9f2db906)
# 
# The formula for expected value (EV) makes it apparent that the bias can also be formulated as the expected value minus the population mean:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/12828b1f927b39d2fa9d75f82c02b91209182911)

# %%
# Take sample
normal_df_sample = normal_df.sample(100)

# Calculate Expected Value (EV), population mean and bias
ev = normal_df_sample.mean()[0]
pop_mean = normal_df.mean()[0]
bias = ev - pop_mean

# %%
print('Sample mean (Expected Value): ', ev)
print('Population mean: ', pop_mean)
print('Bias: ', bias)

# %% [markdown]
# **MSE (Mean Squared Error)** is a formula to measure how much estimators deviate from the true distribution. This can be very useful with for example, evaluating regression models.
# 
# 
# ![](https://i.stack.imgur.com/iSWyZ.png)
# 
# 
# **RMSE (Root Mean Squared Error)** is just the root of the MSE.
# 
# 
# ![](http://file.scirp.org/Html/htmlimages/5-2601289x/fcdba7fc-a40e-4019-9e95-aca3dc2db149.png)
# 
# 

# %%
from math import sqrt

Y = 100 # Actual Value
YH = 94 # Predicted Value

# MSE Formula 
def MSE(Y, YH):
     return np.square(YH - Y).mean()

# RMSE formula
def RMSE(Y, YH):
    return sqrt(np.square(YH - Y).mean())


print('MSE: ', MSE(Y, YH))

print('RMSE: ', RMSE(Y, YH))

# %% [markdown]
# The **Standard Error (SE)** measures how spread the distribution is from the sample mean.
# 
# ![](http://desktopia.net/p/2018/07/standard-deviation-biology-for-life-in-standard-error-of-the-mean-formula.gif)
# 
# The formula can also be defined as the standard deviation divided by the square root of the number of samples.
# 
# ![](https://toptipbio.com/wp-content/uploads/2017/07/Standard-error-formula.jpg)

# %%
# Standard Error (SE)
uni_sample = uniform_dist.sample(100)
norm_sample = normal_dist.sample(100)

print('Standard Error of uniform sample: ', sem(uni_sample))
print('Standard Error of normal sample: ', sem(norm_sample))

# The random samples from the normal distribution should have a higher standard error

# %% [markdown]
# ## Sampling methods <a id="6"></a>

# %% [markdown]
# **Non-Representative Sampling:**
# 
# Convenience Sampling = Pick samples that are most convenient, like the top of a shelf or people that can be easily approached.
# 
# Haphazard Sampling = Pick samples without thinking about it. This often gives the illusion take you are picking out samples at random. 
# 
# Purposive Sampling = Pick samples for a specific purpose. An example is to focus on extreme cases. This can be useful but is limited because it doesn't allow you to make statements about the whole population.
# 
# **Representative Sampling:**
# 
# Simple Random Sampling = Pick samples (psuedo)randomly.
# 
# Systematic Sampling = Pick samples with a fixed interval. For example every 10th sample (0, 10, 20, etc.).
# 
# Stratified Sampling = Pick the same amount of samples from different groups (strata) in the population.
# 
# Cluster Sampling = Divide the population into groups (clusters) and pick samples from those groups.

# %%
# Note that we take very small samples just to illustrate the different sampling methods

print('---Non-Representative samples:---\n')
# Convenience samples
con_samples = normal_dist[0:5]
print('Convenience samples:\n\n{}\n'.format(con_samples))

# Haphazard samples (Picking out some numbers)
hap_samples = [normal_dist[12], normal_dist[55], normal_dist[582], normal_dist[821], normal_dist[999]]
print('Haphazard samples:\n\n{}\n'.format(hap_samples))

# Purposive samples (Pick samples for a specific purpose)
# In this example we pick the 5 highest values in our distribution
purp_samples = normal_dist.nlargest(n=5)
print('Purposive samples:\n\n{}\n'.format(purp_samples))

print('---Representative samples:---\n')

# Simple (pseudo)random sample
rand_samples = normal_dist.sample(5)
print('Random samples:\n\n{}\n'.format(rand_samples))

# Systematic sample (Every 2000th value)
sys_samples = normal_dist[normal_dist.index % 2000 == 0]
print('Systematic samples:\n\n{}\n'.format(sys_samples))

# Stratified Sampling
# We will get 1 person from every city in the dataset
# We have 8 cities so that makes a total of 8 samples
df = pd.read_csv(KAGGLE_DIR + 'toy_dataset.csv')

strat_samples = []

for city in df['City'].unique():
    samp = df[df['City'] == city].sample(1)
    strat_samples.append(samp['Income'].item())
    
print('Stratified samples:\n\n{}\n'.format(strat_samples))

# Cluster Sampling
# Make random clusters of ten people (Here with replacement)
c1 = normal_dist.sample(10)
c2 = normal_dist.sample(10)
c3 = normal_dist.sample(10)
c4 = normal_dist.sample(10)
c5 = normal_dist.sample(10)

# Take sample from every cluster (with replacement)
clusters = [c1,c2,c3,c4,c5]
cluster_samples = []
for c in clusters:
    clus_samp = c.sample(1)
    cluster_samples.extend(clus_samp)
print('Cluster samples:\n\n{}'.format(cluster_samples))    


# %% [markdown]
# ## Covariance <a id="7"></a>

# %% [markdown]
# Covariance is a measure of how much two random variables vary together. variance is similar to covariance in that variance shows you how much one variable varies. Covariance tells you how two variables vary together.
# 
# If two variables are independent, their covariance is 0. However, a covariance of 0 does not imply that the variables are independent.

# %%
# Covariance between Age and Income
print('Covariance between Age and Income: ')

df[['Age', 'Income']].cov()

# %% [markdown]
# ## Correlation <a id="8"></a>

# %% [markdown]
# Correlation is a standardized version of covariance. Here it becomes more clear that Age and Income do not have a strong correlation in our dataset.

# %% [markdown]
# The formula for Pearson's correlation coefficient consists of the covariance between the two random variables divided by the standard deviation of the first random variable times the standard deviation of the second random variable.
# 
# Formula for Pearson's correlation coefficient:
# 
# ![](http://sherrytowers.com/wp-content/uploads/2013/09/correlation_xy-300x97.jpg)

# %%
# Correlation between two normal distributions
# Using Pearson's correlation
print('Pearson: ')
df[['Age', 'Income']].corr(method='pearson')

# %% [markdown]
# Another method for calculating a correlation coefficient is 'Spearman's Rho'. The formula looks different but it will give similar results as Pearson's method. In this example we see almost no difference, but this is partly because it is obvious that the Age and Income columns in our dataset have no correlation.
# 
# Formula for Spearmans Rho:
# 
# ![](http://s3.amazonaws.com/hm_120408/fa/3d86/yhf5/9dwq/4m6e2kcav/original.jpg?1447778688)

# %%
# Using Spearman's rho correlation
print('Spearman: ')
df[['Age', 'Income']].corr(method='spearman')

# %% [markdown]
# ## Linear regression <a id="9"></a>

# %% [markdown]
# Linear Regression can be performed through Ordinary Least Squares (OLS) or Maximum Likelihood Estimation (MLE).
# 
# Most Python libraries use OLS to fit linear models.
# 
# ![](https://image.slidesharecdn.com/simplelinearregressionpelatihan-090829234643-phpapp02/95/simple-linier-regression-9-728.jpg?cb=1251589640)

# %%
# Generate data
x = np.random.uniform(low=20, high=260, size=100)
y = 50000 + 2000*x - 4.5 * x**2 + np.random.normal(size=100, loc=0, scale=10000)

# Plot data with Linear Regression
plt.figure(figsize=(16,5))
plt.title('Well fitted but not well fitting: Linear regression plot on quadratic data', fontsize='xx-large')
sns.regplot(x, y)

# %% [markdown]
# Here we observe that the linear model is well-fitted. However, a linear model is probably not ideal for our data, because the data follows a quadratic pattern. A [polynomial regression model](https://en.wikipedia.org/wiki/Polynomial_regression) would better fit the data, but this is outside the scope of this tutorial.

# %% [markdown]
# We can also implement linear regression with a bare-bones approach. In the following example we measure the vertical distance and horizontal distance between a random data point and the regression line. 
# 
# For more information on implementing linear regression from scratch [I highly recommend this explanation by Luis Serrano](https://aitube.io/video/introduction-to-linear-regression).

# %%
# Linear regression from scratch
import random
# Create data from regression
xs = np.array(range(1,20))
ys = [0,8,10,8,15,20,26,29,38,35,40,60,50,61,70,75,80,88,96]

# Put data in dictionary
data = dict()
for i in list(xs):
    data.update({xs[i-1] : ys[i-1]})

# Slope
m = 0
# y intercept
b = 0
# Learning rate
lr = 0.0001
# Number of epochs
epochs = 100000

# Formula for linear line
def lin(x):
    return m * x + b

# Linear regression algorithm
for i in range(epochs):
    # Pick a random point and calculate vertical distance and horizontal distance
    rand_point = random.choice(list(data.items()))
    vert_dist = abs((m * rand_point[0] + b) - rand_point[1])
    hor_dist = rand_point[0]

    if (m * rand_point[0] + b) - rand_point[1] < 0:
        # Adjust line upwards
        m += lr * vert_dist * hor_dist
        b += lr * vert_dist   
    else:
        # Adjust line downwards
        m -= lr * vert_dist * hor_dist
        b -= lr * vert_dist
        
# Plot data points and regression line
plt.figure(figsize=(15,6))
plt.scatter(data.keys(), data.values())
plt.plot(xs, lin(xs))
plt.title('Linear Regression result')  
print('Slope: {}\nIntercept: {}'.format(m, b))

# %% [markdown]
# The coefficients of a linear model can also be computed using MSE (Mean Squared Error) without an iterative approach. I implemented Python code for this technique as well. The code is in [the second cell of this Github repository](https://github.com/CarloLepelaars/linreg/blob/master/linreg_from_scratch.ipynb)

# %% [markdown]
# ### Anscombe's Quartet

# %% [markdown]
# Anscombe's quartet is a set of four datasets that have the same descriptive statistics and linear regression fit. The datasets are however very different from each other.
# 
# This sketches the issue that although summary statistics and regression models are really helpful in understanding your data, you should always visualize the data to see whats really going on. It also shows that a few outliers can really mess up your model.
# 
# [More information on Anscombe's Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)

# %% [markdown]
# ![](https://www.researchgate.net/profile/Arch_Woodside2/publication/286454889/figure/fig3/AS:669434310037526@1536616985820/Anscombes-quartet-of-different-XY-plots-of-four-data-sets-having-identical-averages.png)

# %% [markdown]
# ## Bootstrapping <a id="10"></a>

# %% [markdown]
# Bootstrapping is a resampling technique to quantify the uncertainty of an estimator given sample data. In other words, we have a sample of data and we take multiple samples from that sample. For example, with bootstrapping we can take means for each bootstrap sample and thereby make a distribution of means.
# 
# Once we created a distribution of estimators we can use this to make decisions. 
# 
# Bootstrapping can be:
# 1. Non-parametric (Take random samples from sample)
# 2. Parametric (Take from a (normal) distribution with sample mean and variance).
#     Downside: You are making assumptions about the distribution.
#     Upside: Computationally more light
# 3. Online bootstrap (Take samples from a stream of data)
# 
# The following code implements a simple non-parametric bootstrap to create a distribution of means, medians and midranges of the Income distribution in our Toy Dataset. We can use this to deduce which income means will make sense for subsequent samples.
# 

# %%
# scikit-learn bootstrap package
from sklearn.utils import resample

# data sample
data = df['Income']

# prepare bootstrap samples
boot = resample(data, replace=True, n_samples=5, random_state=1)
print('Bootstrap Sample: \n{}\n'.format(boot))
print('Mean of the population: ', data.mean())
print('Standard Deviation of the population: ', data.std())

# Bootstrap plot
pd.plotting.bootstrap_plot(data)

# %% [markdown]
# ## Hypothesis testing <a id="11"></a>

# %% [markdown]
# We establish two hypotheses, H0 (Null hypothesis) and Ha (Alternative Hypothesis). 
# 
# We can make four different decisions with hypothesis testing:
# 1. Reject H0 and H0 is not true (no error)
# 2. Do not reject H0 and H0 is true (no error)
# 3. Reject H0 and H0 is true (Type 1 Error)
# 4. Do not reject H0 and H0 is not true (Type 2 error)
# 
# Type 1 error is also called Alpha error.
# Type 2 error is also called Beta error.
# 
# ![](https://qph.fs.quoracdn.net/main-qimg-84121cf5638cbb5919999b2a8d928c91)
# 
# ![](https://i.stack.imgur.com/x1GQ1.png)

# %% [markdown]
# ### P-Value
# 
# A p-value is the probability of finding equal or more extreme results when the null hyptohesis (H0) is true. In other words, a low p-value means that we have compelling evidence to reject the null hypothesis.
# 
# If the p-value is lower than 5% (p < 0.05). We often reject H0 and accept Ha is true. We say that p < 0.05 is statistically significant, because there is less than 5% chance that we are wrong in rejecting the null hypothesis.
# 
# One way to calculate the p-value is through a T-test. We can use [Scipy's ttest_ind function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) to calculate the t-test for the means of two independent samples of scores. In this example we calculate the t-statistic and p-value of two random samples 10 times. 
# 
# We see that the p-value is sometimes very low, but this does not mean that these two random samples are correlated. This is why you have to be careful with relying too heavily of p-values. If you repeat an experiment multiple times you can get trapped in the illusion that there is correlation where there is only randomness.
# 
# [This xkcd comic perfectly illustrates the hazards of relying too much on p-values](https://xkcd.com/882/).

# %%
# Perform t-test and compute p value of two random samples
print('T-statistics and p-values of two random samples.')
for _ in range(10):
    rand_sample1 = np.random.random_sample(10)
    rand_sample2 = np.random.random_sample(10)
    print(stats.ttest_ind(rand_sample1, rand_sample2))

# %%
# To-do
# Equivalence testing

# %% [markdown]
# ### q-q plot (quantile-quantile plot)
# 
# Many statistical techniques require that data is coming from a normal distribution (for example, t-test). Therefore, it is important to verify this before applying statistical techniques.
# 
# One approach is to visualize and make a judgment about the distribution. A q-q plot is very helpful for determining if a distribution is normal. There are other tests for testing 'normality', but this is beyond the scope of this tutorial.
# 
# In the first plot we can easily see that the values line up nicely. From this we conclude that the data is normally distributed.
# 
# In the second plot we can see that the values don't line up. Our conclusion is that the data is not normally distributed. In this case the data was uniformly distributed.
# 

# %%
# q-q plot of a normal distribution
plt.figure(figsize=(15,6))
stats.probplot(normal_dist, dist="norm", plot=plt)
plt.show()

# %%
# q-q plot of a uniform/random distribution
plt.figure(figsize=(15,6))
stats.probplot(uniform_dist, dist="norm", plot=plt) 
plt.show()

# %% [markdown]
# ## Outliers <a id="12"></a>

# %% [markdown]
# An outlier is an observation which deviates from other observations. An outlier often stands out and could be an error.
# 
# Outliers can mess up you statistical models. However, outliers should only be removed when you have established good reasons for removing the outlier.
# 
# Sometimes the outliers are the main topic of interest. This is for example the case with fraud detection. There are many outlier detection methods, but here we will discuss Grubbs test and Tukey’s method. Both tests assume that the data is normally distributed.

# %% [markdown]
# ### Grubbs Test

# %% [markdown]
# In Grubbs test, the null hypothesis is that no observation is an outlier, while the alternative hypothesis is that there is one observation an outlier. Thus the Grubbs test is only searching for one outlier observation.
# 
# The formula for Grubbs test:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bafc310f1dbca658728c73256fed19b6a7f11130)
# 
# Where Y_hat is the sample mean and s is the standard deviation. The Grubbs test statistic is the largest absolute deviation from the sample mean in units of the sample standard deviation.
# 
# [Source](https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers)

# %% [markdown]
# ### Tukey's method

# %% [markdown]
# Tukey suggested that an observation is an outlier whenever an observation is 1.5 times the interquartile range below the first quartile or 1.5 times the interquartile range above the third quartile. This may sound complicated, but is quite intuitive if you see it visually.
# 
# For normal distributions, Tukey’s criteria for outlier observations is unlikely if no outliers are present, but using Tukey’s criteria for other distributions should be taken with a grain of salt.
# 
# The formula for Tukey's method:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a103bbd9233d9f8f711a7c76dfeb9694446f860)
# 
# Ya is the larger of two means being compared. SE is the standard error of the sum of the means.
# 
# [Source](https://en.wikipedia.org/wiki/Tukey%27s_range_test)

# %%
# Detect outliers on the 'Income' column of the Toy Dataset

# Function for detecting outliers a la Tukey's method using z-scores
def tukey_outliers(data) -> list:
    # For more information on calculating the threshold check out:
    # https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32
    threshold = 3
    
    mean = np.mean(data)
    std = np.std(data)
    
    # Spot and collect outliers
    outliers = []
    for i in data:
        z_score = (i - mean) / std
        if abs(z_score) > threshold:
            outliers.append(i)
    return outliers

# Get outliers
income_outliers = tukey_outliers(df['Income'])

# Visualize distribution and outliers
plt.figure(figsize=(15,6))
df['Income'].plot(kind='hist', bins=100, label='Income distribution')
plt.hist(income_outliers, bins=20, label='Outliers')
plt.title("Outlier detection a la Tukey's method", fontsize='xx-large')
plt.xlabel('Income')
plt.legend(fontsize='xx-large')

# %% [markdown]
# ## Overfitting <a id="20"></a>

# %% [markdown]
# Our model is overfitting if it is also modeling the 'noise' in the data. This implies that the model will not generalize well to new data even though the error on the training data becomes very small. Linear models are unlikely to overfit, but as models become more flexible we have to be wary of overfitting. Our model can also underfit which means that it has a large error on the training data. 
# 
# Finding the sweet spot between overfitting and underfitting is called the [Bias Variance Trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). It is nice to know this theorem, but more important to understand how to prevent it. I will explain some techniques for how to do this below.

# %% [markdown]
# ### Prevention of Overfitting
# 
# 1. Split data into training data and test data.
# 2. Regularization: limit the flexibility of the model.
# 3. Cross Validation

# %% [markdown]
# ### Cross Validation
# 
# Cross validation is a technique to estimate the accuracy of our statistical model. It is also called out-of-sample testing or rotation estimation. Cross validation will help us to recognize overfitting and to check if our model generalizes to new (out-of-sample) data.
# 
# A popular cross validation technique is called k-fold cross validation. The idea is simple, we split our dataset up in k datasets and out of each dataset k we pick out a few samples. We then fit our model on the rest of k and try to predict the samples we picked out. We use a metric like Mean Squared Error to estimate how good our predictions are. This procedure is repeated and then we look at the average of the predictions over multiple cross-validation data sets. 
# 
# A special case where we pick out one samples is called 'Leave-One-Out Cross Validation (LOOCV)'. However, the variance of LOOCV is high.
# 
# For more information about cross validation [check out this blog](https://machinelearningmastery.com/k-fold-cross-validation/).
# 
# 

# %% [markdown]
# ## Generalized Linear Models (GLMs) <a id="13"></a>

# %% [markdown]
# ### Link functions

# %% [markdown]
# A Link Function is used in Generalized Linear Models (GLMs) to apply linear models for a continuous response variable given continuous and/or categorical predictors. A link function that is often used is called the inverse logit or logistic sigmoid function.
# 
# The link function provides a relationship between the linear predictor and the mean of a distribution.

# %%
# Inverse logit function (link function)
def inv_logit(x):
    return 1 / (1 + np.exp(-x))

t1 = np.arange(-10, 10, 0.1)
plt.figure(figsize=(15,6))
plt.plot(t1, inv_logit(t1), 
         t1, inv_logit(t1-2),   
         t1, inv_logit(t1*2))
plt.title('Inverse logit functions', fontsize='xx-large')
plt.legend(('Normal', 'Changed intercept', 'Changed slope'), fontsize='xx-large')

# %% [markdown]
# ### Logistic regression
# 
# With logistic regression we use a link function like the inverse logit function mentioned above to model a binary dependent variable. While a linear regression model predicts the expected value of y given x directly, a GLM uses a link function. 
# 
# We can easily implement logistic regression with [sklearn's Logistic Regression function.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# %%
# Simple example of Logistic Regression in Python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Logistic regression classifier
clf = LogisticRegression(random_state=0, 
                         solver='lbfgs',
                         multi_class='multinomial').fit(X, y)

print('Accuracy score of logistic regression model on the Iris flower dataset: {}'.format(clf.score(X, y)))


# %% [markdown]
# ## Frequentist vs. Bayes <a id="14"></a>

# %% [markdown]
# Frequentist:
# 
# 1. Fixed parameters (Processes are fixed)
# 2. Repeated sampling -> Probabilities
# 
# Bayes:
# 
# 1. Probability as "degree of belief"
# 2. P(parameter) -> All plausible values of the parameter
# 3. Updates degree of belief based on a prior belief
# 
# 
# Frequentists and Bayesians agree that Bayes' Theorem is valid. See figure below for explanation of this theorem.
# 
# ![](https://cdn-images-1.medium.com/max/1600/1*LB-G6WBuswEfpg20FMighA.png)
# 
# 
# Bayes theorem extends to distributions and random variables.
# 

# %% [markdown]
# # The end!

# %% [markdown]
# **If you like this Kaggle kernel, feel free to give an upvote and leave a comment! I will try to implement your suggestions in this kernel!**

# %% [markdown]
# ## Bonus: Free statistics courses <a id="15"></a>

# %% [markdown]
# There is a lot of free material online for people who want to dive deeper into statistics. Here is a selection from the Internet.
# 
# Udacity's "Intro to Statistics": https://eu.udacity.com/course/intro-to-statistics--st101
# 
# Udacity's "Intro to Descriptive Statistics": https://eu.udacity.com/course/intro-to-descriptive-statistics--ud827
# 
# Udacity's "Intro to Inferential Statistics": https://eu.udacity.com/course/intro-to-inferential-statistics--ud201
# 
# edX's "Introduction to Probability - The Science of Uncertainty" : https://www.edx.org/course/introduction-probability-science-mitx-6-041x-2
# 
# Khan Academy's videos on statistics and probability: https://www.khanacademy.org/math/statistics-probability
# 
# (Kaggle Kernel) Mathematics of Linear Regression by Nathan Lauga: https://www.kaggle.com/nathanlauga/mathematics-of-linear-regression

# %% [markdown]
# ## Sources <a id="16"></a>

# %% [markdown]
# https://dataconomy.com/2015/02/introduction-to-bayes-theorem-with-python
# 
# https://www.statisticshowto.datasciencecentral.com/discrete-variable/
# 
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/binomial-theorem/binomial-distribution-formula/
# 
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/descriptive-statistics/sample-variance/
# 
# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method
# 
# https://machinelearningmastery.com/k-fold-cross-validation/
# 
# https://en.wikipedia.org/wiki/Poisson_distribution
# 
# https://www.tutorialspoint.com/python/python_p_value.htm
# 
# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
# 
# https://www.slideshare.net/dessybudiyanti/simple-linier-regression
# 
# https://www.youtube.com/channel/UCgBncpylJ1kiVaPyP-PZauQ
# 
# https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff


