# %% [markdown]
# # Statistics
# 
# * Statistics is the science of collecting, analysing, presenting, and interpreting data.
# * Currently the need to turn the large amounts of data available in many applied fields into useful information has stimulated both theoretical and practical developments in statistics.
# * Data are the facts and figures that are collected, analysed, and summarized for presentation and interpretation.
# * Data may be classified as 
# 
#    a) Qualitative
#    
#    b) Quantitative
# 
# * Quantitative data measure either how much or how many of something, and qualitative data provide labels, or names, for categories of like items.
# * Sample survey methods are used to collect data from observational studies, and experimental design methods are used to collect data from experimental studies.
# * The area of descriptive statistics is concerned primarily with methods of presenting and interpreting data using graphs, tables, and numerical summaries. Whenever statisticians use data from a sample to make statements about a data, they are performing statistical inference.
# * Estimation and hypothesis testing are procedures used to make statistical inferences.
# * Methods of probability were developed initially for the analysis of gambling games.
# * Probability plays a key role in statistical inference; it is used to provide measures of the quality and precision of the inferences.
# * The statistical inference (अनुमान) are used primarily for single-variable studies, while others, such as regression and correlation analysis, are used to make inferences about relationships among two or more variables.

# %% [markdown]
# ## Descriptive statisctics
# 
# - Descriptive statistics are tabular, graphical, and numerical summaries of data, i.e.,
#     * tabular,
#     * graphics method or
#     * numerical (for example central tendency and variability).
# - Descriptive statistics are brief informational coefficients that summarize a given data set, which can be either a representation of the entire data or a sample of a data.
# - The purpose of descriptive statistics is to facilitate the presentation and interpretation of data.
# - Descriptive statistics consists of three basic categories of measures:
#     1. measures of central tendency: focus on the average or middle values of data sets
#     2. measures of variability (or spread): aid in analysing how dispersed the distribution is for a set of data
#     3. frequency distribution.
# - Measures of central tendency describe the centre of the data set (mean, median, mode).
# - Measures of variability describe the dispersion of the data set (variance, standard deviation).
# - Measures of frequency distribution describe the occurrence of data within the data set (count)

# %% [markdown]
# ### 1. Tabular methods
# - The most commonly used tabular summary of data for a single variable is a frequency distribution.
# - A frequency distribution shows the number of data values in each of several nonoverlapping classes.
# - Another tabular summary, called a relative frequency distribution, shows the fraction, or percentage, of data values in each class.
# - The most common tabular summary of data for two variables is a cross tabulation, a two-variable analogue of a frequency distribution.
# - Constructing a frequency distribution for a quantitative variable requires more care in defining the classes and the division points between adjacent classes.
# - A frequency distribution would show the number of data values in each of these classes, and a relative frequency distribution would show the fraction of data values in each.
# - A cross tabulation is a two-way table with the rows of the table representing the classes of one variable and the columns of the table representing the classes of another variable.
# 
# ### 2. Graphical Methods
# A number of graphical methods are available for describing data.
# 
# - Dot plots.
# - Histograms.
# - Box-whisker plots.
# - Scatter plots.
# - Bar charts.
# - Pie charts
# 
# ### 3. Numerical statistics This is broken down into-
# - Measures of central tendency: include the
#     * mean,
#     * median,
#     * mode,
#     * percentiles
# 
# - **Measures of variability (spread):** include
#     * standard deviation,
#     * variance,
#     * minimum and maximum variables,
#     * kurtosis, and
#     * skewness.
# 
# - **Outliers:** 
# Sometimes data for a variable will include one or more values that appear unusually large or small and out of place when compared with the other data values. These values are known as outliers and often have been erroneously included in the data set.
# * The mean and standard deviation are used to identify outliers.
# * A z-score can be computed for each data value.
# * With x representing the data value, x̄ the sample mean, and s the sample standard deviation, the z-score is given by
# $$z=\frac{x-\bar{x}}{s}$$
# * The z-score represents the relative position of the data value by indicating the number of standard deviations it is from the mean.
# *  A rule of thumb is that any value with a z-score less than −3 or greater than +3 should be considered an outlier (i.e. 𝑧<−3,𝑜𝑟 𝑧>+3).
# 
# #### Mean or expectation value
# 
# The expected value, or mean, of a random variable—denoted by E(x) or μ—is a weighted average of the values the random variable may assume. The formulas for computing the expected values of discrete and continuous random variables are given by:
# 
# E(x)= $\sum_i$ $x_i$ $p_i(x)$ $~~~~~~~~~~~~~~~~~~~~~~~~$  for discrete variables
# 
# E(x)= $\int$ dx x p(x) $~~~~~~~~~~~~~~~~~~~~~~~~~~~$  for continuous variables
# 
# If $x_1$, $x_2$, $x_3$, ..... $x_i$ ...., $x_k$ have frequency $f_1$, $f_2$, $f_3$,…… $f_k$ then 
# 
# 𝐸(𝑥) = $\sum_i$ $\frac{f_i x_i}{N}$.
# 
# #### Variance
# 
# * In statistics, variance refers to the spread of a data set. It’s a measurement used to identify how far each number in the data set is from the mean.
# * The larger the variance, the more spread in the data set.
# * A large variance means that the numbers in a set are far from the mean and each other. A small variance means that the numbers are closer together in value.
# * Variance is calculated by taking the differences between each number in a data set and the mean, squaring those differences to give them positive value, and dividing the sum of the resulting squares by the number of values in the set.
# * Advantage: One of the primary advantages of variance is that it treats all deviations from the mean of the data set in the same way, regardless of direction. This ensures that the squared deviations cannot sum to zero, which would result in giving the appearance that there was no variability in the data set at all.
# * **Disadvantage:** One of the most commonly discussed disadvantages of variance is that it gives added weight to numbers that are far from the mean, or outliers. Squaring these numbers can at times result in skewed interpretations of the data set as a whole.
# * **Formula:** The variance of a random variable, denoted by Var(x) or σ2, is a weighted average of the squared deviations from the mean. The formulas for computing the variances of discrete and continuous random variables are given by:
# 
# Var(x) = $\sigma^2=\sum_i (x_i-\mu)^2 P_i(x) ~~~~~~~~$  (for discrete variables)
# 
# Var(x) = $\int dx ~ (x-\mu)^2 ~ p(x) ~~~~~~~~$  (for continuous variables)
# 
# In this formula, $x$ represents an individual data point, $\mu$ represents the mean of the data points, and $n$ represents the total number of data points.
# 
# <img src="variance.png" width="700" height="450" />
# 
# ([Reference for the figure](https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1))
# 
# #### Standard deviation
# 
# The standard deviation, denoted σ, is the positive square root of the variance., i.e. $\sigma= \sqrt{Var(x)}$. Since the standard deviation is measured in the same units as the random variable and the variance is measured in squared units, the standard deviation is often the preferred measure.
# 
# #### Covariance
# 
# * Covariance provides insight into how two variables are related to one another.
# * More precisely, covariance refers to the measure of how two random variables in a data set will change together.
# * A positive covariance means that the two variables at hand are positively related, and they move in the same direction.
# * A negative covariance means that the variables are inversely related, or that they move in opposite directions.
# * A zero covariance means that the variables are not related to each other.
# 
# Cov(X, Y) = $\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{N-1} ~~~~~ $ (summed over i, from 0 to N)
#     
# In this formula, $X$ represents the independent variable, $Y$ represents the dependent variable, $N$ represents the number of data points in the sample, $\bar{x}$ represents the mean of the $X$, and $\bar{y}$ represents the mean of the dependent variable $Y$. Note that while calculating a sample variance in order to estimate a population variance, the denominator of the variance equation becomes N – 1. This removes bias from the estimation.
# 
# <img src="Covariance.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
# 
# 
# #### Correlation
# 
# * Covariance and correlation both primarily assess the relationship between variables.
# * The closest analogy to the relationship between them is the relationship between the variance and standard deviation.
# * Covariance measures the total variation of two random variables from their expected values. Using covariance, we can only gauge the direction of the relationship (whether the variables tend to move in tandem or show an inverse relationship). However, it does not indicate the strength of the relationship, nor the dependency between the variables.
# * On the other hand, correlation measures the strength of the relationship between variables. Correlation is the scaled measure of covariance. It is dimensionless. In other words, the correlation coefficient is always a pure value and not measured in any units.
# 
# <img src="Correlaltion.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
# 
# ([for reference click the website](https://www.geeksforgeeks.org/robust-correlation/))
# 
# * **Relation between the covariance and correlation:**
# 
#     $\rho(X,Y)=\frac{Cov(X,Y)}{\sigma_X \sigma_Y}$
# 
#     Where $ρ(X,Y)$ – is the correlation between the variables X and Y
#     
#     COV(X,Y) – is the covariance between the variables X and Y
# 
#     $\sigma_X$ – is the standard deviation of the X-variable
# 
#     $\sigma_Y$ – is the standard deviation of the Y-variable
# 
# * **Advantages of the Correlation Coefficient:**
#     1. Covariance can take on practically any number while a correlation is limited: -1 to +1.
#     2. Because of its numerical limitations, correlation is more useful for determining how strong the relationship is between the two variables.
#     3. Correlation does not have units. Covariance always has units.
#     4. Correlation isn’t affected by changes in the centre (i.e. mean) or scale of the variables.

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
# - If the occurrence or non-occurrence of $E_1$ does not affect the probability of occurrence of $E_2$, then Pr{$E_2$ | $E_1$}=Pr{$E_2$} and we say that $E_1$ and $E_2$ are independent events, they are dependents.
# - If we denote by ( $E_1$ $E_2$) the event that "both $E_1$ and $E_2$ occur,’’ sometimes called a compound event, then
# 
# Pr{ $𝐸_1$ $𝐸_2$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ | $𝐸_1$ }
# 
# - Similarly for three events $(𝐸_1 𝐸_2 𝐸_3)$ 
# 
# Pr{ $𝐸_1$ $𝐸_2$ $𝐸_3$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ | $𝐸_1$ } Pr{ $𝐸_3$ | $𝐸_2$ $𝐸_1$ }
# 
# If these events are independent, then 
# 
# Pr{ $𝐸_1$ $𝐸_2$ } = Pr{ $𝐸_1$ } Pr{ $𝐸_2$ }.
# 
# Similarly 
# 
# Pr{ $𝐸_1$ $𝐸_2$ $𝐸_3$}=Pr{ $𝐸_1$ } Pr{ $𝐸_2$ } Pr{ $𝐸_3$}.
# 
# 
# ### Mutually exclusive events
# 
# - Two or more events are called mutually exclusive if the occurrence of any one of them excludes the occurrence of the others. Thus if $E_1$ and $E_2$ are mutually exclusive events, then
# 
# Pr{ $𝐸_1$ $𝐸_2$ } = 0.
# 
# - If $E_1 + E_2$ denotes the event that ‘‘either $E_1$ or $E_2$ or both occur’’, then
# 
# Pr{ $𝐸_1$ + $𝐸_2$ } = Pr{ $𝐸_1$ } + Pr{ $𝐸_2$ } − Pr{ $𝐸_1$ $𝐸_2$ }.
# 
# ### Random variables
# 
# - A random variable is a numerical description of the outcome of a statistical experiment.
# - A random variable that may assume only a finite number or an infinite sequence of values is said to be discrete; one that may assume any value in some interval on the real number line is said to be continuous
# * Discrete random variables
# * Continuous random variables
# 
# <img src="https://miro.medium.com/max/640/1*7DwXV_h_t7_-TkLAImKBaQ.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>

# %% [markdown]
# ## Probability distribution
# 
# The probability distribution for a random variable describes how the probabilities are distributed over the values of the random variable. Based on the variables, probability distributions are of two type mainly: 
# 
# 1. Discrete probability distribution, and 
# 2. Continuous probability distribution.
# 
# <img src="https://miro.medium.com/max/720/1*4uD1j7NvakmaLmlpgGwk-A.webp" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
# 
# ### 1. Discrete probability distribution
# 
# For a discrete random variable, $x$, the probability distribution is defined by a probability mass function, denoted by $p(x)$. This function provides the probability for each value of the random variable.
# 
# Following two conditions must be satisfied for $p(x)$
# - $p(x)$ must be nonnegative for each value of the random variable, and
# - the sum of the probabilities for each value of the random variable must equal one.
# 
# ### 2. Continuous probability distribution
# 
# - A continuous random variable may assume any value in an interval on the real number line or in a collection of intervals. Since there is an infinite number of values in any interval, it is not meaningful to talk about the probability that the random variable will take on a specific value; instead, the probability that a continuous random variable will lie within a given interval is considered.
# 
# - In the continuous case, the counterpart of the probability mass function is the probability density function, also denoted by $p(x)$. For a continuous random variable, the probability density function provides the height or value of the function at any particular value of $x$; it does not directly give the probability of the random variable taking on a specific value. However, the area under the graph of $p(x)$ corresponding to some interval, obtained by computing the integral of $p(x)$ over that interval, provides the probability that the variable will take on a value within that interval.
# 
# - A probability density function must satisfy two requirements:
# 
#     * $f(x)$ must be nonnegative for each value of the random variable, and
#     * the integral over all values of the random variable must equal one.
# - Following are the special probability distribution functions
#     * The binomial distribution
#     * The Poisson distribution
#     * The normal distribution
# 
# ## Special continuous distribution functions
# 
# ### i. Binomial Distribution function
# 
# If $p$ is the probability that an event will happen in any single trial (called the probability of a success) and $q = (1 - p)$ is the probability that it will fail to happen in any single trial (called the probability of a failure), then the probability that the event will happen exactly $x$ times in $N$ trials (i.e., $x$ successes and $N-x$ failures will occur) is given by
# 
#  
# $$𝑝(X)=𝑁_{𝐶_𝑥} 𝑝^𝑥 𝑞^{𝑁−𝑥}$$
# 
# where $x= 0, 1, 2, . . . ,N;$ where $N! =N(N - 1)(N-2) …….. 1 $; and $0! = 1$ by definition.
# 
# |     Statistics          |       Formula      |
# |-------------------------|--------------------|
# |            Mean         |        $\mu=N p$   |
# |      Variance           | $\sigma^2 = N p q$ |
# | Standard deviation | $\sigma = \sqrt{N p q}$ |
# | Moment coefficient of skewness | $\alpha_3 = \frac{q-p}{\sqrt{N p q}}$ |
# | Moment coefficient of Kurtosis | $\alpha_4 = 3+ \frac{1-6pq}{N p q}$ |
# 
# 
# ### ii. The Poisson distribution function
# 
# The discrete probability distribution
# 
# $$p(X) =  \frac{\lambda^X e^{-\lambda}}{X!}$$
# 
# where $\lambda$ is a given constant, is called the Poisson distribution.
# 
# | Statistics     |    Formula   |
# |----------------|--------------|
# | Mean | $\mu=\lambda $ |
# | Variance | $\sigma^2 = \lambda$ |
# |Standard deviation | $\sigma = \sqrt{\lambda}$ |
# | Moment coefficient of skewness | $\alpha_3 = \frac{1}{\sqrt{\lambda}}$ |
# | Moment coefficient of Kurtosis | $\alpha_4 = 3+ \frac{1}{\lambda}$ |
# 
# ### iii. The normal distribution
# 
# One of the most important examples of a continuous probability distribution is the normal distribution, normal curve, or Gaussian distribution. It is defined by the equation:
# 
# 
# <img src="normal-distri.png" alt= "MArkdown Monster icon" style= "float: center; margin-right: 10px;"/>
# 
# * Area between $( \mu - \sigma )$ to $( \mu + \sigma )$ = 68.27   %
# * Area between $( \mu - 2 \sigma )$ to $( \mu + 2 \sigma )$ = 95.45 %
# * Area between $( \mu - 3 \sigma )$ to $( \mu + 3 \sigma )$ = 99.73  %
# 
# where $\mu$ = mean, $\sigma$ = standard deviation. The total area bounded by the curve Y and X axis is 1. Hence the area under the curve between two ordinates X = a and X = b, where a < b, represents the probability that X lies between a and b. This probability is denoted by Pr{a < X < b }.
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
# * Relations between the Binomial and normal distributions:
# 
#     If N is large and if neither p nor q is too close to zero, the binomial distribution can be closely approximated by a normal distribution with standardized variable given by
# 
#     $$z = \frac{ x - N p}{Npq}.$$
# 
# ### iv. Exponential distribution
# 
# In probability theory and statistics, the exponential distribution is the probability distribution of the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate. The probability density function (pdf) of an exponential distribution is
# 
#    $$f(x; \lambda) = 
#    \begin{cases}
#       \lambda e^{-\lambda x}, & x\geq 0, \\
#       0, & x < 0 .
#     \end{cases}
#    $$
# 
# Here λ > 0 is the parameter of the distribution, often called the rate parameter. The distribution is supported on the interval [0, ∞). The exponential distribution exhibits infinite divisibility.
# 
# |Statistics| Formula|
# |----------|--------|
# | Mean | $𝐸[𝑋]=\frac{1}{\lambda}$ |
# | Median | $m[X] =\frac{ln(2)}{\lambda} < E[X]$ |
# | Variance | $𝑉𝑎𝑟[𝑋]=\frac{1}{\lambda^2}$ |
# | Moments | $E[X^n]=\frac{n!}{\lambda^n}$|
# 
# ### v. Uniform distribution function
# 
# In statistics, uniform distribution refers to a type of probability distribution in which all outcomes are equally likely. The formula for a discrete uniform distribution is
# 
# $$P_x=\frac{1}{n}$$
# 
# where $P_x$= Probability of a discrete value
# 
# n= Number of values in the range

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define the range of k values to plot
k = np.arange(0, 21)

# Define the lambda values to plot
lambdas = [1, 3, 5]

# Plot the Poisson distributions for each lambda value
for lam in lambdas:
    # Calculate the Poisson PMF for the given lambda and k values
    poisson_pmf = poisson.pmf(k, lam)
    
    # Plot the Poisson PMF for the current lambda value
    plt.plot(k, poisson_pmf, label=f'λ = {lam}')

# Add plot labels and legend
plt.xlabel('Number of events (k)')
plt.ylabel('Probability')
plt.title('Poisson Distribution')
plt.legend()

# Show the plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# set the lambda values
lambdas = [1, 2, 3, 4, 5]

# set the range of x values
x = np.arange(0, 21)

# generate and plot the Poisson distributions
for lam in lambdas:
    plt.plot(x, poisson.pmf(x, lam), label=f'$\lambda$={lam}')

# set the plot labels and legend
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Poisson Distribution with Different Lambda Values')
plt.legend()
plt.show()

# %% [markdown]
# ## Central Limit theorem (CLT): 
# 
# In probability theory, the central limit theorem (CLT) states that the distribution of a sample variable approximates a normal distribution (i.e., a “bell curve”) as the sample size becomes larger, assuming that all samples are identical in size, and regardless of the population's actual distribution shape.
# 
# - Sample sizes equal to or greater than 30 are often considered sufficient for the CLT to hold. CLT is useful in finance when analysing a large collection of securities to estimate portfolio distributions and traits for returns, risk, and correlation.
# - According to the central limit theorem, the mean of a sample of data will be closer to the mean of the overall population in question, as the sample size increases, notwithstanding the actual distribution of the data. In other words, the data is accurate whether the distribution is normal or aberrant.
# - Key component of CLT is:
#     * Sampling is successive,
#     * Sampling is random,
#     * Samples should be independent,
#     * Samples should be limited,
#     * Sample size is increasing.
# - A Central Limit Theorem word problem will most likely contain the phrase “assume the variable is normally distributed”, or one like it.


