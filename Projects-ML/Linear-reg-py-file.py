# %% [markdown]
# # Project -1 Sales Prediction 
# 
# **(Simple Linear Regression)**
# 
# ## Problem Statement
# 
# Build a model which predicts sales based on the money spent on different platforms for marketing.
# 
# ## Steps for building a regression model using Python:
# 
# 1. Import pandas and numpy libraries
# 2.  Use `read_csv` to load the dataset into DataFrame.
# 3. Identify the feature(s) (`X`) and outcome (`Y`) variable in the DataFrame for building the model.
# 4. Split the dataset into training and validation sets using `train_test_split()`.
# 5. Import statsmodel library and fit the model using `OLS()` method.
# 6.  Print model summary and conduct model diagnostics.
# 
# ## Data
# - Data is available at in the repository.
# - contains the salary of 50 graduating MBA students of a Business School in 2016 and their corresponding percentage marks in grade 10
#   
# | S. No. | Percentage in Grade 10 |   Salary  |
# |--------|------------------------|-----------|
# | 1 | 62.00 | 270000 |
# | 2 | 76.33 | 200000 |
# | 3 | 72.00 | 240000 |
# | 4 | 60.00 | 250000 |
# | 5 | 61.00 | 180000 |
# | 6 | 55.00 | 300000 |
# | . |   .   |   .    |
# | . |   .   |   .    |
# 
# In this notebook, we'll build a linear regression model to predict Sales using an appropriate predictor variable.

# %% [markdown]
# ## Mathematics used
# 
# Data points = $(x_i, y_i)$
# 
# - We want to fit a line: $\hat{y} = b_0 + b_1 x$.
# 
#     - $\hat{y}$ = predicted values of $y$
#     - $b_0$ = $y$-intercept of the line
#     - $b_1$ = coefficient or slope
# 
# - So next spet would be getting the parameters $b_0$ and $b_1$.
# - We can use the same equation to represent our actual values of $y$:
# 
#     - $y_1 = b_0 + b_1 x_1 + \epsilon_1$
#     - $y_1 = b_0 + b_1 x_1 + \epsilon_1$
#     -   .     .        .         .
#     -   .     .        .         .      
#     - $y_n = b_0 + b_1 x_n + \epsilon_n$.
#    
#    Here we have added $\epsilon$ to each equation for the error we have with respect to the estimated values:
# 
#    - $y$ = actual values
#    - $\epsilon$ = error
# - So we can take above equations for actual points and write a equation like:
#   
#   $y = X* B+\epsilon$
#   
# $$
# \begin{matrix}
# X = \begin{bmatrix} 1 & a_{1}  \\ 1  & a_{2}  \\ 1 & a_{3}  \\ 1 & a_{4} \\ \vdots & \vdots \\ 1 & a_{n}  \end{bmatrix}
# \end{matrix}
# \quad
# \begin{matrix}
# B = \begin{bmatrix} b_{0} \\ b_{1} \end{bmatrix}
# \end{matrix}
# \quad
# \begin{matrix}
# \epsilon = \begin{bmatrix} \epsilon_{1} \\ \epsilon_{2} \\ \epsilon_{3} \\ \epsilon_{4} \\ \vdots \\ \epsilon_n\end{bmatrix}
# \end{matrix}
# $$
# 
#   which means we have $(X*B)_1 = b_0 + b_1 x_1 $, $(X*B)_2 = b_0 + b_1 x_2 $, ...... , $(X*B)_n = b_0 + b_1 x_n $
# 
# 
# 

# %% [markdown]
# ## Building simple linear regression model

# %% [markdown]
# ### Importing important libraries

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

plt.style.use('ggplot')

np.set_printoptions(precision=4, linewidth=100)

# %% [markdown]
# ### Data file path

# %%
# Provide the relative path to the data file
file_path = "../ml-data/MBA-Salary.csv"

# %% [markdown]
# ### Importing the data file

# %%
# importing the data file
mba_salary_df = pd.read_csv(file_path)
mba_salary_df.head( 10 )

# %% [markdown]
# ### Checking infromation about the dataset

# %%
mba_salary_df.info()

# %%
mba_salary_df.dtypes

# %%
mba_salary_df.shape

# %%
mba_salary_df.columns

# %% [markdown]
# ### Checking few plots

# %%
# scatter plot
mba_salary_df.plot.scatter(x='Percentage in Grade 10', y='Salary')
plt.xlabel('Percentage in Grade 10')
plt.ylabel('Salary')
plt.title('Scatter Plot')
plt.show()

# %%
# line plot

mba_salary_df.plot.line(x='Percentage in Grade 10', y='Salary')
plt.xlabel('Percentage in Grade 10')
plt.ylabel('Salary')
plt.title('Line Plot')
plt.show()

# %%
# bar plot
mba_salary_df.plot.bar(x='Percentage in Grade 10', y='Salary')
plt.xlabel('Percentage in Grade 10')
plt.ylabel('Salary')
plt.title('Bar Plot')
plt.show()

# %%
# Histogram

mba_salary_df['Percentage in Grade 10'].plot.hist()
plt.xlabel('Percentage in Grade 10')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# %%
# Box plot

mba_salary_df['Percentage in Grade 10'].plot.box()
plt.ylabel('Percentage in Grade 10')
plt.title('Box Plot')
plt.show()

# %% [markdown]
# ### Importing statsmodel library for linear regression analysis
# 
# - The statsmodel library is used in python for building statistical models. 
# - OLS (Ordinary Least Squares) API available in statsmodel.api is used for estimation of the parameters for simple linear regression model. 
# - The OLS model takes two parameters Y and X.

# %%
# if you encounter `ModuleNotFoundError` for statsmodel then install it using
%pip install statsmodels

# %% [markdown]
# ### Creatin the two parameters
# - In the present case, 'Percentage in Grade 10' will be X and Salary will be Y.
# - OLS API available in statsmodel.api estimates only the coeffient of X parameter $Y=\beta_0+\beta_1 X+\epsilon$.
# - As the value of the columns remains same across all samples, the parameter estimated for this feature or column will be the intercept term.

# %%
# Creating feature Set(X) and Outcome Variable (Y)
import statsmodels.api as sm

X = sm.add_constant(mba_salary_df['Percentage in Grade 10'])
X.head(5)

# %%
Y = mba_salary_df['Salary']

# %% [markdown]
# ### Splitting the Dataset into Training and Validation Sets
# 
# - `train_test_split()` function from `skelarn.model_selection` module provides the ability to split the dataset randomly into 
#   - training and 
#   - validation datasets. 
# - The parameter `train_size` takes a fraction between `0` and `1` for specifying training set size. 
# - The remaining samples in the original set will be 
#   - test or 
#   - validation set.
# - The records that are selected for training and test set are randomly sampled. 
# - The method takes a seed value in parameter named `random_state`, to fix which samples go to training and which ones go to test set.
# - `train_test_split()` method returns four variables as below.
#   1. `train_X` contains `X` features of the training set.
#   2. `train_y` contains the values of response variable for the training set.
#   3. `test_X` contains `X` features of the test set.
#   4. `test_y` contains the values of response variable for the test set.

# %% [markdown]
# #### Importing Sklearn library
# 
# If it is not installed, use
# 
# `pip install scikit-learn`

# %%
from sklearn.model_selection import train_test_split

# %%
train_test_split

# %%
train_X, test_X, train_y, test_y = train_test_split( X, Y, train_size = 0.8, random_state = 100 )

# %% [markdown]
# - `train_size` = 0.8 implies 80% of the data is used for training the model and the remaining 20% is used for validating the model.
# - `random_state` = 100: This parameter sets the seed value for the random number generator. It ensures that the data splitting process is reproducible. 
# - Using the same `random_state` value will produce the same train-test split each time the code is executed, which is useful for consistent results and debugging.

# %%
train_X.head()

# %%
test_X

# %%
test_y

# %% [markdown]
# ### Fitting the Model
# 
# We will fit the model using OLS method and pass `train_y` and `train_X` as parameters.

# %%
mba_salary_lm = sm.OLS(train_y, train_X ).fit()

# %% [markdown]
# The `fit()` method on `OLS()` estimates the parameters and returns model information to the variable `mba_salary_lm`, which contains the model parameters, accuracy measures, and residual values among other details.

# %%
mba_salary_lm

# %% [markdown]
# ### Printing Estimated Parameters and Interpreting Them

# %%
print(mba_salary_lm.params)

# %%
# beta_0
mba_salary_lm.params[0]

# %%
# beta_1
mba_salary_lm.params[1]

# %% [markdown]
# The estimated (predicted) model can be written as
# 
# `MBA Salary = 30587.285 + 3560.587 * (Percentage in Grade 10)`
# 
# The equation can be interpreted as follows: 
# - For every 1% increase in Grade 10, the salary of the MBA students will increase by 3560.587.

# %%
mba_salary_df.plot.scatter(x='Percentage in Grade 10', y='Salary')

# %%
# Scatter plot of original data points
mba_salary_df.plot.scatter(x='Percentage in Grade 10', y='Salary', label = 'Scatter plot')

# Generate predicted values using the linear regression coefficients
MBA_salary = mba_salary_lm.params[0]+mba_salary_lm.params[1]*mba_salary_df['Percentage in Grade 10']

# Plot the fitted line
plt.plot(mba_salary_df['Percentage in Grade 10'], MBA_salary, color='red', label='Fitted Line')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Fitted Line')

# Add legend
plt.legend()

# Show the plot
plt.show()

# %% [markdown]
# ## Model Diagnostics
# 
# It is important to validate the regression model to ensure its validity and goodness of fit before it can be used for practical applications. The following measures are used to validate the simple linear regression models:
# 
# 1. Co-efficient of determination (R-squared).
# 2. Hypothesis test for the regression coefficient.
# 3. Analysis of variance for overall model validity (important for multiple linear regression).
# 4. Residual analysis to validate the regression model assumptions.
# 5. Outlier analysis, since the presence of outliers can significantly impact the regression parameters.
# 

# %% [markdown]
# ### 1. Co-efficient of Determination (R-Squared or $R^2$)
# 
# - The primary objective of regression is to explain the variation in `Y` using the knowledge of `X`. 
# - The co-efficient of determination (R-squared or $R^2$) measures the percentage of variation in `Y` explained by the model $(\beta_0+\beta_1 x)$. 
# - The simple linear regression model can be broken into:
#     1. Variation in outcome variable explained by the model. 
#     2. Unexplained variation as shown in Eq.
# 
#         $\underbrace{Y_i}_{\text{Variation in Y}} = \underbrace{\beta_0 + \beta_1 X_i}_{\text{Variation in Y explained by the model}} + \underbrace{\epsilon_i}_{\text{Variation in Y not explained by the model}}$
#     
#         It can be proven mathematically that
# 
#         $\underbrace{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}_{SST} = \underbrace{\sum_{i=1}^{n} (\hat{Y}_i - \bar{Y})^2}_{SSR} + \underbrace{\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2}_{SSE}$
# 
#         where $\hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_i$ is the predicted value of $Y_i$.
# 
#         $\hat{\beta}_1 = \sum \frac{(X_i-\bar{X})(\bar{Y}_i-\bar{Y})}{(X_i-\bar{X})^2}$
# 
#         $\hat{\beta}_0 = \bar{Y} - \beta_1 \bar{X}$
# 
#         - The hat ($\hat{\beta}_{0 ~\text{or} ~1}$) symbol is used for denoting the estimated value. 
#         - SST = is the sum of squares of total variation $\Rightarrow \sum_{i=1}^{n} (\hat{Y}_i - \bar{Y})^2$, 
#         - SSR = is the sum of squares of explained variation due to the regression model, and
#         - SSE = is the sum of squares of unexplained variation (error) $\Rightarrow \text{SSE}=\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$.
# 
#         $\boxed{R^2 = 1 - \frac{SSE}{SST}}$ 
#         
#         - Mathematically, R-squared ($R^2$) is square of correlation coefficient ($R^2 = r^2$), where $r$ is the Pearson correlation co-efficient.
#         - Higher R-squared indicates better fit; however, one should be careful about the spurious relationship.
# 
# <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 10px;">
# 
# **Assumptions of the Linear Regression Model**
#     
# 1. The errors or residuals $\epsilon_i$ are assumed to follow a normal distribution with expected value of error $\Rightarrow E(\epsilon_i)=0$.
# 2. The variance of error, $\text{VAR}(\epsilon_i)$, is constant for various values of independent variable $X$. This is known as homoscedasticity. When the variance is not constant, it is called heteroscedasticity.
# 3. The error and independent variable are uncorrelated.
# 4. The functional relationship between the outcome variable and feature is correctly defined.
# 
# **Properties of Simple Linear Regression**
# 
# 1. The mean value of $Y_i$ for given $X_i$, $\Rightarrow E(Y_i | X) = \hat{\beta}_0 + \hat{\beta_1} X$.
# 2. $Y_i$ follows a normal distribution with mean $\hat{\beta}_0 + \hat{\beta_1} X$ and variance $\text{Var}(\epsilon_i)$
# </div>

# %% [markdown]
# ### 2. Hypothesis Test for the Regression Co-efficient
# 
# - The regression co-efficient ($\beta_1$) captures the existence of a linear relationship between the outcome variable and the feature. 
# - If $\beta_1 = 0$, we can conclude that there is no statistically significant linear relationship between the two variables. 
# - It can be proved that the sampling distribution of $\beta_1$ is a t-distribution (`Kutner et al., 2013`; `U Dinesh Kumar, 2017`).
# - The null and alternative hypotheses are
#     
#     - $H_0:\beta_1 = 0$
#     - $H_1: \beta_1 \neq 0$.
# 
# - **t-test:** $t_{\alpha/2, n-2} = \frac{\hat{\beta}_1}{S_e(\hat{\beta}_1 )}$.
# - **standard error of estimate of the regression co-efficient**: 
#   
#   $S_e(\hat{\beta}_1) = \frac{S_e}{\sqrt{(X_i - \bar{X})^2}}$
# 
#   where $S_e$ is the standard error of the estimated value of $Y_i$ (and the residuals) and is given by
# 
#   $S_e = \sqrt{\frac{Y_i-\hat{Y}_i}{n-2}}$.
# 
#   The hypothesis test is a two-tailed test. The t-test is a t-distribution with n − 2 degrees of freedom (two degrees of freedom are lost due to the estimation of two regression parameters $\beta_0$ and $beta_1$).
# 
#   $S_e(\beta_1)$ is the standard error of regression co-efficient $\beta_1$.

# %% [markdown]
# ### 3. Analysis of Variance (ANOVA) in Regression Analysis
# 
# We can check the overall validity of the regression model using ANOVA in the case of multiple linear regression model with k features. The null and alternative hypotheses are given by
# 
# - $H_0: \beta_1 = \beta_2 = ... = \beta_k = 0$.
# - $H_A:$ Not all regression coefficients are zero.
# 
# The corresponding F-statistic is given by
# 
# $F=\frac{MSR}{MSE} = \frac{SSR/k}{SSE/(n-k-1)}$
# 
# where 
# 
# - $\text{MSR}= \text{SSR}/k$ and 
# - $\text{MSE} = \text{SSE}/(n − k − 1) \Rightarrow$ are mean squared regression and mean squared error, respectively. 
# - F-test is used for checking whether the overall regression model is statistically significant or not.

# %% [markdown]
# ### 4. Residual Analysis
# 
# - Residuals or errors are the difference between the actual value of the outcome variable and the predicted $(Y_i-\hat{Y}_i)$. 
# - Residual (error) analysis is important to check whether the assumptions of regression models have been satisfied. 
# - It is performed to check the following
# 
#     1. The residuals are normally distributed.
#     2. Variance of residual is constant (homoscedasticity).
#     3. The functional form of regression is correctly specified.
#     4. There are no outliers.

# %% [markdown]
# ## Regression Model Summary Using Python
# 
# The function `summary2()` prints the model summary which contains the information required for diagnosing a regression model 

# %% [markdown]
# ### 1. $R^2$, t-test and p-test

# %%
mba_salary_lm.summary2()

# %% [markdown]
# The output can be separated in three tables as follows:

# %%
mba_salary_lm.summary2().tables[0]

# %%
mba_salary_lm.summary2().tables[1]

# %%
mba_salary_lm.summary2().tables[2]

# %% [markdown]
# - The model R-squared value is $R^2=$`0.211`, that is, the model explains `21.1%` of the variation in salary.
# - The p-value for the t-test is `0.0029` which indicates that there is a statistically significant relationship (at significance value $\alpha = 0.05$) between the feature, percentage in grade `10`, and salary. (If o-value is larger than $\alpha$ then we need to retain the null hypothesis).
# - Also, the probability value of F-statistic of the model is `0.0029` which indicates that the overall model is statistically significant. 
# - Note that, in a simple linear regression, the p-value for t-test and F-test will be the same since the null hypothesis is the same. (Also $F = t^2$ in the case of SLR.)

# %% [markdown]
# ### 2. Check for Normal Distribution of Residual

# %%
mba_salary_resid1 =  mba_salary_lm.resid_pearson
 
mba_salary_resid1 

# %%
# Check for Normal Distribution of Residual
# The normality of residuals can be checked using the probability−probability plot (P-P plot)
# In Python, ProbPlot() method on statsmodel draws the P-P plot

mba_salary_resid = mba_salary_lm.resid
probplot = sm.ProbPlot(mba_salary_resid)
plt.figure( figsize = (12, 8))
probplot.ppplot( line='45' )
plt.title("Normal P-P Plot of Regression Standardized residuals")

plt.show()

# %% [markdown]
# Above can be done again in different way as follows:

# %%
# residue calculation

fitted_values = mba_salary_lm.params[0] + mba_salary_lm.params[1] * mba_salary_df['Percentage in Grade 10']

residuals = mba_salary_df['Salary'] - fitted_values

probplot = sm.ProbPlot(residuals)

plt.figure(figsize=(8, 6))
probplot.ppplot(line='45') 

#The line='45' argument specifies that the plot should include a 45-degree line, which represents perfect normality.


plt.title("P-P Plot of Regression Residuals")
plt.show()


# %% [markdown]
# Clearly the dots in the P-P plot of regression residuals are parallel to the X-axis and do not closely follow the diagonal line (representing the cumulative distribution of a normal distribution), it suggests that the residuals deviate from a normal distribution.
# 
# There could be several reasons for this behavior:
# 
# - Nonlinearity: The relationship between the independent variables and the dependent variable may not be strictly linear. In such cases, the residuals may exhibit non-normal patterns.
# - Heteroscedasticity: The variability of the residuals may change across different ranges of the independent variable. This violates the assumption of constant variance, which can affect the normality of the residuals.
# - Outliers: The presence of outliers in the data can distort the residuals and make them deviate from a normal distribution.
# - Missing variables: If important variables that affect the dependent variable are not included in the model, the residuals may exhibit non-normal patterns.
# - Data transformation: If the data is not transformed appropriately (e.g., using logarithmic transformation), it can lead to non-normal residuals.

# %%
residuals

# %% [markdown]
# ### 3. Test of Homoscedasticity
# 
# - An important assumption of the regression model is that the residuals have constant variance (homoscedasticity) across different values of the predicted value (Y).
# - The homoscedasticity can be observed by drawing a residual plot, which is a plot between standardized residual value and standardized predicted value. 
# - If there is heteroscedasticity (non-constant variance of residuals), then a funnel type shape in the residual plot can be expected. A non-constant variance of the residuals is known as heteroscedasticity.
# - The following custom method `get_standardized_values()` creates the standardized values of a series of values (variable). It subtracts values from mean and divides by standard deviation of the variable.

# %%
def get_standardized_values( vals ): 
    return (vals - vals.mean())/vals.std()

# %%
plt.scatter(get_standardized_values(mba_salary_lm.fittedvalues), 
get_standardized_values(mba_salary_resid)) 
plt.title("Residual Plot: MBA Salary Prediction"); 
plt.xlabel("Standardized predicted values")
plt.ylabel("Standardized Residuals");

# %% [markdown]
# It can be observed in this plot, that the residuals are random and have no funnel shape, which means the residuals have constant variance (homoscedasticity).

# %% [markdown]
# ### 4. Outlier Analysis
# 
# Outliers  are  observations  whose  values  show  a  large  deviation  from  the  mean  value.  Presence  of  an  outlier can have a significant influence on the values of regression coefficients. Thus, it is important to identify the existence of outliers in the data.
# The following distance measures are useful in identifying influential observations:
# 
# 1. Z-Score
# 2. Mahalanobis Distance 
# 3. Cook’s Distance
# 4.  Leverage Values

# %% [markdown]
# #### 4.1. Z-Score
# 
# Z-score is the standardized distance of an observation from its mean value. For the predicted value of the dependent variable Y, the Z-score is given by
# 
# $Z=\frac{Y_i - \bar{Y}}{\sigma_Y}$
# 
# where $Y_i$ is the predicted value of $Y$ for $ith$ observation, $Y$ is the mean or expected value of $Y$, $\sigma_Y$ is the variance of $Y$.
# Any observation with a Z-score of more than 3 may be flagged as an outlier. The Z-score in the data can be obtained using the following code:

# %%
from scipy.stats import zscore

# %%
mba_salary_df['z_score_salary'] = zscore( mba_salary_df.Salary )

# %%
mba_salary_df[(mba_salary_df.z_score_salary > 3.0)|(mba_salary_df.z_score_salary < -3.0) ]

# %% [markdown]
# This code snippet, selects rows where the value of `z_score_salary>3.0` or `z_score_salary< -3.0`. The | symbol represents the logical OR operator, so either condition being true will result in the row being selected.
# 
# 
# So, there are no observations that are outliers as per the Z-score.

# %% [markdown]
# #### 4.2. Cook’s Distance
# 
# - Cook’s distance measures how much the predicted value of the dependent variable changes for all the observations in the sample when a particular observation is excluded from the sample for the estimation of regression parameters.
# - **A  Cook’s  distance  value  of  more  than  1  indicates  highly  influential  observation.**  
# - Python  code  for  calculating Cook’s distance is provided below. 
# - In this `get_influence()` returns the influence of observations in the model and cook_distance variable provides Cook’s distance measures. Then the distances can be plotted against the observation index to find out which observations are influential.

# %%
mba_influence = mba_salary_lm.get_influence() #This retrieves the influence object from the linear regression model

(c, p) = mba_influence.cooks_distance
# This line calculates the Cook's distances for each observation using the cooks_distance method of the influence object. 
# The cooks_distance method returns two arrays: 
# c represents the Cook's distances, and 
# p represents the p-values 
# associated with each distance.

plt.stem(np.arange(len(train_X)), np.round(c, 3), markerfmt=','); #stem plot 

plt.title( "Cooks distance for all observations in MBA Salaray data set" );

plt.xlabel("Row index")
plt.ylabel('Cooks Distance');

# %% [markdown]
# It can be observed that none of the observations’ Cook’s distance exceed 1 and hence none of them are outliers.

# %% [markdown]
# #### 4.3. Leverage Values
# 
# - Leverage  value  of  an  observation  measures  the  influence  of  that  observation  on  the  overall  fit  of  the  regression function and is related to the Mahalanobis distance. 
# - Leverage value of more than `3(k + 1)/n` is treated as highly influential observation, where `k` is the number of features in the model and `n` is the sample size.
# - `statsmodels.graphics.regressionplots` module provides `influence_plot()` which draws a plot between standardized residuals and leverage value. 
# - Mostly, the observations with high leverage value (as men- tioned above) and high residuals [more than value `3(k + 1)/n`] can be removed from the training dataset.

# %%
from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) )

influence_plot( mba_salary_lm, ax = ax ) 
plt.title('Leverage Value Vs Residuals') 
plt.show();

# %% [markdown]
# the size of the circle is proportional to the product of residual and leverage value. The larger the circle, the larger is the residual and hence influence of the observation.

# %% [markdown]
# ### 5. Making Prediction and Measuring Accuracy
# 
# Ideally, the prediction should be made on the validation (or test) data and the accuracy of prediction should be evaluated.
# 
# #### 5.1. Predicting using the Validation Set
# 
# The model variable has a method `predict()`, which takes the X parameters and returns the predicted values.

# %%
pred_y = mba_salary_lm.predict( test_X )
pred_y

# %% [markdown]
# `pred_y` variable contains the predicted value. We can compare the predicted values with the actual values and calculate the accuracy in the next section.

# %%
plt.scatter(test_y, pred_y)
plt.xlabel('Original Y')
plt.ylabel('Predicted Y')
plt.title('Comparison of Original Y and Predicted Y')
plt.show()

# %% [markdown]
# and corresponding metrics are calculated below.

# %% [markdown]
# #### 5.2. Finding R-Squared and RMSE
# 
# - Several measures can be used for measuring the accuracy of prediction. 
# 
#   - Mean Square Error (MSE), 
#   - Root Mean Square Error (RMSE) and 
#   - Mean Absolute Percentage Error (MAPE) 
#   
# are some of the frequently used measures. `sklearn.metrics` has `r2_score` and `mean_squared_error` for measuring R-squared and MSE values. 
# 
# - We need to take the square root of the MSE value to get RMSE value. Both the methods take predicted Y values and actual Y values to calculate the accuracy measures. 
# 
# - Numpy module has `sqrt` method to calculate the square root of a value.

# %%
from sklearn.metrics import r2_score, mean_squared_error

# %%
# R2
np.abs(r2_score(test_y, pred_y))

# %% [markdown]
# So, the model only explains 15.6% of the variance in the validation set.

# %%
# MSE
np.sqrt(mean_squared_error(test_y, pred_y))

# %% [markdown]
# RMSE means the average error the model makes in predicting the outcome. The smaller the value of RMSE, the better the model is.

# %% [markdown]
# #### 5.3. Calculating Prediction Intervals
# 
# - The regression equation gives us the point estimate of the outcome variable for a given value of the independent variable. 
# - In many applications, we would be interested in knowing the interval estimate of $Y_i$ for a given value of explanatory variable. 
# - `wls_prediction_std()` returns the prediction interval while making a prediction. 
# - It takes significance value (a) to calculate the interval. 
# - An a-value of 0.1 returns the prediction at confidence interval of 90%. The code for calculating prediction interval is as follows:

# %%
from statsmodels.sandbox.regression.predstd import wls_prediction_std # Predict the y values

pred_y = mba_salary_lm.predict( test_X )

# Predict the low and high interval values for y
_, pred_y_low, pred_y_high = wls_prediction_std( mba_salary_lm, test_X, alpha = 0.1)

# %%
_

# %%
pred_y_low

# %%
pred_y_high

# %%
# Store all the values in a dataframe
pred_y_df = pd.DataFrame({'grade_10_perc': test_X['Percentage in Grade 10'],
                          'pred_y': pred_y,
                          'pred_y_left': pred_y_low,
                          'pred_y_right': pred_y_high
                          })

# %%
pred_y_df.head()

# %% [markdown]
# Above code performs prediction using a linear regression model (`mba_salary_lm`) and creates a dataframe (`pred_y_df`) to store the predicted values and their confidence intervals.
# 
# Here's a breakdown of the code:
# 
# - `pred_y = mba_salary_lm.predict(test_X)`: This line uses the linear regression model (`mba_salary_lm`) to predict the dependent variable values (`pred_y`) based on the independent variable values (`test_X`).
# 
# - `_, pred_y_low, pred_y_high = wls_prediction_std(mba_salary_lm, test_X, alpha=0.1)`: This line calculates the confidence intervals for the predicted values. The `wls_prediction_std` function is used to obtain the standard errors and confidence intervals based on the linear regression model (`mba_salary_lm`), independent variable values (`test_X`), and a significance level of `0.1`. The resulting `pred_y_low` and `pred_y_high` arrays represent the lower and upper bounds of the confidence intervals, respectively.
# 
# - The first underscore (`_`) is a conventional placeholder variable commonly used when you want to ignore or discard a value returned by a function.
# 
# - `pred_y_df = pd.DataFrame({'grade_10_perc': test_X['Percentage in Grade 10'], 'pred_y': pred_y, 'pred_y_left': pred_y_low, 'pred_y_right': pred_y_high })`: This line creates a dataframe (`pred_y_df`) to store the predicted values (`pred_y`) and their confidence intervals. 
#   
# - The dataframe has four columns: `grade_10_perc` containing the independent variable values from `test_X`, `pred_y` containing the predicted values, `pred_y_left` containing the lower bounds of the confidence intervals, and `pred_y_right` containing the upper bounds of the confidence intervals.

# %% [markdown]
# ##### 5.3.1 Scatter Plot with Prediction Intervals
# 
# Create a scatter plot to visualize the relationship between the grade_10_perc feature and the predicted values (pred_y). Additionally, plot the prediction intervals (pred_y_low and pred_y_high) as error bars or shaded regions to show the uncertainty in the predictions.

# %%
plt.scatter(pred_y_df['grade_10_perc'], pred_y_df['pred_y'], label='Predicted Y')
plt.errorbar(pred_y_df['grade_10_perc'], pred_y_df['pred_y'], yerr=(pred_y_df['pred_y'] - pred_y_df['pred_y_left'], pred_y_df['pred_y_right'] - pred_y_df['pred_y']), fmt='none', color='gray', alpha=0.5, label='Prediction Interval')
plt.xlabel('Grade 10 Percentage')
plt.ylabel('Predicted Y')
plt.title('Scatter Plot of Predicted Y with Prediction Intervals')
plt.legend()
plt.show()

# %% [markdown]
# ##### 5.3.2. Line plot
# 
# Create a line plot to visualize the trend of the predicted values (pred_y) over the grade_10_perc feature.

# %%
plt.plot(pred_y_df['grade_10_perc'], pred_y_df['pred_y'], label='Predicted Y')
plt.xlabel('Grade 10 Percentage')
plt.ylabel('Predicted Y')
plt.title('Line Plot of Predicted Y')
plt.legend()
plt.show()

# %% [markdown]
# ##### 5.3.3. Prediction Interval Plot
# 
# Create a plot that shows the prediction intervals (pred_y_low and pred_y_high) as shaded regions along with the predicted values (pred_y) plotted as a line or markers.

# %%
plt.plot(pred_y_df['grade_10_perc'], pred_y_df['pred_y'], label='Predicted Y')
plt.fill_between(pred_y_df['grade_10_perc'], pred_y_df['pred_y_left'], pred_y_df['pred_y_right'], alpha=0.2, label='Prediction Interval')
plt.xlabel('Grade 10 Percentage')
plt.ylabel('Predicted Y')
plt.title('Prediction Interval Plot')
plt.legend()
plt.show()

# %% [markdown]
# ##### 5.3.4. Actual vs. Predicted Plot
# 
# Create a scatter plot to compare the actual values (test_y) with the predicted values (pred_y).

# %%
plt.scatter(pred_y_df['grade_10_perc'], test_y, label='Actual Y')
plt.scatter(pred_y_df['grade_10_perc'], pred_y, label='Predicted Y')
plt.xlabel('Grade 10 Percentage')
plt.ylabel('Y')
plt.title('Actual vs. Predicted Plot')
plt.legend()
plt.show()

# %% [markdown]
# ##### 5.3.5. Residual Plot
# 
# Plot the residuals (difference between actual and predicted values) against the grade_10_perc feature to assess the model's performance and check for any patterns or systematic errors.

# %%
residuals = test_y - pred_y
plt.scatter(pred_y_df['grade_10_perc'], residuals)
plt.xlabel('Grade 10 Percentage')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# %% [markdown]
# ##### 5.3.6. Histogram of Residuals
# 
# Visualize the distribution of residuals using a histogram to check if they follow a normal distribution or exhibit any skewness.

# %%
residuals = test_y - pred_y
plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# %% [markdown]
# ##### 5.3.7. QQ Plot of Residuals: 
# Plot a QQ plot (Quantile-Quantile plot) to assess whether the residuals follow a normal distribution by comparing them to the theoretical quantiles of a normal distribution.

# %%
sm.qqplot(residuals, line='45')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('QQ Plot of Residuals')
plt.show()

# %% [markdown]
# ##### 5.3.8. Visualize the distribution of predicted values using a histogram with a kernel density estimation.

# %%
sns.histplot(data=pred_y_df, x='pred_y', kde=True)
plt.xlabel('Predicted Y')
plt.ylabel('Frequency')
plt.title('Distribution Plot')
plt.show()

# %% [markdown]
# ##### 5.3.9. Box Plot
# 
# Visualize the distribution of predicted values (pred_y) using a box plot.

# %%
sns.boxplot(data=pred_y_df, y='pred_y')
plt.ylabel('Predicted Y')
plt.title('Box Plot')
plt.show()

# %% [markdown]
# ##### 5.3.10. Violin Plot
# 
# Combine a box plot and a kernel density plot to show the distribution of predicted values.

# %%
sns.violinplot(data=pred_y_df, y='pred_y')
plt.ylabel('Predicted Y')
plt.title('Violin Plot')
plt.show()

# %% [markdown]
# ##### 5.3.11. Pair Plot
# 
# Create a grid of scatter plots to visualize the pairwise relationships between grade_10_perc, pred_y, pred_y_left, and pred_y_right.

# %%
sns.pairplot(data=pred_y_df, vars=['grade_10_perc', 'pred_y', 'pred_y_left', 'pred_y_right'])
plt.title('Pair Plot')
plt.show()

# %% [markdown]
# ##### 5.3.12. Heatmap
# 
# Display the correlation matrix of the pred_y_df DataFrame using a heatmap.

# %%
corr_matrix = pred_y_df[['grade_10_perc', 'pred_y', 'pred_y_left', 'pred_y_right']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


