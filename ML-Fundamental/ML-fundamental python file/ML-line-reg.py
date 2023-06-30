# %% [markdown]
# # Introduction to Linear Regression in Machine Learning
# 
# Linear regression is a popular and widely used algorithm in machine learning for predicting continuous numeric values. It models the relationship between independent variables (input features) and a dependent variable (target variable) by fitting a linear equation to the observed data. In this section, we will provide a brief overview of linear regression, including the mathematical explanation and figures to aid understanding.
# 
# <img src="ML-image/Linear-reg1.png" width="350" height="300" />

# %% [markdown]
# ## Mathematical Explanation:
# The linear regression algorithm aims to find the best-fit line that represents the relationship between the input features ($x$) and the target variable ($y$). The equation for a simple linear regression can be expressed as: 
# 
# $y=mx+b$
# 
# where
# 
# - $y$ represents the target variable or the dependent variable we want to predict.
# - $x$ represents the input feature or the independent variable.
# - $m$ represents the slope of the line, which represents the rate of change of $y$ with respect to $x$.
# - $b$ represents the $y$-intercept, which is the value of $y$ when $x$ is equal to $0$.
# 
# <img src="ML-image/Linear-reg0.png" width="350" height="300" />

# %% [markdown]
# ### Relationship of regression lines
# 
# - A linear line showing the relationship between the dependent and independent variables is called a regression line. 
# - A regression line can show two types of relationship:
# 
# 1. **Positive Linear Relationship:** If the dependent variable increases on the Y-axis and independent variable increases on X-axis, then such a relationship is termed as a Positive linear relationship.
# 
# <img src="ML-image/pos-lin-reg.png" width="300" height="250" />
# 
# 1. **Negative Linear Relationship:** If the dependent variable decreases on the Y-axis and independent variable increases on the X-axis, then such a relationship is called a negative linear relationship.
# 
# <img src="ML-image/neg-lin-reg.png" width="300" height="250" />

# %% [markdown]
# ## Types of Linear Regression
# 
# Linear regression can be further divided into two types of the algorithm:
# 
# 1. **Simple Linear Regression:** If a single independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Simple Linear Regression.
# 2. **Multiple Linear regression:** If more than one independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Multiple Linear Regression.

# %% [markdown]
# ### 1. Simple linear regression 
# 
# Simple linear regression (SLR) is a statistical model that focuses on the relationship between a single independent variable (or feature) and an outcome variable. In SLR, the functional relationship between the outcome variable and the regression coefficient is linear. This means that the mathematical function describing the relationship is a straight line and the impact of the independent variable on the outcome is represented by a linear regression coefficient.
# 
# **Example:** predicting housing prices based on features such as the number of bedrooms, square footage, and location.
# 
# #### 1.1. Mathematical Explanation:
# There are parameters $\beta_0$, $\beta_1$, and $\sigma^2$, such that for any fixed value of the independent variable $x$, the dependent variable is a random variable related to $x$ through the model equation:
# 
# $y=\beta_0 + \beta_1 x +\epsilon$
# 
# where
# 
# - $y$ = Dependent Variable (Target Variable)
# - $x$ = Independent Variable (predictor Variable)
# - $\beta_0$ = intercept of the line (Gives an additional degree of freedom)
# - $\beta_1$ = Linear regression coefficient (scale factor to each input value).
# - $\epsilon$ = random error.
# 
# The goal of linear regression is to estimate the values of the regression coefficients
# 
# <img src="ML-image/Multi-lin-reg.png" width="500" height="350" />
# 
# This algorithm explains the linear relationship between the dependent(output) variable $y$ and the independent(predictor) variable $x$ using a straight line  $y=\beta_0+\beta_1 x$.
# 
# #### 1.2. Goal
# 
# - The goal of the linear regression algorithm is to get the best values for $\beta_0$ and $\beta_1$ to find the best fit line. 
# - The best fit line is a line that has the least error which means the error between predicted values and actual values should be minimum.
# - For a datset with $n$ observation $(x_i, y_i)$, where $i=1,2,3...., n$ the above function can be written as follows
# 
#     $y_i=\beta_0 + \beta_1 x_i +\epsilon_i$
# 
#     where $y_i$ is the value of the observation of the dependent variable (outcome variable) in the smaple, $x_i$ is the value of $ith$ observation of the independent variable or feature in the sample, $\epsilon_i$ is the random error (also known as residuals) in predicting the value of $y_i$, $\beta_0$ and $\beta_i$ are the regression parameters (or regression coefficients or feature weights).
# 
# **Note:**
#  - The quantity $\epsilon$ in the model equation is the “error” -- a random variable, assumed to be symmetrically distributed with
# 
#       $E(\epsilon) = 0 ~~ {\rm and}~~ V(\epsilon) = \sigma_{\epsilon^2} =\sigma^2$
# 
#       It is to be noted here that there are no assumption made about the distribution of $\epsilon$, yet.
#   - The $\beta_0$ (the intercept of the true regression line) parameter is average value of $Y$ when $x$ is zero.
#   - The $\beta_1$ (the slope of the true regression line): The expected (average) change in $Y$ associated with a 1-unit increase in the value of x.
#   - What is $\sigma_{Y}^2$?: is a measure of how much the values of $Y$ spread out about the mean value (homogeneity of variance assumption).
# 
# ![image.png](attachment:image.png)
# 
# #### 1.3. Calculating the regression parameters
# 
# In simple linear regression, there is only one independent variable ($x$) and one dependent variable ($y$). The parameters (coefficients) in simple linear regression can be calculated using the method of **ordinary least squares (OLS)**. The equations and formulas involved in calculating the parameters are as follows:
# 
# 1. **Model Representation:**
# 
#     The simple linear regression model can be represented as:
#     $y = \beta_0 + \beta_1 x + \epsilon$
# 
#     So,
# 
#     $\epsilon = y -\beta_0 - \beta_1 x$.
#     
# 
# 2. **Cost Function or mean squared error (MSE):**
# 
#     The MSE, measures the average squared difference between the predicted values ($\hat{y}$) and the actual values of the dependent variable ($y$). It is given by:
# 
#     MSE = $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$
# 
#     Where:
# 
#     - $n$ is the number of data points.
#     - $y_i$ is the actual value of the dependent variable for the i-th data point.
#     - $\hat{y}_i$ is the predicted value of the dependent variable for the $i-th$ data point.
# 
# 3. **Minimization of the Cost Function:**
# 
#     The parameters $\beta_0$ and $\beta_1$ are estimated by minimizing the cost function. The formulas for calculating the parameter estimates are derived from the derivative of the cost function with respect to each parameter.
# 
#     The parameter estimates are given by:
# 
#     - $\hat{\beta_1} = \frac{\text{Cov}(x,y)}{Var(x)}$ 
#       $\Rightarrow \boxed{\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}}$
#     - $\hat{\beta_0} = \text{y} - \hat{\beta_1}\times \text{mean}(x)$ 
# 
#     Where:
# 
#     - $\hat{\beta_0}$ is the estimated $y$-intercept.
#     - $\hat{\beta_1}$ is the estimated slope.
#     - $\text{Cov}(x, y)$ is the covariance between $x$ and $y$.
#     - $\text{Var}(x)$ is the variance of $x$.
#     - $\text{mean}(x)$ is the mean of $x$.
#     - $\text{mean}(y)$ is the mean of $y$.
# 
#     The estimated parameters $\hat{\beta_0}$ and $\hat{\beta_1}$ provide the values of the intercept and slope that best fit the data according to the simple linear regression model.
# 
# 4. **Prediction:**
# 
#     Once the parameter estimates are obtained, predictions can be made using the equation:
# 
#     $\hat{y} = \hat{\beta_0} + \hat{\beta_1} x$
# 
#     Where:
# 
#     - $\hat{y}$ is the predicted value of the dependent variable.
#     - $\hat{\beta_0}$ is the estimated y-intercept.
#     - $\hat{\beta_1}$ is the estimated slope.
#     - $x$ is the value of the independent variable for which the prediction is being made.
# 
#     These equations and formulas allow for the calculation of the parameters in simple linear regression using the method of **ordinary least squares (OLS)**. By minimizing the sum of squared differences between predicted and actual values, the parameters are determined to best fit the data and enable prediction of the dependent variable.

# %% [markdown]
# > **NOTE on Gradient Descent for Linear Regression:**
# > A regression model optimizes the gradient descent algorithm to update the coefficients of the line by reducing the cost function by randomly selecting coefficient values and then iteratively updating the values to reach the minimum cost function.
# > 
# > Gradient Descent is an iterative optimization algorithm commonly used in machine learning to find the optimal parameters in a model. It can also be applied to linear regression to estimate the parameters (coefficients) that minimize the cost function.
# >
# > The steps involved in using Gradient Descent for Linear Regression are as follows:
# >
# > **Define the Cost Function:**
# > The cost function for linear regression is the Mean Squared Error (MSE), which measures the average squared difference between the predicted values (ŷ) and the actual values (y) of the dependent variable.
# >
# > $MSE = \frac{1}{2n} \sum (y_i - \hat{y}_i)^2$
# > 
# > Where:
# > 
# > - $n$ is the number of data points.
# > - $y_i$ is the actual value of the dependent variable for the i-th data point.
# > $\hat{y}_i$ is the predicted value of the dependent variable for the i-th data point.
# > 
# > **Initialize the Parameters:**
# > 
# > Start by initializing the parameters (coefficients) with random values. Typically, they are initialized as zero or small random values.
# >
# > **Calculate the Gradient:**
# > Compute the gradient of the cost function with respect to each parameter. The gradient represents the direction of steepest ascent in the cost function space.
# >
# > $\frac{\partial (MSE)}{\partial \beta_0} = \frac{1}{n}\sum (\hat{y}_i - y_i)$
# >
# > $\frac{\partial (MSE)}{\partial \beta_1} = \frac{1}{n}\sum (\hat{y}_i - y_i)\times x_i$
# >
# > Where:
# >
# > - $\frac{\partial (MSE)}{\partial \beta_0}$ is the gradient with respect to the y-intercept parameter ($\beta_0$).
# > - $\frac{\partial (MSE)}{\partial \beta_1}$ is the gradient with respect to the slope parameter ($\beta_1$).
# > - $\hat{y}_i$ is the predicted value of the dependent variable for the i-th data point.
# > - $y_i$ is the actual value of the dependent variable for the i-th data point.
# > - $x_i$ is the value of the independent variable for the i-th data point.
# >
# > **Update the Parameters:**
# > Update the parameters using the gradient and a learning rate ($\alpha$), which determines the step size in each iteration.
# >
# > - $\beta_0 = \beta_0 - \alpha \times \frac{\partial (MSE)}{\partial \beta_0}$
# > - $\beta_1 = \beta_1 - \alpha \times \frac{\partial (MSE)}{\partial \beta_1}$
# > 
# > Repeat this update process for a specified number of iterations or until the change in the cost function becomes sufficiently small.
# > 
# > **Predict:**
# > Once the parameters have converged or reached the desired number of iterations, use the final parameter values to make predictions on new data.
# >
# > $\hat{y} = \beta_0 +\beta_1 x$
# >
# > Where:
# >
# > - $\hat{y}$ is the predicted value of the dependent variable.
# > - $\beta_0$ is the $y$-intercept parameter.
# > - $\beta_1$ is the slope parameter.
# > - $x$ is the value of the independent variable for which the prediction is being made.
# >
# > Gradient Descent iteratively adjusts the parameters by updating them in the direction of the negative gradient until it reaches a minimum point in the cost function. This process allows for the estimation of optimal parameters in linear regression, enabling the model to make accurate predictions on unseen data.
# >
# >   <img src="ML-image/optimal-reg2.png" width="1000" height="320" />
# >
# > Let’s take an example to understand this. If we want to go from top left point of the shape to bottom of the pit, a discrete number of steps can be taken to reach the bottom. 
# > - If you decide to take larger steps each time, you may achieve the bottom sooner but, there’s a probability that you could overshoot the bottom of the pit and not even near the bottom. 
# > - In the gradient descent algorithm, the number of steps you’re taking can be considered as the learning rate, and this decides how fast the algorithm converges to the minima.
# >
# > **In the gradient descent algorithm, the number of steps you’re taking can be considered as the learning rate i.e. $\alpha$, and this decides how fast the algorithm converges to the minima.**

# %% [markdown]
# ### Assumptions of Linear Regression
# 
# #### 1. Linearity of residuals
# The relationship between the independent variables and the dependent variable is assumed to be linear. This means that the change in the dependent variable is directly proportional to the change in the independent variables.
# 
# <img src="ML-image/Linearity.png" width="1000" height="320" />
# 
# #### 2. Independence
# The observations in the dataset are assumed to be independent of each other. There should be no correlation or dependence between the residuals (the differences between the actual and predicted values) of the dependent variable for different observations.
# 
# <img src="ML-image/independence.png" width="700" height="320" />
# 
# #### 3. Normal distribution of residuals 
# 
# The mean of residuals should follow a normal distribution with a mean equal to zero or close to zero. This is done in order to check whether the selected line is actually the line of best fit or not. If the error terms are non-normally distributed, suggests that there are a few unusual data points that must be studied closely to make a better model.
# 
# ![image.png](attachment:image.png)
# 
# #### 4. The equal variance of residuals
# The error terms must have constant variance. This phenomenon is known as Homoscedasticity. The presence of non-constant variance in the error terms is referred to as Heteroscedasticity. Generally, non-constant variance arises in the presence of outliers or extreme leverage values.
# 
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ### Evaluation Metrics for Linear Regression
# 
# When performing linear regression, it is essential to evaluate the performance of the model to assess its accuracy and effectiveness. Several evaluation metrics can be used to measure the performance of a linear regression model. Here are some commonly used evaluation metrics:
# 
# #### 1. Mean Squared Error (MSE)
# 
# The Mean Squared Error measures the average squared difference between the predicted values and the actual values of the dependent variable. It is calculated by taking the average of the squared residuals.
# 
# $\boxed{\text{MSE} = \frac{1}{n} \sum \left(y_i - \hat{y}_i\right)^2}$
# 
# Where:
# - $n$ is the number of data points.
# - $y_i$ is the actual value of the dependent variable for the $i-th$ data point.
# - $\hat{y}_i$ is the predicted value of the dependent variable for the $i-th$ data point.
# 
# A lower MSE value indicates better model performance, with zero being the best possible value.
# 
# #### 2. Root Mean Squared Error (RMSE)
# 
# The Root Mean Squared Error is the square root of the MSE and provides a more interpretable measure of the average prediction error.
# 
# $\boxed{\text{RMSE} = \sqrt{\text{MSE}}}$ 
# 
# Like the MSE, a lower RMSE value indicates better model performance.
# 
# #### 3. Mean Absolute Error (MAE)
# 
# The Mean Absolute Error measures the average absolute difference between the predicted values and the actual values of the dependent variable. It is less sensitive to outliers compared to MSE.
# 
# $\boxed{\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|}$
# 
# A lower MAE value indicates better model performance.
# 
# #### 4. R-squared ($R^2$) Coefficient of Determination
# 
# The R-squared value represents the proportion of the variance in the dependent variable that is explained by the independent variables. It ranges from $0$ to $1$, where $1$ indicates that the model perfectly predicts the dependent variable.
# 
# $\boxed{R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}}$
# 
# Where:
# 
# - Residual sum of Squares (RSS) is defined as the sum of squares of the residual for each data point in the plot/data. It is the measure of the difference between the expected and the actual observed output.
# 
#     $\text{RSS} = \sum \left(y_i - \beta_0 - \beta_1 x_i\right)^2$
# 
# - Total Sum of Squares (TSS) is defined as the sum of errors of the data points from the mean of the response variable. Mathematically TSS is
# 
#     $\text{TSS} = \sum \left(y_i - \hat{y}_i\right)^2$
# 
# A higher $R^2$ value indicates a better fit of the model to the data. $R^2$ is commonly interpreted as the percentage of the variation in the dependent variable that is explained by the independent variables. However, it is important to note that $R^2$ does not determine the causal relationship between the independent and dependent variables. It is solely a measure of how well the model fits the data.
# 
# #### 5. Adjusted R-squared
# 
# The Adjusted R-squared accounts for the number of independent variables in the model. It penalizes the inclusion of irrelevant variables and rewards the inclusion of relevant variables.
# 
# $\boxed{\text{Adjusted}~ R^2 = 1-\left[\frac{(1 - R²) * (n - 1)}{(n - p - 1)}\right]}$
# 
# Where:
# 
# - $n$ is the number of data points.
# - $p$ is the number of independent variables.
# 
# A higher Adjusted $R^2$ value indicates a better fit of the model while considering the complexity of the model.
# 
# These evaluation metrics help assess the performance of a linear regression model by quantifying the accuracy of the predictions and the extent to which the independent variables explain the dependent variable. It is important to consider multiple metrics to gain a comprehensive understanding of the model's performance.

# %% [markdown]
# ## Hypothesis in Linear Regression
# 
# In linear regression, hypotheses are formulated to test the significance and validity of the regression coefficients and the overall model. These hypotheses are based on the statistical properties of the regression model and help in drawing conclusions about the relationships between the independent variables and the dependent variable. 
# 
# The two main hypotheses in linear regression are:
# 
# ### 1. Null Hypothesis ($H_0$)
# 
# The null hypothesis states that there is no relationship between the independent variables and the dependent variable. In other words, the regression coefficients are equal to zero, implying that the independent variables have no effect on the dependent variable. Mathematically, the null hypothesis can be expressed as:
# 
# $H_0: \beta_1 = \beta_2 = ..... =  βₚ = 0$
# 
# Where:
# 
# - $\beta_0, \beta_1, .... \beta_p$ are the regression coefficients for the independent variables.
# 
# Rejection of the null hypothesis suggests that there is a significant relationship between the independent variables and the dependent variable.
# 
# ### 2. Alternative Hypothesis ($H_1$)
# The alternative hypothesis states that there is a relationship between the independent variables and the dependent variable. It asserts that at least one of the regression coefficients is not equal to zero, indicating that the independent variables have a significant effect on the dependent variable. Mathematically, the alternative hypothesis can be expressed as:
# 
# $H_1$: At least one $\beta_i \neq 0$, where $i = 1, 2, ..., p$
# 
# Where:
# 
# - $\beta_i$ represents an individual regression coefficient for an independent variable.
# - Acceptance of the alternative hypothesis suggests that there is evidence of a significant relationship between the independent variables and the dependent variable.
# 
# ## Test of hypothesis
# 
# - These hypotheses are tested using statistical methods such as hypothesis testing and p-values. 
# - By examining the estimated regression coefficients, their standard errors, and the corresponding p-values, we can make conclusions about the significance and directionality of the relationships between the independent variables and the dependent variable.
# - It is important to note that the formulation and testing of hypotheses in linear regression are crucial in determining the statistical significance of the model and interpreting the effects of the independent variables on the dependent variable.
# - In linear regression, there are several test statistics commonly used to assess the significance of the regression coefficients and the overall model. The most commonly used test statistics are:
# 
# #### 1. t-statistic
# The t-statistic is used to test the significance of individual regression coefficients. It measures the ratio of the estimated regression coefficient ($\beta$) to its standard error (SE($\beta$)). The formula for calculating the t-statistic for a specific coefficient is:
# 
# $\boxed{t = \frac{\beta}{\text{SE}(\beta)}}$ 
# 
# Where:
# 
# - $\beta$ is the estimated regression coefficient.
# - $\text{SE}(\beta)$ is the standard error of the estimated coefficient and is defined by
# 
#     $\text{SE}(\beta) = \sqrt{\text{Var}(\beta)}$
# 
#     and $\text{Var}(\beta)$ represents the variance of the estimated coefficient.
#  
# The t-statistic follows a t-distribution with ($n - k - 1$) degrees of freedom, where $n$ is the sample size and $k$ is the number of independent variables (including the intercept). The t-statistic is compared to the critical value from the t-distribution to determine the statistical significance.
# 
# <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 10px;">
#     <p>
#         The standard error can be calculated using the following steps:
#     </p>
#     <ul>
#         <li>Calculate the Residual Sum of Squares (RSS):<br>
#             RSS = Σ(yᵢ - ŷᵢ)²<br><br>
#             Where:<br>
#             yᵢ is the observed value of the dependent variable.<br>
#             ŷᵢ is the predicted value of the dependent variable obtained from the regression model.
#         </li>
#         <li>Calculate the Mean Squared Error (MSE):<br>
#             MSE = RSS / (n - k - 1)<br><br>
#             Where:<br>
#             n is the sample size.<br>
#             k is the number of independent variables (including the intercept).
#         </li>
#         <li>Calculate the variance of the estimated coefficient (Var(β)):<br>
#             Var(β) = MSE * (XᵀX)⁻¹<br><br>
#             Where:<br>
#             XᵀX is the matrix product of the transpose of the design matrix X and X.<br>
#             (XᵀX)⁻¹ represents the inverse of the matrix XᵀX.
#         </li>
#         <li>Finally, calculate the standard error of the estimated coefficient (SE(β)):<br>
#             SE(β) = sqrt(Var(β))
#         </li>
#     </ul>
# </div>

# %% [markdown]
# #### 2. F-statistic
# 
# The F-statistic is used to test the overall significance of the regression model. It compares the variation explained by the regression model to the unexplained variation. The formula for calculating the F-statistic is:
# 
# $\boxed{F= \frac{\frac{\text{Explained Variation}}{\text{Degrees of Freedom}}}{\frac{\text{Unexplained Variation}}{\text{Degrees of Freedom}}}}$
# 
# Where:
# 
# - `Explained Variation` represents the sum of squared differences between the predicted values and the mean of the dependent variable.
# - `Unexplained Variation` represents the sum of squared differences between the observed values and the predicted values.
# - `Degrees of Freedom` are calculated as $(k, n - k - 1)$, where $k$ is the number of independent variables and $n$ is the sample size.
# - The `F-statistic` follows an `F-distribution` with $(k, n - k - 1)$ degrees of freedom. It is compared to the critical value from the F-distribution to determine the statistical significance of the overall model.
# 
# **Interpretation:**
# 
# - A higher F-value suggests a greater difference between the explained and unexplained variation, indicating a more significant relationship between the independent variables and the dependent variable. 
# - If the calculated F-value is greater than the critical value at a predetermined significance level (e.g., 0.05), the null hypothesis (no relationship between the independent variables and the dependent variable) is rejected in favor of the alternative hypothesis, indicating statistical significance.
# 
# The F-statistic is valuable for assessing the overall fit of the regression model and determining whether the model provides a significant explanation of the variation in the dependent variable.

# %% [markdown]
# #### 3. p-value
# 
# - The p-value is a statistical measure that represents the probability of observing the test statistic (such as the t-statistic or F-statistic) or a more extreme value if the null hypothesis is true. 
# - It is used to determine the statistical significance of the test. 
# - A low p-value (typically less than a predetermined significance level, such as 0.05) suggests strong evidence against the null hypothesis and supports the alternative hypothesis.
# - Sometimes, p-values are given in terms of stars, which has a following meaning:
# 
# | Stars | Imply | Which means that | ANd therefore |
# |-------|-------|------------------|---------------|
# |   *   | p-value $\leq 0.10$ | There is a $10%$ probability that the coefficient is equal to 0. | No statistical relationship |
# |   **  | p-value $\leq 0.05$ | There is a $5%$ probability that the coefficient is equal to 0. | Statistical significant relationship |
# |   ***  | p-value $\leq 0.01$ | There is a $1%$ probability that the coefficient is equal to 0. | Statistical significant relationship |

# %% [markdown]
# ##### Calculation in Python
# 
# 1. **t-statistics and p-value for individual coefficients:**
# 
# If you want to calculate the t-statistic and corresponding p-value for a specific coefficient in linear regression, you can use the statsmodels library. Here's an example:
# 
# ----- 
# ```python
# import statsmodels.api as sm
# 
# # Fit the linear regression model
# model = sm.OLS(y, X)
# results = model.fit()
# 
# # Get the t-statistic and p-value for a specific coefficient
# t_stat = results.tvalues[coefficient_index]
# p_value = results.pvalues[coefficient_index]
# ```
# -----
# 
# Replace `y` with your dependent variable and `X` with your independent variables. `coefficient_index` refers to the index of the coefficient for which you want to calculate the t-statistic and p-value.
# 
# 2. **F-statistics and p-value for the overall model**
#    
# To calculate the F-statistic and corresponding p-value for the overall model in linear regression, you can use the statsmodels library as well. Here's an example:
# 
# ----- 
# ```python
# import statsmodels.api as sm
# 
# # Fit the linear regression model
# model = sm.OLS(y, X)
# results = model.fit()
# 
# # Get the F-statistic and p-value for the overall model
# f_stat = results.fvalue
# p_value = results.f_pvalue
# ```
# -----
# 
# Again, replace `y` with your dependent variable and `X` with your independent variables.
# 

# %% [markdown]
# ### 2. Multiple linear regression algorithms
# 
# Multiple Linear Regression is an extension of Simple Linear Regression that allows for the analysis of the relationship between a dependent variable ($y$) and multiple independent variables ($x_1$, $x_2$, ..., $x_p$). It assumes a linear relationship between the variables, with the objective of finding the best-fit hyperplane in a multi-dimensional space.
# 
# #### 2.1. Mathematical Explanation:
# 
# The multiple linear regression model can be expressed mathematically as:
# 
# $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$
# 
# Where:
# 
# - $y$ is the dependent variable.
# - $x_1$, $x_2$, ...., $x_p$ are the independent variables.
# - $\beta_0$ is the $y$-intercept, representing the value of $y$ when all the independent variables are zero.
# - $\beta_1$, $\beta_2$, ...,  $\beta_p$ are the coefficients associated with each independent variable, indicating the impact of each variable on the dependent variable while holding other variables constant.
# - $\epsilon$ is the error term, representing the random variability or noise in the relationship between the variables.
# 
# #### 2.2. Goal 
# The goal of multiple linear regression is to estimate the values of $\beta_0$, $\beta_1$, $\beta_2$, ...,  $\beta_p$ that provide the best fit to the observed data, minimizing the sum of squared differences between the predicted values ($\hat{y}$) and the actual values of $y$.
# 
# #### 2.3. Calculating the regression parameters 
# 
# In linear regression, the parameters (coefficients) are estimated using a method called ordinary least squares (OLS). The goal of OLS is to find the values of the parameters that minimize the sum of squared differences between the predicted values and the actual values of the dependent variable. The equations and formulas involved in calculating the linear regression parameters are as follows:
# 
# 1. **Cost Function or mean squared error (MSE):**
# 
#     The cost function, measures the average squared difference between the predicted values ($\hat{y}$) and the actual values of the dependent variable ($y$). It is given by:
# 
#     $\text{MSE} = \frac{1}{n} \sum(y_i - \hat{y}_i)^2$
# 
#     Where:
# 
#     - $n$ is the number of data points.
#     - $y_i$ is the actual value of the dependent variable for the $i-th$ data point.
#     - $\hat{y}_i$ is the predicted value of the dependent variable for the $i-th$ data point.
# 
# 2. **Minimization of the Cost Function:**
#     The parameters $\beta_0$, $\beta_1$, $\beta_2$, ...,  $\beta_p$ are estimated by minimizing the cost function. The formulas for calculating the parameter estimates are derived from the derivative of the cost function with respect to each parameter.
# 
#     The parameter estimates are given by:
# 
#     $\hat{\beta} = \left(X^T X\right)^{-1}\, X^T y$
# 
#     Where:
# 
#     - $\hat{\beta}$ is the vector of parameter estimates.
#     - $X$ is the design matrix consisting of the independent variables.
#     - $X^T$ is the transpose of the design matrix.
#     - $\left(X^T X \right)^{-1}$ is the inverse of the matrix product $X^TX$.
#     - $y$ is the vector of the dependent variable values.
#     
#     The estimated parameter vector $\hat{\beta}$ provides the values of the coefficients that best fit the data according to the linear regression model.
# 
# 3. **Prediction:**
#     Once the parameter estimates are obtained, predictions can be made using the equation:
# 
#     $\hat{y} = \hat{\beta_0} + \hat{\beta_1} x_1 + \hat{\beta_2} x_2 + ... + \hat{\beta_p} x_p$
# 
#     Where:
# 
#     - $\hat{y}$ is the predicted value of the dependent variable.
#     - $\hat{\beta_0}$, $\hat{\beta_1}$, $\hat{\beta_2}$, ...,  $\hat{\beta_p}$ are the estimated coefficients.
#     - $x_1$, $x_2$, ..., $x_p$ are the values of the independent variables for which the prediction is being made.
#     
#     These equations and formulas form the basis of ordinary least squares (OLS) estimation for linear regression. By minimizing the sum of squared differences between predicted and actual values, the parameters are determined to best fit the data and allow for prediction of the dependent variable.

# %% [markdown]
# ## Steps in building a regression model
# 
# - **STEP 1: Collect/Extract Data:** The first step in building a regression model is to collect or extract data on the dependent (outcome) variable and independent (feature) variables from different data sources.
# 
# - **STEP 2: Pre-Process the Data:** Before the model is built, it is essential to ensure the quality of the data for issues such as reliability, com- pleteness, usefulness, accuracy, missing data, and outliers.
# 
#     - Data imputation techniques may be used to deal with missing data. Use of descriptive statistics and visualization (such as box plot and scatter plot) may be used to identify the existence of outliers and variability in the dataset.
#     - Many new variables (such as the ratio of variables or product of variables) can be derived (aka feature engineering) and also used in model building.
#     - Categorical data must be pre-processed using dummy variables as a part of feature engineering, prior to utilizing it in a regression model.
# 
# - **STEP 3: Dividing Data into Training and Validation Datasets:** In this stage the data is divided into two subsets (sometimes more than two subsets):
# 
#     - training dataset and
#     - validation or test dataset.
#     
#     The proportion of training dataset is usually between 70% and 80% of the data and the remaining data is treated as the validation data. The subsets may be created using random/ stratified sampling procedure. This is an important step to measure the performance of the model using dataset not used in model building. It is also essential to check for any overfitting of the model. In many cases, multiple training and multiple test data are used (called cross-validation).
# 
# - **STEP 4: Perform Descriptive Analytics or Data Exploration:** It is always a good practice to perform descriptive analytics before moving to building a predictive analytics model. Descriptive statistics will help us to understand the variability in the model and visualization of the data through, say, a box plot which will show if there are any outliers in the data. Another visualization technique, the scatter plot, may also reveal if there is any obvious relationship between the two variables under consideration. Scatter plot is useful to describe the functional relationship between the dependent or outcome variable and features.
# 
# - **STEP 5: Build the Model:** The model is built using the training dataset to estimate the regression parameters. The method of Ordinary Least Squares (OLS) is used to estimate the regression parameters.
# 
# - **STEP 6: Perform Model Diagnostics:** Regression is often misused since many times the modeler fails to perform necessary diagnostics tests before applying the model. Before it can be applied, it is necessary that the model created is validated for all model assumptions including the definition of the function form. If the model assumptions are violated, then the modeler must use remedial measure.
# 
# - **STEP 7: Validate the Model and Measure Model Accuracy:** A major concern in analytics is over-fitting, that is, the model may perform very well on the training dataset, but may perform badly in validation dataset. It is important to ensure that the model perfor- mance is consistent on the validation dataset as is in the training dataset. In fact, the model may be cross- validated using multiple training and test datasets.
# 
# - **STEP 8: Decide on Model Deployment:** The final step in the regression model is to develop a deployment strategy in the form of actionable items and business rules that can be used by the organization.

# %% [markdown]
# ## Reference
# 
# - **Codes are available at:** https://github.com/arunsinp/Machine-Learning


