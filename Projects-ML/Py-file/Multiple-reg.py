# %% [markdown]
# # Project-2: Auction pricing of players in the Indian premier league (IPL)
# **(Multiple Linear Regression)**
# 
# - Multiple  linear  regression  (MLR)  is  a  supervised  learning  algorithm  for  finding  the  existence  of  an  association relationship between a dependent variable (aka response variable or outcome variable) and several independent variables (aka explanatory variables or predictor variable or features).
# 
# - The functional form of MLR is given by:
# 
#     $Y_i =  \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + .... + + \beta_k X_{ki} + \epsilon_{ki}$
# 
#     where
# 
#     $\beta_1$,  $\beta_2$, $\beta_3$, .... , $\beta_k$ are partial regression coefficients. 
# 
#     - X = independent variables (aka explanatory variables or predictor variable or features)
#     - Y = dependent variable (aka response variable or outcome variable)<br>
# 
# - Since the relationship between an explanatory variable and the response (outcome) variable is calculated after removing (or controlling) the effect all the other explanatory variables (features) in the model.
# 
# - The assumptions that are made in multiple linear regression model are as follows:
#     1. **Linearity:** The regression model is linear in regression parameters (b-values).
#     2. **Normal distribution:** The residuals follow a normal distribution and the expected value (mean) of the residuals is zero.
#     3. **Uncorrelated residuals:** In time series data, residuals are assumed to uncorrelated.
#     4. **Variance of the residuals** The variance of the residuals is constant for all values of $X_i$. When the variance of the residuals is constant for different values of $X_i$, it is called homoscedasticity. A non-constant variance of residuals is called heteroscedasticity.
#     5. **Correlation between independent variables:** There is no high correlation between independent variables in the model (called multi-collinearity). Multi-collinearity can destabilize the model and can result in an incorrect estimation of the regression parameters.
# 
#     The partial regressions coefficients are estimated by minimizing the sum of squared errors (SSE).

# %% [markdown]
# ## Objective: Predicting the SOLD PRICE (Auction Price) of Players
# 
# The Indian Premier League (IPL) is a professional league for Twenty20 (T20) cricket championships that  was started in 2008 in India. IPL was initiated by the BCCI with eight franchises comprising players from  across  the  world.  The  first  IPL  auction  was  held  in  2008  for  ownership  of  the  teams  for  10  years,  with  a base price of USD 50 million. The franchises acquire players through an English auction that is con- ducted every year. However, there are several rules imposed by the IPL. For example, only international  players and popular Indian players are auctioned.
# 
# The performance of the players could be measured through several metrics. Although the IPL fol- lows the Twenty20 format of the game, it is possible that the performance of the players in the other formats of the game such as Test and One-Day matches could influence player pricing. A few players had excellent records in Test matches, but their records in Twenty20 matches were not very impressive. The performances of 130 players who played in at least one season of the IPL (2008−2011) measured through various performance metrics 
# 
# ### Data Code Description
# 
# | Data Code | Description |
# |-----------|-------------|
# | AGE | Age of the player at the time of auction classified into three categories. Category 1 (L25) means the player is less than 25 years old, category 2 means that the age is between 25 and 35 years (B25− 35) and category 3 means that the age is more than 35 (A35). |
# | RUNS-S | Number of runs scored by a player. |
# | RUNS-C | Number of runs conceded by a player. |
# | HS | Highest score by a batsman in IPL. |
# | AVE-B | Average runs scored by a batsman in IPL. |
# | AVE-BL | Bowling average (number of runs conceded/number of wickets taken) in IPL. |
# | SR-B | Batting strike rate (ratio of the number of runs scored to the number of balls faced) in IPL. |
# | SR-BL | Bowling strike rate (ratio of the number of balls bowled to the number of wickets taken) in IPL. |
# | SIXERS | Number of six runs scored by a player in IPL.|
# | WKTS | Number of wickets taken by a player in IPL. |
# | ECON | Economy rate of a bowler (number of runs conceded by the bowler per over) in IPL. |
# | CAPTAINCY EXP | Captained either a T20 team or a national team.|
# | ODI-SR-B | Batting strike rate in One-Day Internationals. |
# | ODI-SR-BL | Bowling strike rate in One-Day Internationals. |
# | ODI-RUNS-S | Runs scored in One-Day Internationals. |
# | ODI-WKTS | Wickets taken in One-Day Internationals.|
# | T-RUNS-S | Runs scored in Test matches. |
# | T-WKTS | Wickets taken in Test matches. |
# | PLAYER-SKILL | Player’s primary skill (batsman, bowler, or allrounder). |
# | COUNTRY | Country of origin of the player (AUS: Australia; IND: India; PAK: Pakistan; SA: South Africa; SL: Sri Lanka; NZ: New Zealand; WI: West Indies; OTH: Other countries). |
# | YEAR-A | Year of Auction in IPL. |
# | IPL TEAM | CSK: Chennai Super Kings; DC: Deccan Chargers; DD: Delhi Dare- devils; KXI: Kings XI Punjab; KKR: Kolkata Knight Riders; MI: Mumbai Indians; PWI: Pune Warriors India; RR: Rajasthan Royals; RCB: Royal Challengers Bangalore |
# 
# *A + sign is used to indicate that the player has played for more than one team. For example, CSK+ would mean that the player has played for CSK as well as for one or more other teams.

# %% [markdown]
# ## Data
# - Data is available at in the repository.

# %% [markdown]
# ## Developing Multiple Linear Regression Model Using Python

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
file_path = "../ml-data/IPL-IMB381IPL2013.csv"

# %% [markdown]
# ### Importing the data file

# %%
# importing the data file

ipl_auction_df = pd.read_csv(file_path)

# %%
ipl_auction_df.info()

# %%
# shape of the dataframe
ipl_auction_df.shape

# %% [markdown]
# There are 130 observations (records) and 26 columns (features) in the data, and there are no missing values.

# %%
# importing first 10 rows
ipl_auction_df.head(5)

# %%
ipl_auction_df.plot.scatter(x='ODI-RUNS-S', y='BASE PRICE')

# %%
# displaying the initial 10 columns for the first 5 rows
ipl_auction_df.iloc[0:5, 0:10]

# %% [markdown]
# ## Building multiple linear regression model
# 
# - We can build a model to understand what features (`X`) of players are influencing their SOLD PRICE or predict the player’s auction prices in future. However, all columns are not features. 
# - For example, Sl. NO. is just a serial number and cannot be considered a feature of the player. 
# - We will build a model using only player’s statistics. So, BASE PRICE can also be removed. 
# - We will create a variable `X_feature` which will contain the list of features that we will finally use for building the model and ignore rest of the columns of the DataFrame. 

# %% [markdown]
# ### 1. Creating a feature columns
# 
# The following function is used for including the features in the model building.

# %%
# Assuming 'ipl_auction_df' is your DataFrame
column_names = ipl_auction_df.columns.tolist()
column_names

# %% [markdown]
# Not all columns are important. We select few of them to make our model for auction price. To do this, we create `X_features` list.

# %%
X_features = ipl_auction_df.columns

# %%
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 
              'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS', 'AVE-BL', 'ECON', 'SR-BL']

# %% [markdown]
# ### 2. Encoding Categorical Features
# 
# - Qualitative variables or categorical variables need to be encoded using dummy variables before incorporating them in the regression model. 
# - If a categorical variable has `n` categories (e.g., the player role in the data has four categories, namely, batsman, bowler, wicket-keeper and allrounder), then we will need `n − 1` dummy variables. So, in the case of PLAYING ROLE, we will need **three dummy variables** since there are four categories.

# %%
# Finding unique values of column PLAYING ROLE
ipl_auction_df['PLAYING ROLE'].unique()

# %%
ipl_auction_df['COUNTRY'].unique()

# %% [markdown]
# The variable can be converted into four dummy variables. 
# - Set the variable value to `1` to indicate the role of the player. 
# - This can be done using `pd.get_dummies()` method. 
# - We will create dummy variables for only PLAYING ROLE to understand and then create dummy variables for the rest of the categorical variables.
# 

# %%
pd.get_dummies(ipl_auction_df['PLAYING ROLE'])[0:5] 
#[0:5] is a slicing operation that selects the first five rows of the resulting DataFrame

# %% [markdown]
# As shown in the table above, the `pd.get_dummies()` method has created four dummy variables and has already set the variables to `1` as variable value in each sample.

# %% [markdown]
# - We must create dummy variables for all categorical (qualitative) variables present in the dataset.

# %%
categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']

# %%
ipl_auction_encoded_df = pd.get_dummies(ipl_auction_df[X_features], 
                                        columns = categorical_features, 
                                        drop_first = True)

# %%
ipl_auction_encoded_df.columns

# %% [markdown]
# - The dataset contains the new dummy variables that have been created. 
# - We can reassign the new features to the variable `X_features`, which we created earlier to keep track of all features that will be used to build the model finally.

# %%
X_features = ipl_auction_encoded_df.columns
X_features

# %% [markdown]
# ### 3. Splitting the Dataset into Train and Validation Sets
# 
# - Before building the model, we will split the dataset into 80:20 ratio
# - The split function allows using a parameter `random_state`, which is a seed function for reproducibility of randomness. This parameter is not required to be passed. 
# - Setting this variable to a fixed number will make sure that the records that go into **training** and **test set** remain unchanged and hence the results can be reproduced. We will use the value 42 (it is again selected randomly). 

# %%
# Creating feature Set(X) and Outcome Variable (Y)
import statsmodels.api as sm
X = sm.add_constant( ipl_auction_encoded_df )
Y = ipl_auction_df['SOLD PRICE']

# %%
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X , Y,
train_size = 0.8, random_state = 42 )

# %% [markdown]
# ### 4. Fitting the Model
# 
# We will fit the model using OLS method and pass `train_y` and `train_X` as parameters.

# %%
ipl_model_1 = sm.OLS(train_y, train_X).fit()

# %% [markdown]
# Printing Estimated Parameters and Interpreting Them

# %%
print(ipl_model_1.params)

# %% [markdown]
# #### Scatter Plot

# %%
# Scatter plot of original data points
ipl_auction_df.plot.scatter(x='Percentage in Grade 10', y='Salary', label = 'Scatter plot')

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
# ### 5. Regression Model Summary Using Python
# 
# The function `summary2()` prints the model summary which contains the information required for diagnosing a regression model 

# %%
ipl_model_1.summary2()

# %%



