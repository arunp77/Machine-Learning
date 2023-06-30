# %% [markdown]
# # Data Science & Machine Learning
# 
# ------
# 
# <span style='color:Blue'><q><i>Machine learning enables a machine to automatically learn from data, improve performance from experiences, and predict things without being explicitly programmed. </i></q></span>
# 
# -------

# %% [markdown]
# **Prerequisites:** Before learning machine learning, you must have the basic knowledge of followings so that you can easily understand the concepts of machine learning:
# 
# - Fundamental knowledge of probability and linear algebra.
# - The ability to code in any computer language, especially in Python language.
# - Knowledge of Calculus, especially derivatives of single variable and multivariate functions.
# 
# In the present tutorial, I will be mainly focused on Python programming languages where, I will mainly be using Numpy, Pandas, Matplotlib. Seaborn, Scikit-learn libraries. When necessary, other libraries will also be mentioned. 

# %% [markdown]
# ## Python and data science
# 
# <img src="https://user-images.githubusercontent.com/15100077/209703557-f22b143b-8b42-4c5d-b8dd-f180522f33d8.png"  width="700" height="600" />
# 
# * Data science projects need extraction of data from various sources, data cleaning, data imputation beside model building, validation, and making predictions. 
# * Data analysis is mostly an iterative process, where lots of exploration needs to be done in an ad-hoc manner. 
# * Python being an interpreted language provides an interactive interface for accomplishing this. Python is an interpreted, high-level, general-purpose programming language. 
# * Python’s strong community, continuously evolves its data science libraries and keeps it cutting edge.
# * It has libraries for **linear algebra computations**, **statistical analysis**, **machine learning**, **visualization**, **optimization**, **stochastic models**, etc.
# 
# <img src="ML-image/python-libraries.png" width="700" height="450" />
# 
# (**Reference for the figure:** Machine learning using Python, Manaranjan Pradhan & U Dinesh Kumar)

# %% [markdown]
# ### Core Python Libraries for Data Analysis
# 
# | Areas of Application | Library | Description | 
# |----------------------|---------|-------------|
# | Mathematical Computations | [NumPy](www.numpy.org)| NumPy is the fundamental package for scientific computing involving large arrays and matrices. It provides useful mathematical computation capabilities. |
# | Data Structure Operations (Dataframes) | [Pandas](https://pandas.pydata.org/) | Pandas provides high-performance, easy-to-use data structures called DataFrame for exploration and analysis. DataFrames are the key data structures that feed into most of the statistical and machine learning models. |
# | Visualization | [Matplotlib](https://matplotlib.org/)| Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python|
# | More elegant Visualization | [Seaborn](https://seaborn.pydata.org/) | Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics. |
# | Machine Learning Algorithm | [Scikit-learn (aka sklearn)](https://scikit-learn.org/stable/) | Scikit-learn provides a range of supervised and unsupervised learning algorithms. |
# | Statistical Computations | [SciPy](www.scipy.org) | SciPy contains modules for optimization and computation. It provides libraries for several statistical distributions and statistical tests. |
# | Statistical Modelling | [StatsModels](www.statsmodels.org/stable/index.html) | StatsModels is a Python module that provides classes and functions for various statistical analyses. | 
# | IDE (Integrated Development Environment) | [Jupyter Notebook](jupyter.org) | The Jupyter Notebook is an opensource web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.|

# %% [markdown]
# ### Top  tools for ML model training
# 
# There are several tools available for training machine learning models, and the best one for you depends on your specific needs and preferences. Here are some popular tools for ML model training:
# 
# 1. [TensorFlow](https://github.com/tensorflow): TensorFlow is an open-source machine learning framework developed by Google. It is widely used for building and training neural networks and deep learning models. TensorFlow offers a high degree of flexibility and can be used for a range of tasks, from image recognition to natural language processing.
# 
# 2. [PyTorch](https://pytorch.org/): PyTorch is an open-source machine learning library developed by Facebook. It is popular for its dynamic computational graph and ease of use. PyTorch is especially well-suited for building and training deep learning models.
# 
# 3. [Keras](https://keras.io/): Keras is an open-source neural network library written in Python. It provides a user-friendly API for building and training neural networks, making it a popular choice for beginners. Keras supports multiple backends, including TensorFlow, and can run on both CPU and GPU.
# 
# 4. [Scikit-learn](https://scikit-learn.org/stable/): Scikit-learn is an open-source machine learning library for Python. It provides a range of algorithms for classification, regression, clustering, and dimensionality reduction, as well as tools for model selection and evaluation. Scikit-learn is easy to use and integrates well with other Python libraries.
# 
# 5. [Microsoft Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/): CNTK is an open-source toolkit developed by Microsoft for building and training deep learning models. It is optimized for performance on multi-GPU and multi-machine systems and provides a range of neural network primitives for building custom models.
# 
# 6. [Theano](http://deeplearning.net/software/theano/): Theano is an open-source numerical computation library for Python. It provides a range of optimization techniques for building and training neural networks and can be used for a range of machine learning tasks. Theano is especially well-suited for working with large datasets and complex models.
# 
# These are just a few of the popular tools available for ML model training. Each of them has its strengths and weaknesses, so it's important to choose the one that best meets your needs.

# %% [markdown]
# # Python Installation
# 
# There are many ways, you can install python on your operating system. Best way is to go to Anaconda page and download the Anaconda installer and install it. Anaconda is a distribution of the Python and R programming languages for scientific computing, data science, machine learning, and related fields. It includes a package manager, environment manager, and numerous libraries and tools to help simplify the installation and configuration of scientific packages. Anaconda also includes Jupyter Notebook, an interactive development environment for working with code and data, and Spyder, an integrated development environment for Python. Anaconda is widely used in data science and machine learning for its ease of use, package management capabilities, and large community support.
# 
# It comes with all the necessary libraries (Please see: https://www.anaconda.com/products/distribution). Here, download latest version of the Anconda installer and follow the onscreen steps. It is one of the most easiest way to isntall the python. Normally Anaconda comes with the Anaconda-Navigator. The Anaconda Navigator is the graphical user interface that allows you to manage your environments, packages, and applications.
# 
# <img src="https://miro.medium.com/max/314/0*mPlfXXt1on7czWtf.png" width="200" height="200" />
# 
# 
# After installation:
# 
# - **On windows**: On Windows, Anaconda can be found in the Start menu or by searching for "Anaconda Navigator" in the search bar. Additionally, you can also access Anaconda Prompt, which is a command-line interface for Anaconda, by searching for "Anaconda Prompt" in the search bar. 
# 
# - **on mac:** Anaconda-Nevigator can be found in Lanunchpad; or you can find the Anaconda installation in the following directory: '`/Users/<your username>/opt/anaconda3/`'. You can also find it by opening a terminal window and typing the following command: '`conda info`'. This will display information about the Anaconda installation, including the path where it is installed.
# - **on linux:** On Linux, Anaconda is typically installed in the home directory of the user who installed it. The default installation path is '`/home/username/anaconda3`, where '`username`' is the name of the user who installed Anaconda. You can check the installation directory by running the following command in the terminal: `echo $HOME/anaconda3`'. This will output the installation path of Anaconda on your system. If you installed Anaconda in a different location, you can replace $HOME/anaconda3 with the actual installation path.
# 
# <img src="ML-image/Anaconda.png" width="1000" height="550" />
# 
# You can open Jupyter or Spyder notebooks directly from here by clicking 'Launch'. 
# 
# For writing my codes, I usually usw 'VSCODE', which can be installed by Anaconda-Navigator directly.

# %% [markdown]
# # Example project
# 
# ### How to start with ML?
# 
# Here, we will start with a example. We will follow following steps here-
# 1. Importing Libraries
# 2. Get the Dataset
# 3. Importing the Datasets into the jupyter notebook
# 4. Handling Missing data
# 5. Encoding Categorical data
# 6. Splitting the Dataset into the Training set and Test set
# 7. Feature Scaling

# %% [markdown]
# 1. **Importing the libraries**

# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')

# %% [markdown]
# 2. **Getting the dataset**
# 
# We have added a example data set: 'Country-Data.csv' here.

# %% [markdown]
# 3. Importing the Datasets into the jupyter notebook

# %%
# Imorting files
con_df = pd.read_csv('Country-Data.csv')

# %%
con_df

# %% [markdown]
# 4. **Handling Missing data**
# 
# There are two main ways, we can do this
# - **By deleting the particular row:** The first way is used to commonly deal with null values. In this way, we just delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give the accurate output.
# - **By calculating the mean:** In this way, we will calculate the mean of that column or row which contains any missing value and will put it on the place of missing value. This strategy is useful for the features which have numeric data such as age, salary, year, etc. Here, we will use this approach.
# 
# To handle missing values, we will use Scikit-learn library in our code, which contains various libraries for building machine learning models

# %%
# Extracting independent variable:
x= con_df.iloc[:,:-1].values
pd.DataFrame(x)

# %%
x

# %% [markdown]
# In the above code, the first colon(:) is used to take all the rows, and the second colon(:) is for all the columns. Here we have used :-1, because we don't want to take the last column as it contains the dependent variable. So by doing this, we will get the matrix of features.

# %%
# Extracting dependent variable:
y= con_df.iloc[:,3].values
y

# %%
pd.DataFrame(y)

# %% [markdown]
# Here we have taken all the rows with the last column only. It will give the array of dependent variables.

# %%
from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with strategy 'mean'
imputer = SimpleImputer(strategy='mean')

# Fit the imputer object to the independent variables x
imputer.fit(x[:, 1:3])

# Replace missing data with the calculated mean value
x[:, 1:3] = imputer.transform(x[:, 1:3])
x

# %% [markdown]
# 5. **Encoding Categorical data**
# 
# Categorical data is data which has some categories such as, in our dataset; there are two categorical variable, Country, and Purchased.
# 
# Since machine learning model completely works on mathematics and numbers, but if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.
# 
# For Country variable:
# 
# Firstly, we will convert the country variables into categorical data. So to do this, we will use `LabelEncoder()` class from preprocessing library.

# %%
#Catgorical data  
#for Country Variable  
from sklearn.preprocessing import LabelEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])  
x

# %%
z= pd.DataFrame(x) # just to see in a dataframe format. 
z

# %% [markdown]
# In above code, we have imported LabelEncoder class of sklearn library. This class has successfully encoded the variables into digits.
# 
# But in our case, there are three country variables, and as we can see in the above output, these variables are encoded into 0, 1, and 2. By these values, the machine learning model may assume that there is some correlation between these variables which will produce the wrong output. So to remove this issue, we will use **dummy encoding**.
# 
# > **Dummy Variables:** Dummy variables are those variables which have values 0 or 1. The 1 value gives the presence of that variable in a particular column, and rest variables become 0. With dummy encoding, we will have a number of columns equal to the number of categories.
# 
# In our dataset, we have 3 categories so it will produce three columns having 0 and 1 values. For Dummy Encoding, we will use OneHotEncoder class of preprocessing library.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Encode the Country variable using LabelEncoder
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Apply one-hot encoding using ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = ct.fit_transform(x)
x

# %% [markdown]
# As we can see in the above output, all the variables are encoded into numbers 0 and 1 and divided into three columns.

# %% [markdown]
# **NOTE:** In this code, we first encode the Country variable using `LabelEncoder` as before. Then we define a `ColumnTransformer` object that applies `OneHotEncoder` to the first column of the input data. We pass a list of transformers to the `ColumnTransformer` constructor, where each transformer is defined as a tuple (name, transformer, columns). Here, we give the transformer a name of `one_hot_encoder`, use an instance of OneHotEncoder as the transformer, and specify the index of the column to transform as [0]. We set the remainder parameter to 'passthrough' so that any remaining columns are passed through without any transformation.
# 
# Finally, we use the `fit_transform` method of the `ColumnTransformer` object to apply the transformation to the input data. The resulting data will have one-hot encoded columns for the Country variable, followed by any remaining columns.

# %%
z1=pd.DataFrame(x)
z1

# %%
labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y)

# %%
z2=pd.DataFrame(y)
z2

# %% [markdown]
# For the second categorical variable, we will only use labelencoder object of LableEncoder class. Here we are not using `OneHotEncoder` class because the purchased variable has only two categories yes or no, and which are automatically encoded into `0` and `1`.

# %% [markdown]
# 6. **Splitting the Dataset into the Training set and Test set**
# 
# In machine learning data preprocessing, we divide our dataset into a training set and test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.
# 
# Suppose, if we have given training to our machine learning model by a dataset and we test it by a completely different dataset. Then, it will create difficulties for our model to understand the correlations between the models.
# 
# If we train our model very well and its training accuracy is also very high, but we provide a new dataset to it, then it will decrease the performance. So we always try to make a machine learning model which performs well with the training set and also with the test dataset.
# 
# > **training Set:** A subset of dataset to train the machine learning model, and we already know the output.
# 
# > **Test set:** A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.

# %%
# For splitting the dataset, we will use the below lines of code:

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

# %% [markdown]
# - In the above code, the first line is used for splitting arrays of the dataset into random train and test subsets.
# - In the second line, we have used four variables for our output that are
# 
#     - `x_train`: features for the training data
#     - `x_test`: features for testing data
#     - `y_train`: Dependent variables for training data
#     - `y_test`: Independent variable for testing data
# 
# - In `train_test_split()` function, we have passed four parameters in which first two are for arrays of data, and `test_size` is for specifying the size of the test set. 
# - The `test_size` maybe `.5`, `.3`, or `.2`, which tells the dividing ratio of training and testing sets.
# - The last parameter `random_state` is used to set a seed for a random generator so that you always get the same result, and the most used value for this is 42.

# %%
con_df

# %%
x

# %%
x_test

# %%
x_train

# %%
y

# %%
y_test

# %%
y_train

# %% [markdown]
# 7. **Feature Scaling**
# 
# Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.

# %% [markdown]
# Let us first consider a new datset : 'con-new.csv'

# %%
con_new_df  = pd.read_csv('con-new.csv')
con_new_df

# %% [markdown]
# As we can see, the age and salary column values are not on the same scale. A machine learning model is based on Euclidean distance, and if we do not scale the variable, then it will cause some issue in our machine learning model.
# 
# ![image.png](attachment:image.png)
# 
# If we compute any two values from age and salary, then salary values will dominate the age values, and it will produce an incorrect result. So to remove this issue, we need to perform feature scaling for machine learning.
# 
# There are two ways to perform feature scaling in machine learning:
# 
# **Standardization:**
# 
# ![image.png](attachment:image-2.png)
# 
# **Normalization:**
# 
# ![image.png](attachment:image-3.png)
# 
# Here, we will use the standardization method for our dataset.

# %%
# For feature scaling, we will import StandardScaler class of sklearn.preprocessing library as:
from sklearn.preprocessing import StandardScaler  

# %% [markdown]
# Now, we will create the object of StandardScaler class for independent variables or features. And then we will fit and transform the training dataset.

# %%
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  

# %% [markdown]
# For test dataset, we will directly apply `transform()` function instead of `fit_transform()` because it is already done in training set.

# %%
x_test= st_x.transform(x_test)  

# %%
z4 = pd.DataFrame(x_train)
z4

# %%
z5 = pd.DataFrame(x_test)
z5

# %% [markdown]
# As we can see in the above output, all the variables are scaled between values -1 to 1.
# 
# > **Note:** Here, we have not scaled the dependent variable because there are only two values 0 and 1. But if these variables will have more range of values, then we will also need to scale those variables.

# %% [markdown]
# # References
# 
# 1. https://www.javatpoint.com/machine-learning
# 2. https://www.geeksforgeeks.org/machine-learning/#dp
# 3. https://jovian.com/learn/machine-learning-with-python-zero-to-gbms
# 4. Book: Machine learning using Python, Manaranjan Pradhan & U Dinesh Kumar
# 5. https://www.superdatascience.com/pages/machine-learning


