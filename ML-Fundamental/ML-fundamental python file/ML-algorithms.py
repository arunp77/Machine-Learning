# %% [markdown]
# <div style="background-color: #008080; border: 2px solid #ddd; padding: 5px; text-align: left; color: white;">
#   <h1>Machine Learning: An Introductory Tutorial for Beginners</h1>
#   <h4>Author: Dr. Arun Kumar Pandey</h4>
# </div>
# 
# 
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# # Modern analytics and tools
# 
# Analytics is a collection of techniques and tools used for creating value from data. Techniques include concepts such as artificial intelligence (AI), machine learning (ML), and deep learning (DL) algorithms.
# 
# <img src="ML-image/ML1.png" width="800" height="550" />
# 
# ## 1. Artificial Intelligence (AI):
# 
# AI is a broad field of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. It encompasses various techniques, algorithms, and approaches to mimic human cognitive abilities, such as learning, reasoning, problem-solving, perception, and natural language understanding. AI can be categorized into two types:
# 
# - **Narrow AI (or Weak AI):** Narrow AI refers to AI systems designed to perform specific tasks or functions. These systems are highly specialized and excel at a particular domain, such as voice assistants, recommendation systems, or autonomous vehicles. Narrow AI does not possess general intelligence and is limited to the tasks it is trained or programmed for.
# 
#     Example: Optical Character Recognition (OCR), Virtual Personal Assistants, etc.
# 
# - **General AI (or Strong AI):** General AI aims to develop machines that possess human-level intelligence and can understand, learn, and perform any intellectual task that a human can do. Achieving true general AI is still a subject of ongoing research and remains a significant challenge.
# 
#     Example: Human-level Conversational AI, Superintelligent AI etc.
# 
# ![image.png](attachment:image.png)
# 
# (Image credit: https://www.scs.org.sg/articles/machine-learning-vs-deep-learning)
# 
# ## 2. Machine Learning (ML):
# 
# Machine Learning is a subset of AI that focuses on developing algorithms and models that enable computers to learn and make predictions or decisions without explicit programming. ML algorithms learn from data and improve their performance through experience. They analyze patterns and relationships within the data to make predictions, classifications, or discover insights. ML can be further classified into several types:
# 
#   - **Supervised Learning:** In supervised learning, models are trained on labeled data, where each example has input features and corresponding target labels or outcomes. The model learns to generalize from the labeled data and make predictions on unseen data.
# 
#   - **Unsupervised Learning:** Unsupervised learning involves training models on unlabeled data, where the algorithm learns patterns and structures without explicit target labels. It focuses on finding hidden patterns, groupings, or generating meaningful representations of the data.
# 
#   - **Semi-Supervised Learning:** Semi-supervised learning combines elements of both supervised and unsupervised learning. It uses a small amount of labeled data along with a larger amount of unlabeled data to improve the model's performance.
# 
#   - **Reinforcement Learning:** Reinforcement learning involves an agent that learns to interact with an environment. The agent takes actions, receives feedback or rewards, and learns to maximize its cumulative reward over time.
# 
#   ![image-2.png](attachment:image-2.png)
# 
#   (Image credit: https://www.scs.org.sg/articles/machine-learning-vs-deep-learning)
# 
# ## 3. Deep Learning (DL):
# 
# Deep learning is a subset of machine learning that focuses on artificial neural networks with multiple layers, enabling the model to learn hierarchical representations of data. These neural networks are composed of multiple layers of interconnected nodes, called neurons, that can learn hierarchical representations of the data. Deep learning algorithms excel at processing complex, high-dimensional data, such as images, speech, and text. DL has gained significant attention and success in recent years, achieving state-of-the-art performance in various domains, including computer vision, natural language processing, and speech recognition.
# 
# There are several categories or architectures within deep learning. Here are three prominent ones along with examples:
# 
# - **Convolutional Neural Networks (CNNs):** CNNs are primarily used for image and video processing tasks. They excel at learning spatial hierarchies and capturing local patterns by applying convolutional layers that perform feature extraction and pooling layers that downsample the data. CNNs have achieved breakthrough performance in tasks like image classification, object detection, and image segmentation. An example is the famous image recognition model called "AlexNet," which won the ImageNet Large Scale Visual Recognition Challenge in 2012.
# 
# - **Recurrent Neural Networks (RNNs):** RNNs are designed for sequential data and have internal memory to process information with temporal dependencies, making them suitable for tasks like speech recognition, language modeling, and machine translation. They have loops within their architecture, allowing information to persist and be passed from one step to another. An example is the "Long Short-Term Memory" (LSTM) network, which is capable of handling long-term dependencies and is widely used in natural language processing applications.
# - 
# 
# - **Generative Adversarial Networks (GANs):** GANs consist of two neural networks—a generator and a discriminator—competing against each other. The generator network learns to create realistic data instances, such as images, while the discriminator network tries to distinguish between the generated data and real data. GANs are popular for generating synthetic data, image synthesis, image-to-image translation, and even creating deepfake videos. One notable example is the "Deepfake" technology, which uses GANs to manipulate and create realistic video content.
# 
# ![image-3.png](attachment:image-3.png)
# 
# (Image credit: https://www.scs.org.sg/articles/machine-learning-vs-deep-learning)

# %% [markdown]
# ## Key Differences Between Machine Learning And Deep Learning
# 
# |                         | Machine Learning | Deep Learning |
# |-------------------------|------------------|---------------|
# | Approach        | Requires structure data | Does not require structure data |
# | Human intervention | Requires human intervention for mistakes | Does not require human intervention for mistakes |
# | Hardware | Can function on CPU | Requires GPU / Significant computing power |
# | Time | Takes seconds to hours | Takes weeks |
# | Uses | Forecasting, predicting and other simple applications |  More complex applications like autonomus vehicles |

# %% [markdown]
# # Machine learning
# 
# **How does Machine Learning work?:** A Machine Learning system learns from historical data, builds the prediction models, and whenever it receives new data, predicts the output for it. The accuracy of predicted output depends upon the amount of data, as the huge amount of data helps to build a better model which predicts the output more accurately.
# 
# ## How does Machine Learning Work?
# 
# The following picture presents a simplified diagram that demonstrates the functioning of machine learning.
# 
# ![image.png](attachment:image.png)
# 
# Machine learning works by training algorithms on data to identify patterns, make predictions, or make informed decisions without being explicitly programmed for each specific task. Here's a simplified explanation of how machine learning works:
# 
# - **Define Objective:** First step is to find a objective.
#   
# - **Data Collection:** The first step is to gather relevant data for the problem at hand. This data can come from various sources such as databases, spreadsheets, sensors, or web scraping. The data should include both input features (attributes) and the corresponding output or target variable (the desired prediction or outcome).
# 
# - **Data Preprocessing:** Before training a machine learning model, the data needs to be preprocessed to ensure its quality and suitability for analysis. This step may involve handling missing values, encoding categorical variables, normalizing or scaling numeric features, and splitting the data into training and testing sets.
# 
# - **Model (algorithm) Selection:** Select an appropriate machine learning algorithm based on the problem type and the available data. There are various types of algorithms, including regression, classification, clustering, and reinforcement learning. The choice of algorithm depends on the specific problem, the nature of the data, and the desired outcome.
# 
# - **Model Training:** In the training phase, the selected machine learning algorithm is presented with the labeled training data. The algorithm learns the patterns, relationships, and rules present in the data by adjusting its internal parameters through an optimization process. The goal is to minimize the error or difference between the predicted output and the actual target variable.
# 
# - **Model Evaluation:** After training, the model's performance needs to be assessed on unseen data to evaluate its effectiveness. This is typically done using evaluation metrics appropriate for the problem, such as accuracy, precision, recall, or mean squared error. By evaluating the model's performance, you can assess its ability to generalize to new, unseen data.
# 
# - **Model Optimization:** If the model's performance is not satisfactory, it may require further optimization. This can involve fine-tuning hyperparameters (settings that control the learning process), feature engineering (selecting or creating better input features), or exploring different algorithms or architectures. The iterative process of optimization aims to improve the model's accuracy and generalization capabilities.
# 
# - **Model Deployment and Prediction:** Once the model is trained and evaluated, it can be deployed to make predictions on new, unseen data. The trained model takes input features from the unseen data and generates predictions or classifications based on the patterns it has learned during training.
# 
# - **Model Monitoring and Maintenance:** Machine learning models require monitoring and maintenance to ensure they continue to perform well as new data becomes available. Monitoring can involve tracking the model's performance, retraining the model periodically with updated data, or adapting the model to changes in the problem domain.

# %% [markdown]
# ## Categories of ML algorithms
# 
# There are three main categories of machine learning algorithms:
# 
# | Algorithms | Definition | Goal | Example |
# |------------|------------|------|---------|
# | Supervised Learning | In supervised learning, the machine is trained on a labeled dataset, where the target variable (or output/dependent variable) is known for each observation in the dataset. | The goal of supervised learning is to learn a mapping from the input variables (or features) to the output variable, so that the machine can make accurate predictions on new, unseen data. | - linear regression, -logistic regression, -decision trees, and -neural networks |
# | Unsupervised Learning | In unsupervised learning, the machine is trained on an unlabeled dataset, where the target variable (or output) is not known for each observation in the dataset. | The goal of unsupervised learning is to learn patterns in the data, such as similarities or clusters, without a specific goal or target variable in mind. | -k-means clustering, -hierarchical clustering, -principal component analysis (PCA), and -t-distributed stochastic neighbor embedding (t-SNE)| 
# | Reinforcement Learning | In reinforcement learning, the machine learns by interacting with an environment, receiving feedback in the form of rewards or penalties for each action taken. | The goal of reinforcement learning is to learn an optimal policy, or set of actions, that maximize the cumulative reward over time | - Q-learning, - SARSA, and - deep reinforcement learning  |
# 
# It's worth noting that there are also hybrid approaches that combine elements of supervised and unsupervised learning, such as semi-supervised learning and transfer learning. Additionally, there are other subfields of machine learning, such as deep learning, which focuses on neural networks with many layers, and natural language processing (NLP), which focuses on processing and understanding human language.
# 
# ![image.png](attachment:image.png)
# 
# We can also classify the ML algorithums as:
# 
# <img src="ML-image/ML-claasification.png"  width="900" height="380" />
# 
# 
# **NOTE:** Labeled data has both the input and output parameters in a completely machine-readable pattern, but requires a lot of human labor to label the data, to begin with. Unlabeled data only has one or none of the parameters in a machine-readable form. This negates the need for human labor but requires more complex solutions.

# %% [markdown]
# ## Use of the different algorithm
# 
# These are the following examples:
# 
# <img src="ML-image/Uses-ML.png" width="950" height="370" />

# %% [markdown]
# ## 1. Supervised Machine Learning
# 
# - Supervised learning is the types of machine learning in which machines are trained using well "labelled" training data, and on basis of that data, machines predict the output. The labelled data means some input data is already tagged with the correct output.
# - In supervised learning, the training data provided to the machines work as the supervisor that teaches the machines to predict the output correctly. It applies the same concept as a student learns in the supervision of the teacher.
# - Supervised learning is a process of providing input data as well as correct output data to the machine learning model. The aim of a supervised learning algorithm is to find a mapping function to map the input variable(x) with the output variable(y).
# - In the real-world, supervised learning can be used for Risk Assessment, Image classification, Fraud Detection, spam filtering, etc.
# 
# ### Example
# 
# Suppose you are building a spam email classifier that can automatically label incoming emails as spam or not spam. You would start by collecting a dataset of labeled emails, where each email is assigned a label of either "spam" or "not spam". You could then train a supervised learning algorithm, such as logistic regression or random forests, on this labeled dataset to learn a mapping function from the input email features (such as sender, subject, and body text) to the output label (spam or not spam). Once the model is trained, you can use it to predict the label of new, unseen emails.
# 
# ### How Supervised Learning Works?
# 
# In supervised learning, models are trained using labelled dataset, where the model learns about each type of data. Once the training process is completed, the model is tested on the basis of test data (a subset of the training set), and then it predicts the output.
# 
# <img src="ML-image/Model.regresion.png" width="700" height="350" />
# 
# or more specific example:
# 
# <img src="ML-image/Supervised-process.png" width="700" height="350" />
# 
# ### Steps Involved in Supervised Learning
# 
# - First Determine the type of training dataset
# - Collect/Gather the labelled training data.
# - Split the training dataset into 
#   - training dataset, 
#   - test dataset, and 
#   - validation dataset.
# - Determine the input features of the training dataset, which should have enough knowledge so that the model can accurately predict the output.
# - Determine the suitable algorithm for the model, such as support vector machine, decision tree, etc.
# - Execute the algorithm on the training dataset. Sometimes we need validation sets as the control parameters, which are the subset of training datasets.
# - Evaluate the accuracy of the model by providing the test set. If the model predicts the correct output, which means our model is accurate.
# 
# ### Types of supervised Machine learning Algorithms
# 
# Supervised learning can be further divided into two types of problems:
# 
# <img src="ML-image/super-alg.png" width="850" height="250" />
# 
# 1. **Regression:** Regression algorithms are used when the target variable is continuous. They predict the value of the target variable based on the input features. Regression algorithms are used if there is a relationship between the input variable and the output variable. It is used for the prediction of continuous variables, such as Weather forecasting, Market Trends, etc.
#     - linear regression, 
#     - polynomial regression, and 
#     - support vector regression.
# 2. **Classification:** Classification algorithms are used when the target variable is categorical. They predict the class of the target variable based on the input features. Classification algorithms are used to predict discrete values, such as a binary (Yes-No, Male-Female, True-false, etc.) outcome or a categorical label.
#     - logistic regression, 
#     - decision trees, 
#     - random forests, and 
#     - support vector machines.
# 3. **Ensemble methods:** Ensemble methods combine multiple models to improve the accuracy of predictions. 
#     - bagging, 
#     - boosting, and 
#     - stacking.
# 4. **Neural networks:** Neural networks are a type of machine learning model that is inspired by the structure and function of the human brain. They consist of layers of interconnected nodes that can learn complex patterns in data. 
#     - feedforward neural networks, 
#     - convolutional neural networks, and 
#     - recurrent neural networks.
# 5. **Naive Bayes:** Naive Bayes is a simple probabilistic algorithm that is used for classification tasks. It is based on Bayes' theorem and assumes that the input features are independent of each other.
# 6. **Support vector machines:** Support vector machines are a type of algorithm that can be used for both regression and classification tasks. They find the best boundary that separates the data into different classes or predicts the target variable based on the input features.
# 
# ### Advantage and disadvantage of Supervvised learning
# 
# | Advantage | Disadvantage |
# |-----------|--------------|
# | With the help of supervised learning, the model can predict the output on the basis of prior experiences. | Supervised learning models are not suitable for handling the complex tasks. |
# | In supervised learning, we can have an exact idea about the classes of objects. | Supervised learning cannot predict the correct output if the test data is different from the training dataset. |
# | Supervised learning model helps us to solve various real-world problems such as fraud detection, spam filtering, etc. | Training required lots of computation times. |
# || In supervised learning, we need enough knowledge about the classes of object. |
# 

# %% [markdown]
# ## 2. Unsupervised Machine Learning
# 
# Unsupervised learning is another type of machine learning in which the computer is trained on unlabeled data, meaning the data does not have any pre-existing labels or targets. The goal of unsupervised learning is to find hidden patterns or structure in the data without the guidance of a labeled dataset.
# 
# ### Example
# 
# Suppose you are working with a dataset of customer purchase histories for a retail store, and you want to identify groups or clusters of customers who have similar purchase patterns. Since the dataset does not have any pre-defined labels or categories, you would use an unsupervised learning algorithm, such as k-means clustering or hierarchical clustering, to group the customers based on their purchase histories. The algorithm would analyze the input features (such as purchase frequency, amount spent, and product categories) to discover patterns or similarities in the data, and then group the customers into different clusters based on these patterns. Once the clustering is complete, you can use it to identify different customer segments and tailor marketing strategies accordingly.
# 
# ### How Unsupervised Learning works?
# 
# As the name suggests, unsupervised learning is a machine learning technique in which models are not supervised using training dataset. Instead, models itself find the hidden patterns and insights from the given data. It can be compared to learning which takes place in the human brain while learning new things. Unlike supervised learning, where the data is labeled with a target variable, unsupervised learning algorithms work on unlabeled data.
# 
# The goal of unsupervised learning is to find the underlying structure of dataset, group that data according to similarities, and represent that dataset in a compressed format.
# 
# <img src="ML-image/unsuper-process-png.png" width="800" height="400" />
# 
# ([Image credit](https://www.diegocalvo.es/en/learning-non-supervised/))
# 
# ### Steps Involved in Supervised Learning
# 
# The basic steps in unsupervised learning are as follows:
# 
# - **Data preparation:** The first step is to prepare the data for analysis. This involves cleaning the data, handling missing values, and scaling the features as needed.
# - **Choosing an algorithm:** Next, you choose an unsupervised learning algorithm that is appropriate for your data and task. There are several types of unsupervised learning algorithms, including clustering, dimensionality reduction, and association rule mining.
# - **Training the model:** The next step is to train the unsupervised learning model on the data. During training, the model will analyze the patterns and structure in the data to identify relationships and groupings.
# - **Model evaluation:** Unlike supervised learning, there is no clear metric for evaluating the performance of unsupervised learning algorithms. Instead, you must manually inspect the results to determine if they make sense and align with your domain knowledge.
# - **Applying the model:** Once you have trained and evaluated the model, you can apply it to new, unseen data to make predictions or uncover hidden patterns.
# 
# In summary, unsupervised learning works by analyzing the structure and patterns in unlabeled data to find meaningful relationships and groupings. It can be a powerful tool for discovering hidden insights in data, but it requires careful attention to data quality and model selection to achieve good performance.
# 
# ### Types of Unsupervised Learning Algorithm
# 
# There are several types of unsupervised learning algorithms, each with their own unique approach to discovering patterns and structure in unlabeled data. Some of the most common types of unsupervised learning algorithms are:
# 
# <img src="ML-image/unsuper-alg2.png" width="850" height="250" />
# 
# 1. **Clustering algorithms:** Clustering algorithms group similar data points together based on their similarity or distance. 
#     - k-means clustering, 
#     - hierarchical clustering, and 
#     - density-based clustering.
# 2. **Dimensionality reduction algorithms:** Dimensionality reduction algorithms reduce the number of features in the data while retaining as much useful information as possible. 
#     - principal component analysis (PCA), 
#     - t-SNE, and 
#     - autoencoders.
# 3. **Association rule mining algorithms:** Association rule mining algorithms identify relationships and dependencies between variables in the data. 
#     - Apriori and 
#     - FP-Growth.
# 4. **Generative algorithms:** Generative algorithms learn the underlying distribution of the data and can generate new samples that are similar to the original data.      
#     - Gaussian mixture models and 
#     - variational autoencoders.
# 5. **Anomaly detection algorithms:** Anomaly detection algorithms identify data points that are significantly different from the rest of the data. 
#     - isolation forest and 
#     - one-class SVM.
# 
# Each of these unsupervised learning algorithms has its own strengths and weaknesses and can be applied to a wide range of tasks, including data clustering, outlier detection, feature engineering, and data visualization. The choice of algorithm will depend on the specific problem you are trying to solve and the characteristics of your data.
# 
# ### Advantages & Disadvantages of Unsupervised Learning
# 
# | Advantages | Disadvantages |
# |------------|---------------|
# | Unsupervised learning is used for more complex tasks as compared to supervised learning because, in unsupervised learning, we don't have labeled input data. | Unsupervised learning is intrinsically more difficult than supervised learning as it does not have corresponding output. |
# | Unsupervised learning is preferable as it is easy to get unlabeled data in comparison to labeled data. | The result of the unsupervised learning algorithm might be less accurate as input data is not labeled, and algorithms do not know the exact output in advance. |
# 
# ### Difference between Supervised and Unsupervised Learning
# 
# | Supervised Learning	| Unsupervised Learning |
# |-----------------------|-----------------------|
# | Supervised learning algorithms are trained using labeled data. |	Unsupervised learning algorithms are trained using unlabeled data. |
# | Supervised learning model takes direct feedback to check if it is predicting correct output or not.	| Unsupervised learning model does not take any feedback. |
# | Supervised learning model predicts the output.	| Unsupervised learning model finds the hidden patterns in data. |
# | In supervised learning, input data is provided to the model along with the output.	| In unsupervised learning, only input data is provided to the model. |
# | The goal of supervised learning is to train the model so that it can predict the output when it is given new data.	| The goal of unsupervised learning is to find the hidden patterns and useful insights from the unknown dataset. |
# | Supervised learning needs supervision to train the model.	| Unsupervised learning does not need any supervision to train the model. |
# | Supervised learning can be categorized in Classification and Regression problems.	| Unsupervised Learning can be classified in Clustering and Associations problems. |
# | Supervised learning can be used for those cases where we know the input as well as corresponding outputs.	| Unsupervised learning can be used for those cases where we have only input data and no corresponding output data. |
# | Supervised learning model produces an accurate result.	| Unsupervised learning model may give less accurate result as compared to supervised learning. |
# | Supervised learning is not close to true Artificial intelligence as in this, we first train the model for each data, and then only it can predict the correct output.	| Unsupervised learning is more close to the true Artificial Intelligence as it learns similarly as a child learns daily routine things by his experiences. |
# | It includes various algorithms such as Linear Regression, Logistic Regression, Support Vector Machine, Multi-class Classification, Decision tree, Bayesian Logic, etc.	| It includes various algorithms such as Clustering, KNN, and Apriori algorithm.| 

# %% [markdown]
# ## 3. Reinforcement Learning
# Reinforcement Learning (RL) is a type of machine learning algorithm that allows an agent to learn through trial and error by interacting with an environment. The agent receives feedback in the form of rewards or penalties for the actions it takes in the environment. The goal of the agent is to learn a policy that maximizes the total reward it receives over time.
# 
# In RL, the agent learns by exploring the environment, taking actions, receiving rewards, and updating its policy based on the feedback it receives. The agent’s policy is a mapping from states to actions that maximizes the expected future reward.
# 
# ### Some Examples for Reinforcement Learning
# 
# Imagine you are training an agent to play a game of chess. The agent interacts with the game board and makes moves based on the current state of the game. The goal is for the agent to learn to make moves that lead to a win, while avoiding moves that lead to a loss.
# 
# In reinforcement learning, the agent receives a reward signal based on the outcome of each move it makes. For example, the agent might receive a positive reward for making a winning move, or a negative reward for making a losing move. The agent uses this feedback to adjust its strategy and make better moves in the future.
# 
# To train the agent, you would use a reinforcement learning algorithm such as Q-learning or policy gradient methods. These algorithms learn to optimize the agent's behavior by maximizing the expected cumulative reward over time. As the agent interacts with the game board and receives feedback, it gradually learns which moves are more likely to lead to a win and adjusts its strategy accordingly.
# 
# Over time, the agent should become better and better at playing chess, as it learns from its experiences and improves its decision-making. Reinforcement learning has been successfully applied to a wide range of problems, including robotics, game playing, and autonomous driving.
# 
# ### Components in RL
# There are several components in RL:
# 
# 1. **The Environment:** The environment is the external world with which the agent interacts. The agent receives observations of the environment, takes actions based on those observations, and receives rewards or penalties as feedback.
# 
# 2. **The Agent:** The agent is the entity that interacts with the environment. It selects actions based on its current policy, which is updated based on the rewards it receives.
# 
# 3. **Reward Function:** The reward function defines the goal of the agent. It maps states and actions to rewards or penalties. The agent tries to maximize the cumulative reward it receives over time.
# 
# 4. **Policy:** The policy is the mapping from states to actions. The agent learns the policy through trial and error, and updates it based on the rewards it receives.
# 
# 5. **Value Function:** The value function estimates the expected cumulative reward that the agent will receive if it follows a particular policy.
# 
# RL is a powerful approach that has been successfully applied to a wide range of applications, including robotics, game playing, recommendation systems, and autonomous driving.

# %% [markdown]
# ## Machine learning life cycle
# 
# Machine learning life cycle is a cyclic process to build an efficient machine learning project. The main purpose of the life cycle is to find a solution to the problem or project. Machine learning life cycle involves seven major steps, which are given below:
# 
# 1. **Gathering Data:** In this step, we need to identify the different data sources, as data can be collected from various sources such as files, database, internet, or mobile devices. This step includes the below tasks:
# 
#     - Identify various data sources
#     - Collect data
#     - Integrate the data obtained from different sources
# 
# 2. **Data preparation:**
# 
#     - **Data exploration:** It is used to understand the nature of data that we have to work with. We need to understand the characteristics, format, and quality of data. A better understanding of data leads to an effective outcome. In this, we find Correlations, general trends, and outliers.
#     - **Data pre-processing:** Now the next step is preprocessing of data for its analysis.
# 
# 3. **Data Wrangling:** Data wrangling is the process of cleaning and converting raw data into a useable format. It is not necessary that data we have collected is always of our use as some of the data may not be useful. In real-world applications, collected data may have various issues, including:
# 
#     - Missing Values
#     - Duplicate data
#     - Invalid data
#     - Noise
# 
#     So, we use various filtering techniques to clean the data.
# 
# 4. **Analyse Data:** Now the cleaned and prepared data is passed on to the analysis step. This step involves:
# 
#     - Selection of analytical techniques
#     - Building models
#     - Review the result
#     
#     The aim of this step is to build a machine learning model to analyze the data using various analytical techniques and review the outcome. It starts with the determination of the type of the problems, where we select the machine learning techniques such as Classification, Regression, Cluster analysis, Association, etc. then build the model using prepared data, and evaluate the model. Hence, in this step, we take the data and use machine learning algorithms to build the model.
# 
# 5. **Train the model:** Now the next step is to train the model, in this step we train our model to improve its performance for better outcome of the problem. We use datasets to train the model using various machine learning algorithms. Training a model is required so that it can understand the various patterns, rules, and, features.
# 
# 6. **Test the model:** Once our machine learning model has been trained on a given dataset, then we test the model. In this step, we check for the accuracy of our model by providing a test dataset to it. Testing the model determines the percentage accuracy of the model as per the requirement of project or problem.
# 
# 7. **Deployment:** The last step of machine learning life cycle is deployment, where we deploy the model in the real-world system. If the above-prepared model is producing an accurate result as per our requirement with acceptable speed, then we deploy the model in the real system. But before deploying the project, we will check whether it is improving its performance using available data or not. The deployment phase is similar to making the final report for a project.
# 
# <img src="https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-life-cycle.png"  width="450" height="450" />
# 
# 
# ### Framework for develping ML models
# 
# The framework for ML algorithm development can be divided into five integrated stages:
# - **Problem and opportunity identification:** A good ML project starts with the ability of the organization to define the problem clearly. Domain knowledge is very important at this stage of the project. Problem definition or opportunity identification will be a major challenge for many companies who do not have capabilities to ask right questions.
# - **Data Collection & Feature Extraction:** Once the problem is defined clearly, the project team should identify and collect the relevant data. This is an iternative processess since 'relevant data' may not be known in advance in many analytics projects. The existence of ERP systems will be very useful at this stage. In addition to the data available within the organization, they have to collect data from external sources. The data needs to be integrated to create a data lake. Quality of the data is major impediment for successful ML model development. 
# - **data pre-processing:** Anecdotal evidence suggests that data preparation and data processing from a significant proportion of any analytics project. This would include data cleaning and data imputation and the creation of additional variables such as interaction variables and dummy variables. 
# - **ML model building:** ML model building is an iterative process that aims to find the best model. Several analytical tools and solution procedures will be used to find the best ML model. To avoid overfitting, it is important to create sevral training and validation datasets.
# - **Communication and development of the data analysis:** The primary objective of machine learning is to come up with actionable items that can be deployed. The communication of the ML algorithm output to the top managment and clients plays a crucial role. Innovative data visualization techniques may be used in this stage. Development of the model may involve developing software solutions and products, such as recommender engine.
# 

# %% [markdown]
# # Introduction to Data
# 
# Data is a crucial component in the field of Machine Learning. It refers to the set of observations or measurements that can be used to train a machine-learning model. The quality and quantity of data available for training and testing play a significant role in determining the performance of a machine-learning model. Data can be in various forms such as numerical, categorical, or time-series data, and can come from various sources such as databases, spreadsheets, or APIs.
# 
# ### Properties of Data
# 
# - **Volume:** Scale of Data. With the growing world population and technology at exposure, huge data is being generated each and every millisecond.
# - **Variety:** Different forms of data – healthcare, images, videos, audio clippings.
# - **Velocity:** Rate of data streaming and generation.
# - **Value:** Meaningfulness of data in terms of information that researchers can infer from it.
# - **Veracity:** Certainty and correctness in data we are working on.
# - **Viability:** The ability of data to be used and integrated into different systems and processes.
# - **Security:** The measures taken to protect data from unauthorized access or manipulation.
# - **Accessibility:** The ease of obtaining and utilizing data for decision-making purposes.
# - **Integrity:** The accuracy and completeness of data over its entire lifecycle.
# - **Usability:** The ease of use and interpretability of data for end-users.
# 
# ### Categorries of dataset
# 
# Data is typically divided into two types: 
# 
# 1. Labeled data
# 2. Unlabeled data
# 
# ### Need of Dataset
# In building ML applications, datasets are divided into two parts:
# 
# - **Training dataset:**  The part of data we use to train our model. This is the data that your model actually sees(both input and output) and learns from.
# - **Validation Data:** The part of data that is used to do a frequent evaluation of the model, fit on the training dataset along with improving involved hyperparameters (initially set parameters before the model begins learning). This data plays its part when the model is actually training.
# - **Test Dataset:** Once our model is completely trained, testing data provides an unbiased evaluation. When we feed in the inputs of Testing data, our model will predict some values(without seeing actual output). After prediction, we evaluate our model by comparing it with the actual output present in the testing data. This is how we evaluate and see how much our model has learned from the experiences feed in as training data, set at the time of training.
# 
# <img src="https://static.javatpoint.com/tutorial/machine-learning/images/how-to-get-datasets-for-machine-learning.png"  width="500" height="500" />
# 
# (Image credit: https://www.javatpoint.com/how-to-get-datasets-for-machine-learning)
# 
# **Example:**
# 
# Imagine you’re working for a car manufacturing company and you want to build a model that can predict the fuel efficiency of a car based on the weight and the engine size. In this case, the target variable (or label) is the fuel efficiency, and the features (or input variables) are the weight and engine size. You will collect data from different car models, with corresponding weight and engine size, and their fuel efficiency. This data is labeled and it’s in the form of (weight,engine size,fuel efficiency) for each car. After having your data ready, you will then split it into two sets: training set and testing set, the training set will be used to train the model and the testing set will be used to evaluate the performance of the model. Preprocessing could be needed for example, to fill missing values or handle outliers that might affect your model accuracy.
# 
# ### Types of data in datasets
# - **Numerical data:** Such as house price, temperature, etc.
# - **Categorical data:** Such as Yes/No, True/False, Blue/green, etc.
# - **Ordinal data:** These data are similar to categorical data but can be measured on the basis of comparison.
# 
# ### Popular sources for datasets
# 
# 1. [Kaggle Datasets](https://www.kaggle.com/datasets)
# 2. [Datasets via AWS](https://registry.opendata.aws/) and [AWS Data Exchange](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE)
# 3. [Google's Dataset Search Engine](https://datasetsearch.research.google.com/https://datasetsearch.research.google.com/)
# 4. [Microsoft Datasets](https://msropendata.com/)
# 5. [Awesome Public Dataset Collection](https://github.com/awesomedata/awesome-public-datasets)
# 6. [Scikit-learn dataset](https://scikit-learn.org/stable/datasets.html#datasets)

# %% [markdown]
# ![image.png](attachment:image.png)
# 
# (Image via [Abdul Rahid](https://www.slideshare.net/awahid/big-data-and-machine-learning-for-businesses))

# %% [markdown]
# ## References
# 
# - https://github.com/arunsinp/Machine-Learning/
# - https://github.com/arunsinp/Machine-Learning/blob/main/ML-Fundamental/0.1-ML-algorithms.ipynb
# - https://github.com/arunsinp/Machine-Learning/tree/main
# - https://www.diegocalvo.es/en/learning-non-supervised/
# - https://www.coursera.org/learn/machine-learning (One of the best)


