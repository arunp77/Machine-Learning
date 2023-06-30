# %% [markdown]
# # Deep Learning
# 
# ![image.png](attachment:image.png)
# 
# [Image reference](https://www.ait.de/en/deep-learning/)

# %% [markdown]
# ## Definition
# Deep learning is a subfield of machine learning that involves the use of artificial neural networks with multiple layers to model and solve complex problems. 
# 
# Artificial Neural Networks (ANNs) are a type of deep learning model that is designed to simulate the way the human brain works. ANNs consist of interconnected nodes (neurons) that transmit signals and perform computations on input data to produce output values. These networks can be used for a wide range of applications, including image recognition, speech recognition, natural language processing, and more.
# 
# ## Classification
# 
# ANNs can be classified into several categories based on their structure and function, including:
# 
# 1. **Feedforward neural networks:** These are neural networks that have a series of interconnected layers, where the output of each layer serves as the input for the next layer.
# 
# 2. **Convolutional neural networks (CNNs):** CNNs are primarily used for image and video processing tasks. They consist of layers of convolutional filters that can identify patterns in the input images.
# 
# 3. **Recurrent neural networks (RNNs):** These are neural networks that are well-suited to sequence analysis tasks, such as natural language processing or speech recognition. They use a type of neural network layer called a recurrent layer that can maintain an internal state and process inputs one at a time.
# 
# 4. **Generative adversarial networks (GANs):** GANs are a type of network that can generate new data based on a set of input data. They consist of two networks: a generator network that creates new data, and a discriminator network that evaluates the quality of the generated data.
# 
# 5. **Autoencoders:** Autoencoders are designed to learn a compressed representation of input data. They consist of an encoder network that compresses the input data into a low-dimensional representation, and a decoder network that can reconstruct the original input data from the compressed representation.

# %% [markdown]
# ![image.png](attachment:image.png)
# 
# https://www.sciencedirect.com/science/article/pii/S2352914822000612

# %% [markdown]
# ![image.png](attachment:image.png)
# 
# (simple form: https://developer.ibm.com/articles/cc-machine-learning-deep-learning-architectures/ good link)

# %% [markdown]
# ## Deep Learning framework
# 
# In the context of deep learning, a framework is a software library or tool that provides a set of APIs (application programming interfaces) and abstractions to simplify the development of deep neural networks. Frameworks typically include pre-implemented building blocks for common neural network layers, such as convolutional, recurrent, and fully connected layers, as well as optimization algorithms and training routines.
# 
# There are many popular frameworks for deep learning, some of which include:
# 
# - **TensorFlow:** Developed by Google, it is an open-source software library for dataflow and differentiable programming across a range of tasks.
# 
# - **PyTorch:** Developed by Facebook, it is an open-source machine learning framework used for applications such as computer vision and natural language processing.
# 
# - **Keras:** An open-source neural network library written in Python, it runs on top of other deep learning frameworks such as TensorFlow and Theano.
# 
# - **Caffe:** Developed by Berkeley Vision and Learning Center, it is a deep learning framework that specializes in image recognition.
# 
# - **Theano:** Another popular open-source numerical computation library, it is used for deep learning and other mathematical computations.
# 
# - **MXNet:** An open-source deep learning framework that is highly scalable and supports a range of programming languages including Python, R, and Julia.
# 
# - **Chainer:** A Python-based, open-source deep learning framework that is highly flexible and allows for dynamic computation graphs.
# 
# These frameworks provide a range of features and tools for developing and training deep neural networks, making it easier for developers and researchers to experiment with different architectures and optimize their models for specific tasks.

# %% [markdown]
# ## Difference between Deep learning and Machine learning
# 
# |                         | Machine Learning | Deep Learning |
# |-------------------------|------------------|---------------|
# | Approach        | Requires structure data | Does not require structure data |
# | Human intervention | Requires human intervention for mistakes | Does not require human intervention for mistakes |
# | Hardware | Can function on CPU | Requires GPU / Significant computing power |
# | Time | Takes seconds to hours | Takes weeks |
# | Uses | Forecasting, predicting and other simple applications |  More complex applications like autonomus vehicles |

# %% [markdown]
# ## Practical uses and applications of deep learning 
# 
# Here are a few examples of the practical uses and applications of deep learning across various domains:
# 
# 1. **Image and Object Recognition:** Deep learning has significantly improved image classification, object detection, and recognition tasks. Examples include:
# 
#     - Autonomous vehicles use deep learning algorithms to recognize and interpret objects in real-time, enabling them to navigate and make informed driving decisions.
# 
#     - Facial recognition systems, such as those used for biometric identification or security purposes, employ deep learning techniques to accurately recognize and verify individuals' faces.
# 
# 2. **Natural Language Processing (NLP):** Deep learning has greatly advanced natural language processing tasks, allowing computers to understand and generate human language. Examples include:
# 
#     - Chatbots and virtual assistants utilize deep learning models to understand user queries, provide relevant responses, and engage in conversational interactions.
# 
#     - Machine translation systems, like Google Translate, employ deep learning to improve the accuracy and fluency of translations between different languages.
# 
#     - Sentiment analysis algorithms analyze text data from social media, customer reviews, or surveys, using deep learning models to determine the sentiment expressed in the text (e.g., positive, negative, neutral).
# 
# 3. **Medical Diagnostics:** Deep learning has shown promising results in medical imaging analysis and disease diagnostics. Examples include:
# 
#     - Deep learning models can detect anomalies and classify medical images, such as X-rays, MRIs, or CT scans, assisting radiologists in diagnosing diseases like cancer or identifying abnormalities.
# 
#     - Deep learning algorithms have been used to predict the risk of certain diseases, such as diabetic retinopathy or cardiovascular diseases, based on patient data, enabling early detection and intervention.
# 
# 4. **Recommendation Systems:** Deep learning models are used in recommendation systems to personalize and improve user experiences. Examples include:
# 
#     - Streaming platforms like Netflix and Spotify employ deep learning algorithms to recommend personalized movies, TV shows, or music based on a user's viewing or listening history.
# 
#     - E-commerce platforms, such as Amazon, utilize deep learning-based recommendation systems to suggest products based on a user's browsing history, purchase behavior, and similar user profiles.
# 
# 5. **Speech Recognition:** Deep learning has significantly enhanced speech recognition accuracy and enabled voice-controlled applications. Examples include:
# 
#     - Voice assistants like Apple's Siri, Amazon's Alexa, or Google Assistant utilize deep learning models to accurately recognize and respond to spoken commands and queries.
# 
#     - Transcription services employ deep learning algorithms to convert spoken language into written text, facilitating tasks such as transcription services, voice search, or closed captioning.
# 
# These are just a few examples showcasing the broad range of applications where deep learning has made significant advancements. The versatility and effectiveness of deep learning models have enabled breakthroughs in many fields, revolutionizing industries and improving various aspects of our lives.

# %%



