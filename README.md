# DeepLearningProject - Exploring Deep Learning Approaches for Classifying Sentiment using the IMDB Movie Review Dataset.

SCS 3546-010 Deep Learning Project
Group: Liam Callaghan, Peter Kiedrowski and Prashant Sharma

The project consists of three notebooks. Each of these notebooks contains a link to run in Colab, which is the recommended environment.
A more detailed analysis of each model is included in the notebooks.
* [Recurrent_Autoencoder](https://github.com/peterkd/DeepLearningProject/blob/main/Recurrent%20LSTM%20Autoencoder/Recurrent_Autoencoder.ipynb) - [Open in Colab](https://colab.research.google.com/github/peterkd/DeepLearningProject/blob/main/BERT/BERT_Classifier.ipynb)
* [BERT Classifier](https://github.com/peterkd/DeepLearningProject/blob/main/BERT/BERT_Classifier.ipynb) - [Open in Colab](https://colab.research.google.com/github/peterkd/DeepLearningProject/blob/main/Recurrent%20LSTM%20Autoencoder/Recurrent_Autoencoder.ipynb)
* [GloVe_Classifier](https://github.com/peterkd/DeepLearningProject/blob/main/GloVe/Glove_Classifier.ipynb)

Sentiment analysis is a common and highly beneficial application of Deep Learning for many organizations. Sentiment analysis can be applied to social media posts, reviews and product complaints to name a few. There are numerous approaches to classify sentiment in Python. This project aims to explore just some of the available methodologies that are available in order to improve the accuracy for sentiment classification based on the TensorFlow Sentiment Analysis tutorial ([RNN](https://www.tensorflow.org/tutorials/text/text_classification_rnn), [BERT](https://www.tensorflow.org/tutorials/text/classify_text_with_bert) and [GloVe](https://www.tensorflow.org/tutorials/text/word_embeddings)) - [Open in Colab](https://colab.research.google.com/github/peterkd/DeepLearningProject/blob/main/GloVe/Glove_Classifier.ipynb): 

In this project, we focus on the most popular text dataset for Deep Learning sentiment analysis: the IMDB movie review set. Using this data, we build and train a recurrent LSTM autoencoder to compress reviews and transfer the dense lower dimensional output from the encoder layers for further training in an additional dense network for classifying sentiment. Training the sentiment classifier with frozen encoder layers yielded only modest performance in accuracy. The approach was then expanded upon to fine-tune the LSTM encoder layers with the dense neural network classifier layers resulting in a test accuracy score of 82%. 

Next, we fine-tuned Google's BERT model: "Bidirectional Encoder Representations from Transformers". Unlike other models that read text input sequentially, the Transformed encoder utilized by BERT reads the entire text at once. This allows the model to learn the context of a word based on all of its surroundings. The Encoder then passes word sequences into the model, masking a subset of the words. The model then attempts to predict the value of the masked words, based on the context of other non-masked words and uses this approach in loss calculation. It was also the longest-running approach.
This approach achieved an accuracy score of 92%.

Finally, we investigated the GloVe model. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
This approach achieved an accuracy score of 90%.

Training a encoder for a dataset is a challenging and educational exercise allowing the user to customize the encoder to a given dataset and task. However, the approach often requires large amounts of data and computational processing power to create representations of the data that generalize well to new unobserved input. The pretrained solutions available today (e.g. GloVe and BERT) offer deep architectures that generalize well to new data with little to no required fine-tuning of the weights. 
