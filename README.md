# DeepLearningProject - Exploring Deep Learning Approaches for Classifying Sentiment using the IMDB Movie Review Dataset.

SCS 3546-010 Deep Learning Project
Group: Liam Callaghan, Peter Kiedrowski and Prashant Sharma

The project consists of three notebooks:
* [Recurrent_Autoencoder](https://github.com/peterkd/DeepLearningProject/blob/main/Recurrent%20LSTM%20Autoencoder/Recurrent_Autoencoder.ipynb)
* [BERT Classifier](https://github.com/peterkd/DeepLearningProject/blob/main/BERT/BERT_Classifier.ipynb)
* [GloVe_Classifier](https://github.com/peterkd/DeepLearningProject/blob/main/Glove_Classifier.ipynb)

Sentiment analysis is a common and highly beneficial application of Deep Learning for many organizations. Sentiment analysis can be applied to social media posts, reviews and product complaints to name a few. There are numerous approaches to classify sentiment in Python. This project aims to explore just some of the available methodologies that are available in order to improve the accuracy for sentiment classification based on the TensorFlow Sentiment Analysis tutorial ([RNN](https://www.tensorflow.org/tutorials/text/text_classification_rnn){:target="_blank"}, [BERT](https://www.tensorflow.org/tutorials/text/classify_text_with_bert){:target="_blank"} and [GloVe](https://www.tensorflow.org/tutorials/text/word_embeddings){:target="_blank"}: 

In this project, we focus on the most popular text dataset for Deep Learning sentiment analysis: the IMDB movie reivew set. Using this data, we build and train a recurrent LSTM autoencoder to compress reviews and further extract the encoder layer for further training in an additional network for classifying sentiment. This approach achieved an accuracy score of 77.5%

Next, we fine-tune Google's BERT model: Bidirectional Encoder Representations from Transformers. Unlike other models that read text input sequentially, the Transformed encoder utilized by BERT reads the entire text at once. This allows the model to learn the context of a word based on all of its surrpoundings. The Encoder then passes word sequences into the model, masking a subset of the words. The model then attempts to predict the value of the masked words, based on the context of other non-masked words and uses this approach in loss calculation.
This approach achieved an accuracy score of 89% accuracy.

Finally, we investigated the GloVe model. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
This approach achieved an accuracy score of 89% accuracy.

