# Next-word-predictor
NLP model using LSTM layers to predict the next word based on previous 3 words.


This project involves developing a neural network model to predict the next word in a sequence of text. The implementation is in Python using the Keras deep learning library. Below is a detailed technical description of the project:

Key Components:
Data Loading and Preprocessing:

Tokenizer: The text data is tokenized using Keras' Tokenizer, which converts text into a sequence of integers.
Padding: The sequences are padded to ensure uniform input length for the neural network.
Model Architecture:

Sequential Model: A sequential model is used, consisting of embedding, LSTM (Long Short-Term Memory), and dense layers.
Embedding Layer: Converts integer sequences to dense vectors of fixed size.
LSTM Layers: Two LSTM layers are used to capture the sequential dependencies in the text data.
Dense Layer: A fully connected dense layer with softmax activation is used for the final output, which predicts the probability distribution over the vocabulary.
Training the Model:

Compilation: The model is compiled using the categorical cross-entropy loss function and the Adam optimizer.
Early Stopping and Model Checkpoint: Early stopping is used to monitor the validation loss and save the best model based on the lowest validation loss.
Evaluation:

Loss Monitoring: The training process monitors the loss, and the best model is saved during the training process.
Prediction:

Prediction Function: A function Predict_Next_Words is defined to take a sequence of words and predict the next word using the trained model.
User Interaction: The model predicts the next word based on user input in a loop until the user decides to stop by entering "0".
Saving and Loading Model and Tokenizer:

The trained model and the tokenizer are saved to disk (next_words.h5 and token.pkl respectively) and later loaded for making predictions.
