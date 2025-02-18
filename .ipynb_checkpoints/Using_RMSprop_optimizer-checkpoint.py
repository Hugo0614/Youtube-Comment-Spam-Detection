import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import random
import pickle
import numpy as np

# only used once!
# nltk.download('stopwords')
# nltk.download('wordnet')

# can set other number
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# read and get data from CSV file
def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    text = data['TEXT']
    labels = data['CLASS']
    return text, labels

# read and get data from XLSX file
def read_xlsv_file(file_path):
    data = pd.read_excel(file_path)
    text = data['TEXT']
    labels = data['CLASS']
    return text, labels

# Pre-processing data
def preprocess_text(text):
    """
    Pre-processing of text, including removal of special characters, 
    conversion tolowercasee, deletion of stop words and word restoration
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_text = []
    for t in text:
        # remove the special letter
        t = re.sub(r'[^a-zA-Z]', ' ', t)
        # convert to lowercase
        t = t.lower()
        # Participle
        words = t.split()
        # Removal of stop words and morphological reduction
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        processed_text.append(' '.join(words))
    return processed_text

# 模型训练函数
def train_model_RMSprop(text, labels, num_epochs):
    text = preprocess_text(text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, train_size=0.8, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=32, epochs=num_epochs, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    evaluation = model.evaluate(X_test, y_test)

    print("Test set loss:", evaluation[0])
    print("Training set accuracy:", evaluation[1])

    return model, tokenizer, history

def test_model(model, tokenizer, text, labels):
    text = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])
    predictions = model.predict(padded_sequences)
    y_pred = (predictions > 0.5).astype(int).flatten()
    accuracy = accuracy_score(labels, y_pred)
    print("Testing set accuracy:", accuracy)
    df = pd.DataFrame({'Text': text, 'Predicted Label': y_pred, 'Actual Label': labels})
    print(df)

# Plotting training loss curves
def plot_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    train_text, train_labels = read_csv_file("./Topic_3_Data/Topic1-youtube_spam_train.csv")
    num_epochs = 100

    model, tokenizer, history = train_model_RMSprop(train_text, train_labels, num_epochs)
    plot_loss(history)

    test_text, test_labels = read_xlsv_file("./Topic_3_Data/Topic1-youtube_spam_test.xlsx")
    test_model(model, tokenizer, test_text, test_labels)

    # Save the training model
    model.save('youtube_spam_detection_model.h5')
    print("Model saved as: youtube_spam_detection_model.h5")

    # Save Tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved as tokenizer.pickle")

if __name__ == '__main__':
    main()