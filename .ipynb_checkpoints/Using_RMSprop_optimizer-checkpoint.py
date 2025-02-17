import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

# read the train .csv file and just take two useful col TEXT & CLASS
def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    text = data['TEXT']
    labels = data['CLASS']
    return text, labels

# read the train .xlsv file and just take two useful col TEXT & CLASS
def read_xlsv_file(file_path):
    data = pd.read_excel(file_path)
    text = data['TEXT']
    labels = data['CLASS']
    return text, labels

# Feature extraction and model training
def train_model_RMSprop(text, labels, num_epochs):
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    # Padding sequences to have the same length
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, train_size=0.8, random_state=42)

    # Model architecture
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0001) 
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Model training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=32, epochs=num_epochs, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    evaluation = model.evaluate(X_test, y_test)
    
    print("Test set loss:", evaluation[0])
    print("Training set accuracy:", evaluation[1])

    return model, tokenizer, history

# Testing the trained model and calculating accuracy
def test_model(model, tokenizer, text, labels):
    # Tokenize the text data
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])

    # Making predictions using the model
    predictions = model.predict(padded_sequences)
    y_pred = (predictions > 0.5).astype(int).flatten()
    accuracy = accuracy_score(labels, y_pred)
    print("Testing set accuracy:", accuracy)

    # Create a table to compare the actual CLASS col and the predicted CLASS col
    df = pd.DataFrame({'Text': text, 'Predicted Label': y_pred, 'Actual Label': labels})
    print(df)


def plot_loss(history):
    # Get the loss values from training history
    loss = history.history['loss']

    # Create a list of epochs
    epochs = range(1, len(loss) + 1)

    # Plot the loss values
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    train_text, train_labels = read_csv_file("./Topic_3_Data/Topic1-youtube_spam_train.csv")

    num_epochs = 100  # can change

    model, tokenizer, history = train_model_RMSprop(train_text, train_labels, num_epochs)
    
    plot_loss(history)

    test_text, test_labels = read_xlsv_file("./Topic_3_Data/Topic1-youtube_spam_test.xlsx")

    test_model(model, tokenizer, test_text, test_labels)


if __name__ == '__main__':
    main()