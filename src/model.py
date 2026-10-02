import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Embedding

def build_model(seq_length, n_vocab):
    """
    Builds and compiles the LSTM Neural Network with an Embedding layer.
    """
    model = Sequential()
    
    # Keras 3 standard practice is to use an explicit Input layer
    model.add(Input(shape=(seq_length,)))
    
    # Embedding layer to learn vector representations of the notes
    model.add(Embedding(n_vocab, 128))
    
    # First LSTM layer
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    
    # Second LSTM layer (no return_sequences since it's the last LSTM)
    model.add(LSTM(256))
    
    # Fully connected layer with relu activation
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(n_vocab, activation='softmax'))
    
    # Compile the model with sparse categorical crossentropy and accuracy
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    print("Loading data to determine shapes...")
    
    network_input = np.load(os.path.join("data", "network_input.npy"))
    
    with open(os.path.join("data", "mappings.pkl"), "rb") as filepath:
        mappings = pickle.load(filepath)
        
    n_vocab = len(mappings)
    
    # The shape is (samples, 100). The sequence length is the second dimension.
    seq_length = network_input.shape[1]
    
    print(f"Sequence length: {seq_length}")
    print(f"Vocabulary size: {n_vocab}")
    print("-" * 40)
    
    # Build the model
    model = build_model(seq_length, n_vocab)
    
    # Print the network architecture
    model.summary()
