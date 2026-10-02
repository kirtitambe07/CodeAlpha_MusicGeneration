import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# Import our build_model function
from model import build_model

def train():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Train the Music Generation LSTM")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size", help="Batch size (default: 64)")
    args = parser.parse_args()
    
    print("Loading prepared data...")
    network_input = np.load(os.path.join("data", "network_input.npy"))
    network_output = np.load(os.path.join("data", "network_output.npy"))
    
    with open(os.path.join("data", "mappings.pkl"), "rb") as filepath:
        mappings = pickle.load(filepath)
        
    n_vocab = len(mappings)
    
    # The new shape is just the sequence length
    seq_length = network_input.shape[1]
    
    print(f"Building model (Sequence length: {seq_length}, Vocab: {n_vocab})...")
    model = build_model(seq_length, n_vocab)
    
    # Setup ModelCheckpoint to save the best model weights
    filepath = os.path.join("models", "best_model.keras")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    
    print(f"Starting training for {args.epochs} epochs (Batch size: {args.batch_size})...")
    # Train the model
    history = model.fit(
        network_input, 
        network_output, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        callbacks=callbacks_list
    )
    
    print("Training complete! Generating loss and accuracy curves...")
    
    # Plot and save the loss and accuracy curves
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.title('Model Accuracy During Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join("outputs", "loss.png")
    plt.savefig(plot_path)
    print(f"Loss and accuracy curves saved to: {plot_path}")

if __name__ == "__main__":
    train()
