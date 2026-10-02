# CodeAlpha AI Internship - Task 3: Music Generation with AI

## Overview
This project generates original piano music using a Long Short-Term Memory (LSTM) neural network. By training on a corpus of classical piano compositions, the AI learns the patterns, harmonies, and structures of the music to generate brand-new, entirely original sequences in the same style.

## How it Works
1. **Parsing MIDI:** The raw `.mid` files are flattened into a single timeline using the `music21` library, extracting a sequential list of Notes and Chords.
2. **Vocabulary Mapping:** The unique notes and chords are mapped to integer IDs to create a vocabulary.
3. **Training Data Preparation:** We create sequences of 100 consecutive notes to serve as the input, with the 101st note as the target output.
4. **Model Architecture:** The integers are passed through an `Embedding` layer to learn rich, multi-dimensional representations of musical meaning, followed by stacked LSTM layers that capture the long-term structural context.
5. **Generation & Temperature Sampling:** During generation, we seed the network with a random sequence and ask it to predict the next note. We use "Temperature Sampling" to scale the output probabilities, ensuring the AI is creative and doesn't get stuck in repetitive loops.
6. **MIDI Output:** The predicted IDs are decoded back into `music21` Note and Chord objects and exported as a new `.mid` file.

## Dataset
The model was trained on a dataset of **48 Chopin MIDI files**, resulting in approximately 58,000 extracted musical elements (notes and chords) and a vocabulary size of 397 unique elements.

## Model Architecture
The neural network is built using Keras (TensorFlow 3) and consists of 1.13 million trainable parameters:
*   **Input Layer:** Expects sequences of length 100.
*   **Embedding Layer:** (vocab_size, 128) - learns dense vector representations of notes.
*   **LSTM Layer 1:** 256 units, returns sequences.
*   **Dropout Layer:** 30% dropout to prevent overfitting.
*   **LSTM Layer 2:** 256 units.
*   **Dense Layer:** 256 units with ReLU activation.
*   **Dropout Layer:** 30% dropout.
*   **Output Layer:** Dense layer with `vocab_size` units and Softmax activation.
*   **Compilation:** `sparse_categorical_crossentropy` loss and `adam` optimizer.

## Results
The model was trained for 50 epochs using a Google Colab T4 GPU.
*   **Final Training Loss:** ~0.78
*   **Final Training Accuracy:** ~75.8%

This indicates the model successfully learned to predict the next note with high accuracy, producing stylistically coherent musical phrases.

## Folder Structure
```text
CodeAlpha_MusicGeneration/
├── data/
│   ├── midi/            # Raw Chopin MIDI files
│   ├── mappings.pkl     # Vocabulary translation dictionary
│   └── network_input.npy/network_output.npy # Processed training data
├── models/
│   └── best_model.keras # Saved weights of the best training epoch
├── outputs/
│   ├── generated_music.mid # Final generated AI music
│   ├── chopin_ai_v1.mid    # Sample generated MIDI track
│   └── chopin_ai_v1.mp3    # Audio render of the sample track
├── src/
│   ├── check_data.py    # Utility to verify MIDI files
│   ├── preprocess.py    # Data extraction and preparation
│   ├── model.py         # Model architecture definition
│   ├── train.py         # Training loop script
│   └── generate.py      # Music generation script
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation
1. Clone the repository and navigate into it.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   * Windows: `.\.venv\Scripts\activate`
   * Mac/Linux: `source .venv/bin/activate`
4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
**1. Prepare the Data:**
Extract notes from the MIDI files and build the training sequences:
```bash
python src/preprocess.py
```

**2. Train the Model:**
Train the LSTM network. The best model will be saved automatically:
```bash
python src/train.py --epochs 50 --batch-size 64
```

**3. Generate Music:**
Generate a new composition using the trained model:
```bash
python src/generate.py --length 500 --temperature 0.8
```

## Future Improvements
*   **Time and Duration:** The current model spaces all notes 0.5 beats apart. Training a dual-output model to predict both the *pitch* and the *duration* of the next note would make the music rhythmically expressive.
*   **Velocity:** Adding MIDI velocity (how hard the key is pressed) to the training data to capture the emotional dynamics of the performance.
*   **Transformer Models:** Upgrading from LSTM to an Attention-based Transformer architecture (like MusicLM) to capture even longer structural dependencies in classical pieces.
