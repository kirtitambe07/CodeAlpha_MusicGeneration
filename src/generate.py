import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream

def sample_with_temperature(probabilities, temperature=1.0):
    """
    Applies temperature scaling to the probabilities to add variability to the output.
    """
    # Prevent divide by zero or negative probabilities
    probabilities = np.asarray(probabilities).astype('float64')
    log_probs = np.log(probabilities + 1e-7) / temperature
    exp_probs = np.exp(log_probs)
    scaled_probs = exp_probs / np.sum(exp_probs)
    
    # Randomly pick the next note based on the newly scaled probabilities
    return np.random.choice(len(scaled_probs), p=scaled_probs)

def generate():
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Generate music with trained LSTM")
    parser.add_argument("--length", type=int, default=500, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity/Randomness scale (0.1 to 2.0+)")
    args = parser.parse_args()
    
    print("Loading data and mapping dictionary...")
    # Load integer sequences
    network_input = np.load(os.path.join("data", "network_input.npy"))
    
    # Load mappings dictionary (which maps "C4" -> 5)
    with open(os.path.join("data", "mappings.pkl"), "rb") as filepath:
        note_to_int = pickle.load(filepath)
        
    # We need the reverse mapping to turn the AI's integer guess back into a note name!
    int_to_note = {number: note_name for note_name, number in note_to_int.items()}
    
    print("Loading trained model...")
    model = load_model(os.path.join("models", "best_model.keras"))
    
    print("Picking a random starting sequence (seed)...")
    # Pick a random 100-note sequence from our training data to "jumpstart" the AI
    start_index = np.random.randint(0, len(network_input) - 1)
    current_pattern = list(network_input[start_index])
    
    print(f"Generating {args.length} notes...")
    generated_notes = []
    
    for i in range(args.length):
        # Format the pattern for the model: shape (1, 100)
        prediction_input = np.array([current_pattern])
        
        # Ask the model for the probabilities of the next note
        probabilities = model.predict(prediction_input, verbose=0)[0]
        
        # Apply temperature sampling to pick the final ID
        next_note_id = sample_with_temperature(probabilities, args.temperature)
        
        # Convert integer ID back into a Note/Chord string
        result_string = int_to_note[next_note_id]
        generated_notes.append(result_string)
        
        # Slide the window: append the new ID and remove the oldest ID
        current_pattern.append(next_note_id)
        current_pattern = current_pattern[1:]
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{args.length} notes...")
            
    print("Converting AI output into MIDI file...")
    offset = 0
    midi_elements = []
    
    # Assign the Piano instrument
    midi_elements.append(instrument.Piano())
    
    for pattern in generated_notes:
        # Check if the pattern is a Chord (e.g., "4.7.11" or just a single integer string)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_pitch in notes_in_chord:
                new_note = note.Note(int(current_pitch))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            midi_elements.append(new_chord)
            
        # Otherwise, the pattern is a single Note (e.g., "C4")
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_elements.append(new_note)
            
        # Increase the offset to place the next note later in time
        offset += 0.5
        
    midi_stream = stream.Stream(midi_elements)
    
    output_file = os.path.join("outputs", "generated_music.mid")
    midi_stream.write('midi', fp=output_file)
    print(f"\nDone! Music successfully saved to: {output_file}")

if __name__ == "__main__":
    generate()
