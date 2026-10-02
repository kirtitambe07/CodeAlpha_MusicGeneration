import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes():
    """ Parse all MIDI files and extract notes and chords """
    midi_dir = os.path.join("data", "midi")
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi"))
    
    notes = []
    
    print(f"Parsing {len(midi_files)} MIDI files...")
    for i, file in enumerate(midi_files):
        try:
            midi = converter.parse(file)
            notes_to_parse = None
            
            # Flatten the stream so that notes from all tracks/hands are in time order
            notes_to_parse = midi.flatten().notes
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    # For notes, extract the pitch name (e.g. C4)
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # For chords, convert to integer format and join with dots (e.g. 4.7.11)
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    
            if (i + 1) % 10 == 0:
                print(f"Parsed {i + 1} files")
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            
    # Save the full ordered note list
    with open(os.path.join("data", "notes.pkl"), "wb") as filepath:
        pickle.dump(notes, filepath)
        
    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    
    # Get all unique pitch names (vocabulary)
    pitches = sorted(set(notes))
    
    # Create a mapping from pitch string to an integer
    note_to_int = dict((note, number) for number, note in enumerate(pitches))
    
    # Save the mapping
    with open(os.path.join("data", "mappings.pkl"), "wb") as filepath:
        pickle.dump(note_to_int, filepath)
        
    network_input = []
    network_output = []
    
    # Create input sequences of length 100, and the corresponding target output
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # Convert lists to numpy arrays (samples, 100) and (samples,)
    network_input = np.array(network_input)
    network_output = np.array(network_output)
    
    # Save arrays to data/ folder so we don't need to re-parse every time
    np.save(os.path.join("data", "network_input.npy"), network_input)
    np.save(os.path.join("data", "network_output.npy"), network_output)
    
    return network_input, network_output, n_patterns

if __name__ == "__main__":
    notes = get_notes()
    print(f"Total notes and chords extracted: {len(notes)}")
    n_vocab = len(set(notes))
    print(f"Vocabulary size (Unique notes/chords): {n_vocab}")
    
    network_input, network_output, n_patterns = prepare_sequences(notes, n_vocab)
    print(f"Number of training sequences generated: {n_patterns}")
