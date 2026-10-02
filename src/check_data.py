import os
import glob
from music21 import converter, note, chord

def check_data():
    midi_dir = os.path.join("data", "midi")
    # Find all .mid and .midi files
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi"))
    
    print(f"Found {len(midi_files)} MIDI files in {midi_dir}.")
    
    success_count = 0
    total_notes_and_chords = 0
    
    print("Loading and parsing MIDI files (this may take a minute)...")
    for i, file in enumerate(midi_files):
        try:
            # Parse the MIDI file into a music21 Stream
            midi_parsed = converter.parse(file)
            
            # Extract all Notes and Chords from the parsed stream
            # recurse() searches through all tracks and measures
            elements = midi_parsed.recurse().getElementsByClass([note.Note, chord.Chord])
            
            count = len(elements)
            total_notes_and_chords += count
            success_count += 1
            
            # Print progress every 10 files
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(midi_files)} files...")
                
        except Exception as e:
            print(f"Failed to load {os.path.basename(file)}: {e}")
            
    print("-" * 40)
    print(f"Successfully loaded {success_count} out of {len(midi_files)} files.")
    print(f"Total notes and chords extracted: {total_notes_and_chords}")

if __name__ == "__main__":
    check_data()
