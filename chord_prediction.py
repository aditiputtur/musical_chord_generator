import pretty_midi
import joblib #for saving prediction model as pickle file for future reference
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def extract_melody_from_midi(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        if len(midi_data.instruments) < 1:
            print(f"Error processing {midi_file}: No instruments found")
            return None
        
        #first track is melody, second track is chords/harmony usually
        melody = midi_data.instruments[0]
        melody_notes = []
        for note in melody.notes:
            pitch = (note.pitch % 12) + 1  #chromatic scale conversion (1-12)
            melody_notes.append(pitch)
        return melody_notes
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None

def extract_chords_from_track(instrument):
    chords = []
    current_chord = []
    current_start_time = None

    for note in instrument.notes:
        if current_start_time is None or note.start != current_start_time:
            if current_chord:
                chords.append(current_chord)
            current_chord = [note.pitch]
            current_start_time = note.start
        else:
            current_chord.append(note.pitch)
    
    if current_chord:
        chords.append(current_chord)
    
    return chords

def extract_chords_from_midi(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        if len(midi_data.instruments) < 2:
            print(f"Error processing {midi_file}: No chord track found")
            return None
        
        #first track is melody, second track is chords/harmony usually
        chords = extract_chords_from_track(midi_data.instruments[1])  
        return chords
        
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None
    

def create_sequences(melody, chords, seq_length):
    melody_sequences = []
    chord_labels = []
    
    for i in range(0, min(len(melody), len(chords)) - seq_length):
        melody_seq = melody[i:i + seq_length]
        chord_seq = chords[i]  #group corresponding chords
        melody_sequences.append(melody_seq)
        chord_labels.append(chord_seq)
    
    return melody_sequences, chord_labels

#folder containing midi files
midi_folder = 'MIDI'

all_melody_sequences = []
all_chord_labels = []

sequence_length = 8

#process midi files in the folder
for midi_file in os.listdir(midi_folder):
    #confirm it ends with .mid
    if midi_file.endswith('.mid'):
        midi_path = os.path.join(midi_folder, midi_file)
        
        #extract melody and chords
        melody_notes = extract_melody_from_midi(midi_path)
        chord_notes = extract_chords_from_midi(midi_path)
        
        #check if both melody and chords are extracted successfully
        if melody_notes is not None and chord_notes is not None:
            melody_sequences, chord_labels = create_sequences(melody_notes, chord_notes, sequence_length)
            all_melody_sequences.extend(melody_sequences)
            all_chord_labels.extend(chord_labels)

#check for enough data to train on
if len(all_melody_sequences) == 0 or len(all_chord_labels) == 0:
    raise ValueError("No valid data to train on. Check the MIDI files or extraction process.")

def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        # Pad with zeros (or another value) if the sequence is shorter than max_length
        padded_seq = seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq
        padded_sequences.append(padded_seq[:max_length])  # Truncate if longer than max_length
    return padded_sequences

#after you collect all_chord_labels, find the maximum length for padding
max_chord_length = max(len(chord) for chord in all_chord_labels)

#pad chord labels to ensure uniform length
all_chord_labels_padded = pad_sequences(all_chord_labels, max_chord_length)

X = np.array(all_melody_sequences)
y = np.array(all_chord_labels_padded) 

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize and fit decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'chord_prediction_model.pkl')
