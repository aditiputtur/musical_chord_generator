import streamlit as st
import pretty_midi
import joblib #for getting pickle file
import numpy as np
import os
import librosa

os.system("python chord_prediction.py")

#make sequences from a new melody (subset of full melody)
def create_sequences_from_melody(melody, seq_length):
    melody_sequences = []
    for i in range(0, len(melody) - seq_length):
        melody_seq = melody[i:i+seq_length]
        melody_sequences.append(melody_seq)
    return melody_sequences

#create midi file with the predicted chords
def create_midi_with_chords(new_melody_file, predicted_chords, output_file='predicted_chords.mid'):
    midi = pretty_midi.PrettyMIDI()
    melody_og = pretty_midi.PrettyMIDI(new_melody_file).instruments[0]  # Original melody track

    chord_instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    starts = []
    ends = []

    # If there's an original chord track, extract timing from it
    if len(pretty_midi.PrettyMIDI(new_melody_file).instruments) >= 2:
        for i in range(len(pretty_midi.PrettyMIDI(new_melody_file).instruments[1].notes)):
            if pretty_midi.PrettyMIDI(new_melody_file).instruments[1].notes[i].start not in starts:
                starts.append(pretty_midi.PrettyMIDI(new_melody_file).instruments[1].notes[i].start)
                ends.append(pretty_midi.PrettyMIDI(new_melody_file).instruments[1].notes[i].end)
    else:
        # If no chord track exists, use the melody's note timings to create chord timings
        for beat in pretty_midi.PrettyMIDI(new_melody_file).get_downbeats():
            if beat not in starts:
                starts.append(beat)
                ends.append(beat + 0.5)

    # Add predicted chords to the chord track
    for i, chord in enumerate(predicted_chords):
        if i < len(starts):
            for pitch in list(set(chord)):  # Remove duplicate pitches
                note = pretty_midi.Note(velocity=75, pitch=pitch + 24,
                                        start=starts[i], end=ends[i])
                if note.pitch > 24:
                    chord_instrument.notes.append(note)

    midi.instruments.append(melody_og)
    midi.instruments.append(chord_instrument)

    midi.write(output_file)
    return output_file

# Load the pre-trained model
model = joblib.load('chord_prediction_model.pkl')

# Streamlit app starts here
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title('Melody to Chord Generator')

# Upload midi file
uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid"])
midi_data = pretty_midi.PrettyMIDI(uploaded_file)

if len(midi_data.instruments) == 0:
    st.error("Please upload a MIDI file.")
elif midi_data.instruments[0].is_drum:
    melody_instrument = midi_data.instruments[1]
    audio_data = melody_instrument.fluidsynth(fs = 16000)
    st.write("Uploaded Melody:")
    st.audio(audio_data, sample_rate = 16000)
else:
    melody_instrument = midi_data.instruments[0]
    audio_data = melody_instrument.fluidsynth(fs = 16000)
    st.write("Uploaded Melody:")
    st.audio(audio_data, sample_rate = 16000)

if uploaded_file is not None:
    midi_path = os.path.join(uploaded_file.name)
    
    #extract melody notes from midi file
    new_melody_notes = extract_melody_from_midi(midi_path)
    
    #create short sequences for prediction
    sequence_length = 8  #shorter sequence
    new_melody_sequences = create_sequences_from_melody(new_melody_notes, sequence_length)
    
    #predict chords using prediction model
    X_new = np.array(new_melody_sequences)
    predicted_chords = model.predict(X_new)

    pred_chord_names = []
    for chord in predicted_chords:
        chord_notes = []
        for note in chord:
            if (note + 24) not in chord_notes:
                chord_notes.append(pretty_midi.note_number_to_name(note + 24))
        pred_chord_names.append(chord_notes[0:3])
        
        
    #display chord names on app
    st.write("Predicted Chords:")
    st.write(np.array(pred_chord_names))
    
    #generate new midi file with chords
    output_file = create_midi_with_chords(midi_path, predicted_chords)
    
    #provide a download button for the new midi file
    with open(output_file, "rb") as f:
        st.download_button("Download Generated Music", f, file_name="generated_music.mid", mime="audio/midi")

    midi_data2 = pretty_midi.PrettyMIDI(output_file)
    audio_data2 = midi_data2.fluidsynth(fs = 16000)

    st.write("Uploaded Melody + Predicted Chords:")
    st.audio(audio_data2, sample_rate = 16000)