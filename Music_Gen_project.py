import os
import music21
import numpy as np
import tensorflow as tf
from music21 import stream, note, chord
from play import play_midi

dataSet_path = "Data/archive"  # Folder where all MIDI data is stored
model_path = "Model/music_generation_model.h5"  # Path to save/load the model

# Loading all MIDI files from the entire dataset
def load_dataset_files():
    dataset_files = []
    for root, _, files in os.walk(dataSet_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                dataset_files.append(os.path.join(root, file))
    return dataset_files


def preprocess_midi(file_path):
    try:
        midi = music21.converter.parse(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    notes = []
    pitched_elements = [el for el in midi.flat.notes if not isinstance(el, music21.percussion.PercussionChord)]
    
    # Collect notes or chords only, skipping instrument details
    for element in pitched_elements:
        if isinstance(element, music21.note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, music21.chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    
    return notes



# Creating sequences for training
def create_sequences(notes, sequence_length=100):
    pitch_names = sorted(set(item for item in notes))
    note_to_int = {note: num for num, note in enumerate(pitch_names)}

    input_sequences = []
    output_notes = []
    
    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        input_sequences.append([note_to_int[note] for note in seq_in])
        output_notes.append(note_to_int[seq_out])

    return np.array(input_sequences), np.array(output_notes), note_to_int


# Adding a Bidirectional LSTM layer and Attention mechanism
def attention_layer(inputs):
    attention = tf.keras.layers.Dense(1, activation="tanh")(inputs)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(inputs.shape[-1])(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    output_attention = tf.keras.layers.Multiply()([inputs, attention])
    return output_attention


# Creating the model with Bidirectional LSTM and Attention
def create_model(input_shape, output_dim):
    inputs = tf.keras.Input(shape=input_shape)
    lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    lstm_out = attention_layer(lstm_out)
    lstm_out = tf.keras.layers.LSTM(128)(lstm_out)
    
    dense_out = tf.keras.layers.Dense(128, activation="relu")(lstm_out)
    outputs = tf.keras.layers.Dense(output_dim, activation="softmax")(dense_out)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# Sampling with temperature
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    probabilities = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probabilities), p=probabilities)


# Generating notes using the trained model with error handling
def generate_notes(model, start_seq, num_notes, note_to_int, int_to_note, temperature=1.0):
    generated_notes = []
    current_seq = start_seq
    
    for i in range(num_notes):
        current_seq = np.reshape(current_seq, (1, current_seq.shape[0], 1))
        
        try:
            prediction = model.predict(current_seq, verbose=0)
        except Exception as e:
            print(f"Error during prediction: {e}")
            break
        
        index = sample_with_temperature(prediction[0], temperature)
        
        if index not in int_to_note:
            index = np.random.choice(list(int_to_note.keys()))
        
        note = int_to_note[index]
        generated_notes.append(note)
        
        current_seq = np.append(current_seq[0][1:], index)
        
    return generated_notes


# Convert generated notes to MIDI, with randomization of rhythm and humanization of volume
def notes_to_midi(notes, output_file="output.midi"):
    midi_stream = stream.Stream()
    
    for n in notes:
        if ('.' in n) or n.isdigit():
            midi_note = chord.Chord([int(note) for note in n.split('.')])
        else:
            midi_note = note.Note(n)
            
        midi_note.quarterLength = np.random.choice([0.25, 0.5, 1.0, 2.0])
        midi_note.volume.velocity = np.random.randint(60, 100)
        
        midi_stream.append(midi_note)
    
    midi_stream.write('midi', fp=output_file)


# Main function to run the system with enhanced functionality
def music_generation_system():
    print("Loading the complete dataset...")
    dataset_files = load_dataset_files()
    
    if not dataset_files:
        print("No MIDI files found!")
        return

    # Load and process all notes from dataset
    notes = []
    for file in dataset_files:
        notes += preprocess_midi(file)
        
    if not notes:
        print("No notes found in any file!")
        return
    
    sequence_length = 100
    input_sequences, output_notes, note_to_int = create_sequences(notes, sequence_length)
    input_sequences = np.reshape(input_sequences, (input_sequences.shape[0], sequence_length, 1))
    input_sequences = input_sequences / float(len(note_to_int))
    output_notes = tf.keras.utils.to_categorical(output_notes)

    # Model training or loading
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating and training new model...")
        model = create_model(input_sequences.shape[1:], output_notes.shape[1])
        model.fit(input_sequences, output_notes, epochs=10, batch_size=64)
        model.save(model_path)
        print(f"Model saved as {model_path}")

    # Filter options post-training
    key_filter = input("Enter key (e.g., C, Gm, D#) for the music or press Enter to skip: ").strip()
    time_signature = input("Enter time signature (e.g., 4/4, 3/4) or press Enter to skip: ").strip()
    mood = input("Enter the mood (e.g., happy, sad, suspense) or press Enter to skip: ").strip().lower()

    # Start sequence and generate notes
    start_seq = input_sequences[np.random.randint(0, len(input_sequences)-1)]
    int_to_note = {num: note for note, num in note_to_int.items()}
    generated_notes = generate_notes(model, start_seq, num_notes=50, note_to_int=note_to_int, int_to_note=int_to_note)

    output_file = "generated_music.mid"
    notes_to_midi(generated_notes, output_file)
    print(f"Generated music saved as {output_file}")
    play_midi(output_file)


if __name__ == '__main__':
    music_generation_system()
