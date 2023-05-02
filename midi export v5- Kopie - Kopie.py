# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QWidget, QLabel, QFormLayout, QLineEdit
import time
from mido import MidiFile, MidiTrack, Message, MetaMessage, tick2second
from music21 import stream, note, chord, tempo, midi, instrument, interval, scale

def deep_list_to_tuple(l):
    if isinstance(l, list):
        return tuple(deep_list_to_tuple(x) for x in l)
    elif hasattr(l, '__iter__') and not isinstance(l, str):
        return tuple(l)
    else:
        return l

def get_unique_values(input_data):
    unique_set = set()

    for item in input_data:
        if isinstance(item, list):
            item = frozenset(item)
        unique_set.add(item)

    return unique_set




def hashable(item):
    try:
        hash(item)
        return True
    except TypeError:
        return False



def save_midi_from_notes(notes, output_path, bpm=120):
    stream_output = stream.Stream()

    # Fügen Sie das gewünschte Tempo hinzu
    stream_output.append(tempo.MetronomeMark(number=bpm))

    # Fügen Sie die Noten und Akkorde dem Stream hinzu
    for element in notes:
        if isinstance(element, tuple) and len(element) == 3:  # Note, Akkord und Instrument
            note_element = note.Note(element[0], quarterLength=0.5) #Wenn Sie möchten, dass die Melodie langsamer ist, können Sie den Wert von quarterLength erhöhen.
            note_element.lyrics.append(element[1])
            stream_output.append(instrument.fromString(element[2]))
            stream_output.append(note_element)
        elif isinstance(element, str):  # Einzelne Note
            note_element = note.Note(element, quarterLength=0.5)
            stream_output.append(note_element)

    # Konvertieren Sie den Stream in eine MIDI-Datei und speichern Sie ihn
    midi_file = midi.translate.streamToMidiFile(stream_output)
    midi_file.open(output_path, 'wb')
    midi_file.write()
    midi_file.close()






folder_path = r"C:\Users\ManUt\source\repos\Python magenta music KI maker midi v1\models\modell auswahlen und midi export"

try:
    # Convert to absolute path and remove relative elements
    folder_path = os.path.abspath(folder_path)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
except OSError as e:
    print(f"Error creating directory {folder_path}: {e}")

class MainWindow(QMainWindow):
    input_data = None  # Definieren Sie input_data als Klassenattribut
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MIDI Melody Generator")
        self.setFixedSize(800, 400)

        layout = QVBoxLayout()

        self.load_data_button = QPushButton("Load Training Data")
        self.load_data_button.clicked.connect(self.load_training_data)
        layout.addWidget(self.load_data_button)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_button)

        form_layout = QFormLayout()

        self.sequence_length_edit = QLineEdit()
        self.sequence_length_edit.setPlaceholderText("100")
        form_layout.addRow(QLabel("Sequence Length:"), self.sequence_length_edit)

        self.epochs_edit = QLineEdit()
        self.epochs_edit.setPlaceholderText("100")
        form_layout.addRow(QLabel("Number of Epochs:"), self.epochs_edit)

        self.batch_size_edit = QLineEdit()
        self.batch_size_edit.setPlaceholderText("64")
        form_layout.addRow(QLabel("Batch Size:"), self.batch_size_edit)

        layout.addLayout(form_layout)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        self.generate_button = QPushButton("Generate Melody")
        self.generate_button.clicked.connect(self.generate_melody)
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_training_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data", "", "NPY Files (*.npy);;All Files (*)", options=options)
        if file_path:
            self.input_data = np.load(file_path, allow_pickle=True)
            self.input_data = deep_list_to_tuple(self.input_data)
            MainWindow.input_data = self.input_data  # Setze das Klassenattribut
            self.input_data = [tuple(x) if isinstance(x, list) else x for x in self.input_data]
            self.input_data = [tuple(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x for x in self.input_data]
            self.train_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Training data loaded successfully.")
        print(self.input_data)

         


    def load_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if file_path:
            self.model_save_path = file_path
            self.generate_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Model loaded successfully.")
        print(load_model)

    def train_model(self):
        sequence_length = int(self.sequence_length_edit.text() or "100")
        num_epochs = int(self.epochs_edit.text() or "100")
        batch_size = int(self.batch_size_edit.text() or "64")

        # Convert lists to tuples
        input_data_as_tuples = deep_list_to_tuple(self.input_data)

        unique_values = len(set(input_data_as_tuples))
        unique_values = len(get_unique_values(input_data_as_tuples))
        note_to_int = {note: num for num, note in enumerate(get_unique_values(input_data_as_tuples))}
        int_to_note = {num: note for num, note in enumerate(get_unique_values(input_data_as_tuples))}
        print(note_to_int)
        print(list(input_data_as_tuples))




        # Bereite die Eingabe- und Ausgabedaten für das Netzwerk vor
        network_input = []
        network_output = []

        for i in range(0, len(self.input_data) - sequence_length, 1):
            sequence_in = self.input_data[i : i + sequence_length]
            sequence_out = self.input_data[i + sequence_length]
            network_input.append([note_to_int[tuple(note) if isinstance(note, list) else note] for note in sequence_in])
            network_output.append(note_to_int[tuple(sequence_out) if isinstance(sequence_out, list) else sequence_out])

        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(unique_values)
        network_output = tf.keras.utils.to_categorical(network_output)

        model = Sequential()
        model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(unique_values))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        model.fit(network_input, network_output, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)

        self.model_save_path = "trained_model.h5"
        model.save(self.model_save_path)
        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Model trained and saved successfully.")

    def harmonize_melody(self, notes, chords):
        harmonized_notes = []
        for i, note_element in enumerate(notes):
            if isinstance(note_element, tuple):
                note_obj = note.Note(note_element[0])
                chord_name = note_element[1]
                instrument_name = note_element[2]
                chord_obj = chord.Chord(chord_name)
                harmony_notes = chord_obj.closedPosition(forceOctave=note_obj.octave)
                harmony_notes.insert(0, note_obj)
                harmonized_notes.append((harmony_notes, chord_name, instrument_name))
            elif isinstance(note_element, str):
                note_obj = note.Note(note_element)
                harmonized_notes.append(note_obj)
        return harmonized_notes

    def generate_melody(self, output_path=None, num_steps=360, temperature=1.0):
        generated_melody = []
        for event_dict in self.input_data:
            event_type = event_dict['type']
            # Rest of the code
        if MainWindow.input_data is None:
            QMessageBox.warning(self, "Error", "Training data not loaded.")
            return

        if output_path is None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Generated Melody", "", "MIDI Files (*.mid);;All Files (*)", options=options)
            if not file_name:
                return

            output_path = os.path.join(folder_path, file_name if file_name.endswith(".mid") else file_name + ".mid")

        model = load_model(self.model_save_path)

        sequence_length = int(self.sequence_length_edit.text() or "100")
        output_length = 360

        unique_set = set()

        for event_tuple in self.input_data:
            event_type = event_dict['type']
            for key, value in event_dict.items():
                if key != 'type':
                    unique_set.add((event_type, key, value))

        unique_values = len(unique_set)
        
        # Definieren der initial_sequence
        initial_sequence = self.input_data[:sequence_length]

   


        for i in range(num_steps):  # updated
            input_data = np.reshape(initial_sequence, (1, len(initial_sequence), 1)).astype(np.float32)
            input_data = input_data / float(unique_values)
            prediction = model.predict(input_data, verbose=0)
            prediction = np.power(prediction, 1/temperature)  # updated
            prediction = prediction / np.sum(prediction)  # updated
            index = np.random.choice(len(prediction[0]), p=prediction[0])  # updated



        note_to_int = {note: num for num, note in enumerate(set(input_data_as_tuples))}
        int_to_note = {num: note for num, note in enumerate(set(input_data_as_tuples))}
     

        initial_sequence = self.input_data[:sequence_length]
        generated_sequence = []

        # Funktion zum Erzeugen des Instruments basierend auf dem Quintenzirkel
        def get_instrument_for_chord(chord):
            # Sie können die Logik anpassen, um das Instrument basierend auf dem Quintenzirkel auszuwählen
            if chord == 'Cmaj':
                return 'Piano'
            elif chord == 'Gmaj':
                return 'Violin'
            elif chord == 'Dmaj':
                return 'Flute'
            # Fügen Sie weitere Bedingungen für andere Akkorde hinzu
            else:
                return 'Piano'  # Standardinstrument


        for i in range(output_length):
            input_data = np.reshape(initial_sequence, (1, len(initial_sequence), 1)).astype(np.float32)
            input_data = input_data / float(unique_values)
            prediction = model.predict(input_data, verbose=0)
            index = np.argmax(prediction)
            note_or_chord = int_to_note[index]

            if isinstance(note_or_chord, tuple):
                note = note_or_chord[0]  # extract the first element if the note is a tuple
                chord = note_or_chord[1]
                instrument_name = get_instrument_for_chord(chord)
                generated_sequence.append((note, chord, instrument_name))
            elif isinstance(note_or_chord, str):  # Einzelne Note
                generated_sequence.append(note_or_chord)

            initial_sequence = np.append(initial_sequence, note_to_int[note_or_chord])
            initial_sequence = initial_sequence[1:]

        harmonized_sequence = self.harmonize_melody(generated_sequence, int_to_note)
        save_midi_from_notes(harmonized_sequence, output_path)



        QMessageBox.information(self, "Success", "Melody generated and saved as 'generated_melody.mid'.")
        print(self.input_data)
        print("Generated and harmonized sequence:", generated_sequence)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

input("Press Enter to exit...")
