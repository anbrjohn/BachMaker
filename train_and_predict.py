### Train model and make predictions

from formatting import *
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
import subprocess
import random


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

def train(epoch_set, batches, seqlen, start_epoch=0, save=True, load_weights=False, make_music=[]):
    # Set save to number of starting epochs and load to filename to load from
    # Make music is a list of seeds
    if load_weights:
        model.load_weights(load)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    for i in range(batches):
        model.fit(x,y, batch_size=64, nb_epoch=epoch_set, verbose=1)
        start_epoch += epoch_set
        if save:
            title = str(start_epoch) + "epochs.h5"
            model.save_weights(title)
            print("Weights saves as", title)
        if len(make_music) > 0:
            for ii in range(len(make_music)):
                filename = str(start_epoch) + "e" + str(ii) + ".csv"
                compose(make_music[ii], 240, seqlen, new_tempo=240, shift_pitch=15, save=filename)
                midiname = filename[:-3] + "mid"
                command = "csvmidi " + filename + " > " + midiname
                subprocess.call(command, shell=True)
                
                
# eg: x = np.load("X_DATA.npy")
#     y = np.load("Y_DATA.npy")
#     train(25, 3, start_epoch=0, load_weights="75epochs.h5", make_music=seeds)


### Make predictions


# Narrows down notes to predict from
# For 1-hot encoding of y (not float or 4hot)
def get_notes(prediction, considered_notes=4):
    # Given probability distribution corresponding to
    # a 1hot vector for a single voice, 
    # predicts next note for each voice
    
    # considered_notes takes the top-n predictions
    # and samples based on their relative probabilites
    #　Too high and get weirder music, too low and get
    # repetitive music
    note_values = []
    for i in range(considered_notes):
        note = np.argmax(prediction)
        # Append pitch and probability
        note_values.append((note, prediction[note]))
        # Turn off so next-highest will now be returned
        prediction[note] = 0
    return note_values

# Samples from the probability distribution
def guess_note(prediction):
    note_values = []
    noteprobs = get_notes(prediction)
    prob_sum = sum([n[1] for n in noteprobs])
    probs = [n[1]/prob_sum for n in noteprobs] # Normalize probabilities
    notes = [n[0] for n in noteprobs]
    guess = np.random.choice(notes, 1, p=probs)
    guess = guess.tolist()
    return guess[0]

# Repeatedly guesses following note
def consec(seed, iterations, seqlen):
    voices = int(seed.shape[0] / seqlen)
    s = seed * 100 # Back to non-decimal pitch representations
    s = s.T.tolist()[0]
    total = []
    # Add seed to running total
    for i in range(seqlen):
        total.append(s[voices * i:voices * (i + 1)])
    next_seed = seed
    for i in range(iterations):
        # Large array corresponding to 1 hot vector for each voice concatenated together
        prediction = model.predict(np.array([next_seed]))
        notes = int(len(prediction[0]) / voices)
        new_line = np.zeros(voices)
        for voice in range(voices):
            # 1 hot vector probability array for a single voice
            voice_prediction = prediction[0][notes*voice:notes*(voice+1)]
            guess = guess_note(voice_prediction)
            new_line[voice] = guess
        total.append(new_line.tolist())        
        next_seed = next_seed[voices:].tolist()
        for guess in new_line.tolist():
            next_seed.append([guess/100])
        next_seed = np.array(next_seed)
    return np.array(total)

    
def adjust(text, pitch, speed):
    header = text[0].split()
    header[5] = str(speed) + "\n"
    header = " ".join(header)
    text[0] = header
    adjusted = final_transpose(text, pitch)
    return adjusted


# Compose piece, optionally transpose and change tempo, optionally save
def compose(seed, iterations, seqlen, new_tempo=False, shift_pitch=0, save=False):
    total = consec(seed, iterations, seqlen)
    final = decode(total)
    if new_tempo:
        adjust(final, shift_pitch, new_tempo)
    elif shift_pitch != 0:
        adjust(final, shift_pitch, int(final[0].split()[-1]))
    if save:
        make_csv(str(save), final)

# eg: compose(x[0], 50, 2, new_tempo=False, shift_pitch=0, save="sample.csv")
