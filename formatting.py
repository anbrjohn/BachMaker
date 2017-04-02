#! /usr/bin/env python3
### CSV --> new format (generate training data)

import numpy as np
import subprocess # Only used to convert mid files to csv
gran = 32 # Granularity (smallest note duration captured)

# Initial formatting from raw string to list of commands in 
# order of timestep (originally was in order of voice)
def do_format(text):
    formatted = []
    for line in text:
        line = line.split(", ")
        line[1] = int(line[1])
        if line[2].lower() == "note_on_c" or line[2].lower() == "note_off_c":
            formatted += [line]
    formatted = sorted(formatted, key=lambda i:i[1])
    return formatted


# Makes note values relative to the first one
def transpose(data, offset=40):
    start = int(data[0][4]) - offset # Zero reserved for silence
    for line in data:
        line[4] = str(int(line[4]) - start)  
    return data


# Changes timestamp to relative durations
def timing(data, metronome, granularity=gran): 
    last = 0
    for line in data:
        last = str(line[1])
        line[1] = line[1] / metronome #Normalized (1.0 = one quarter note)
        line[1] = round(line[1] * (np.log2(granularity) - 1), 0)
    return data


# Further organizes format, sets note_off command to pitch value of 0
def trim(data):
    output = []
    for line in data:
        voice = int(line[0])
        timing = int(line[1])
        note = int(line[4])
        on_off = line[2]
        if on_off.lower() == "note_on_c":
            note = int(line[4])
        else: #Note_off_c
            note = 0
        output.append([timing, voice, note])
    output.sort()
    return output


# Instead of saying "turn this note N on at timestamp 2 and off at 5,
# it just says [0, N, N, N, N, 0, 0, ...]
def expand(data, number_of_voices, end_buffer=10):
    voices = number_of_voices - 1
    start = data[0][0]
    stop = data[-1][0] 
    data = data[::-1] # Reverse it for popping
    # Initialize array of size: Incremental timesteps x Number of voices
    timesteps = np.zeros(((stop - start), voices))
    all_voices = np.zeros(voices)
    time, voice, pitch = data.pop()
    for i in range(start, stop):
        while time == i:
            all_voices[voice - 2] = pitch
            try:
                time, voice, pitch = data.pop()
            except: # If no data left to pop
                time = "skip" # To break out of while loop
        timesteps[i - start] = all_voices
        # Buffer at end to help signal end of song
    buffer = np.zeros((end_buffer, voices))
    timesteps = np.vstack((timesteps, buffer))
    return timesteps

# Combine all the steps together to convert
# midi or csv file into our training format
def encode(filename):
    if filename[-4:] == ".mid":
        midiname = filename[:-4] + ".csv"
        command = "midicsv " + filename + " > " + midiname
        with open(midiname) as f:
            text = f.readlines()
    else: #eg: csv or txt extension
        with open(filename) as f:
            text = f.readlines()
    header = text[0].split(", ")
    number_of_voices = int(header[4])
    metronome = int(header[5])
    f1 = do_format(text)
    f2 = transpose(f1, offset=50)
    f3 = timing(f2, metronome)
    f4 = trim(f3)
    f5 = expand(f4, number_of_voices)
    return f5


### New format --> CSV (generate predictions from model)


# Removes repeat information for timesteps
def collapse(data):
    data = data.T
    change_log = []
    voice_num = 2
    for voice in data:
        i = 0
        # Find first note for a track that is not 0
        while i < len(voice) - 1 and voice[i] == 0: 
            i += 1
        change_log.append((i, voice_num, voice[i])) #Timestep, voice, and note
        for time in range(i, len(voice)):
            note = round(voice[time])
            # If current note is different than the last one
            if note != change_log[-1][2]: 
                change_log.append((time, voice_num, note))
        voice_num += 1
    return change_log

    
# Redoes some formatting and adds in note_off commands at pitch transitions
def un_organize(data, metronome=480, granularity=gran):
    time_factor = metronome / (np.log2(granularity) - 2)
    new = []
    prev_voice = 0
    for time, voice, note in data:
        time = time * time_factor
        if note != 0:
            command = 'Note_on_c'
            volume = 70
        else:
            # TODO - Could be a problem if 1st note of a voice is 0
            command = 'Note_off_c' 
            volume = 0
        # Adds note_off command for previous pitch at current timestep
        if prev_voice != voice: # If this is the first note of a new track
            prev_voice = voice
            prev_note = note
        else:
            off_line = [voice, int(time), 'Note_off_c', 0, int(prev_note), 0]
            new.append(off_line)
        line = [voice, int(time), command, 0, int(note), volume]
        new.append(line)
        prev_note = note
    return new
    

# Converts back into the string format for a csv file
def undo_format(data, metronome=480):
    number_of_voices = data[-1][0]
    last_timestep = max([x[1] for x in data]) + 2
    line1 = "0, 0, Header, 1, "+str(number_of_voices)+", "+str(metronome)+"\n"
    line2 = "1, 0, Start_track\n"
    line3 = "1, " + str(last_timestep+5) + ", End_track\n"
    line4 = "2, 0, Start_track\n"
    formatted = [line1, line2, line3, line4]
    data = sorted(data)
    last = 2
    indecies = []
    for i in range(len(data)):
        voice = data[i][0]
        if voice != last:
            indecies += [i-3]
            last = voice
    for line in data:
        line[0] = str(line[0]) # Voice
        line[1] = str(line[1]) # Time
        line[3] = str(line[3]) # Instrument
        line[4] = str(line[4]) # Note
        line[5] = str(line[5]) # Volume
        line = ", ".join(line)
        formatted += [line + "\n"]
    # Add markers for start/stop of tracks
    voice = 2
    for point in indecies:
        end = str(voice) + ", " + str(last_timestep) + ", End_track\n"
        voice += 1
        start = str(voice) + ", 0, Start_track\n"
        insert_point = point + (voice * 2) + 1
        formatted.insert(insert_point, end)
        formatted.insert(insert_point+1, start)
    end = str(voice) + ", " + str(last_timestep) + ", End_track\n"
    formatted.append(end)
    formatted.append("0, 0, End_of_file\n")
    return formatted


# Puts all these steps together
def decode(data):
    u3 = collapse(data)
    u2 = un_organize(u3)
    u1 = undo_format(u2)    
    return u1

# To convert into midi
def make_csv(filename, data): 
    with open(filename, "w") as f:
        for line in data:
            f.write(line)
    print("Saved as", filename)
    
# eg: new_format = encode(filename)
#     original_format = decode(new_format)
#     make_csv("output.csv", original_format)
