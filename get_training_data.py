#! /usr/bin/env python3
###Getting data

from formatting import *
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import wget
import time
import os


# Location of download links
webpage = "http://www.bachcentral.com/midiindexcomplete.html"
# What all the download links begin with
file_prefix = "http://www.bachcentral.com/"


# Web scraper. Downloads every file ending in .mid from a webpage
def scrape(webpage, extension=".mid"):
    # Get all the files of a given extension from a webpage
    http = httplib2.Http()
    status, response = http.request(webpage)
    files = []
    for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
        if link.has_attr('href'):
            linkname = link['href']
            if linkname[-len(extension):] == extension:
                files += [linkname]
    return files    
     
    
def download(files, delay=0, prefix=file_prefix, convert=False):
    # Given a list of files, downloads each, optionally also converts to csv
    total = len(files)
    i = 1
    for file in files:
        filename = prefix+file
        wget.download(filename)
        print("Downloaded", i, "files out of", total)
        i += 1
        # In case the webpage doesn't like many rapid downloads
        time.sleep(delay) 


def get_midi_info():
    # Get stats on all the midi files - tempo and number of voices for each
    all_files = os.listdir()
    # All midi csv files in the directory
    midis = [file for file in all_files if file[-4:] in [".csv", ".txt"]]
    info = []
    for file in midis:
        with open(file) as f:
            try:
                text = f.readlines()
                line = text[0]
                line = line.split(", ")
                tempo = int(line[-1])
                voices = int(line[-2])
                #print(file)
            except:
                print("Skipped", file)
        info.append([voices - 1, tempo, file]) 
        # Substract one because first voice never has notes, only commands
        # like adjusting speed and loudness
    return info
            

# Process all of files together
def cull_midis(voice_list):
    # Input list of acceptable number of voices, eg: [1,2,3]
    if type(voice_list) == int:
        voice_list = [voice_list]
    info = get_midi_info()
    file_info = [x for x in info if x[0] in voice_list]
    files = [x[-1] for x in file_info]
    print(len(files), "files found") 
    # Process all files
    all_data = []
    for file in files:
        data = encode(file)
        # Add empty lines for any unused voices
        # so array has uniform shape
        width = max(voice_list)
        while data.shape[1] < width:
            empty_voice = np.zeros([len(data),1])
            data = np.hstack([data,empty_voice])
        for line in data:
            all_data.append(line)
    print("Done")
    return all_data    
    
    
# eg: all_data = cull_midis([1,2,3,4])


### Convert data to x and y for training


def get_xy(data, seqlen=3, y_type=1, save=False):
    """Prepare data. y_type can be float, 1-hot, or 4-hot"""
    voices = all_data[0].shape[0]
    dataX = [] # Represenation of consecutive notes (of length seqlen)
    dataY = [] # Representation of the following notes
    length = len(data[0]) # Number of features for each element
    for i in range(len(data) - seqlen):
        x_data = np.array(data[i:i+seqlen])
        x_data = x_data.flatten()
        dataX.append(x_data)
        dataY.append(data[i+seqlen])
    x = np.reshape(dataX, (len(dataX), seqlen*length, 1))
    x = x/100 #NN prefers floats
    dataY[-1] = dataY[-2][:] #Last element is empty and has wrong length
    y = np.array(dataY)
    # Desired y format
    y_type = str(y_type)[0].lower()
    if y_type == "f": # Not preffered, susceptible to slight changes
        y /= 100
    elif y_type == "1": # Preferred format
        notes = int(max(np.array(data).flatten())) + 1
        print("Possible different notes:", notes) # Length of 1-hot vector for each voice
        size = len(x)
        all_lines = np.zeros([size, notes * voices])
        for timestep in range(size):
            # A 1-hot vector for each voice, concatenated together
            concat_lines = np.array([])
            for voice in y[timestep]:
                line = np.zeros([notes])
                line[int(voice)] = 1
                concat_lines = np.hstack([concat_lines, line])
            all_lines[timestep] = concat_lines
        y = all_lines
    elif y_type == "4": # Not preffered, can't distinguish unique voices
        y = to_categorical(y) #Convert to "four-hot" encoding
    if save:
        np.save("X_DATA", x)
        np.save("Y_DATA", y)
        print("Saved as X_DATA and Y_DATA")
    return x, y

# eg: x, y = get_xy(all_data, seqlen=3, save=True)
