Status:
-----

Video explanation: https://youtu.be/N6s26JI8tnI?t=9m33s

[Listen here](https://soundcloud.com/user-758753778/1epochs) to training after 1 epoch (7 minutes on laptop CPU). Converted to standard notation, [this](https://github.com/anbrjohn/BachMaker/blob/master/1e5.pdf) is what the model produced. [This](https://soundcloud.com/user-758753778/10epochs) is after 10 epochs.

Still a work-in-progress.
I hope to steadily chip away at it whenever I get tired of the homework that I actually should be doing.

Goal:
-----

Automatically generate music in the style of J.S. Bach. Specifically, train a neural network on midi files of Bach works.

Inspiration Taken from the DeepBach Project:
- https://www.youtube.com/watch?v=QiBM7-5hA6o
- https://arxiv.org/pdf/1612.01010.pdf

And from other similar projects:
- http://bachbot.com/#/?_k=evep3j
- http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/


Structure of midi files:
-----

Instead of directly encoding an audio signal, midi files act as an electronic musical score, with information on volume, pitch, and timing for different tracks. I use this [delightful program](http://www.fourmilab.ch/webtools/midicsv/) to easily convert midi files into human-readable csv files. 

eg: `$ midicsv chorale.mid > chorale.csv`

The meat of the file consists of lines like this:

```
2, 120, Note_on_c,  0, 67, 64
2, 180, Note_off_c, 0, 67, 44
3, 180, Note_on_c,  3, 72, 64
[Track Number, timestamp, command, instrument number, pitch, volume]
```

Currently, I process these into the following format:

```
array([[ 50.,   0.,   0.,   0.],
       [ 50.,   0.,   0.,   0.],
       [ 50.,   0.,  38.,  38.],
       [ 50.,   0.,  41.,  38.]])
```

Each row represents a time step. During processing, the "granularity" can be adjusted. For example, having one row for every 16th note versus one for every 64th note. Too high and the data grows in size. Too low and I risk not capturing details like trills, tuplets, ...). Each column represents a different track. The value inside corresponds to the note value (more on this in  bit), with zero representing no note (silence). Note that I don't bother keeping information on the instrumentation or exact volume. I relativize the pitch as an approximate way to transpose pieces written in different keys together, so this network can more easily learn common patterns. However, this approach technically doesn't transpose everything into the same key if we consider pieces that don't begin on the tonic. (To quote [reddit](https://www.reddit.com/r/musictheory/comments/2pv3a7/why_arent_everyone_starting_songs_with_tonics/): "If you want to be a basic bitch you start with the tonic"). My expectation (hope) is that this will not be an issue.

My Model:
-----

I scraped [this site](http://www.bachcentral.com/midiindexcomplete.html) for Bach midis, and for now only processed the ones with 4 instrument tracks or fewer (5 tracks total because the first track never is always just metadata like speed and volume changes, which I ignore). This is the majority of the files. For files with fewer than 4 tracks, I currently just return silence for the other tracks. I used a NN with 2 LSTM laters of 100-nodes each with 0.2 percent dropout, one more hidden layer of 100-nodes, and an activation layer.

I debated how to best organize my data and tried various approaches. I believe that converting all the input training data (with a sequence length of 3) to one-hot encoding for each track and doing the same for the output training data would lead to the best results, but it takes much, much longer to train. I toyed around with "4hot" encoding, where each track it marked with a 1 in a vector that represents every possible note and silence (around 100), but that doesn't capture which voice is responsible for which note, nor does it distinguish when multiple voices are playing the same note.

I settled on dividing the input data as is by 100 to feed the network as floats in the range [0.0, 1.0). I didn't like the idea of a minor change in value (eg. 0.56 to 0.57) potentially meaning the difference between a well-formed chord and dissonance, but this was my compromise. I did, however, convert the output (y) training data into a one-hot vector for each track. When generating a new file, it then uses the predict function and samples from the probability distribution (restricted to the top-n) for some variety.

As a proof of concept, I trained this model for just one epoch. The good news is that its output is well-formatted (Bach joke: well-tempered) in that it can sucessfully be converted back into a midi file. It even has chords and a somewhat complex rhythm in which the voices play well together, which I think is very promising! Still, it is a far cry from a fugue. When I have some time, I will train it longer and see what I get. So far, I have also created a MIDI after training for 10 epochs. I also made one that was generated completely at random for comparison (read: to make me feel better about my model).

