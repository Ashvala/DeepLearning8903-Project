import json
import librosa
import torch
import os
import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]


data_path = "./data/nsynth/"
setting = "train" # can be train, test, valid
output_dir = "./data/nsynth/"

def loadClasses():
    examples_path = data_path + "nsynth-" + setting + "/examples.json"
    with open(examples_path) as f:
        data = json.load(f)
    vals = data.values()

    #classes = [512, 401, 141, 0, 515, 2, 1, 775, 143, 418, 4]
    classVals = [v for v in vals]
    print("class length dataset", len(classVals))
    classValsDict = {}

    for v in classVals:
        inst = v[u'instrument']
        if not inst in classValsDict:
            classValsDict[inst] = []
        classValsDict[inst].append(v)

    #print(len(classValsDict))
    return classValsDict

def parseIntoSpectrogram(obj, setting=setting, n_fft=1024):
#    print("Parsing", obj["note_str"])
    hop_length = n_fft//4
    path = data_path + "nsynth-" + setting + "/audio/" + obj["note_str"] + ".wav"
    # load into librosa
    sig, sr = librosa.load(path, sr=16000)
    S = librosa.core.stft(sig, n_fft=1024, hop_length=hop_length)
    magnitude, phase = librosa.magphase(S)
    return magnitude, phase


flatten = lambda l: [item for sublist in l for item in sublist]
compose_str = lambda x : output_dir + x + setting + "-stft" + '.npy'

if __name__ == "__main__":
    classValsDict = loadClasses()
    labels_list = list()

    for classId in classValsDict:
        classVals = classValsDict[classId]        
        for i, val in enumerate(classVals):
            print(classId, i, val["pitch"], len(classValsDict))
            mag, phase  = parseIntoSpectrogram(val)
            output_array = np.concatenate((mag, phase), axis=0)
            np.save(compose_str(val['note_str']), output_array)
            labels_list.append([compose_str(val['note_str']), val['pitch']])        
        
    y = np.asarray(flatten(labels_list))
    print(y)
    np.save(compose_str("labels"), y)
