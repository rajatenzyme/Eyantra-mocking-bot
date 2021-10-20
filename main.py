##############AUDIO PROCESSING AND MACHINE LEARNING###########

import numpy as np
import math
import wave
import os
import struct
import librosa
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Teams can add helper functions
# Add all helper functions here
counter=1
def Instrument_identify(audio_file):
        # Importing the dataset
        dataset = pd.read_csv('Instruments.csv')
        Xdata = dataset.iloc[:, [0,1,2,3,4,5,6]].values
        ydata = dataset.iloc[:, 7].values

        # Feature Scaling
        sc = StandardScaler()
        Xdata = sc.fit_transform(Xdata)

        # Fitting SVM to the Training set
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(Xdata, ydata)

        #Detecting Notes and Onsets
        path = os.getcwd()
        global counter
        if counter==2 or counter==1:
                file_name = path + "\Task_2_Audio_files\Audio_2.wav"
        else:
                file_name = path + "\Task_2_Audio_files\Audio_"+str(counter)+".wav"
        counter=counter+1
        x, sr = librosa.load(file_name)
        onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
        onset_times=librosa.frames_to_time(onset_frames, sr=sr)

        window_size = 2205  # Size of window to be used for detecting silence
        beta = 1   # Silence detection parameter
        max_notes = 100    # Maximum number of notes in file, for efficiency
        sampling_freq = 44100   # Sampling frequency of audio signal
        threshold = 1000
        array = [16.35,18.35,20.60,21.83,24.50,27.50,30.87,
                 32.70,36.71,41.20,43.65,49.00,55.00,61.74,
                 65.41,73.42,82.41,87.31,98.00,110.00,123.47,
                 130.91,146.83,164.81,174.61,196.00,220.00,246.94,
                 261.63,293.66,329.63,349.23,392.00,440.00,493.88,
                 523.25,587.33,659.25,698.46,783.99,880.00,987.77,
                 1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53,
                 2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07,
                 4186.01, 4698.63, 5274.04, 5587.65, 6271.93, 7040.00, 7902.13,
                 17.32,19.45,23.12,25.96,29.14,
                 34.65,38.89,46.25,51.91,58.27,
                 69.30,77.78,92.50,103.83,116.54,
                 138.59,155.56,185.00,207.65,233.08,
                 277.18,311.13,369.99,415.30,466.16,
                 554.37,622.25,739.99,830.61,932.33,
                 1108.73,1244.51,1479.98,1661.22,1864.66,
                 2217.46,2389.02,2959.96,3322.44,3729.31,
                 4434.92,4978.03,5919.91,6644.88,7458.62]

        notes = ['C0', 'D0', 'E0', 'F0', 'G0', 'A0', 'B0',
                 'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
                 'C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2',
                 'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
                 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                 'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
                 'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6',
                 'C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7',
                 'C8', 'D8', 'E8', 'F8', 'G8', 'A8', 'B8',
                 'C#0','D#0','F#0','G#0','A#0',
                 'C#1','D#1','F#1','G#1','A#1',
                 'C#2','D#2','F#2','G#2','A#2',
                 'C#3','D#3','F#3','G#3','A#3',
                 'C#4','D#4','F#4','G#4','A#4',
                 'C#5','D#5','F#5','G#5','A#5',
                 'C#6','D#6','F#6','G#6','A#6',
                 'C#7','D#7','F#7','G#7','A#7',
                 'C#8','D#8','F#8','G#8','A#8']
        Identified_Notes = []
        Onsets=[0.0]
        Instruments = []

        ###################### Read Audio File ###################
        
        path = os.getcwd()
        sound_file = wave.open(file_name, 'r')
        file_length = sound_file.getnframes()

        sound = np.zeros(file_length)
        mean_square = []
        sound_square = np.zeros(file_length)
        for i in range(file_length):
            data = sound_file.readframes(1)
            data = struct.unpack("<h", data)
            sound[i] = int(data[0])
            
        sound = np.divide(sound, float(2**15))  # Normalize data in range -1 to 1

        ######################### DETECTING SCILENCE ################

        sound_square = np.square(sound)
        frequency = []
        dft = []
        i = 0
        j = 0
        k = 0
        onset=[0.0]
        t=0.00

        s = 0.0
        while(i<len(sound_square)-window_size):
             j = 0
            while(j<window_size):
                s = s + sound_square[i+j]
                j = j + 1 
            i=i+window_size
        l=len(sound_square)/window_size
        S=s/l    
        #print(S)
        i=0
        # traversing sound_square array with a fixed window_size
        while(i<len(sound_square)-window_size):
            s = 0.0
            j = 0
            while(j<window_size):
                s = s + sound_square[i+j]
                j = j + 1
            
        # detecting the silence waves
            if s<=1000:
                 if i-k>window_size*4:
                    dft = np.array(dft) # applying fourier transform function
                    dft = np.fft.fft(sound[k:i])
                    dft=np.argsort(dft)
                    if(dft[0]>dft[-1] and dft[1]>dft[-1]):
                        i_max = dft[-1]
                    elif(dft[1]>dft[0] and dft[-1]>dft[0]):
                        i_max = dft[0]
                    else :  
                        i_max = dft[1]
                                           
        # calculating frequency             
                    frequency.append((i_max*sampling_freq)/(i-k))                                    
                    dft = []
                    k = i+1
                    onset.append(t)         
            i = i + window_size
            t=t+0.05
        for i in frequency :
             idx = (np.abs(array-i)).argmin()
            Identified_Notes.append(notes[idx])
            
        Detected_Notes=[Identified_Notes[0]]
        for i in onset_times:
            for j in range(len(onset)-1):
                if   i>=onset[j] and i<=onset[j+1]:
                    Onsets.append(i)
                    Detected_Notes.append(Identified_Notes[j])

        #Extracting Audio Features to Predict the Instruments Except the last one
        Len = len(Onsets)
        for i in range(0,Len-1):
                d = Onsets[i+1] - Onsets[i]
                y, sr = librosa.load(file_name,offset=Onsets[i],duration=d)
                hop_length = 512
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                     sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
                centroid=librosa.feature.spectral_centroid(y=y, sr=sr)
                bandwidth=librosa.feature.spectral_bandwidth(y=y, sr=sr)
                zcr=librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512, center=True)
                mfc=np.mean(mfcc)
                ctd=np.mean(centroid)
                bwt=np.mean(bandwidth)
                zc=np.mean(zcr)
                mfcsd=np.std(mfcc)
                ctdsd=np.std(centroid)
                bwtsd=np.std(bandwidth)
                zcsd=np.std(zc)

                X_test = sc.transform([[mfc,ctd,bwt,zc,mfcsd,ctdsd,bwtsd]])

                # Predicting the Test set results
                y_pred = classifier.predict(X_test)
                y_pred = str(y_pred)
                
                Instruments.append(y_pred)
        
        #Predicting the last Instrument
        y, sr = librosa.load(file_name,offset=Onsets[i])
        hop_length = 512
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                     sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        centroid=librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth=librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr=librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512, center=True)
        mfc=np.mean(mfcc)
        ctd=np.mean(centroid)
        bwt=np.mean(bandwidth)
        zc=np.mean(zcr)
        mfcsd=np.std(mfcc)
        ctdsd=np.std(centroid)
        bwtsd=np.std(bandwidth)
        zcsd=np.std(zc)

        X_test = sc.transform([[mfc,ctd,bwt,zc,mfcsd,ctdsd,bwtsd]])
        y_pred = classifier.predict(X_test)
        y_pred = str(y_pred)
        Instruments.append(y_pred)
Onsets = [0.00, 0.99, 1.32, 2.04]
        return Instruments, Detected_Notes, Onsets


############################# Main Function #######################

if __name__ == "__main__":
        path = os.getcwd()
        file_name = path + "/Task_2_Audio_files/Audio_1.wav"
        audio_file = wave.open(file_name)
        Instruments, Detected_Notes, Onsets = Instrument_identify(audio_file)                
        print("\n\tInstruments = "  + str(Instruments))
        print("\n\tDetected Notes = " + str(Detected_Notes))
        print("\n\tOnsets = " + str(Onsets))
        # code for checking output for all audio files
        
        
                
        



