import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import soundfile as sf

import time

class DataExtractor:
    def __init__(self, n_mfcc = 20):
        self.n_mfcc = n_mfcc

    def load_data(self, y, sr):
        self.y = y
        self.sr = sr
        self.feature_extract()

    def load_file(self, filename):
        self.y, self.sr = librosa.load(filename)
        self.feature_extract()

    def feature_extract(self):
        # Tempo information
        self.tempo = librosa.feature.tempo(y=self.y, sr=self.sr).round()

        # Separate harmonic and percussive components, Tonnetz
        self.y_harmonic, self.y_percussive = librosa.effects.hpss(self.y)
        self.tonnetz = librosa.feature.tonnetz(y=self.y, sr=self.sr)

        # Mathematical features
        features_list = {}

        features_list['tempo'] = [self.tempo.min(), self.tempo.mean(), self.tempo.max(), self.tempo.var()]
        features_list['y_harmoic'] = [self.y_harmonic.min(), self.y_harmonic.mean(), self.y_harmonic.max(), self.y_harmonic.var()]
        features_list['y_percussive'] = [self.y_percussive.min(), self.y_percussive.mean(), self.y_percussive.max(), self.y_percussive.var()]
        features_list['tonnetz'] = [self.tonnetz.min(), self.tonnetz.mean(), self.tonnetz.max(), self.tonnetz.var()]

        # Other Sound features
        cstft=librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        features_list['cstft'] = [cstft.min(), cstft.mean(), cstft.max(), cstft.var()]

        srms=librosa.feature.rms(y=self.y)
        features_list['srms'] = [srms.min(), srms.mean(), srms.max(), srms.var()]

        specband=librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        features_list['specband'] = [specband.min(), specband.mean(), specband.max(), specband.var()]

        speccent=librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        features_list['speccent'] = [speccent.min(), speccent.mean(), speccent.max(), speccent.var()]

        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        features_list['rolloff'] = [rolloff.min(), rolloff.mean(), rolloff.max(), rolloff.var()]

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.y)
        features_list['zero_crossing_rate'] = [zero_crossing_rate.min(), zero_crossing_rate.mean(), zero_crossing_rate.max(), zero_crossing_rate.var()]

        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc= self.n_mfcc)
        for i in range(self.n_mfcc):
            features_list[f'mfcc_{i}'] = [mfcc[i].min(), mfcc[i].mean(), mfcc[i].max(), mfcc[i].var()]

        self.features_df = pd.DataFrame(self.features_list).transpose()
        self.features_df.columns = ['min', 'mean', 'max', 'var']

    def get_data(self, data_print = False):
        # print the data  
        if(data_print):
            self.print_features()

        np_data = self.features_df.to_numpy()
        np_data = np_data.reshape((120, 1))

        return (np_data)

    def print_features(self):
        print(f"Tempo: {self.tempo}")

        print(self.features_df)

    def plot_waveform(self):
        # Plot
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y= self.y, sr=self.sr)
        plt.title("Waveform")
        plt.show()

    def plot_tonnetz(self):
        pass

    def plot_Harmonic_Percussion(self):
        # Plot
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.subplots() #Creates the Axis
        ax2 = ax1.twinx()    #Creates twin axis

        librosa.display.waveshow(self.y_harmonic, sr=self.sr, color='r', ax= ax1)
        librosa.display.waveshow(self.y_percussive, sr=self.sr, color='b', ax = ax2)
        plt.title("Harmonic and Percussive Component")
        plt.show()

if __name__ == "__main__":
    DataExtrac = DataExtractor()

    print(time.time())
    DataExtrac.load_file('')
    print(time.time())
