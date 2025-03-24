import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import csv

import librosa
import soundfile as sf

TIME_INTERVAL = 10  # in seconds
IGNORE_MIN = 1      # in minutes

OUTPUT_FOLDER = "processdata"
INPUT_FOLDER = "data"

class DataExtractor:
    def __init__(self, n_mfcc = 20):
        self.n_mfcc = n_mfcc

    def load_data(self, y, sr):
        self.y = y
        self.sr = sr
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

        self.features = features_list

    def get_data(self, data_print = False):
        flattened_list = []
        for key in sorted(self.features.keys()):
            flattened_list.extend(self.features[key])
        return flattened_list

def writer_process(queue, file_path):
    try:
        with open(file_path, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            while True:
                item = queue.get()
                if item is None:  # Sentinel value to stop
                    break
                writer.writerow(item)
    except Exception as e:
        print(f"Writer process error: {e}")

def data_processor(y, sr, queue):
    try:
        DataExtrac = DataExtractor()
        DataExtrac.load_data(y, sr)

        X = DataExtrac.get_data()
        queue.put(X)
    except Exception as e:
        print(f"Processor error: {e}")

def segment_mp3(input_file, input_folder = INPUT_FOLDER, output_folder= OUTPUT_FOLDER):
    output_file = input_file.split('.')[0] + '.csv'

    queue = multiprocessing.Queue()
    writer = multiprocessing.Process(target=writer_process, args=(queue, output_folder + '\\' + output_file))
    writer.start()

    try:
        # print(input_folder + "\\" + input_file)
        y, sr = librosa.load(input_folder + "\\" + input_file)  # Load with original sampling rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    interval_samples = sr * TIME_INTERVAL
    num_intervals = int(np.floor(len(y) / interval_samples))

    processes = []
    for i in range(IGNORE_MIN, num_intervals + 1):
        start_sample = i * interval_samples
        end_sample = (i + 1) * interval_samples
        interval_audio = None

        # Save the remaining part
        if(i < num_intervals):
            interval_audio = y[start_sample:end_sample]
        else:
            interval_audio = y[start_sample:]

        # Save the output
        # output_file = output_folder + '\\' + input_file.split('.')[0] + f"_{i+1}.mp3"
        # sf.write(output_file, interval_audio, sr)

        p = multiprocessing.Process(target=data_processor, args=(interval_audio, sr, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    queue.put(None)  # Sentinel to signal writer to stop
    writer.join()

    print(f"Audio file '{input_file}' split and saved to '{output_folder}' successfully.")

if __name__ == "__main__":
    genres= os.listdir(INPUT_FOLDER)
    processes = []
    
    for genre in genres:
        print("Genre:",genre)
        #args = []

        for _,_,files in os.walk(INPUT_FOLDER + "\\" + genre):
            for i,file in enumerate(files):
                # print(file, i)
                os.makedirs(OUTPUT_FOLDER + "\\" + genre, exist_ok= True)

                #segment_mp3(file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre)

                p = multiprocessing.Process(target= segment_mp3, 
                                      args= (file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            processes.clear()

    print("Segments done!")